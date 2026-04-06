"""백그라운드 워커 - 이미지 생성 및 영상 조립"""

import json
import os
import glob

from db.database import SessionLocal
from db.models import Job
from jobs_queue.job_manager import update_job_progress, mark_job_failed, set_video_path
from api.deps import resolve_user_api_keys
from core.r2_storage import is_r2_enabled
from config import settings


def _update_r2_sync(job_id: str, status: str):
    """R2 동기화 상태 업데이트"""
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.r2_synced = status
            db.commit()
    finally:
        db.close()


async def generate_images_for_job(job_id: str):
    """이미지 생성 백그라운드 태스크"""
    from core.gemini_client import generate_all_images

    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return

        update_job_progress(job_id, "generating_images", 0.0, "이미지 생성 준비 중...")

        lines = json.loads(job.script_json)
        job_dir = os.path.join(settings.STORAGE_DIR, job_id)

        keys = resolve_user_api_keys(db, job.user_id)

        try:
            await generate_all_images(
                job_id=job_id,
                lines=lines,
                style=job.style,
                storage_dir=job_dir,
                progress_callback=update_job_progress,
                api_key=keys["gemini"],
            )

            update_job_progress(job_id, "preview_ready", 0.4, "이미지 생성 완료 - 미리보기 확인")

            # R2 업로드
            from core.r2_storage import upload_job_files, is_r2_enabled
            if is_r2_enabled():
                ok = await upload_job_files(job_id, "images")
                _update_r2_sync(job_id, "partial" if ok else "none")

        except Exception as e:
            mark_job_failed(job_id, f"이미지 생성 실패: {str(e)}")
    finally:
        db.close()


async def generate_clips_for_job(job_id: str):
    """AI 영상 클립 생성 백그라운드 태스크"""
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return

        update_job_progress(job_id, "generating_clips", 0.25, "AI 영상 클립 생성 준비 중...")

        lines = json.loads(job.script_json)
        job_dir = os.path.join(settings.STORAGE_DIR, job_id)
        clips_dir = os.path.join(job_dir, "clips")

        images = sorted(glob.glob(os.path.join(job_dir, "images", "img_*.png")))

        model_key = getattr(job, "video_mode", "hailuo") or "hailuo"
        keys = resolve_user_api_keys(db, job.user_id)

        try:
            if model_key == "veo_lite":
                from core.google_video import generate_clips_batch as google_generate_clips_batch
                await google_generate_clips_batch(
                    images=images,
                    output_dir=clips_dir,
                    progress_callback=update_job_progress,
                    job_id=job_id,
                    api_key=keys["gemini"],
                )
            else:
                from core.fal_video import generate_clips_batch
                await generate_clips_batch(
                    images=images,
                    output_dir=clips_dir,
                    model_key=model_key,
                    progress_callback=update_job_progress,
                    job_id=job_id,
                    api_key=keys["fal"],
                )

            update_job_progress(job_id, "clips_ready", 0.50, "AI 영상 클립 생성 완료 - 미리보기 확인")

            from core.r2_storage import upload_job_files, is_r2_enabled
            if is_r2_enabled():
                await upload_job_files(job_id, "clips")

        except Exception as e:
            mark_job_failed(job_id, f"AI 클립 생성 실패: {str(e)}")
    finally:
        db.close()


async def regenerate_clip_for_job(job_id: str, line_index: int):
    """단일 AI 영상 클립 재생성"""
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return

        lines = json.loads(job.script_json)
        line = lines[line_index]
        job_dir = os.path.join(settings.STORAGE_DIR, job_id)
        image_path = os.path.join(job_dir, "images", f"img_{line_index:02d}.png")
        output_path = os.path.join(job_dir, "clips", f"clip_raw_{line_index:02d}.mp4")

        model_key = getattr(job, "video_mode", "hailuo") or "hailuo"
        keys = resolve_user_api_keys(db, job.user_id)

        try:
            if model_key == "veo_lite":
                from core.google_video import generate_video_clip as google_generate_video_clip
                await google_generate_video_clip(
                    image_path=image_path,
                    output_path=output_path,
                    api_key=keys["gemini"],
                )
            else:
                from core.fal_video import generate_video_clip
                await generate_video_clip(
                    image_path=image_path,
                    output_path=output_path,
                    model_key=model_key,
                    api_key=keys["fal"],
                )

            from core.r2_storage import upload_file as r2_upload, is_r2_enabled
            if is_r2_enabled():
                await r2_upload(output_path, f"jobs/{job_id}/clips/clip_raw_{line_index:02d}.mp4")

        except Exception as e:
            mark_job_failed(job_id, f"클립 재생성 실패: {str(e)}")
    finally:
        db.close()


async def render_video_for_job(job_id: str):
    """영상 조립 백그라운드 태스크"""
    from core.video_assembler import assemble_shorts

    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return

        update_job_progress(job_id, "generating_tts", 0.4, "TTS 나레이션 생성 중...")

        lines = json.loads(job.script_json)
        job_dir = os.path.join(settings.STORAGE_DIR, job_id)

        # 이미지 파일 목록
        images = sorted(glob.glob(os.path.join(job_dir, "images", "img_*.png")))

        # BGM 파일 결정: 로컬 → R2 다운로드 폴백
        bgm_path = None
        if job.bgm_filename:
            # 1. 로컬에 있으면 그대로 사용 (개발 모드)
            local_bgm = os.path.join(settings.BGM_DIR, job.bgm_filename)
            if os.path.exists(local_bgm):
                bgm_path = local_bgm
            # 2. 없으면 R2에서 다운로드 (배포 모드)
            elif is_r2_enabled() and job.user_id:
                from core.r2_storage import download_file_sync
                import asyncio
                r2_key = f"bgm/{job.user_id}/{job.bgm_filename}"
                temp_bgm = os.path.join(job_dir, "temp", f"bgm_{job.bgm_filename}")
                ok = await asyncio.to_thread(download_file_sync, r2_key, temp_bgm)
                if ok:
                    bgm_path = temp_bgm
        if not bgm_path:
            bgm_files = glob.glob(os.path.join(settings.BGM_DIR, "*.mp3"))
            if bgm_files:
                bgm_path = bgm_files[0]

        # AI 클립이 있으면 경로 수집
        clips_dir = os.path.join(job_dir, "clips")
        ai_clips = sorted(glob.glob(os.path.join(clips_dir, "clip_raw_*.mp4")))

        keys = resolve_user_api_keys(db, job.user_id)

        config = {
            "job_dir": job_dir,
            "images": images,
            "lines": lines,
            "title": job.title,
            "title_line1": getattr(job, "title_line1", None),
            "title_line2": getattr(job, "title_line2", None),
            "video_mode": getattr(job, "video_mode", "kenburns") or "kenburns",
            "ai_clips": ai_clips if ai_clips else None,
            "tts_engine": job.tts_engine,
            "tts_speed": job.tts_speed,
            "voice_id": job.voice_id,
            "emotion": job.emotion,
            "bgm_path": bgm_path,
            "bgm_volume": job.bgm_volume,
            "bgm_start_sec": job.bgm_start_sec or 0.0,
            "font_title": settings.FONT_TITLE,
            "font_sub": settings.FONT_SUB,
            "typecast_api_key": keys["typecast"],
        }

        try:
            video_path = await assemble_shorts(
                job_id=job_id,
                config=config,
                progress_callback=update_job_progress,
            )
            set_video_path(job_id, video_path)

            from core.r2_storage import upload_job_files, is_r2_enabled
            if is_r2_enabled():
                ok = await upload_job_files(job_id, "output")
                _update_r2_sync(job_id, "synced" if ok else "partial")

        except Exception as e:
            mark_job_failed(job_id, f"영상 조립 실패: {str(e)}")
    finally:
        db.close()


async def regenerate_image_for_job(job_id: str, line_index: int, korean_request: str = None, english_prompt: str = None):
    """단일 이미지 재생성 — 영어 프롬프트 직접 사용 또는 한글→영어 변환"""
    from core.gemini_client import generate_image, korean_to_nb2_prompt

    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return

        lines = json.loads(job.script_json)
        line = lines[line_index]
        job_dir = os.path.join(settings.STORAGE_DIR, job_id)
        output_path = os.path.join(job_dir, "images", f"img_{line_index:02d}.png")

        keys = resolve_user_api_keys(db, job.user_id)

        try:
            if english_prompt:
                prompt = english_prompt
            else:
                prompt = await korean_to_nb2_prompt(
                    korean_request=korean_request or line["text"],
                    narration_text=line["text"],
                    api_key=keys["gemini"],
                )

            # 새 프롬프트를 script_json에 반영
            line["image_prompt"] = prompt
            job.script_json = json.dumps(lines, ensure_ascii=False)
            db.commit()

            await generate_image(
                prompt=prompt,
                style=job.style,
                output_path=output_path,
                api_key=keys["gemini"],
            )

            job.status = "preview_ready"
            db.commit()

            from core.r2_storage import upload_file as r2_upload, is_r2_enabled
            if is_r2_enabled():
                await r2_upload(output_path, f"jobs/{job_id}/images/img_{line_index:02d}.png")

        except Exception as e:
            mark_job_failed(job_id, f"이미지 재생성 실패: {str(e)}")
    finally:
        db.close()
