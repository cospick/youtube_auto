"""미리보기 API — 이미지 미리보기 + AI 클립 미리보기"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Path, Request, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
from api.models import PreviewResponse, ClipPreviewResponse, ScriptLine, AI_VIDEO_MODES
from api.deps import get_approved_user, get_user_job
from db.database import get_db
from db.models import Job, User
from config import settings
from PIL import Image, ImageOps
import json
import os
import io

router = APIRouter(prefix="/api/jobs", tags=["preview"])


@router.get("/{job_id}/preview", response_model=PreviewResponse)
async def get_preview(
    job_id: str = Path(..., pattern=r"^[a-f0-9]{12}$"),
    db: Session = Depends(get_db),
    _user: User = Depends(get_approved_user),
):
    """생성된 이미지 + 대본 미리보기"""
    job = get_user_job(db, job_id, _user)
    if job.status not in ("preview_ready", "awaiting_confirmation", "regenerating_image"):
        raise HTTPException(status_code=400, detail=f"미리보기 불가 (상태: {job.status})")

    lines = [ScriptLine(**l) for l in json.loads(job.script_json)]
    image_urls = [f"/api/jobs/{job_id}/images/{i}" for i in range(len(lines))]

    return PreviewResponse(title=job.title, lines=lines, image_urls=image_urls)


@router.post("/{job_id}/confirm")
async def confirm_and_render(
    request: Request,
    job_id: str = Path(..., pattern=r"^[a-f0-9]{12}$"),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
    _user: User = Depends(get_approved_user),
):
    """미리보기 확인 → AI 영상이면 클립 생성, Ken Burns면 바로 영상 조립"""
    # Request body에서 video_mode 직접 읽기
    try:
        body = await request.json()
    except Exception:
        body = {}
    video_mode = body.get("video_mode", "kenburns") or "kenburns"
    print(f"[DEBUG confirm] job_id={job_id}, raw_body={body}, video_mode={video_mode}")

    job = get_user_job(db, job_id, _user)
    if job.status not in ("preview_ready", "awaiting_confirmation"):
        raise HTTPException(status_code=400, detail=f"확정 불가 (상태: {job.status})")

    # 미리보기에서 선택한 영상 모드를 DB에 저장
    job.video_mode = video_mode

    if video_mode in AI_VIDEO_MODES:
        # AI 영상 모드: 이미지 확인 → AI 클립 생성 단계로
        job.status = "generating_clips"
        job.current_step = "AI 영상 클립 생성 준비 중..."
        db.commit()
        background_tasks.add_task(_generate_clips_task, job_id)
        return {"message": "AI 영상 클립 생성을 시작합니다", "job_id": job_id, "next": "clips"}
    else:
        # Ken Burns 모드: 바로 영상 조립
        job.status = "awaiting_confirmation"
        job.current_step = "영상 제작 준비 중..."
        db.commit()
        background_tasks.add_task(_render_video_task, job_id)
        return {"message": "영상 제작을 시작합니다", "job_id": job_id, "next": "render"}


class RegenerateRequest(BaseModel):
    korean_request: Optional[str] = None
    english_prompt: Optional[str] = None


@router.post("/{job_id}/regenerate-image/{line_index}")
async def regenerate_image(
    job_id: str = Path(..., pattern=r"^[a-f0-9]{12}$"),
    line_index: int = 0,
    body: RegenerateRequest = RegenerateRequest(),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
    _user: User = Depends(get_approved_user),
):
    """특정 이미지 재생성 (한글 요청어 → 영어 프롬프트 변환)"""
    job = get_user_job(db, job_id, _user)

    lines = json.loads(job.script_json)
    if line_index < 0 or line_index >= len(lines):
        raise HTTPException(status_code=400, detail="잘못된 이미지 인덱스")

    job.status = "regenerating_image"
    db.commit()

    background_tasks.add_task(_regenerate_single_image, job_id, line_index, body.korean_request, body.english_prompt)
    return {"message": f"이미지 {line_index + 1} 재생성 시작"}


@router.post("/{job_id}/upload-image/{line_index}")
async def upload_image(
    job_id: str = Path(..., pattern=r"^[a-f0-9]{12}$"),
    line_index: int = 0,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _user: User = Depends(get_approved_user),
):
    """사용자 이미지 업로드 — AI 이미지 대체"""
    job = get_user_job(db, job_id, _user)

    lines = json.loads(job.script_json)
    if line_index < 0 or line_index >= len(lines):
        raise HTTPException(status_code=400, detail="잘못된 이미지 인덱스")

    if file.content_type not in ("image/png", "image/jpeg", "image/webp"):
        raise HTTPException(status_code=400, detail="PNG, JPG, WebP 이미지만 업로드 가능합니다")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="파일 크기는 10MB 이하만 가능합니다")

    img = Image.open(io.BytesIO(contents))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    # 9:16 비율로 cover-crop
    target_w, target_h = 1080, 1920
    target_ratio = target_w / target_h

    src_w, src_h = img.size
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        offset = (src_w - new_w) // 2
        img = img.crop((offset, 0, offset + new_w, src_h))
    else:
        new_h = int(src_w / target_ratio)
        offset = (src_h - new_h) // 2
        img = img.crop((0, offset, src_w, offset + new_h))

    img = img.resize((target_w, target_h), Image.LANCZOS)

    output_path = os.path.join(settings.STORAGE_DIR, job_id, "images", f"img_{line_index:02d}.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, "PNG")

    from core.r2_storage import upload_file as r2_upload, is_r2_enabled
    if is_r2_enabled():
        await r2_upload(output_path, f"jobs/{job_id}/images/img_{line_index:02d}.png")

    return {"message": f"이미지 {line_index + 1} 업로드 완료", "image_url": f"/api/jobs/{job_id}/images/{line_index}"}


async def _render_video_task(job_id: str):
    """백그라운드: TTS + 영상 조립"""
    from jobs_queue.worker import render_video_for_job

    await render_video_for_job(job_id)


async def _generate_clips_task(job_id: str):
    """백그라운드: AI 영상 클립 생성"""
    from jobs_queue.worker import generate_clips_for_job

    await generate_clips_for_job(job_id)


async def _regenerate_single_image(job_id: str, line_index: int, korean_request: str = None, english_prompt: str = None):
    """백그라운드: 단일 이미지 재생성"""
    from jobs_queue.worker import regenerate_image_for_job

    await regenerate_image_for_job(job_id, line_index, korean_request, english_prompt)


# ─────────────────────────────────────
# AI 클립 미리보기 / 재생성 / 확인
# ─────────────────────────────────────


@router.get("/{job_id}/clip-preview", response_model=ClipPreviewResponse)
async def get_clip_preview(
    job_id: str = Path(..., pattern=r"^[a-f0-9]{12}$"),
    db: Session = Depends(get_db),
    _user: User = Depends(get_approved_user),
):
    """AI 클립 미리보기 데이터"""
    job = get_user_job(db, job_id, _user)
    if job.status not in ("clips_ready", "awaiting_confirmation"):
        raise HTTPException(status_code=400, detail=f"클립 미리보기 불가 (상태: {job.status})")

    lines = [ScriptLine(**l) for l in json.loads(job.script_json)]
    clip_urls = [f"/api/jobs/{job_id}/clips/{i}" for i in range(len(lines))]
    image_urls = [f"/api/jobs/{job_id}/images/{i}" for i in range(len(lines))]

    return ClipPreviewResponse(title=job.title, lines=lines, clip_urls=clip_urls, image_urls=image_urls)


@router.get("/{job_id}/clips/{index}")
async def get_clip_file(
    job_id: str = Path(..., pattern=r"^[a-f0-9]{12}$"),
    index: int = 0,
    db: Session = Depends(get_db),
    _user: User = Depends(get_approved_user),
):
    """개별 클립 파일 서빙"""
    from fastapi.responses import StreamingResponse
    from core.r2_storage import is_r2_enabled, r2_file_exists, stream_from_r2

    get_user_job(db, job_id, _user)
    job_dir = os.path.join(settings.STORAGE_DIR, job_id)
    clip_path = os.path.join(job_dir, "clips", f"clip_raw_{index:02d}.mp4")

    if os.path.exists(clip_path):
        return FileResponse(clip_path, media_type="video/mp4")

    r2_key = f"jobs/{job_id}/clips/clip_raw_{index:02d}.mp4"
    if is_r2_enabled() and r2_file_exists(r2_key):
        return StreamingResponse(stream_from_r2(r2_key), media_type="video/mp4")

    raise HTTPException(status_code=404, detail="클립 파일 없음")


@router.post("/{job_id}/regenerate-clip/{line_index}")
async def regenerate_clip(
    job_id: str = Path(..., pattern=r"^[a-f0-9]{12}$"),
    line_index: int = 0,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
    _user: User = Depends(get_approved_user),
):
    """특정 AI 클립 재생성"""
    job = get_user_job(db, job_id, _user)

    lines = json.loads(job.script_json)
    if line_index < 0 or line_index >= len(lines):
        raise HTTPException(status_code=400, detail="잘못된 클립 인덱스")

    background_tasks.add_task(_regenerate_single_clip, job_id, line_index)
    return {"message": f"클립 {line_index + 1} 재생성 시작"}


@router.post("/{job_id}/confirm-clips")
async def confirm_clips_and_render(
    job_id: str = Path(..., pattern=r"^[a-f0-9]{12}$"),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
    _user: User = Depends(get_approved_user),
):
    """AI 클립 확인 → TTS + 영상 조립 시작"""
    job = get_user_job(db, job_id, _user)
    if job.status not in ("clips_ready", "awaiting_confirmation"):
        raise HTTPException(status_code=400, detail=f"확정 불가 (상태: {job.status})")

    job.status = "awaiting_confirmation"
    job.current_step = "영상 제작 준비 중..."
    db.commit()

    background_tasks.add_task(_render_video_task, job_id)
    return {"message": "영상 제작을 시작합니다", "job_id": job_id}


@router.post("/{job_id}/upload-clip/{line_index}")
async def upload_clip(
    job_id: str = Path(..., pattern=r"^[a-f0-9]{12}$"),
    line_index: int = 0,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _user: User = Depends(get_approved_user),
):
    """사용자 영상 업로드 — AI 클립 대체"""
    job = get_user_job(db, job_id, _user)

    lines = json.loads(job.script_json)
    if line_index < 0 or line_index >= len(lines):
        raise HTTPException(status_code=400, detail="잘못된 클립 인덱스")

    allowed_types = ("video/mp4", "video/quicktime", "video/webm", "video/x-msvideo")
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="MP4, MOV, WebM, AVI 영상만 업로드 가능합니다")

    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="파일 크기는 50MB 이하만 가능합니다")

    clips_dir = os.path.join(settings.STORAGE_DIR, job_id, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    output_path = os.path.join(clips_dir, f"clip_raw_{line_index:02d}.mp4")

    if file.content_type == "video/mp4":
        # MP4는 그대로 저장
        with open(output_path, "wb") as f:
            f.write(contents)
    else:
        # MOV/WebM/AVI → FFmpeg로 MP4 변환
        import subprocess
        import tempfile

        ext = {
            "video/quicktime": ".mov",
            "video/webm": ".webm",
            "video/x-msvideo": ".avi",
        }.get(file.content_type, ".tmp")

        tmp_path = os.path.join(clips_dir, f"_upload_tmp{ext}")
        with open(tmp_path, "wb") as f:
            f.write(contents)

        try:
            cmd = f'ffmpeg -y -i "{tmp_path}" -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p -an "{output_path}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr[:300])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    from core.r2_storage import upload_file as r2_upload, is_r2_enabled
    clip_output = os.path.join(clips_dir, f"clip_raw_{line_index:02d}.mp4")
    if is_r2_enabled() and os.path.exists(clip_output):
        await r2_upload(clip_output, f"jobs/{job_id}/clips/clip_raw_{line_index:02d}.mp4")

    return {"message": f"클립 {line_index + 1} 업로드 완료", "clip_url": f"/api/jobs/{job_id}/clips/{line_index}"}


async def _regenerate_single_clip(job_id: str, line_index: int):
    """백그라운드: 단일 AI 클립 재생성"""
    from jobs_queue.worker import regenerate_clip_for_job

    await regenerate_clip_for_job(job_id, line_index)
