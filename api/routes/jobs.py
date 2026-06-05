"""작업 관리 API"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Path, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from api.models import JobCreateRequest, JobResponse, JobStatus, DraftJobRequest, DraftJobResponse, ScriptLine
from api.deps import get_approved_user, get_user_job, get_user_job_by_uid
from db.database import get_db
from db.models import Job, JobTask, User, UserProduct
from config import settings
from core.time_utils import utc_isoformat, utc_now_naive
from core.user_assets_visual import new_line_id, ensure_line_ids, line_asset_rel_candidates
from core.r2_storage import (
    require_r2_for_generation,
    is_r2_enabled,
    download_job_tts_to_local,
    delete_job_intermediate_files,
    delete_job_all_files,
)
from pydantic import BaseModel
from jobs_queue.task_queue import enqueue_task
import asyncio
import glob
import json
import os
import shutil
import uuid

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _require_generation_storage():
    try:
        require_r2_for_generation()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


def _job_to_response(job: Job, user_map: dict | None = None) -> JobResponse:
    video_url = None
    if job.video_path and (os.path.exists(job.video_path) or getattr(job, "r2_synced", "") == "synced"):
        video_url = f"/api/jobs/{job.id}/video"

    files_expired = bool(job.files_expired_at)
    days_remaining = None
    if job.completed_at and not files_expired:
        age = (utc_now_naive() - job.completed_at).days
        days_remaining = max(0, 30 - age)
        if days_remaining == 0:
            files_expired = True

    topic = job.topic if user_map is not None else None
    owner_nickname = None
    owner_email = None
    if user_map is not None and job.user_id and job.user_id in user_map:
        owner = user_map[job.user_id]
        owner_nickname = owner.nickname
        owner_email = owner.email

    # reopen_job 라우트(jobs.py: reopen_job)의 검증 조건과 동일하게 유지.
    # 활성 task 체크는 N+1 회피 위해 생략 — 드물게 클릭 후 409 발생 시 사용자 재시도.
    # preview_ready 포함: reopen이 멱등이라 이미 편집 중인 job에 다시 진입하는 흐름도 허용.
    # (사용자가 편집 화면을 떠났다가 작업이력으로 돌아와 다시 들어가는 케이스)
    #
    # intermediates_purged 조건은 'completed' 작업에만 적용한다. 이 플래그는 카드 B 영상 완성
    # 시점에야 worker가 False로 내려주고(그 전까진 컬럼 기본값 True), 다운로드/정리(finalize) 후
    # 다시 True가 된다. 따라서 '아직 완성 전 초안(preview_ready)'은 기본값이 True여도 중간 산출물이
    # 실제로 살아있는 정상 편집 대상이다. discard로 비운 경우는 files_expired가 따로 잡아낸다.
    can_reopen = (
        job.generation_mode == "user_assets"
        and job.status in ("completed", "preview_ready")
        and not files_expired
        and (job.status == "preview_ready" or not bool(job.intermediates_purged))
    )

    return JobResponse(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        current_step=job.current_step,
        created_at=utc_isoformat(job.created_at) or "",
        completed_at=utc_isoformat(job.completed_at),
        video_url=video_url,
        error=job.error_message,
        files_expired=files_expired,
        days_remaining=days_remaining,
        topic=topic,
        owner_nickname=owner_nickname,
        owner_email=owner_email,
        can_reopen=can_reopen,
    )


def _latest_job_task(db: Session, job_id: str) -> JobTask | None:
    return (
        db.query(JobTask)
        .filter(JobTask.job_id == job_id)
        .order_by(JobTask.created_at.desc())
        .first()
    )


def _with_latest_task_state(db: Session, job: Job, user_map: dict | None = None) -> JobResponse:
    response = _job_to_response(job, user_map)
    task = _latest_job_task(db, job.id)
    if not task:
        return response

    response.task_id = task.id
    response.task_kind = task.kind
    response.task_status = task.status
    response.task_error = task.error_message

    if job.status not in ("completed", "failed") and task.status in ("failed", "blocked"):
        response.status = "failed"
        response.current_step = "작업 실패"
        response.error = task.error_message or "작업 큐가 실패했습니다"
    return response


def _copy_product_snapshot(product: UserProduct, dest_path: str):
    """제품 이미지를 job 폴더로 스냅샷 복사. 로컬 우선, 없으면 R2에서 다운로드."""
    from api.routes.products import _local_path
    from core.r2_storage import is_r2_enabled, stream_from_r2

    local_src = _local_path(product.user_id, product.id)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.exists(local_src):
        shutil.copy(local_src, dest_path)
        return

    if is_r2_enabled() and product.r2_key:
        with open(dest_path, "wb") as f:
            for chunk in stream_from_r2(product.r2_key):
                f.write(chunk)
        if os.path.getsize(dest_path) > 0:
            return
        os.remove(dest_path)

    raise RuntimeError("제품 이미지 원본을 찾을 수 없습니다")


@router.post("/", response_model=JobResponse)
async def create_job(
    request: JobCreateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _user: User = Depends(get_approved_user),
):
    """작업 생성 → 이미지 생성 시작"""
    _require_generation_storage()
    # Job ID를 미리 생성 (스냅샷 경로 계산용)
    job_id = uuid.uuid4().hex[:12]

    # 제품 이미지 검증 + 스냅샷 준비 (Job 커밋 전에 먼저)
    product = None
    if request.product_image_id:
        product = db.query(UserProduct).filter(
            UserProduct.id == request.product_image_id,
            UserProduct.user_id == _user.id,
        ).first()
        if not product:
            raise HTTPException(status_code=400, detail="선택한 제품을 찾을 수 없습니다")

        snapshot_path = os.path.join(settings.STORAGE_DIR, job_id, "product", "product.png")
        try:
            _copy_product_snapshot(product, snapshot_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"제품 이미지 준비 실패: {e}")

    # TTS 세션 존재 확인 (미리 생성된 TTS 재사용 경로)
    tts_session_dir = None
    if request.tts_session_id:
        tts_session_dir = os.path.join(
            settings.STORAGE_DIR, "tts_sessions", request.tts_session_id
        )
        if not os.path.exists(tts_session_dir):
            raise HTTPException(
                status_code=400,
                detail="TTS 세션을 찾을 수 없습니다. 음성 설정 단계에서 다시 생성해주세요.",
            )

    job = Job(
        id=job_id,
        user_id=_user.id,
        topic=request.topic,
        style=request.style.value,
        video_mode=request.video_mode.value,
        tts_engine=request.tts_engine.value,
        tts_speed=request.tts_speed,
        voice_id=request.voice_id,
        emotion=request.emotion,
        title=request.title,
        title_line1=request.title_line1,
        title_line2=request.title_line2,
        script_json=json.dumps(
            [line.model_dump() for line in request.lines], ensure_ascii=False
        ),
        product_image_id=request.product_image_id,
        bgm_volume=request.bgm_volume,
        bgm_filename=request.bgm_filename,
        bgm_start_sec=request.bgm_start_sec,
        tts_session_id=request.tts_session_id,
        generation_mode=request.generation_mode,
        line_sources_json=json.dumps(request.line_sources, ensure_ascii=False),
        status="pending",
        current_step="작업 대기 중...",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # 작업 디렉토리 생성
    job_dir = os.path.join(settings.STORAGE_DIR, job.id)
    for sub in ["images", "clips", "tts", "temp", "output"]:
        os.makedirs(os.path.join(job_dir, sub), exist_ok=True)

    # TTS 세션 파일을 job_dir/tts/로 이동 (있으면)
    if tts_session_dir:
        try:
            for fname in os.listdir(tts_session_dir):
                shutil.move(
                    os.path.join(tts_session_dir, fname),
                    os.path.join(job_dir, "tts", fname),
                )
            os.rmdir(tts_session_dir)
        except Exception as e:
            # 이동 실패는 치명적이지 않음 — 영상 조립 시 TTS 재생성 경로가 살아있음
            # 다만 tts_session_id가 DB에 남아있으면 worker가 오판 가능 → 지우기
            job.tts_session_id = None
            db.commit()
            print(f"[create_job] TTS 세션 이동 실패, 재생성 경로로 폴백: {e}")

    enqueue_task(
        db,
        job=job,
        kind="card_a_images",
        payload={},
        dedupe_key="card_a_images",
        max_attempts=80,
    )

    return _job_to_response(job)


@router.post("/draft", response_model=DraftJobResponse)
async def create_draft_job(
    request: DraftJobRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(get_approved_user),
):
    """카드 B 전용: 쪼개진 대본만으로 draft Job 생성.

    이 시점에서는 음성/BGM/제목이 아직 정해지지 않았다.
    줄별 자산 편집 화면에서 업로드/AI 생성을 거친 뒤, /confirm 시점에
    음성·BGM 정보가 body로 함께 전송돼 Job이 보강된 후 영상 조립이 시작된다.
    """
    job_id = uuid.uuid4().hex[:12]

    n = len(request.lines)
    script_lines = [
        {
            "line_id": new_line_id(),
            "text": text,
            "image_prompt": "",
            "motion": "zoom_in",
            "asset_version": 0,
            "status": "pending",
            "fail_reason": None,
        }
        for text in request.lines
    ]

    job = Job(
        id=job_id,
        user_id=_user.id,
        topic="",
        title="",
        script_json=json.dumps(script_lines, ensure_ascii=False),
        generation_mode="user_assets",
        line_sources_json=json.dumps(["ai"] * n, ensure_ascii=False),
        status="preview_ready",
        current_step="자산 편집 대기 중",
        # 초안은 줄별 자산을 지금부터 만들어 채우는 '편집 중' 상태 → 중간 산출물이 살아있음.
        # 컬럼 기본값(True)을 그대로 두면 작업이력의 can_reopen 판정이 어긋나므로 명시적으로 False.
        intermediates_purged=False,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # 작업 디렉토리 생성
    job_dir = os.path.join(settings.STORAGE_DIR, job.id)
    for sub in ["images", "clips", "tts", "temp", "output"]:
        os.makedirs(os.path.join(job_dir, sub), exist_ok=True)

    return DraftJobResponse(job_id=job.id, lines=[ScriptLine(**l) for l in script_lines])


@router.get("/", response_model=list[JobResponse])
async def list_jobs(limit: int = 20, db: Session = Depends(get_db), _user: User = Depends(get_approved_user)):
    """작업 목록 (최신순, 본인 작업만)"""
    jobs = db.query(Job).filter(Job.user_id == _user.id).order_by(Job.created_at.desc()).limit(limit).all()
    return [_job_to_response(j) for j in jobs]


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str = Path(..., pattern=r"^[a-f0-9]{12}$"),
    db: Session = Depends(get_db),
    _user: User = Depends(get_approved_user),
):
    """작업 상태 조회"""
    job = get_user_job(db, job_id, _user)
    return _with_latest_task_state(db, job)


@router.get("/{job_id}/stream")
async def stream_progress(
    request: Request,
    job_id: str = Path(..., pattern=r"^[a-f0-9]{12}$"),
):
    """SSE로 실시간 진행률 전송"""
    # SSE는 Depends 사용 불가 → 쿠키에서 직접 토큰 검증
    from core.security import decode_token
    import jwt as _jwt
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다")
    try:
        payload = decode_token(token)
    except (_jwt.ExpiredSignatureError, _jwt.InvalidTokenError):
        raise HTTPException(status_code=401, detail="토큰이 만료되었습니다")

    token_user_id = payload.get("sub")
    token_role = payload.get("role", "user")

    async def event_generator():
        db = next(get_db())
        try:
            while True:
                if await request.is_disconnected():
                    break
                db.expire_all()
                job = get_user_job_by_uid(db, job_id, token_user_id, token_role)
                if not job:
                    yield f"data: {json.dumps({'error': '작업을 찾을 수 없습니다'})}\n\n"
                    break
                data = {
                    "status": job.status,
                    "progress": job.progress,
                    "current_step": job.current_step,
                }
                if job.status == "completed":
                    data["video_url"] = f"/api/jobs/{job.id}/video"
                if job.error_message:
                    data["error"] = job.error_message

                latest_task = _latest_job_task(db, job_id)
                if latest_task:
                    data["task_id"] = latest_task.id
                    data["task_kind"] = latest_task.kind
                    data["task_status"] = latest_task.status
                    if latest_task.error_message:
                        data["task_error"] = latest_task.error_message
                    if job.status not in ("completed", "failed") and latest_task.status in ("failed", "blocked"):
                        data["status"] = "failed"
                        data["current_step"] = "작업 실패"
                        data["error"] = latest_task.error_message or "작업 큐가 실패했습니다"

                # 이미지 생성 단계: 대본 + 완성된 이미지 인덱스 전송
                if job.status in ("pending", "generating_images", "preview_ready"):
                    try:
                        lines = json.loads(job.script_json) if job.script_json else []
                        data["lines"] = [
                            {"text": l.get("text", ""), "motion": l.get("motion", "")}
                            for l in lines
                        ]
                        job_dir = os.path.join(settings.STORAGE_DIR, job_id)
                        completed = []
                        for i in range(len(lines)):
                            img_path = os.path.join(
                                job_dir, "images", f"img_{i:02d}.png"
                            )
                            if os.path.exists(img_path):
                                completed.append(i)
                        data["completed_images"] = completed
                    except Exception:
                        pass

                # AI 클립 생성 단계: 완성된 클립 인덱스 전송
                if job.status in ("generating_clips", "clips_ready"):
                    try:
                        lines = json.loads(job.script_json) if job.script_json else []
                        data["lines"] = [
                            {"text": l.get("text", ""), "motion": l.get("motion", "")}
                            for l in lines
                        ]
                        job_dir = os.path.join(settings.STORAGE_DIR, job_id)
                        completed_clips = []
                        for i in range(len(lines)):
                            clip_path = os.path.join(
                                job_dir, "clips", f"clip_raw_{i:02d}.mp4"
                            )
                            if os.path.exists(clip_path):
                                completed_clips.append(i)
                        data["completed_clips"] = completed_clips
                    except Exception:
                        pass
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                if data["status"] in ("completed", "failed"):
                    break
                await asyncio.sleep(1)
        finally:
            db.close()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/{job_id}/retry-images")
async def retry_images(
    job_id: str = Path(..., pattern=r"^[a-f0-9]{12}$"),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
    _user: User = Depends(get_approved_user),
):
    """실패한 이미지 생성 재시도"""
    _require_generation_storage()
    job = get_user_job(db, job_id, _user)

    # 이미 진행 중이면 거부
    if job.status == "generating_images":
        raise HTTPException(status_code=409, detail="이미지 생성이 이미 진행 중입니다")

    # 기존 이미지 파일 삭제 (SSE가 파일 존재로 완료를 판단하므로)
    images_dir = os.path.join(settings.STORAGE_DIR, job_id, "images")
    for f in glob.glob(os.path.join(images_dir, "img_*.png")):
        os.remove(f)

    job.status = "pending"
    job.error_message = None
    db.commit()

    task, already_running = enqueue_task(
        db,
        job=job,
        kind="card_a_images",
        payload={"retry": True},
        dedupe_key="card_a_images",
        max_attempts=80,
    )
    return {"message": "이미지 생성 재시도 시작", "task_id": task.id, "already_running": already_running}


@router.get("/{job_id}/tasks/{task_id}")
async def get_task_status(
    job_id: str = Path(..., pattern=r"^[a-f0-9]{12}$"),
    task_id: str = Path(..., pattern=r"^[a-f0-9]{12}$"),
    db: Session = Depends(get_db),
    _user: User = Depends(get_approved_user),
):
    """작업 큐 상태 조회."""
    get_user_job(db, job_id, _user)
    task = db.query(JobTask).filter(JobTask.id == task_id, JobTask.job_id == job_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="작업 큐를 찾을 수 없습니다")
    try:
        payload = json.loads(task.payload_json or "{}")
    except Exception:
        payload = {}
    line_ids = payload.get("line_ids") or []
    completed = payload.get("completed_line_ids") or []
    return {
        "task_id": task.id,
        "job_id": task.job_id,
        "kind": task.kind,
        "status": task.status,
        "attempt_count": task.attempt_count,
        "max_attempts": task.max_attempts,
        "next_run_at": utc_isoformat(task.next_run_at),
        "current_line_index": payload.get("current_line_index"),
        "total": len(line_ids),
        "completed": len(completed),
        "error": task.error_message,
        "payload": payload,
    }


async def _generate_images_task(job_id: str):
    """백그라운드: 이미지 생성"""
    from jobs_queue.worker import generate_images_for_job

    await generate_images_for_job(job_id)


# ─── 카드 B: 완료 후 재편집 (reopen) + 복원 (draft-state) ──────────────

class DraftStateResponse(BaseModel):
    job_id: str
    status: str
    generation_mode: str
    intermediates_purged: bool
    video_path: str | None = None
    video_url: str | None = None
    title: str | None = ""
    title_line1: str | None = None
    title_line2: str | None = None
    tts_engine: str | None = None
    tts_speed: float | None = None
    voice_id: str | None = None
    emotion: str | None = None
    tts_session_id: str | None = None
    bgm_filename: str | None = None
    bgm_volume: float | None = None
    bgm_start_sec: float | None = None
    product_image_id: str | None = None
    lines: list[dict] = []
    line_sources: list[str] = []
    last_render_signature: dict | None = None


def _build_draft_state(job: Job) -> DraftStateResponse:
    """카드 B Job → 프론트가 복원에 필요한 전체 상태 페이로드."""
    try:
        lines = json.loads(job.script_json or "[]")
    except Exception:
        lines = []
    try:
        sources = json.loads(job.line_sources_json or "[]")
    except Exception:
        sources = []
    try:
        last_sig = json.loads(job.last_render_signature) if job.last_render_signature else None
    except Exception:
        last_sig = None

    video_url = None
    if job.video_path and (os.path.exists(job.video_path) or getattr(job, "r2_synced", "") == "synced"):
        video_url = f"/api/jobs/{job.id}/video"

    return DraftStateResponse(
        job_id=job.id,
        status=job.status,
        generation_mode=job.generation_mode or "ai_full",
        intermediates_purged=bool(job.intermediates_purged),
        video_path=job.video_path,
        video_url=video_url,
        title=job.title or "",
        title_line1=job.title_line1,
        title_line2=job.title_line2,
        tts_engine=job.tts_engine,
        tts_speed=job.tts_speed,
        voice_id=job.voice_id,
        emotion=job.emotion,
        tts_session_id=job.tts_session_id,
        bgm_filename=job.bgm_filename,
        bgm_volume=job.bgm_volume,
        bgm_start_sec=job.bgm_start_sec,
        product_image_id=job.product_image_id,
        lines=lines,
        line_sources=sources,
        last_render_signature=last_sig,
    )


@router.get("/{job_id}/draft-state", response_model=DraftStateResponse)
async def get_draft_state(
    job_id: str = Path(..., pattern=r"^[a-f0-9]{12}$"),
    db: Session = Depends(get_db),
    _user: User = Depends(get_approved_user),
):
    """카드 B 전용: 편집 화면 복원에 필요한 전체 상태 응답."""
    job = get_user_job(db, job_id, _user)
    if (job.generation_mode or "ai_full") != "user_assets":
        raise HTTPException(status_code=404, detail="카드 B 작업이 아닙니다")
    return _build_draft_state(job)


def _active_tasks_count(db: Session, job_id: str) -> int:
    return (
        db.query(JobTask)
        .filter(JobTask.job_id == job_id)
        .filter(JobTask.status.in_(["queued", "running", "retrying"]))
        .count()
    )


async def _restore_local_assets_from_r2(job_id: str, lines: list[dict], sources: list[str]) -> list[str]:
    """카드 B reopen 시 누락된 로컬 자산을 R2에서 복구. 모자란 파일 목록(에러용)을 반환."""
    from jobs_queue.worker import _ensure_r2_asset_local_any

    missing: list[str] = []
    for i, line in enumerate(lines):
        kind = "clip" if (i < len(sources) and sources[i] == "clip") else "image"
        candidates = line_asset_rel_candidates(kind, line, i)
        # 로컬 우선 확인
        from api.routes.preview import _job_asset_exists_any
        if _job_asset_exists_any(job_id, candidates):
            continue
        # R2에서 복구 시도
        if is_r2_enabled():
            rel = await _ensure_r2_asset_local_any(job_id, candidates)
            if rel:
                continue
        missing.append(f"{i + 1}번째 줄 {kind}")

    # TTS 디렉토리도 복구 (있으면)
    tts_dir = os.path.join(settings.STORAGE_DIR, job_id, "tts")
    has_local_tts = os.path.isdir(tts_dir) and any(
        f.endswith(".wav") for f in os.listdir(tts_dir)
    )
    if not has_local_tts and is_r2_enabled():
        await download_job_tts_to_local(job_id)

    return missing


def _restore_tts_session_dir(tts_session_id: str | None, job_id: str) -> None:
    """카드 B reopen 후 incremental TTS 재빌드를 위해 tts_sessions/{id}/ 를 job_dir/tts/ 로부터 복원.

    첫 빌드 시 confirm이 session_dir → job_dir/tts/ 로 move하면서 session_dir이 사라진다.
    재빌드 시점에 session_dir이 없으면 /preview-build가 full rebuild로 폴백돼 최적화가 무효화되므로,
    reopen 시점에 session_dir 을 복사로 재구성한다."""
    import shutil

    if not tts_session_id:
        return
    if len(tts_session_id) != 12 or any(c not in "0123456789abcdef" for c in tts_session_id):
        return

    sessions_root = os.path.join(settings.STORAGE_DIR, "tts_sessions")
    session_dir = os.path.join(sessions_root, tts_session_id)
    src_dir = os.path.join(settings.STORAGE_DIR, job_id, "tts")

    if os.path.isdir(session_dir):
        return  # 이미 존재 — 그대로
    if not os.path.isdir(src_dir):
        return  # 원본도 없음 — 복원 불가, 다음 단계에서 검증 실패

    os.makedirs(session_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        if not os.path.isfile(src):
            continue
        try:
            shutil.copy2(src, os.path.join(session_dir, name))
        except Exception:
            pass


@router.post("/{job_id}/reopen", response_model=DraftStateResponse)
async def reopen_job(
    job_id: str = Path(..., pattern=r"^[a-f0-9]{12}$"),
    db: Session = Depends(get_db),
    _user: User = Depends(get_approved_user),
):
    """카드 B 완료 후 편집 화면으로 복귀.

    검증:
    - generation_mode == "user_assets"
    - status == "completed"
    - intermediates_purged == False (다운로드/discard 후엔 편집 불가)
    - 활성 task 0건 (재제작 중 동시 재진입 차단)
    - 자산 파일(images/clips) 실재 (R2에서 자동 복구 시도)

    처리:
    - 누락된 로컬 자산을 R2에서 복구
    - status = "preview_ready"
    - DraftStateResponse 반환
    """
    job = get_user_job(db, job_id, _user)
    if (job.generation_mode or "ai_full") != "user_assets":
        raise HTTPException(status_code=400, detail="이 작업은 카드 B 모드가 아닙니다")

    # 이미 preview_ready 상태면 멱등 응답 (active task 없는 경우에 한해)
    if job.status == "preview_ready":
        if _active_tasks_count(db, job_id) > 0:
            raise HTTPException(status_code=409, detail="이미 편집 가능 상태이며 진행 중인 작업이 있습니다")
        # 중단된 초안 재진입: 서버 재시작 등으로 로컬에서 사라진 자산을 R2에서 best-effort 복구.
        # 완성본 경로와 달리 missing은 무시한다 — 아직 업로드/생성 전인 pending 줄이 정상적으로
        # 존재할 수 있고, 여기서 차단하면 "이어서 편집"이 불가능해지기 때문. 실제 자산 누락은
        # 이후 /confirm 단계의 줄별 검증(preview.py)에서 차단된다.
        try:
            draft_lines = json.loads(job.script_json or "[]")
        except Exception:
            draft_lines = []
        try:
            draft_sources = json.loads(job.line_sources_json or "[]")
        except Exception:
            draft_sources = []
        if draft_lines:
            ensure_line_ids(draft_lines)
            job.script_json = json.dumps(draft_lines, ensure_ascii=False)
            await _restore_local_assets_from_r2(job_id, draft_lines, draft_sources)
            _restore_tts_session_dir(job.tts_session_id, job_id)
            db.commit()
            db.refresh(job)
        return _build_draft_state(job)

    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"완료된 작업만 다시 편집할 수 있습니다 (상태: {job.status})")

    if bool(job.intermediates_purged):
        raise HTTPException(status_code=410, detail="편집 가능 기한이 지났습니다 (이미 다운로드/정리됨)")

    if _active_tasks_count(db, job_id) > 0:
        raise HTTPException(status_code=409, detail="진행 중인 작업이 있어 편집 화면으로 돌아갈 수 없습니다")

    try:
        lines = json.loads(job.script_json or "[]")
    except Exception:
        lines = []
    try:
        sources = json.loads(job.line_sources_json or "[]")
    except Exception:
        sources = []

    # line_id 보장 (구버전 row 안전망)
    ensure_line_ids(lines)
    job.script_json = json.dumps(lines, ensure_ascii=False)

    missing = await _restore_local_assets_from_r2(job_id, lines, sources)
    if missing:
        raise HTTPException(
            status_code=410,
            detail=f"자산을 복구할 수 없습니다: {', '.join(missing)}",
        )

    # incremental TTS 재빌드를 위해 tts_sessions/{id}/ 도 복원
    _restore_tts_session_dir(job.tts_session_id, job_id)

    job.status = "preview_ready"
    job.error_message = None
    db.commit()
    db.refresh(job)
    return _build_draft_state(job)


# ─── 카드 B: 영상 다운로드 후 정리 (finalize) / 새 영상 시작 (discard) ──

@router.post("/{job_id}/finalize")
async def finalize_job(
    job_id: str = Path(..., pattern=r"^[a-f0-9]{12}$"),
    db: Session = Depends(get_db),
    _user: User = Depends(get_approved_user),
):
    """사용자가 '영상 다운로드'를 누른 시점에 호출. 중간 산출물(images/clips/tts/temp)을
    R2 + 로컬에서 삭제. 최종 영상(output/shorts_final.mp4)은 유지되어 작업이력에서 30일
    동안 재다운로드 가능. 멱등성 — 이미 purged면 그대로 200 반환."""
    job = get_user_job(db, job_id, _user)
    if (job.generation_mode or "ai_full") != "user_assets":
        raise HTTPException(status_code=400, detail="이 작업은 카드 B 모드가 아닙니다")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"완료된 작업만 정리할 수 있습니다 (상태: {job.status})")

    # 이미 정리됨 → 멱등 응답
    if bool(job.intermediates_purged):
        return {"ok": True, "already_purged": True}

    await delete_job_intermediate_files(job_id)

    # tts_sessions/{tts_session_id}/도 함께 정리 (잔존하면 디스크 낭비)
    if job.tts_session_id and len(job.tts_session_id) == 12:
        session_dir = os.path.join(settings.STORAGE_DIR, "tts_sessions", job.tts_session_id)
        if os.path.isdir(session_dir):
            shutil.rmtree(session_dir, ignore_errors=True)

    job.intermediates_purged = True
    db.commit()
    return {"ok": True, "already_purged": False}


@router.post("/{job_id}/discard")
async def discard_job(
    job_id: str = Path(..., pattern=r"^[a-f0-9]{12}$"),
    db: Session = Depends(get_db),
    _user: User = Depends(get_approved_user),
):
    """사용자가 '새 영상 만들기'를 누른 시점에 호출. 최종 영상 포함 전체 산출물을 삭제.
    Job 레코드는 남기되 video_path=None / files_expired_at=now() 로 마킹.
    멱등성 — 이미 files_expired면 그대로 200 반환."""
    job = get_user_job(db, job_id, _user)
    if (job.generation_mode or "ai_full") != "user_assets":
        raise HTTPException(status_code=400, detail="이 작업은 카드 B 모드가 아닙니다")

    if job.files_expired_at:
        return {"ok": True, "already_discarded": True}

    await delete_job_all_files(job_id)

    if job.tts_session_id and len(job.tts_session_id) == 12:
        session_dir = os.path.join(settings.STORAGE_DIR, "tts_sessions", job.tts_session_id)
        if os.path.isdir(session_dir):
            shutil.rmtree(session_dir, ignore_errors=True)

    job.video_path = None
    job.files_expired_at = utc_now_naive()
    job.intermediates_purged = True
    db.commit()
    return {"ok": True, "already_discarded": False}
