"""작업 관리 API"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Path, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from api.models import JobCreateRequest, JobResponse, JobStatus
from api.deps import get_current_user, get_user_job, get_user_job_by_uid
from db.database import get_db
from db.models import Job, User
from config import settings
import asyncio
import glob
import json
import os

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _job_to_response(job: Job) -> JobResponse:
    import datetime as _dt
    video_url = None
    if job.video_path and os.path.exists(job.video_path):
        video_url = f"/api/jobs/{job.id}/video"

    files_expired = bool(job.files_expired_at)
    days_remaining = None
    if job.completed_at and not files_expired:
        age = (_dt.datetime.utcnow() - job.completed_at).days
        days_remaining = max(0, 30 - age)
        if days_remaining == 0:
            files_expired = True

    return JobResponse(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        current_step=job.current_step,
        created_at=job.created_at.isoformat() if job.created_at else "",
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        video_url=video_url,
        error=job.error_message,
        files_expired=files_expired,
        days_remaining=days_remaining,
    )


@router.post("/", response_model=JobResponse)
async def create_job(
    request: JobCreateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """작업 생성 → 이미지 생성 시작"""
    job = Job(
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
        bgm_volume=request.bgm_volume,
        bgm_filename=request.bgm_filename,
        bgm_start_sec=request.bgm_start_sec,
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

    # 백그라운드에서 이미지 생성 시작
    background_tasks.add_task(_generate_images_task, job.id)

    return _job_to_response(job)


@router.get("/", response_model=list[JobResponse])
async def list_jobs(limit: int = 20, db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    """작업 목록 (최신순, 본인 작업만 / admin은 전체)"""
    query = db.query(Job)
    if _user.role != "admin":
        query = query.filter(Job.user_id == _user.id)
    jobs = query.order_by(Job.created_at.desc()).limit(limit).all()
    return [_job_to_response(j) for j in jobs]


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str = Path(..., pattern=r"^[a-f0-9]{12}$"),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """작업 상태 조회"""
    job = get_user_job(db, job_id, _user)
    return _job_to_response(job)


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
                if job.status in ("completed", "failed"):
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
    _user: User = Depends(get_current_user),
):
    """실패한 이미지 생성 재시도"""
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

    background_tasks.add_task(_generate_images_task, job_id)
    return {"message": "이미지 생성 재시도 시작"}


async def _generate_images_task(job_id: str):
    """백그라운드: 이미지 생성"""
    from jobs_queue.worker import generate_images_for_job

    await generate_images_for_job(job_id)
