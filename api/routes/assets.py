"""이미지/영상/BGM 파일 서빙 API"""

from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import FileResponse
from config import settings
from core.video_assembler import get_duration
import os

router = APIRouter(prefix="/api/jobs", tags=["assets"])
bgm_router = APIRouter(prefix="/api/assets", tags=["bgm"])


@router.get("/{job_id}/images/{idx}")
async def get_image(
    job_id: str = Path(..., pattern=r"^[a-f0-9]{12}$"),
    idx: int = Path(..., ge=0, le=20),
):
    """생성된 이미지 파일 서빙"""
    path = os.path.join(settings.STORAGE_DIR, job_id, "images", f"img_{idx:02d}.png")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다")
    return FileResponse(path, media_type="image/png")


@router.get("/{job_id}/video")
async def get_video(
    job_id: str = Path(..., pattern=r"^[a-f0-9]{12}$"),
):
    """최종 영상 파일 서빙"""
    path = os.path.join(settings.STORAGE_DIR, job_id, "output", "shorts_final.mp4")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="영상을 찾을 수 없습니다")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"shorts_{job_id}.mp4",
    )


# ── BGM 관련 엔드포인트 ──


@bgm_router.get("/bgm", response_model=list)
async def list_bgm():
    """BGM 파일 목록 + 메타데이터 반환"""
    bgm_dir = settings.BGM_DIR
    if not os.path.isdir(bgm_dir):
        return []
    files = []
    for fname in sorted(os.listdir(bgm_dir)):
        if fname.lower().endswith((".mp3", ".wav", ".ogg")):
            fpath = os.path.join(bgm_dir, fname)
            try:
                duration = get_duration(fpath)
            except Exception:
                duration = 0
            files.append({
                "filename": fname,
                "duration": round(duration, 1),
                "url": f"/api/assets/bgm/{fname}",
            })
    return files


@bgm_router.get("/bgm/{filename:path}")
async def get_bgm_file(filename: str):
    """BGM 파일 스트리밍 (미리듣기용)"""
    fpath = os.path.join(settings.BGM_DIR, filename)
    abs_path = os.path.abspath(fpath)
    abs_bgm_dir = os.path.abspath(settings.BGM_DIR)
    if not abs_path.startswith(abs_bgm_dir):
        raise HTTPException(status_code=403, detail="접근 불가")
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail="BGM 파일을 찾을 수 없습니다")
    return FileResponse(abs_path, media_type="audio/mpeg")
