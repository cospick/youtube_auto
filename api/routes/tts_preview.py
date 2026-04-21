"""TTS 음성 미리듣기 엔드포인트"""

import os
import re

from fastapi import APIRouter, Query, HTTPException, Depends
from fastapi.responses import FileResponse

import requests as http_requests

from sqlalchemy.orm import Session
from config import settings
from core.tts_engines import generate_tts_typecast
from api.deps import get_approved_user, resolve_user_api_keys
from db.database import get_db
from db.models import User

router = APIRouter(prefix="/api/tts", tags=["tts"])

PREVIEW_DIR = os.path.join(settings.STORAGE_DIR, "tts_preview")
SAMPLE_TEXT = "안녕하세요, 반갑습니다."

_SAFE_FILENAME = re.compile(r"^[\w\-]+$")


def _cache_path(user_id: str, engine: str, voice_id: str, speed: float, emotion: str) -> str:
    safe_id = voice_id.replace("-", "_")
    return os.path.join(
        PREVIEW_DIR,
        f"{user_id}_{engine}_{safe_id}_s{speed}_{emotion}.mp3"
    )


EMOTION_LABELS = {
    "normal": "보통", "happy": "기쁨", "sad": "슬픔", "angry": "화남",
    "whisper": "속삭임", "toneup": "밝게", "tonedown": "차분하게",
    "tonemid": "중간톤", "regret": "후회", "urgent": "급박한",
    "scream": "외침", "shout": "소리침", "trustful": "신뢰감",
    "soft": "부드럽게", "cold": "차갑게", "sarcasm": "비꼼",
    "inspire": "영감", "cute": "귀엽게", "cheer": "응원", "casual": "캐주얼",
}


@router.get("/emotions")
async def get_voice_emotions(voice_id: str = Query(..., min_length=1), db: Session = Depends(get_db), _user: User = Depends(get_approved_user)):
    """Typecast 성우의 지원 감정 목록 반환"""
    keys = resolve_user_api_keys(db, _user.id)
    tc_key = keys["typecast"]
    if not tc_key:
        raise HTTPException(500, "Typecast API 키가 설정되지 않았습니다. 설정 페이지에서 키를 입력하세요.")

    resp = http_requests.get(
        f"https://api.typecast.ai/v1/voices/{voice_id}",
        headers={"X-API-KEY": tc_key},
    )
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, "성우 정보 조회 실패")

    # 보통 → 긍정 → 부정 → 특수 순서
    EMOTION_ORDER = [
        "normal",
        "happy", "cheer", "toneup", "inspire", "cute", "casual", "trustful", "soft",
        "sad", "angry", "cold", "sarcasm", "regret", "tonedown", "tonemid",
        "whisper", "urgent", "scream", "shout",
    ]

    raw_emotions = []
    entries = resp.json()
    # ssfm-v30 우선, 없으면 ssfm-v21 사용
    for entry in entries:
        if entry.get("model") == "ssfm-v30":
            raw_emotions = entry.get("emotions", [])
            break
    if not raw_emotions:
        for entry in entries:
            if entry.get("model") == "ssfm-v21":
                raw_emotions = entry.get("emotions", [])
                break

    order_map = {e: i for i, e in enumerate(EMOTION_ORDER)}
    sorted_emotions = sorted(raw_emotions, key=lambda e: order_map.get(e, 99))

    return [
        {"value": e, "label": EMOTION_LABELS.get(e, e)}
        for e in sorted_emotions
    ]


@router.get("/preview")
async def tts_preview(
    engine: str = Query(..., pattern="^typecast$"),
    voice_id: str = Query(..., min_length=1, max_length=100),
    speed: float = Query(default=1.0, ge=0.5, le=2.0),
    emotion: str = Query(default="normal", max_length=20),
    db: Session = Depends(get_db),
    _user: User = Depends(get_approved_user),
):
    """선택한 엔진+음성+속도+감정으로 샘플 오디오 생성/반환"""
    keys = resolve_user_api_keys(db, _user.id)

    if not _SAFE_FILENAME.match(voice_id.replace("-", "_").replace(".", "_")):
        raise HTTPException(400, "잘못된 voice_id 형식입니다")

    os.makedirs(PREVIEW_DIR, exist_ok=True)
    cached = _cache_path(_user.id, engine, voice_id, speed, emotion)

    if os.path.exists(cached):
        media_type = "audio/mpeg" if cached.endswith(".mp3") else "audio/wav"
        return FileResponse(cached, media_type=media_type)

    tmp_dir = os.path.join(PREVIEW_DIR, f"tmp_{engine}_{voice_id.replace('-', '_')}")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        sentences = [SAMPLE_TEXT]

        emo = emotion if emotion != "normal" else None
        await generate_tts_typecast(
            tmp_dir, sentences,
            voice_id=voice_id, speed=speed, emotion=emo,
            api_key=keys["typecast"],
        )
        wav_path = os.path.join(tmp_dir, "sent_00.wav")
        if os.path.exists(wav_path):
            os.replace(wav_path, cached)
        else:
            raise HTTPException(500, "Typecast 오디오 생성 실패")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"미리듣기 생성 실패: {e}")
    finally:
        import shutil
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if not os.path.exists(cached):
        raise HTTPException(500, "오디오 파일 생성 실패")

    media_type = "audio/mpeg" if cached.endswith(".mp3") else "audio/wav"
    return FileResponse(cached, media_type=media_type)
