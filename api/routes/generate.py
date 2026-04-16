"""대본 생성 API — 멀티스텝 (제목 → 나레이션 → 이미지 프롬프트)"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from api.models import (
    TitleRequest, TitleResponse,
    NarrationRequest, NarrationResponse,
    ImagePromptRequest, ImagePromptResponse,
)
from api.deps import get_approved_user, resolve_user_api_keys
from db.database import get_db
from db.models import User
import traceback

router = APIRouter(prefix="/api/generate", tags=["generate"])


@router.post("/titles", response_model=TitleResponse)
async def generate_titles_endpoint(request: TitleRequest, db: Session = Depends(get_db), _user: User = Depends(get_approved_user)):
    """Step 2: 제목 3~4개 생성"""
    from core.gemini_client import generate_titles as gen
    keys = resolve_user_api_keys(db, _user.id)

    try:
        result = await gen(
            topic=request.topic,
            category=request.category,
            pain_point=request.pain_point,
            ingredient=request.ingredient,
            content_type=request.content_type,
            keyword=request.keyword,
            api_key=keys["gemini"],
        )
        return TitleResponse(**result)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] 제목 생성 실패:\n{tb}")
        raise HTTPException(status_code=500, detail=f"제목 생성 실패: {repr(e)}")


@router.post("/narration", response_model=NarrationResponse)
async def generate_narration_endpoint(request: NarrationRequest, db: Session = Depends(get_db), _user: User = Depends(get_approved_user)):
    """Step 3: 선택된 제목 기반 나레이션 생성"""
    from core.gemini_client import generate_narration as gen
    keys = resolve_user_api_keys(db, _user.id)

    try:
        result = await gen(
            topic=request.topic,
            selected_title=request.selected_title,
            num_lines=request.num_lines,
            category=request.category,
            pain_point=request.pain_point,
            ingredient=request.ingredient,
            content_type=request.content_type,
            keyword=request.keyword,
            api_key=keys["gemini"],
        )
        return NarrationResponse(**result)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] 나레이션 생성 실패:\n{tb}")
        raise HTTPException(status_code=500, detail=f"나레이션 생성 실패: {repr(e)}")


@router.post("/image-prompts", response_model=ImagePromptResponse)
async def generate_image_prompts_endpoint(request: ImagePromptRequest, db: Session = Depends(get_db), _user: User = Depends(get_approved_user)):
    """Step 4: 확정된 나레이션 기반 이미지 프롬프트 + 모션 생성"""
    from core.gemini_client import generate_image_prompts as gen
    keys = resolve_user_api_keys(db, _user.id)

    try:
        result = await gen(
            narration_lines=request.narration_lines,
            style=request.style.value,
            category=request.category,
            topic=request.topic,
            content_type=request.content_type,
            api_key=keys["gemini"],
        )
        return ImagePromptResponse(**result)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] 이미지 프롬프트 생성 실패:\n{tb}")
        raise HTTPException(status_code=500, detail=f"이미지 프롬프트 생성 실패: {repr(e)}")
