"""대본 생성 API — 멀티스텝 (제목 → 나레이션 → 이미지 프롬프트)"""

from fastapi import APIRouter, HTTPException
from api.models import (
    TitleRequest, TitleResponse,
    NarrationRequest, NarrationResponse,
    ImagePromptRequest, ImagePromptResponse,
)
import traceback

router = APIRouter(prefix="/api/generate", tags=["generate"])


@router.post("/titles", response_model=TitleResponse)
async def generate_titles_endpoint(request: TitleRequest):
    """Step 2: 제목 3~4개 생성"""
    from core.gemini_client import generate_titles as gen

    try:
        result = await gen(
            topic=request.topic,
            category=request.category,
            pain_point=request.pain_point,
            ingredient=request.ingredient,
            mention_type=request.mention_type,
        )
        return TitleResponse(**result)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] 제목 생성 실패:\n{tb}")
        raise HTTPException(status_code=500, detail=f"제목 생성 실패: {repr(e)}")


@router.post("/narration", response_model=NarrationResponse)
async def generate_narration_endpoint(request: NarrationRequest):
    """Step 3: 선택된 제목 기반 나레이션 생성"""
    from core.gemini_client import generate_narration as gen

    try:
        result = await gen(
            topic=request.topic,
            selected_title=request.selected_title,
            num_lines=request.num_lines,
            category=request.category,
            pain_point=request.pain_point,
            ingredient=request.ingredient,
            mention_type=request.mention_type,
        )
        return NarrationResponse(**result)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] 나레이션 생성 실패:\n{tb}")
        raise HTTPException(status_code=500, detail=f"나레이션 생성 실패: {repr(e)}")


@router.post("/image-prompts", response_model=ImagePromptResponse)
async def generate_image_prompts_endpoint(request: ImagePromptRequest):
    """Step 4: 확정된 나레이션 기반 이미지 프롬프트 + 모션 생성"""
    from core.gemini_client import generate_image_prompts as gen

    try:
        result = await gen(
            narration_lines=request.narration_lines,
            style=request.style.value,
        )
        return ImagePromptResponse(**result)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] 이미지 프롬프트 생성 실패:\n{tb}")
        raise HTTPException(status_code=500, detail=f"이미지 프롬프트 생성 실패: {repr(e)}")
