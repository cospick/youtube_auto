"""Gemini API 래퍼 - 텍스트 생성 + Nano Banana 2 이미지 생성"""

from google import genai
from google.genai import types
from config import settings
import asyncio
import base64
import json
import os
import re
import time

_nb2_guide = None


def _load_nb2_guide() -> str:
    """Nano Banana 2 공식 프롬프트 가이드 로드 (캐싱)"""
    global _nb2_guide
    if _nb2_guide is None:
        guide_path = os.path.join(os.path.dirname(__file__), "nb2_prompt_guide.txt")
        with open(guide_path, "r", encoding="utf-8") as f:
            _nb2_guide = f.read()
    return _nb2_guide


STYLE_SUFFIXES = {
    "realistic": "photorealistic, 8k, ultra-detailed, high resolution photography",
    "anime": "anime style, vibrant colors, Japanese animation, detailed illustration",
    "illustration": "digital illustration, artistic, painterly style, concept art",
}

MOTION_TYPES = ["zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down"]


def get_client(api_key: str = None) -> genai.Client:
    key = api_key or settings.GEMINI_API_KEY
    if not key:
        raise RuntimeError("Gemini API 키가 설정되지 않았습니다. 환경변수 또는 api_key 파라미터를 확인해주세요.")
    return genai.Client(
        api_key=key,
        http_options={"timeout": 120_000},
    )


def _build_category_context(
    category: str,
    pain_point: str = None,
    ingredient: str = None,
    mention_type: str = None,
    product_name: str = None,
) -> str:
    """카테고리별 컨텍스트 문자열 생성"""
    if category != "cosmetics":
        return ""

    ctx = "\n[화장품/스킨케어 필수 반영사항 — 아래 내용을 대본에 반드시 녹여야 합니다]\n"
    if pain_point:
        ctx += f"타겟 고민: {pain_point}\n"
    if ingredient:
        ctx += f"핵심 성분: {ingredient}\n"
    if mention_type == "comment":
        ctx += "제품 언급 방식: 댓글 유도 (제품명을 직접 말하지 않고 '궁금하면 댓글!' 등으로 유도)\n"
    elif mention_type == "direct":
        if product_name:
            ctx += f"제품 언급 방식: 직접 언급\n"
            ctx += f"[필수] 제품명 '{product_name}'을(를) 대본 6줄 중 최소 1줄에 반드시 포함해야 합니다. 이 규칙은 절대 생략할 수 없습니다.\n"
        else:
            ctx += "제품 언급 방식: 직접 언급 (성분명이나 제품을 직접 언급 가능)\n"
    return ctx


def _parse_gemini_json(raw_text: str) -> dict:
    """Gemini 응답에서 JSON 추출 (마크다운 코드블록 제거, 유니코드 정규화)"""
    raw = raw_text.strip()
    raw = raw.replace('\u2014', '-').replace('\u2013', '-')
    raw = raw.replace('\u201c', '"').replace('\u201d', '"')
    raw = raw.replace('\u2018', "'").replace('\u2019', "'")
    raw = re.sub(r"^```json\s*\n?", "", raw)
    raw = re.sub(r"\n?\s*```$", "", raw.strip())
    return json.loads(raw)


# ──────────────────────────────────────────────
# Step 2: 제목 생성 (3~4개 옵션)
# ──────────────────────────────────────────────

async def generate_titles(
    topic: str,
    category: str = "general",
    pain_point: str = None,
    ingredient: str = None,
    mention_type: str = None,
    product_name: str = None,
    api_key: str = None,
) -> dict:
    """
    Gemini로 제목 3~4개 생성.
    반환: {"titles": [{"title": "...", "hook": "..."}, ...]}
    """
    client = get_client(api_key)
    category_context = _build_category_context(category, pain_point, ingredient, mention_type, product_name)

    prompt = f"""당신은 YouTube Shorts 전문 카피라이터입니다.
다음 주제에 대해 시선을 사로잡는 한국어 제목을 4개 만들어주세요.

주제: "{topic}"
{category_context}

제목 작성 규칙:
- 최대 16자 이내 (공백 포함)
- 다음 기법 중 하나 이상 활용:
  · FOMO 자극: "이것 모르면 손해", "아직도 이렇게?"
  · 명령형: "절대 하지 마세요", "당장 바꾸세요"
  · 궁금증 유발: "이게 진짜 원인?", "아무도 안 알려주는"
  · 숫자 활용: "3가지 비밀", "5초만에"
- 각 제목마다 왜 이 제목이 효과적인지 한줄 설명(hook)을 달아주세요.

Output ONLY valid JSON:
{{
    "titles": [
        {{"title": "제목1", "hook": "이 제목이 효과적인 이유"}},
        {{"title": "제목2", "hook": "이 제목이 효과적인 이유"}},
        {{"title": "제목3", "hook": "이 제목이 효과적인 이유"}},
        {{"title": "제목4", "hook": "이 제목이 효과적인 이유"}}
    ]
}}"""

    response = await client.aio.models.generate_content(
        model=settings.GEMINI_TEXT_MODEL,
        contents=prompt,
        config=genai.types.GenerateContentConfig(temperature=0.9),
    )
    result = _parse_gemini_json(response.text)

    if "titles" not in result or len(result["titles"]) < 2:
        raise ValueError("Gemini 응답에 titles가 부족합니다")
    return result


# ──────────────────────────────────────────────
# Step 3: 나레이션 생성
# ──────────────────────────────────────────────

async def generate_narration(
    topic: str,
    selected_title: str,
    num_lines: int = 6,
    category: str = "general",
    pain_point: str = None,
    ingredient: str = None,
    mention_type: str = None,
    product_name: str = None,
    api_key: str = None,
) -> dict:
    """
    선택된 제목 기반으로 나레이션 생성.
    반환: {"lines": [{"text": "...", "role": "hook"}, ...]}
    """
    client = get_client(api_key)
    category_context = _build_category_context(category, pain_point, ingredient, mention_type, product_name)

    # 카테고리별 라인 지시 강화
    line_instructions = ""
    if category == "cosmetics":
        if pain_point:
            line_instructions += f"- Line 1~2에서 반드시 '{pain_point}' 고민을 직접 언급하며 공감하세요.\n"
        if ingredient:
            line_instructions += f"- Line 3~5에서 반드시 '{ingredient}' 성분이 왜 효과적인지 설명하세요.\n"
        if mention_type == "comment":
            line_instructions += "- Line 6(CTA)에서 반드시 '궁금하면 댓글!', '댓글로 알려드려요' 등으로 끝내세요. 제품명은 절대 직접 언급하지 마세요.\n"
        elif mention_type == "direct":
            if product_name:
                line_instructions += f"- [필수] Line 4~6 중 최소 1줄에 '{product_name}'이라는 제품명을 반드시 포함하세요. 성분명만 언급하고 제품명을 빠뜨리면 실패입니다.\n"
            else:
                line_instructions += "- 대본에서 성분명이나 제품을 자연스럽게 직접 언급하세요.\n"

    prompt = f"""당신은 YouTube Shorts 나레이션 작가입니다.
아래 제목의 쇼츠 영상을 위한 나레이션 {num_lines}줄을 작성하세요.

제목: "{selected_title}"
주제: "{topic}"
{category_context}

나레이션 작성 규칙:

1. 글자 수: 각 줄 24자 이내 (?,!,. 제외). 24자 초과 시 자막이 3줄이 되어 불가.

2. 말투: 실제 사람이 유튜브 쇼츠를 나레이션하듯 자연스럽고 흐르는 구어체.
   - 어미를 다양하게 섞으세요. 모든 줄이 "~요"나 "~세요"로 끝나면 안 됩니다.
   - 활용 어미: ~죠?, ~잖아요, ~거든요, ~래요, ~다고 해요, ~이에요, ~입니다, ~면 돼요, ~한다는 거!
   - 문장이 자연스럽게 다음 문장으로 이어져야 합니다.

3. 스토리 아크 (반드시 준수):
   Line 1: hook — 공감/충격적 질문
   Line 2: problem — 문제 심화
   Line 3: insight — 반전/핵심 사실
   Line 4: solution1 — 해결책 1
   Line 5: solution2 — 해결책 2
   Line 6: cta — 행동 유도

{line_instructions}

Output ONLY valid JSON:
{{
    "lines": [
        {{"text": "나레이션 문장", "role": "hook"}},
        {{"text": "나레이션 문장", "role": "problem"}},
        {{"text": "나레이션 문장", "role": "insight"}},
        {{"text": "나레이션 문장", "role": "solution1"}},
        {{"text": "나레이션 문장", "role": "solution2"}},
        {{"text": "나레이션 문장", "role": "cta"}}
    ]
}}"""

    response = await client.aio.models.generate_content(
        model=settings.GEMINI_TEXT_MODEL,
        contents=prompt,
        config=genai.types.GenerateContentConfig(temperature=0.8),
    )
    result = _parse_gemini_json(response.text)

    if "lines" not in result or len(result["lines"]) < num_lines:
        raise ValueError("Gemini 응답에 lines가 부족합니다")
    return result


# ──────────────────────────────────────────────
# Step 4: 이미지 프롬프트 + 모션 생성
# ──────────────────────────────────────────────

async def generate_image_prompts(
    narration_lines: list[str],
    style: str,
    category: str = "general",
    api_key: str = None,
) -> dict:
    """
    확정된 나레이션 기반으로 이미지 프롬프트 + 모션 생성.
    반환: {"lines": [{"text": "...", "image_prompt": "...", "motion": "..."}, ...]}
    """
    client = get_client(api_key)
    style_desc = STYLE_SUFFIXES.get(style, style)
    nb2_guide = _load_nb2_guide()

    lines_text = "\n".join([f"  Line {i+1}: \"{line}\"" for i, line in enumerate(narration_lines)])

    cosmetics_guide = ""
    if category == "cosmetics":
        cosmetics_guide = """
[COSMETICS SHOT TYPE GUIDE]
- When narration describes skin problems, damage, texture, or barrier issues, actively use EXTREME CLOSE-UP or MACRO shots (show every pore, crack, flake, redness in microscopic detail).
- When narration describes ingredients or scientific explanation, use MACRO shots of product texture, serum droplets, or skin surface absorbing the product.
- Use photography terms like: 100mm macro lens, clinical lighting, visible micro-cracks, flaky texture, glistening, pearlescent glow, dermatological photography style.
"""

    prompt = f"""You are a visual director for YouTube Shorts.
For each narration line below, create an English image generation prompt and assign a camera motion type.

Narration lines:
{lines_text}

Image style: {style_desc}

[IMAGE PROMPT RULES]
- Refer to the following official Nano Banana 2 prompt guide and apply its techniques:

--- NANO BANANA 2 PROMPT GUIDE ---
{nb2_guide}
--- END GUIDE ---

- Write prompts as NARRATIVE descriptions, not keyword lists.
- Structure: Describe the scene like a story — subject, environment, lighting, mood.
- When depicting people, ALWAYS specify "Korean" (e.g., "a young Korean woman", "a Korean man").
- Keep each prompt under 60 words.
- Do NOT include any text, words, letters, or watermarks.
- Each prompt must describe ONE clear scene.
{cosmetics_guide}
[MOTION RULES]
- Assign a motion type from: {MOTION_TYPES}
- Vary motions — do not repeat the same motion consecutively.
- zoom_in: dramatic reveals, emotional close-ups
- zoom_out: establishing shots, wide scenes
- pan_left/pan_right: horizontal movement
- pan_up: hope/aspiration, pan_down: grounding/reality

Output ONLY valid JSON:
{{
    "lines": [
        {{"text": "나레이션 원문", "image_prompt": "English image description...", "motion": "zoom_in"}},
        ...
    ]
}}"""

    response = await client.aio.models.generate_content(
        model=settings.GEMINI_TEXT_MODEL,
        contents=prompt,
        config=genai.types.GenerateContentConfig(temperature=0.7),
    )
    result = _parse_gemini_json(response.text)

    if "lines" not in result:
        raise ValueError("Gemini 응답에 lines가 없습니다")
    for line in result["lines"]:
        if line.get("motion") not in MOTION_TYPES:
            line["motion"] = "zoom_in"
    return result


# ──────────────────────────────────────────────
# 이미지 프롬프트 변형 (재생성용)
# ──────────────────────────────────────────────

async def korean_to_nb2_prompt(korean_request: str, narration_text: str, api_key: str = None) -> str:
    """
    한글 요청어를 Nano Banana 2용 영어 이미지 프롬프트로 변환.
    """
    client = get_client(api_key)
    nb2_guide = _load_nb2_guide()

    prompt = f"""당신은 이미지 생성 프롬프트 전문가입니다.
사용자의 한글 요청을 Nano Banana 2에 최적화된 영어 이미지 프롬프트로 변환하세요.

나레이션 문맥: "{narration_text}"
사용자 요청: "{korean_request}"

다음 Nano Banana 2 가이드를 참고하세요:
{nb2_guide}

규칙:
- 사용자의 요청 의도를 정확히 반영한 영어 프롬프트를 작성
- 키워드 나열이 아닌 서술형 문장으로 작성 (Narrative over Keywords)
- 사람이 등장할 경우 반드시 "Korean"을 명시
- 60단어 이내
- 텍스트/글자/워터마크 절대 포함 금지
- 카메라, 조명, 분위기를 서술적으로 묘사

영어 프롬프트만 출력하세요. 다른 설명 없이."""

    response = await client.aio.models.generate_content(
        model=settings.GEMINI_TEXT_MODEL,
        contents=prompt,
        config=genai.types.GenerateContentConfig(temperature=0.7),
    )
    return response.text.strip().strip('"')


# ──────────────────────────────────────────────
# Nano Banana 2 이미지 생성
# ──────────────────────────────────────────────

async def generate_image(
    prompt: str,
    style: str,
    output_path: str,
    max_retries: int = 3,
    progress_callback=None,
    job_id: str = None,
    api_key: str = None,
) -> str:
    """
    Nano Banana 2 (Gemini 3.1 Flash Image)로 이미지 생성.
    429 할당량 초과 시 자동 재시도 (최대 max_retries회).
    반환: 저장된 파일 경로
    """
    client = get_client(api_key)
    style_suffix = STYLE_SUFFIXES.get(style, "")
    full_prompt = f"{prompt}, {style_suffix}" if style_suffix else prompt

    for attempt in range(max_retries + 1):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=settings.GEMINI_IMAGE_MODEL,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio="9:16",
                    ),
                ),
            )

            # 응답에서 이미지 데이터 추출
            image_bytes = None
            candidates = response.candidates
            if candidates and candidates[0].content and candidates[0].content.parts:
                for part in candidates[0].content.parts:
                    if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                        image_bytes = part.inline_data.data
                        break

            if not image_bytes:
                if attempt < max_retries:
                    print(f"[RETRY] 이미지 없는 응답, 재시도 ({attempt + 1}/{max_retries})")
                    await asyncio.sleep(3)
                    continue
                raise RuntimeError("이미지 생성 실패: 응답에 이미지 없음")

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            return output_path

        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            is_server_error = "503" in err_str or "UNAVAILABLE" in err_str or "500" in err_str or "INTERNAL" in err_str
            if (is_rate_limit or is_server_error) and attempt < max_retries:
                if is_rate_limit:
                    wait = 30
                    msg = f"1분에 보낼 수 있는 요청 수를 초과했어요. 약 {wait}초 후 자동으로 재시도합니다"
                else:
                    wait = 5 * (attempt + 1)
                    msg = f"AI 서버가 일시적으로 불안정해요. 약 {wait}초 후 자동으로 재시도합니다"
                print(f"[RETRY] {msg} ({attempt + 1}/{max_retries}): {err_str[:80]}")
                if progress_callback and job_id:
                    progress_callback(
                        job_id=job_id,
                        status="generating_images",
                        progress=0.1,
                        step=msg,
                    )
                await asyncio.sleep(wait)
                continue
            raise

    raise RuntimeError(f"이미지 생성 실패: {prompt[:50]}...")


async def generate_all_images(
    job_id: str,
    lines: list[dict],
    style: str,
    storage_dir: str,
    progress_callback=None,
    api_key: str = None,
) -> list[str]:
    """대본의 모든 이미지를 병렬 생성. 반환: 이미지 경로 목록"""
    total = len(lines)

    if progress_callback:
        progress_callback(
            job_id=job_id,
            status="generating_images",
            progress=0.05,
            step=f"이미지 생성 중... 0 / {total}장 완료",
        )

    completed = 0

    async def _generate_and_track(i):
        nonlocal completed
        output_path = os.path.join(storage_dir, "images", f"img_{i:02d}.png")
        result = await generate_image(
            prompt=lines[i]["image_prompt"],
            style=style,
            output_path=output_path,
            progress_callback=progress_callback,
            job_id=job_id,
            api_key=api_key,
        )
        completed += 1
        if progress_callback:
            progress_callback(
                job_id=job_id,
                status="generating_images",
                progress=0.05 + (completed / total) * 0.35,
                step=f"이미지 생성 중... {completed} / {total}장 완료",
            )
        return result

    results = await asyncio.gather(*[_generate_and_track(i) for i in range(total)])
    image_paths = list(results)

    return image_paths
