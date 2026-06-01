"""Gemini API 래퍼 - 텍스트 생성 + Nano Banana 2 이미지 생성"""

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from config import settings
import asyncio
import base64
import json
import os
import re
import time
import logging

logger = logging.getLogger(__name__)

_nb2_guide = None
_promo_comment_template = None


def _load_nb2_guide() -> str:
    """Nano Banana 2 공식 프롬프트 가이드 로드 (캐싱)"""
    global _nb2_guide
    if _nb2_guide is None:
        guide_path = os.path.join(os.path.dirname(__file__), "nb2_prompt_guide.txt")
        with open(guide_path, "r", encoding="utf-8") as f:
            _nb2_guide = f.read()
    return _nb2_guide


def _load_promo_comment_template() -> str:
    """화장품 '홍보성 (고정댓글 유도형)' 나레이션 템플릿 로드 (캐싱)"""
    global _promo_comment_template
    if _promo_comment_template is None:
        path = os.path.join(os.path.dirname(__file__), "prompts", "promo_comment.md")
        with open(path, "r", encoding="utf-8") as f:
            _promo_comment_template = f.read()
    return _promo_comment_template


# ── promo_comment 나레이션 Structured Output 스키마 ──
# Gemini가 자유 산문으로 작성하면 SDK가 이 스키마에 맞춰 JSON으로 구조화.
# role 필드는 제외 — 서버가 인덱스 기준으로 사후 할당 (line1~4, cta).
class _PromoCommentLine(BaseModel):
    text: str


class _PromoCommentNarration(BaseModel):
    lines: list[_PromoCommentLine] = Field(..., min_length=5, max_length=5)


class _UserAssetsVisualBible(BaseModel):
    main_subject: str | None = None
    primary_settings: list[str] = Field(default_factory=list)
    tone: str = ""
    continuity_rules: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)


class _UserAssetsContinuityAnchor(BaseModel):
    id: str
    description: str


class _UserAssetsPlanLine(BaseModel):
    line_id: str
    line_index: int
    text: str
    beat_role: str = ""
    visual_intent: str = ""
    continuity_anchor: str | None = None
    image_prompt: str
    motion: str = "zoom_in"


class _UserAssetsVisualPlan(BaseModel):
    version: int = 1
    inferred_topic: str = ""
    narrative_summary: str = ""
    visual_bible: _UserAssetsVisualBible = Field(default_factory=_UserAssetsVisualBible)
    continuity_anchors: list[_UserAssetsContinuityAnchor] = Field(default_factory=list)
    lines: list[_UserAssetsPlanLine]


class _UserAssetImageQa(BaseModel):
    relevance_score: float = 0.0
    continuity_score: float = 0.0
    has_readable_text_or_logo: bool = False
    issues: list[str] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    should_retry: bool = False
    retry_instruction: str = ""


STYLE_SUFFIXES = {
    "realistic": "photorealistic, 8k, ultra-detailed, high resolution photography",
    "anime": "anime style, vibrant colors, Japanese animation, detailed illustration",
    "illustration": "digital illustration, artistic, painterly style, concept art",
}

MOTION_TYPES = ["zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down"]

PRODUCT_REFERENCE_PREFIX = (
    "IMPORTANT: This is the FINAL CTA shot — make the PRODUCT the visual hero. "
    "The reference image shows the exact product that must appear — "
    "reproduce it faithfully (same shape, color, label design, packaging).\n\n"
    "Composition rules:\n"
    "- The product dominates the foreground, filling roughly half the frame\n"
    "- Camera close enough that the product feels prominent and detailed\n"
    "- Use shallow depth of field — product tack-sharp, background softly blurred\n"
    "- A simple Korean hand may hold or present the product (no detailed person, "
    "no face in frame)\n"
    "- Soft studio or natural lighting that highlights the product's texture and label\n"
    "- Clean, minimal background — plain surface or softly blurred lifestyle context\n\n"
    "Scene description:\n"
)


def get_client(api_key: str = None) -> genai.Client:
    key = api_key
    if not key:
        raise RuntimeError("Gemini API 키가 설정되지 않았습니다. 설정 화면에서 사용자 본인의 Gemini API 키를 저장해주세요.")
    return genai.Client(api_key=key)


def _build_category_context(
    category: str,
    pain_point: str = None,
    ingredient: str = None,
    content_type: str = None,
    keyword: str = None,
) -> str:
    """화장품 카테고리 + 영상 목적(info/promo)별 컨텍스트 문자열 생성"""
    if category != "cosmetics":
        return ""
    # 안전 디폴트: content_type이 없거나 예상 외 값이면 지시사항 없음
    # (구 프론트 탭이 content_type을 안 보내도 자동 promo로 떨어지지 않도록 방어)
    if content_type not in ("info", "promo"):
        return ""

    if content_type == "info":
        ctx = "\n[영상 목적: 정보성]\n"
        if keyword:
            ctx += f"핵심 키워드: {keyword}\n"
        ctx += "- 제품명·브랜드명 언급 금지.\n"
        return ctx

    # promo
    has_required_inputs = bool(pain_point) or bool(ingredient)
    if has_required_inputs:
        ctx = "\n[화장품 홍보성 — 아래 내용을 대본에 반드시 녹여야 합니다]\n"
    else:
        ctx = "\n[영상 목적: 홍보성]\n"
    if pain_point:
        ctx += f"타겟 고민: {pain_point}\n"
    if ingredient:
        ctx += f"핵심 성분: {ingredient}\n"
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
    content_type: str = None,
    keyword: str = None,
    api_key: str = None,
) -> dict:
    """
    Gemini로 제목 3~4개 생성.
    반환: {"titles": [{"title": "...", "hook": "..."}, ...]}
    """
    if category == "cosmetics" and content_type == "promo_comment":
        return await _generate_titles_promo_comment(topic, api_key)

    client = get_client(api_key)
    category_context = _build_category_context(category, pain_point, ingredient, content_type, keyword)
    if keyword:
        logger.info("제목 생성 — 핵심 키워드: %s", keyword)

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

    response = client.models.generate_content(
        model=settings.GEMINI_TEXT_MODEL,
        contents=prompt,
        config=genai.types.GenerateContentConfig(temperature=0.9),
    )
    result = _parse_gemini_json(response.text)

    if "titles" not in result or len(result["titles"]) < 2:
        raise ValueError("Gemini 응답에 titles가 부족합니다")
    return result


async def _generate_titles_promo_comment(topic: str, api_key: str = None) -> dict:
    """화장품 '홍보성 (고정댓글 유도형)' 제목 생성 — 반말·숫자 훅 포맷."""
    client = get_client(api_key)
    prompt = f"""주제: "{topic}"
이 주제로 YouTube Shorts 제목 4개를 반말 구어체로 만들어.
- 숫자 훅 선호 (예: "단 3초면 끝", "99%가 모르는")
- 강한 호기심·충격으로 시청자가 본문을 반드시 열게 만드는 톤
- 16자 이내
- 제목에 괄호, 태그, 해시태그, 접미사("(댓글)" 같은 것) 절대 넣지 마

Output ONLY valid JSON:
{{
    "titles": [
        {{"title": "제목1", "hook": "왜 효과적인지 한 줄"}},
        {{"title": "제목2", "hook": "왜 효과적인지 한 줄"}},
        {{"title": "제목3", "hook": "왜 효과적인지 한 줄"}},
        {{"title": "제목4", "hook": "왜 효과적인지 한 줄"}}
    ]
}}"""
    response = client.models.generate_content(
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
    content_type: str = None,
    keyword: str = None,
    api_key: str = None,
) -> dict:
    """
    선택된 제목 기반으로 나레이션 생성.
    반환: {"lines": [{"text": "...", "role": "hook"}, ...]}
    """
    if category == "cosmetics" and content_type == "promo_comment":
        return await _generate_narration_promo_comment(topic, selected_title, api_key)

    client = get_client(api_key)
    category_context = _build_category_context(category, pain_point, ingredient, content_type, keyword)
    if keyword:
        logger.info("나레이션 생성 — 핵심 키워드: %s", keyword)

    # 카테고리별 라인 지시
    line_instructions = ""
    if category == "cosmetics" and content_type in ("info", "promo"):
        if content_type == "info":
            if keyword:
                line_instructions += (
                    f"- Line 3~5에서 '{keyword}'를 중심으로 설명하세요.\n"
                )
            line_instructions += (
                "- 순수 정보 전달. 구매 권유 금지.\n"
                "- Line 6(CTA): 정보 마무리 (블로그·구매 유도 금지).\n"
            )
        else:  # promo
            if pain_point:
                line_instructions += (
                    f"- Line 1~2에서 반드시 '{pain_point}' 고민을 "
                    f"직접 언급하며 공감하세요.\n"
                )
            if ingredient:
                line_instructions += (
                    f"- Line 3~5에서 반드시 '{ingredient}' 성분이 "
                    f"왜 효과적인지 설명하세요. 이때 '{ingredient}'를 "
                    f"'작은 입자', '좋은 성분' 같은 추상 개념으로 희석하지 말고 "
                    f"단어 그대로 최소 한 번은 등장시키세요.\n"
                )
            line_instructions += (
                "- Line 6(CTA): '자세한 건 블로그에서', '더보기 눌러 확인' 등 "
                "블로그 유입 유도.\n"
            )

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

    response = client.models.generate_content(
        model=settings.GEMINI_TEXT_MODEL,
        contents=prompt,
        config=genai.types.GenerateContentConfig(temperature=0.8),
    )
    result = _parse_gemini_json(response.text)

    if "lines" not in result or len(result["lines"]) < num_lines:
        raise ValueError("Gemini 응답에 lines가 부족합니다")

    # 글자수 경고 (운영 품질 지표)
    for i, line in enumerate(result["lines"]):
        text = line.get("text", "")
        clean_len = len(re.sub(r"[?!.,~…]", "", text))
        if clean_len > 24:
            logger.warning(
                "Line %d 글자수 초과(%d자, 허용 24자): %s",
                i + 1, clean_len, text,
            )

    # 키워드 반영 경고 (운영 품질 지표)
    joined = " ".join(line.get("text", "") for line in result["lines"])
    if pain_point:
        pain_tokens = [t.strip() for t in pain_point.split(",") if t.strip()]
        if pain_tokens and not any(t in joined for t in pain_tokens):
            logger.warning("나레이션에 pain_point 미반영: %s", pain_point)
    if ingredient:
        ingredient_head = ingredient.split()[0] if ingredient.split() else ingredient
        if ingredient_head and ingredient_head not in joined:
            logger.warning("나레이션에 ingredient 미반영: %s", ingredient)

    return result


async def _generate_narration_promo_comment(
    topic: str,
    selected_title: str,  # unused — 시그니처 유지
    api_key: str = None,
) -> dict:
    """화장품 '홍보성 (고정댓글 유도형)' 나레이션 — Structured Output 방식.

    프롬프트는 JSON 지시 없이 자유 산문으로 요청하고, SDK의 response_schema로
    JSON 구조를 강제한다. 이 구조가 Gemini Web UI와 동일한 입력 조건을 재현해
    본연의 창작 품질을 확보한다. role 필드는 서버가 인덱스 기준 사후 할당.
    """
    client = get_client(api_key)
    template = _load_promo_comment_template()
    prompt = template.replace("{{TOPIC}}", topic)

    response = client.models.generate_content(
        model=settings.GEMINI_TEXT_MODEL,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=0.9,
            response_mime_type="application/json",
            response_schema=_PromoCommentNarration,
        ),
    )

    # response.parsed 우선 사용 (SDK가 Pydantic 인스턴스로 변환), 실패 시 텍스트 파싱 폴백
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        lines_data = [{"text": line.text} for line in parsed.lines]
    else:
        result = _parse_gemini_json(response.text)
        raw_lines = result.get("lines") or []
        lines_data = [{"text": l.get("text", "")} for l in raw_lines]

    if len(lines_data) != 5:
        raise ValueError(f"promo_comment 나레이션은 5줄이어야 합니다 (실제: {len(lines_data)}줄)")

    # role 필드를 서버에서 인덱스 기준 할당 (다운스트림 파이프라인 호환)
    role_labels = ["line1", "line2", "line3", "line4", "cta"]
    for i, line in enumerate(lines_data):
        line["role"] = role_labels[i]

    return {"lines": lines_data}


# ──────────────────────────────────────────────
# Step 4: 이미지 프롬프트 + 모션 생성
# ──────────────────────────────────────────────

async def generate_image_prompts(
    narration_lines: list[str],
    style: str,
    category: str = "general",
    topic: str = "",
    content_type: str = None,
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

    # CTA 라인 비주얼 가이드 — content_type에 따라 분기
    if content_type == "promo":
        cta_guide = """
[CTA LINE GUIDE — LAST NARRATION LINE]
The LAST line is always a CTA (call-to-action). Compose this as a
PRODUCT-HERO shot, NOT a person shot:
- The PRODUCT is the main subject, dominating the frame (roughly half the frame)
- Close-up or medium close-up of the product on a clean surface,
  or held by a simple Korean hand
- NO face in frame (a hand is OK; if any person appears, no face visible)
- Shallow depth of field — product tack-sharp, background softly blurred
- Premium commercial aesthetic, soft studio or natural lighting
- Clean minimal background

EXCEPTION: This line is exempt from the "never use the same distance as
previous line" rule above. Use CLOSE-UP or MEDIUM regardless of line 5's distance.
"""
    elif content_type in ("info", "promo_comment"):
        cta_guide = """
[CTA LINE GUIDE — INFO CLOSURE]
The LAST line closes the topic naturally with an informational wrap-up.
- Compose as a lifestyle or topic-relevant shot matching the narration
- NO product hero shot, NO specific product featured
- Natural continuation of the previous scenes' tone
"""
    else:
        cta_guide = ""

    cosmetics_guide = ""
    if category == "cosmetics":
        cosmetics_guide = """
[COSMETICS VISUAL DIRECTION GUIDE]

When the narration describes a skin/body concern, complete this analysis
and output it in the "symptom_analysis" field BEFORE writing the image prompt.
Skip this analysis for lines about ingredients, product, or general scenes.

STEP 1 — VISUAL TRANSLATION
  What does this symptom actually look like?
  - Visible condition (홍조, 여드름, 다크서클): describe exact visual appearance
  - Sensation (가려움, 당김, 따가움): find the visible behavior or sign
  - Non-skin (탈모, 손톱): identify the correct body area and visual indicator

STEP 2 — DISTINGUISH FROM LOOKALIKES
  What could this be visually confused with?
  What visual detail differentiates them?

STEP 3 — PRECISE WORD SELECTION
  Choose English words that specifically describe THIS condition's visual signature
  and cannot be confused with the lookalike from Step 2.

[EXAMPLES]
Example — 홍조:
  symptom_analysis: "Smooth redness on cheeks/nose. Confused with acne — but flush is a gradient, acne is raised bumps. USE: flushed, deep red hue"
  image_prompt: "her cheeks and nose noticeably flushed with a deep red hue"

Example — 가려움증:
  symptom_analysis: "Invisible sensation → show scratching behavior. Confused with injury — but scratching is gentle/repetitive. USE: scratching, irritated"
  image_prompt: "a Korean woman uncomfortably scratching her forearm, faint pink streaks on sensitive skin"

[CAMERA DISTANCE GUIDE]
Vary the camera distance across lines. NEVER use the same distance for consecutive lines.

EXTREME MACRO — frame fills with ~2cm² of the target surface.
  Individual pores, microscopic cracks, flaky layers, or product texture visible.
  Subject must be a single body part: "a Korean woman's cheek" NOT "a Korean woman".
  Keywords: extreme macro photography, microscopic details, sharp focus,
  100mm macro lens at minimum focus distance, harsh/dramatic clinical lighting.
  Use for: skin/hair/nail problem detail (lines 1-2), product texture on target area.
  At least ONE extreme macro MUST appear in lines 1-3.

CLOSE-UP — a single feature fills the frame (cheek, nose bridge, scalp, nail).
  Keywords: extreme close-up, highly detailed, clinical lighting.
  Use for: symptom close-up, product application moment.

PORTRAIT — full face visible, expressions clear. 85mm lens, shallow depth of field.
  Use for: emotional reaction, before/after transformation.

MEDIUM — face + shoulders + environment context.
  Use for: lifestyle scene, product-in-hand (NOT for CTA — see CTA LINE GUIDE below).

[INGREDIENT/SOLUTION LINE GUIDE]
When narration describes an ingredient or how the product works:
- The PRODUCT TEXTURE is the protagonist, NOT the person.
- Show the product meeting its target area at EXTREME MACRO or CLOSE-UP distance.
- Focus on the product's texture (glistening, translucent, viscous, milky, pearlescent).
- Determine the target area from the video topic:
  skincare → skin surface, shampoo → hair/scalp, lip care → lips, nail care → cuticle
- Keywords: being applied, sinking into, absorbed, melting into, lathering, soft studio lights.
- Do NOT show a person's expression — show only the product interacting with the target.

__CTA_GUIDE_PLACEHOLDER__

[REALISM RULE]
- ALWAYS depict real people, real skin, and real products in photorealistic style.
- NEVER use metaphors, diagrams, 3D renders, scientific visualizations, or abstract art.
- "피부 장벽이 무너졌다" → show real damaged/irritated skin, NOT a crumbling brick wall.
- "성분이 흡수된다" → show product being applied to real skin, NOT molecular diagrams.
"""
        cosmetics_guide = cosmetics_guide.replace("__CTA_GUIDE_PLACEHOLDER__", cta_guide)

    if category == "cosmetics":
        output_format = """Output ONLY valid JSON:
{{
    "lines": [
        {{"text": "나레이션 원문", "symptom_analysis": "Steps 1-3 reasoning here (or null if not a symptom line)", "image_prompt": "English image description...", "motion": "zoom_in"}},
        ...
    ]
}}"""
    else:
        output_format = """Output ONLY valid JSON:
{{
    "lines": [
        {{"text": "나레이션 원문", "image_prompt": "English image description...", "motion": "zoom_in"}},
        ...
    ]
}}"""

    prompt = f"""You are a visual director for YouTube Shorts.
For each narration line below, create an English image generation prompt and assign a camera motion type.

Video topic: {topic}

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

{output_format}"""

    import asyncio as _asyncio

    last_err = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=settings.GEMINI_TEXT_MODEL,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.7,
                    response_mime_type="application/json",
                ),
            )
            result = _parse_gemini_json(response.text)

            # 관대한 파싱: Gemini가 가끔 lines 키 없이 배열을 바로 주거나
            # 다른 키로 감싸서 주는 현상을 복구한다.
            if "lines" not in result:
                if isinstance(result, list):
                    result = {"lines": result}
                elif isinstance(result, dict):
                    list_values = [v for v in result.values() if isinstance(v, list)]
                    if len(list_values) == 1:
                        result = {"lines": list_values[0]}
                    else:
                        raise ValueError(
                            f"Gemini 응답 구조 불일치 (lines 키 없음, 배열 후보 {len(list_values)}개)"
                        )
                else:
                    raise ValueError(f"Gemini 응답이 dict/list가 아님: {type(result).__name__}")

            if not isinstance(result.get("lines"), list) or len(result["lines"]) == 0:
                raise ValueError("lines가 비어있거나 배열이 아님")

            for line in result["lines"]:
                if line.get("motion") not in MOTION_TYPES:
                    line["motion"] = "zoom_in"
                analysis = line.pop("symptom_analysis", None)
                if analysis:
                    logger.info("증상 분석: %s → %s", analysis, line.get("image_prompt", "")[:60])
            return result
        except Exception as e:
            last_err = e
            logger.warning(
                "[generate_image_prompts] 시도 %d/3 실패: %s", attempt + 1, e
            )
            if attempt < 2:
                await _asyncio.sleep(1.5 ** attempt)

    raise ValueError(f"Gemini 이미지 프롬프트 생성 3회 모두 실패: {last_err}")


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
- 사용자의 한글 요청과 나레이션이 인물(성별/연령/실명)을 명시하지 않으면 "a Korean woman in her 20s"를 기본값으로 사용
- 사용자가 명시적으로 다른 인물(남자, 노인, 아이 등)을 적었다면 그 묘사를 그대로 따른다
- 60단어 이내
- 텍스트/글자/워터마크 절대 포함 금지
- 카메라, 조명, 분위기를 서술적으로 묘사

영어 프롬프트만 출력하세요. 다른 설명 없이."""

    response = client.models.generate_content(
        model=settings.GEMINI_TEXT_MODEL,
        contents=prompt,
        config=genai.types.GenerateContentConfig(temperature=0.7),
    )
    return response.text.strip().strip('"')


# ──────────────────────────────────────────────
# 카드 B: 전체 대본 기반 Visual Plan
# ──────────────────────────────────────────────

async def generate_user_assets_visual_plan(
    lines: list[dict],
    sources: list[str],
    style: str,
    api_key: str = None,
) -> dict:
    """카드 B 전용: 주제 없는 사용자 대본에서 visual bible + 줄별 shot plan 생성."""
    from core.user_assets_visual import VISUAL_PLAN_VERSION

    client = get_client(api_key)
    nb2_guide = _load_nb2_guide()
    style_desc = STYLE_SUFFIXES.get(style, style)

    script_lines = []
    for i, line in enumerate(lines):
        script_lines.append({
            "index": i + 1,
            "line_id": line.get("line_id") or f"idx:{i}",
            "text": line.get("text") or "",
            "source": sources[i] if i < len(sources) else "ai",
        })

    prompt = f"""You are a visual director for a Korean YouTube Shorts editor.
The user did NOT provide a separate topic and there is NO CTA line. Infer the topic only from the full script.

Create a Card B visual plan for coherent AI image generation.

Full script lines as JSON:
{json.dumps(script_lines, ensure_ascii=False, indent=2)}

Image style: {style_desc}

Official Nano Banana 2 guide to apply:
--- GUIDE ---
{nb2_guide}
--- END GUIDE ---

Planning rules:
- First infer the actual topic, thesis, emotional arc, and visual continuity from the full script.
- Treat the last line as a normal narrative ending, NOT a CTA, NOT a product hero shot.
- Use all lines as context, including lines whose source is image or clip, because they affect story continuity.
- For each line, write a specific English image prompt that fits that exact line and the surrounding lines.
- Avoid generic stock scenes. Make each shot carry the script's logic.
- Maintain continuity using anchors such as same person, same workspace, same device setup, same mood, or a deliberate contrast.
- If people appear, default to "a Korean woman in her 20s" unless the script's wording clearly specifies a different person (e.g., an elderly man, a child, a named real person, an explicitly male or older subject). When the script is ambiguous about who appears, always choose a Korean woman in her 20s.
- If the script names a real product, app, platform, or brand, keep that context naturally when it is important to the scene.
- Avoid captions, subtitles, watermarks, and readable screen text.
- Each image prompt must be narrative, one clear scene, under 70 words.
- Assign one motion from: {MOTION_TYPES}
- Vary motion naturally; do not repeat one motion for every line.

Output ONLY valid JSON in this exact shape:
{{
  "version": {VISUAL_PLAN_VERSION},
  "inferred_topic": "one sentence in Korean",
  "narrative_summary": "one concise Korean paragraph",
  "visual_bible": {{
    "main_subject": "stable subject; if the script does not clearly state who the subject is, set this to 'a Korean woman in her 20s'",
    "primary_settings": ["..."],
    "tone": "...",
    "continuity_rules": ["..."],
    "avoid": ["..."]
  }},
  "continuity_anchors": [
    {{"id": "anchor_short_id", "description": "what should remain visually consistent"}}
  ],
  "lines": [
    {{
      "line_id": "same line_id from input",
      "line_index": 1,
      "text": "same Korean text from input",
      "beat_role": "hook|context|example|turn|contrast|explanation|ending",
      "visual_intent": "what this shot must communicate",
      "continuity_anchor": "anchor_short_id or null",
      "image_prompt": "English Nano Banana 2 prompt",
      "motion": "zoom_in"
    }}
  ]
}}"""

    last_err = None
    result = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=settings.GEMINI_TEXT_MODEL,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.35,
                    response_mime_type="application/json",
                    response_schema=_UserAssetsVisualPlan,
                ),
            )
            parsed = getattr(response, "parsed", None)
            if isinstance(parsed, _UserAssetsVisualPlan):
                result = parsed.model_dump()
            elif isinstance(parsed, dict):
                result = parsed
            else:
                result = _parse_gemini_json(response.text or "")
            break
        except Exception as e:
            last_err = e
            logger.warning(
                "[generate_user_assets_visual_plan] 시도 %d/3 실패: %s",
                attempt + 1,
                e,
            )
            if attempt < 2:
                await asyncio.sleep(1.2 ** attempt)

    if result is None:
        raise ValueError(f"카드 B visual plan JSON 생성 실패: {last_err}")

    if not isinstance(result, dict):
        raise ValueError("카드 B visual plan 응답이 JSON 객체가 아닙니다")

    plan_lines = result.get("lines")
    if not isinstance(plan_lines, list) or not plan_lines:
        raise ValueError("카드 B visual plan에 lines가 없습니다")

    input_by_id = {str(line["line_id"]): line for line in script_lines}
    normalized_lines = []
    by_id = {
        str(line.get("line_id")): line
        for line in plan_lines
        if isinstance(line, dict) and line.get("line_id")
    }
    for i, input_line in enumerate(script_lines):
        line_id = str(input_line["line_id"])
        item = by_id.get(line_id)
        if not isinstance(item, dict):
            raise ValueError(f"카드 B visual plan 누락 line_id={line_id}")
        item["line_id"] = line_id
        item["line_index"] = i + 1
        item["text"] = input_by_id[line_id]["text"]
        if item.get("motion") not in MOTION_TYPES:
            item["motion"] = MOTION_TYPES[i % len(MOTION_TYPES)]
        if not str(item.get("image_prompt") or "").strip():
            raise ValueError(f"카드 B visual plan 프롬프트 누락 line_id={line_id}")
        normalized_lines.append(item)

    result["version"] = VISUAL_PLAN_VERSION
    result["lines"] = normalized_lines
    result.setdefault("visual_bible", {})
    result.setdefault("continuity_anchors", [])
    result.setdefault("inferred_topic", "")
    result.setdefault("narrative_summary", "")
    return result


def _is_soft_text_or_logo_issue(issue: str) -> bool:
    """Incidental laptop/device details should not hard-fail a usable image."""
    s = (issue or "").lower()
    soft_terms = (
        "small logo",
        "tiny logo",
        "subtle logo",
        "laptop lid",
        "apple logo",
        "brand logo",
        "brand mark",
        "keyboard",
        "keyboard characters",
        "keyboard letters",
        "ui-like",
        "blurred ui",
        "incidental",
    )
    hard_terms = (
        "watermark",
        "caption",
        "subtitle",
        "large text",
        "central text",
        "readable sentence",
        "readable paragraph",
        "screen text",
        "ui screenshot",
        "title text",
    )
    if any(term in s for term in hard_terms):
        return False
    return any(term in s for term in soft_terms)


async def evaluate_user_asset_image(
    image_path: str,
    plan: dict,
    line_plan: dict,
    final_prompt: str,
    api_key: str = None,
) -> dict:
    """생성 이미지가 카드 B shot plan과 맞는지 vision QA. 실패 시 skipped를 반환."""
    try:
        from PIL import Image

        client = get_client(api_key)
        image = Image.open(image_path)
        image.load()
        qa_prompt = f"""You are a practical visual QA reviewer.
Evaluate whether this generated vertical image matches the intended YouTube Shorts shot.

Inferred topic: {plan.get("inferred_topic", "")}
Narrative summary: {plan.get("narrative_summary", "")}
Visual bible: {json.dumps(plan.get("visual_bible", {}), ensure_ascii=False)}
Line text: {line_plan.get("text", "")}
Visual intent: {line_plan.get("visual_intent", "")}
Image prompt sent to the image model: {final_prompt}

Check for:
- Relevance to the line and overall inferred topic
- Continuity with the visual bible
- Blocking visual text problems: large captions, subtitles, watermarks, title text, readable screen text, or UI screenshots that draw attention
- Non-blocking warnings: small laptop logos, tiny incidental brand marks, keyboard letters, or blurred UI-like details
- Generic stock-photo feeling that fails to communicate the line

Important policy:
- Do NOT block an otherwise relevant image only because a small laptop logo, keyboard letters, or tiny incidental brand mark is visible.
- Put those small incidental details in "warnings".
- Put only serious text/watermark/UI problems or relevance/continuity failures in "blocking_issues".

Output ONLY valid JSON:
{{
  "relevance_score": 0.0,
  "continuity_score": 0.0,
  "has_readable_text_or_logo": false,
  "issues": ["short issue strings"],
  "blocking_issues": ["serious issues that make the image unusable"],
  "warnings": ["minor issues that should be improved but can be accepted"],
  "should_retry": false,
  "retry_instruction": "one concise English correction, or empty string"
}}"""
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=settings.GEMINI_TEXT_MODEL,
            contents=[qa_prompt, image],
            config=genai.types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=_UserAssetImageQa,
            ),
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, _UserAssetImageQa):
            result = parsed.model_dump()
        elif isinstance(parsed, dict):
            result = parsed
        else:
            result = _parse_gemini_json(response.text or "")
        relevance = float(result.get("relevance_score", 0.0) or 0.0)
        continuity = float(result.get("continuity_score", 0.0) or 0.0)
        has_text = bool(result.get("has_readable_text_or_logo"))
        retry_instruction = str(result.get("retry_instruction") or "").strip()

        issues = result.get("issues") if isinstance(result.get("issues"), list) else []
        blocking_issues = result.get("blocking_issues") if isinstance(result.get("blocking_issues"), list) else []
        warnings = result.get("warnings") if isinstance(result.get("warnings"), list) else []

        if has_text and not blocking_issues and not warnings:
            for issue in issues:
                if _is_soft_text_or_logo_issue(str(issue)):
                    warnings.append(str(issue))
                else:
                    blocking_issues.append(str(issue))

        low_relevance = relevance < 0.62
        low_continuity = continuity < 0.5
        if low_relevance:
            blocking_issues.append("Image is not relevant enough to the line or inferred topic")
        if low_continuity:
            blocking_issues.append("Image continuity is too weak for the visual plan")

        blocking = bool(blocking_issues)
        should_retry = blocking or bool(warnings) or bool(result.get("should_retry"))
        if should_retry and not retry_instruction:
            if blocking:
                retry_instruction = "Regenerate the image with clearer relevance to the line, stronger visual continuity, and no large readable text, UI screenshots, or watermarks."
            else:
                retry_instruction = "Regenerate with an unbranded device, no visible logo, and a blank or softly blurred screen while keeping the same composition."
        return {
            "status": "checked",
            "relevance_score": relevance,
            "continuity_score": continuity,
            "has_readable_text_or_logo": has_text,
            "issues": issues,
            "blocking": blocking,
            "blocking_issues": blocking_issues,
            "warnings": warnings,
            "should_retry": should_retry,
            "retry_instruction": retry_instruction,
        }
    except Exception as e:
        logger.warning("[user_assets_image_qa] skipped: %s", e)
        return {
            "status": "skipped",
            "blocking": False,
            "should_retry": False,
            "issues": [str(e)[:160]],
            "blocking_issues": [],
            "warnings": [str(e)[:160]],
            "retry_instruction": "",
        }


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
    reference_images: list = None,
) -> str:
    """
    Nano Banana 2 (Gemini 3.1 Flash Image)로 이미지 생성.
    429 할당량 초과 시 자동 재시도 (최대 max_retries회).
    reference_images: PIL.Image 리스트. 제공 시 contents에 함께 전달되어 이미지 편집/합성에 사용.
    반환: 저장된 파일 경로
    """
    client = get_client(api_key)
    style_suffix = STYLE_SUFFIXES.get(style, "")
    full_prompt = f"{prompt}, {style_suffix}" if style_suffix else prompt

    contents = [full_prompt]
    if reference_images:
        contents.extend(reference_images)

    for attempt in range(max_retries + 1):
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.models.generate_content,
                    model=settings.GEMINI_IMAGE_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(
                            aspect_ratio="9:16",
                        ),
                    ),
                ),
                timeout=240,
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
            is_server_error = (
                "503" in err_str or "UNAVAILABLE" in err_str
                or "500" in err_str or "502" in err_str or "504" in err_str
                or "INTERNAL" in err_str or "DEADLINE_EXCEEDED" in err_str
                or "TimeoutError" in type(e).__name__ or "timed out" in err_str.lower()
            )
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
    product_image=None,
) -> list[str]:
    """대본의 모든 이미지를 생성. 반환: 이미지 경로 목록

    무료 티어 분당 한도(429) 보호를 위해 한꺼번에 보내지 않고,
    동시 IMAGE_GEN_CONCURRENCY장 + 장 사이 IMAGE_GEN_INTERVAL_SEC초 간격으로
    천천히 순차 전송한다. (settings에서 조정 가능)

    product_image: PIL.Image — 제공 시 마지막 라인(CTA)에만 참조 이미지로 전달.
    """
    total = len(lines)

    if progress_callback:
        progress_callback(
            job_id=job_id,
            status="generating_images",
            progress=0.05,
            step=f"이미지 생성 중... 0 / {total}장 완료",
        )

    completed = 0
    # 동시 생성 장수 제한 (무료 티어 분당 한도 보호)
    sem = asyncio.Semaphore(max(1, settings.IMAGE_GEN_CONCURRENCY))

    async def _generate_and_track(i):
        nonlocal completed
        # 장마다 출발을 IMAGE_GEN_INTERVAL_SEC초씩 늦춰, 분당 요청이 몰리지 않게 분산.
        # (동시 1장이면 사실상 i번째 장은 i*간격 시점에 시작)
        start_delay = i * settings.IMAGE_GEN_INTERVAL_SEC
        if start_delay > 0:
            await asyncio.sleep(start_delay)
        output_path = os.path.join(storage_dir, "images", f"img_{i:02d}.png")

        # CTA 라인이고 제품 이미지가 있으면 접두어 + 참조 이미지 전달
        # (접두어는 호출 시점에만 붙이고 script_json에는 저장하지 않음)
        prompt = lines[i]["image_prompt"]
        refs = None
        if i == total - 1 and product_image is not None:
            prompt = PRODUCT_REFERENCE_PREFIX + prompt
            refs = [product_image]

        async with sem:
            result = await generate_image(
                prompt=prompt,
                style=style,
                output_path=output_path,
                progress_callback=progress_callback,
                job_id=job_id,
                api_key=api_key,
                reference_images=refs,
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
