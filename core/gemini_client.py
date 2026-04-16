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
import logging

logger = logging.getLogger(__name__)

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
    key = api_key or settings.GEMINI_API_KEY
    if not key:
        raise RuntimeError("Gemini API 키가 설정되지 않았습니다. 환경변수 또는 api_key 파라미터를 확인해주세요.")
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
            ctx += f"핵심 키워드(이 키워드를 중심축으로 정보 전개): {keyword}\n"
        ctx += (
            "- 사용자에게 가치 있는 지식을 전달하는 영상입니다.\n"
            "- 제품명·브랜드명 언급 금지.\n"
            "- 성분에 대한 '정보 전달'은 허용하나 '구매 권유'는 금지.\n"
            "  OK 예: '효모수는 탈모 예방에 효과적이에요'\n"
            "  NG 예: '효모수 성분 든 제품을 지금 구매하세요'\n"
            "- CTA(마지막 줄)는 블로그·구매·댓글 유도 문구 없이 "
            "정보 마무리로만 자연스럽게 끝내세요.\n"
        )
        return ctx

    # promo
    ctx = "\n[영상 목적: 홍보성 — 네이버 클립 쇼츠용]\n"
    if pain_point:
        ctx += f"타겟 고민: {pain_point}\n"
    if ingredient:
        ctx += f"핵심 성분 또는 제품 역할: {ingredient}\n"
    ctx += (
        "- 제품명은 절대 직접 언급하지 마세요 (성분명·역할 위주).\n"
        "- CTA(마지막 줄)는 네이버 클립 특성상 시청자를 블로그로 유도해야 합니다. "
        "'자세한 건 아래 블로그', '더보기 눌러서 확인', '프로필 링크 확인' 등 "
        "블로그/더보기/프로필 링크로 유도하는 문구로 끝내세요.\n"
    )
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
    client = get_client(api_key)
    category_context = _build_category_context(category, pain_point, ingredient, content_type, keyword)
    if keyword:
        logger.info("나레이션 생성 — 핵심 키워드: %s", keyword)

    # 영상 목적별 라인 지시 강화
    line_instructions = ""
    if category == "cosmetics" and content_type in ("info", "promo"):
        if content_type == "info":
            if keyword:
                line_instructions += (
                    f"- Line 3~5에서 '{keyword}'를 정보의 중심축으로 삼아, "
                    f"이 키워드와 관련된 원리·효능·팁만 선별해서 설명하세요. "
                    f"키워드와 무관한 일반론으로 흐르지 마세요.\n"
                )
            line_instructions += (
                "- 제품명·브랜드명 언급 금지. 성분 정보 전달은 OK "
                "(단 구매 권유는 금지).\n"
                "- Line 1~2: 주제와 관련된 흥미로운 질문이나 공감 유도.\n"
                "- Line 3~5: 원리·효능·팁 중심으로 사실을 설명 "
                "(성분의 효과 설명은 가능, 특정 제품 권유는 불가).\n"
                "- Line 6(CTA): 주제에 어울리는 정보 마무리 "
                "(블로그·구매·댓글 유도 문구 금지).\n"
            )
        else:  # promo
            if pain_point:
                line_instructions += (
                    f"- Line 1~2에서 반드시 '{pain_point}' 고민을 "
                    f"직접 언급하며 공감하세요.\n"
                )
            if ingredient:
                line_instructions += (
                    f"- Line 3~5에서 '{ingredient}'의 핵심 포인트를 요약해서, "
                    f"왜 효과적인지 또는 어떻게 문제를 해결하는지 "
                    f"자연스럽게 설명하세요. 내용이 길면 핵심만 뽑아서 라인에 녹이세요.\n"
                )
            line_instructions += (
                "- 제품명은 절대 말하지 마세요. 성분·역할 중심으로만 표현.\n"
                "- Line 6(CTA)는 반드시 '자세한 내용은 아래 블로그', "
                "'더보기 눌러 확인', '프로필 링크에서 자세히' 중 한 가지 톤으로 "
                "블로그/더보기/프로필 유입을 유도하는 문장으로 마무리하세요. "
                "제품명은 여전히 언급 금지.\n"
            )

    prompt = f"""당신은 유튜브 쇼츠 나레이션 작가입니다. 구독자 수십만의 채널을 만들어본 경험이 있고, "사람이 진짜 말하듯" 자연스러운 스크립트를 쓰는 게 특기입니다.

제목: "{selected_title}"
주제: "{topic}"
{category_context}

# 작업 프로세스 (반드시 2단계로 수행)

## 1단계 — 초안(draft_story): 자유롭게 이야기 완성
먼저 글자 수 제약은 **완전히 무시**하고, 실제 사람이 친구에게 말하듯 자연스러운 {num_lines}줄을 써보세요.
- 각 줄은 앞 줄을 이어받아 **대화가 흐르듯** 연결되어야 합니다.
- 쉼표로 이어지는 라인은 다음 줄과 의미가 실제로 이어져야 함.

## 2단계 — 다듬기(lines): 자연스러운 한 문장으로 완성
초안을 의미 변경 없이 자연스러운 한 문장으로 완성합니다.
- 각 라인은 Veo AI가 만든 4초짜리 영상 한 장면에 대응합니다.
  TTS(1.0~1.2배)로 4초를 자연스럽게 채우는 분량은 대략 24~28자.
- **최대 28자. 초과 금지.**
- 하한은 없습니다. 자연스러움이 우선 — 짧게 써야 자연스러우면 짧게 쓰세요.
  (단, 너무 짧으면 영상이 비어 보이는 점만 참고하세요.)
- 초안이 이미 자연스러우면 그대로 사용 (억지 압축·확장 금지).

# 역할 재정의 — "슬롯"이 아니라 "목적"입니다
- Line 1 (hook): 시청자가 스크롤을 멈출 이유. 강한 질문·공감·충격.
- Line 2 (problem): "그래서 더 답답한 이유"를 드러냄. Line 1을 자연스럽게 받아야 함.
- Line 3 (insight): 판을 뒤집는 반전. "사실은 그게 아니라…" 류.
- Line 4 (solution1): 구체적 해결 방법 1. 명사형보다 동사·상황으로 생생하게.
- Line 5 (solution2): 해결 방법 2 또는 1과의 시너지.
- Line 6 (cta): 독립된 한 문장. Line 5와 자연스럽게 전환되어야 하며, 갑자기 다른 광고 멘트로 튀면 실패.

※ 완결성 규칙:
  - Line 1(hook)만 "~하는 분?", "~는 경험 있으시죠?" 같은
    [대상 지칭 + 의문·감탄] 형태를 짧게 허용합니다.
  - Line 2~6은 주어·술어가 모두 드러나는 완결된 평서문 또는 의문문.
    "~단 거!", "~할 뿐이에요" 같은 생략 파편으로 끝나지 마세요.

{line_instructions}

# 말투 규칙
- 실제 한국인 유튜버가 쇼츠를 녹음하듯 **입말**로. 문어체 금지.
- 구어체 표현(진짜·막·완전·딱 등)은 주제와 맞을 때만 자연스럽게.
- 같은 어미가 2줄 연속 오지 않게 가볍게 변화를 주세요.
  (단, 다양성 때문에 "~대요" 같은 전언형을 억지로 넣지 마세요 — 신뢰감 약화됨.)

# 자주 나오는 실패 패턴 (피해야 할 것들)

## ❌ 실패 1 — "주어 생략으로 의미 붕괴"
NG: "머리카락과 단백질 구조가 비슷해 흡수가 잘 돼요."
    → 뭐가 비슷하다는 건지 음성으로는 파악 불가.
OK: "맥주효모는 머리카락과 단백질 구조가 비슷해요."
    → 주어 명확.

## ❌ 실패 2 — "반전을 무력화하는 절망 비유 (Line 2)"
NG: "좋다는 건 다 해봐도 밑 빠진 독에 물 붓기죠."
    → "모든 시도가 소용없다"로 읽혀 Line 3의 반전("핵심은 X") 무력화.
OK: "좋다는 건 다 해봐도 뿌리부터 튼튼해지는 느낌은 없으셨죠?"
    → "효과가 없었던 원인"을 남겨둬 Line 3 반전이 자연스럽게 연결됨.

## ❌ 실패 3 — "키워드를 추상화해 중심축 이탈 (Line 3·6)"
NG Line 3: "알고 보면 핵심은 '맥주효모' 같은 영양 공급이죠."
NG Line 6: "결국 꾸준한 영양 공급이 가장 중요한 습관이에요."
    → 키워드가 "영양 공급"이라는 일반 개념으로 희석. 시청자 기억에 키워드가 안 남음.
OK Line 3: "사실 두피엔 맥주효모 같은 단백질 성분이 필요해요."
OK Line 6: "맥주효모처럼 모발에 닿는 성분을 꾸준히 챙겨보세요."
    → 키워드가 주어/귀결로 명확히 등장. Line 6에서 재등장해 각인.

## ❌ 실패 4 — "전언형 어미로 신뢰감 약화 (정보성 특화)"
NG: "비타민 B군이 두피 환경까지 건강하게 만든대요."
    → "~대요"는 남의 말 옮기는 느낌. 정보성에서 확신 약화.
OK: "비타민 B군은 두피 환경까지 건강하게 만들어줘요."
    → "~어요/답니다"로 직접 단언. 정보성 신뢰감↑.

## ❌ 실패 5 — "같은 어미 2줄 연속 + 감정 라벨링"
NG:
  Line 1: "밤에 자다가 긁어서 깨는 경험 있으시죠?"
  Line 2: "보습제를 발라도 그때뿐이라 답답하셨죠?"
    → (1) "~죠?" 두 줄 연속으로 단조롭고 뻔함.
    → (2) "답답하셨죠?"가 시청자 감정을 작가가 대신 라벨링 — 강압적.
OK:
  Line 1: "밤에 자다가 긁어서 깼던 경험 있으시죠?"
  Line 2: "보습제를 발라도 그때뿐이잖아요."
    → "~죠?" → "~잖아요"로 어미 자연 전환 + 공감을 공유형으로.
    ("~잖아요" = "우리 둘 다 아는 사실" 톤, 라벨링 없이 공감 형성)

※ 키워드를 따옴표('', "")로 감싸지 마세요 — 음성 나레이션이라 어색합니다.

# 좋은 예시 (참고만 — 이 주제는 예시일 뿐, 실제 입력 주제에 맞춰 작성하세요)

## 예시 A — 홍보성: "모공 속 피지 + 클레이 마스크"
draft_story (자유, 글자수 무제한):
  1. 세수를 아무리 해도 코 주변 피지가 까맣게 보이시죠?
  2. 사실 일반 클렌저로는 모공 깊숙한 피지를 녹이기 어려워요.
  3. 알고 보면 피지는 기름이라 물보다 흡착제로 잘 빠진대요.
  4. 천연 클레이가 모공 속 피지를 자석처럼 끌어당기고,
  5. 시간이 지날수록 피부 결이 부드럽게 정돈되는 걸 느끼실 거예요.
  6. 자세한 사용법은 프로필 아래 블로그에서 확인하실 수 있어요!
lines (자연스러운 완결형, 24~28자 이상적):
  1. 세수해도 코 주변 피지가 까맣게 보이시죠?         (22자)
  2. 일반 클렌저로는 모공 속까지 닿기 어렵거든요.      (23자)
  3. 피지는 기름이라 흡착제로 잘 빠진다고 해요.        (22자)
  4. 천연 클레이가 모공 속 피지를 쫙 끌어당기고,       (22자)
  5. 쓸수록 피부 결이 부드럽게 정돈되는 게 느껴져요.   (24자)
  6. 자세한 사용법은 블로그 링크에서 확인하세요.       (22자)

## 예시 B — 정보성: "자외선이 피부 노화에 미치는 영향"
draft_story (자유):
  1. 흐린 날에는 자외선 차단제를 건너뛰는 분이 많으시죠?
  2. 그런데 흐린 날에도 자외선의 80%가 그대로 도달한다고 해요.
  3. 사실 피부 노화의 주범은 자외선 A가 일으키는 진피층 손상이에요.
  4. 비타민 C는 이렇게 생긴 활성산소를 중화시키는 데 도움을 줘요.
  5. 레티놀은 진피층에서 콜라겐 재생을 돕는 걸로 알려져 있어요.
  6. 매일 바르는 습관이 1년 뒤 피부를 완전히 바꿔놓는다고 합니다.
lines (24~28자):
  1. 흐린 날엔 자외선 차단제 건너뛰시죠?              (19자)
  2. 흐린 날에도 자외선의 80%가 그대로 도달해요.       (24자)
  3. 노화의 주범은 진피층까지 뚫는 자외선 A거든요.     (24자)
  4. 비타민 C는 활성산소 중화에 도움을 줘요.           (21자)
  5. 레티놀은 진피층에서 콜라겐 재생을 돕는다고 해요.   (25자)
  6. 매일 바르는 습관이 1년 뒤 피부를 바꾼다고 합니다.  (25자)

## 예시 C — 정보성(키워드 有): 주제 "탈모 해결법" + 키워드 "맥주효모"
draft_story (자유):
  1. 샤워 후 수챗구멍 머리카락 보면 한숨 나오시죠?
  2. 좋다는 샴푸 여러 개 써봐도 뿌리부터 튼튼해진 적은 없잖아요.
  3. 사실 두피에 진짜 필요한 건 단백질 구조가 비슷한 맥주효모 같은 성분이에요.
  4. 맥주효모는 모발의 케라틴과 구조가 비슷해서 흡수가 잘 됩니다.
  5. 맥주효모엔 비타민 B군도 풍부해서 두피 환경까지 같이 건강해지거든요.
  6. 그러니 맥주효모처럼 모발에 직접 닿는 성분을 챙겨보세요.
lines (최대 28자 — 어미를 다양하게):
  1. 샤워 후 수챗구멍 보면 한숨 나오시죠?                   (20자, ~죠?)
  2. 좋다는 샴푸 써봐도 뿌리 튼튼해진 적 없잖아요.           (22자, ~잖아요)
  3. 사실 두피엔 맥주효모 같은 단백질 성분이 필요해요.       (24자, ~요)
  4. 맥주효모는 모발과 구조가 비슷해 흡수가 잘 됩니다.       (23자, ~습니다)
  5. 비타민 B군도 풍부해서 두피 환경까지 건강해지거든요.     (24자, ~거든요)
  6. 맥주효모처럼 모발에 닿는 성분을 챙겨보세요.             (22자, ~세요)

# 출력 형식

Output ONLY valid JSON:
{{
    "draft_story": [
        "자유 초안 1줄",
        "자유 초안 2줄",
        "자유 초안 3줄",
        "자유 초안 4줄",
        "자유 초안 5줄",
        "자유 초안 6줄"
    ],
    "lines": [
        {{"text": "자연스러운 완결형 1줄 (최대 28자, 24~28자가 이상적)", "role": "hook"}},
        {{"text": "자연스러운 완결형 2줄 (최대 28자, 24~28자가 이상적)", "role": "problem"}},
        {{"text": "자연스러운 완결형 3줄 (최대 28자, 24~28자가 이상적)", "role": "insight"}},
        {{"text": "자연스러운 완결형 4줄 (최대 28자, 24~28자가 이상적)", "role": "solution1"}},
        {{"text": "자연스러운 완결형 5줄 (최대 28자, 24~28자가 이상적)", "role": "solution2"}},
        {{"text": "자연스러운 완결형 6줄 (최대 28자, 24~28자가 이상적)", "role": "cta"}}
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

    # 2-Pass 생성의 초안 단계를 로그로 남겨 품질 점검에 활용
    draft = result.pop("draft_story", None)
    if draft:
        logger.info("나레이션 초안(draft_story): %s", draft)

    # 글자수 경고 (하드 리젝 아님 — 운영 품질 지표로 활용)
    for i, line in enumerate(result["lines"]):
        text = line.get("text", "")
        clean_len = len(re.sub(r"[?!.,~…]", "", text))
        if clean_len > 28:
            logger.warning(
                "Line %d 글자수 초과(%d자, 허용 28자): %s",
                i + 1, clean_len, text,
            )
        elif clean_len < 20 and i != 0:  # Line 1(hook)은 짧아도 OK
            logger.warning(
                "Line %d 너무 짧음(%d자, 이상 24~28자): %s",
                i + 1, clean_len, text,
            )

    return result


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
    elif content_type == "info":
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

    response = client.models.generate_content(
        model=settings.GEMINI_TEXT_MODEL,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=0.7,
            response_mime_type="application/json",
        ),
    )
    result = _parse_gemini_json(response.text)

    if "lines" not in result:
        raise ValueError("Gemini 응답에 lines가 없습니다")
    for line in result["lines"]:
        if line.get("motion") not in MOTION_TYPES:
            line["motion"] = "zoom_in"
        analysis = line.pop("symptom_analysis", None)
        if analysis:
            logger.info("증상 분석: %s → %s", analysis, line.get("image_prompt", "")[:60])
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

    response = client.models.generate_content(
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
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=settings.GEMINI_IMAGE_MODEL,
                contents=contents,
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
    product_image=None,
) -> list[str]:
    """대본의 모든 이미지를 병렬 생성. 반환: 이미지 경로 목록

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

    async def _generate_and_track(i):
        nonlocal completed
        output_path = os.path.join(storage_dir, "images", f"img_{i:02d}.png")

        # CTA 라인이고 제품 이미지가 있으면 접두어 + 참조 이미지 전달
        # (접두어는 호출 시점에만 붙이고 script_json에는 저장하지 않음)
        prompt = lines[i]["image_prompt"]
        refs = None
        if i == total - 1 and product_image is not None:
            prompt = PRODUCT_REFERENCE_PREFIX + prompt
            refs = [product_image]

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
