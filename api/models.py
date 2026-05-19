"""Pydantic 요청/응답 스키마"""

from pydantic import BaseModel, Field, model_validator
from typing import Optional, Literal
from enum import Enum
import datetime


class StylePreset(str, Enum):
    REALISTIC = "realistic"
    ANIME = "anime"
    ILLUSTRATION = "illustration"


class TTSEngine(str, Enum):
    TYPECAST = "typecast"


class MotionType(str, Enum):
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    PAN_UP = "pan_up"
    PAN_DOWN = "pan_down"


class VideoMode(str, Enum):
    KENBURNS = "kenburns"
    HAILUO = "hailuo"
    HAILUO23 = "hailuo23"
    WAN = "wan"
    KLING = "kling"
    VEO = "veo"
    VEO_LITE = "veo_lite"


class JobStatus(str, Enum):
    PENDING = "pending"
    GENERATING_IMAGES = "generating_images"
    PREVIEW_READY = "preview_ready"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    GENERATING_CLIPS = "generating_clips"
    CLIPS_READY = "clips_ready"
    GENERATING_TTS = "generating_tts"
    ASSEMBLING_VIDEO = "assembling_video"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Step 2: 제목 생성 ──

class TitleRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=200)
    category: str = "general"
    pain_point: Optional[str] = None
    ingredient: Optional[str] = None
    content_type: Optional[str] = None  # "info" | "promo" | "promo_comment" (cosmetics 전용)
    keyword: Optional[str] = None  # "info" 전용 — 주제를 좁히는 서브키워드


class TitleOption(BaseModel):
    title: str
    hook: str


class TitleResponse(BaseModel):
    titles: list[TitleOption]


# ── Step 3: 나레이션 생성 ──

class NarrationRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=200)
    selected_title: str = Field(..., min_length=1, max_length=30)
    num_lines: int = Field(default=6, ge=5, le=8)
    category: str = "general"
    pain_point: Optional[str] = None
    ingredient: Optional[str] = None
    content_type: Optional[str] = None  # "info" | "promo" | "promo_comment" (cosmetics 전용)
    keyword: Optional[str] = None  # "info" 전용 — 주제를 좁히는 서브키워드


class NarrationLine(BaseModel):
    text: str
    role: str


class NarrationResponse(BaseModel):
    lines: list[NarrationLine]


# ── Step 4: 이미지 프롬프트 생성 ──

class ImagePromptRequest(BaseModel):
    narration_lines: list[str]
    style: StylePreset = StylePreset.REALISTIC
    category: str = "general"
    topic: str = ""
    content_type: Optional[str] = None  # "info" | "promo" | "promo_comment" (cosmetics 전용)


class ScriptLine(BaseModel):
    line_id: Optional[str] = None
    text: str
    image_prompt: str = ""
    motion: MotionType = MotionType.ZOOM_IN
    asset_version: int = 0
    # 카드 B에서 사용: 줄별 자산 상태 ("pending" | "ready" | "failed")
    status: Literal["pending", "ready", "failed"] = "pending"
    fail_reason: Optional[str] = None
    asset_action: Optional[Literal["ai_image", "ai_clip", "image_upload", "clip_upload"]] = None
    asset_step: Optional[Literal[
        "queued",
        "planning",
        "generating",
        "qa",
        "retrying",
        "saving",
        "converting",
    ]] = None
    asset_message: Optional[str] = None


class ImagePromptResponse(BaseModel):
    lines: list[ScriptLine]


class JobCreateRequest(BaseModel):
    topic: str = ""
    style: StylePreset = StylePreset.REALISTIC
    video_mode: VideoMode = VideoMode.KENBURNS
    tts_engine: TTSEngine = TTSEngine.TYPECAST
    tts_speed: float = Field(default=1.0, ge=0.5, le=2.0)
    voice_id: Optional[str] = None
    emotion: Optional[str] = None
    title: Optional[str] = ""
    title_line1: Optional[str] = None
    title_line2: Optional[str] = None
    lines: list[ScriptLine]
    bgm_volume: float = Field(default=0.12, ge=0.0, le=0.5)
    bgm_filename: Optional[str] = None
    bgm_start_sec: float = Field(default=0.0, ge=0.0)
    product_image_id: Optional[str] = None
    # 음성 단계에서 사전 생성된 TTS 세션 ID (있으면 영상 조립 시 재사용)
    tts_session_id: Optional[str] = None
    # 카드 A("AI가 모두 생성") vs 카드 B("사용자 직접 제공") 분기
    generation_mode: Literal["ai_full", "user_assets"] = "ai_full"
    # 카드 B: 줄별 자산 출처. ["ai"|"image"|"clip", ...] 길이는 lines와 일치해야 함.
    line_sources: list[Literal["ai", "image", "clip"]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_mode_b(self):
        if self.generation_mode == "user_assets":
            if len(self.line_sources) != len(self.lines):
                raise ValueError("line_sources 길이가 lines 길이와 일치해야 합니다")
        return self


# ── 카드 B 전용 ──

class SplitScriptRequest(BaseModel):
    """자유 길이 사용자 대본을 문장 단위로 쪼개기 요청."""
    script: str = Field(..., min_length=10, max_length=5000)


class SplitScriptResponse(BaseModel):
    lines: list[str]


class DraftJobRequest(BaseModel):
    """카드 B용 draft Job 생성 요청 (쪼개진 대본 보유)."""
    lines: list[str] = Field(..., min_length=1, max_length=50)


class DraftJobResponse(BaseModel):
    job_id: str
    lines: list[ScriptLine] = Field(default_factory=list)


class SplitLineRequest(BaseModel):
    """카드 B 카드 안에서 Enter로 줄 분할 요청."""
    line_index: int = Field(..., ge=0)
    before: str = Field(..., min_length=0, max_length=5000)
    after: str = Field(..., min_length=0, max_length=5000)


class SplitLineResponse(BaseModel):
    lines: list[ScriptLine]
    sources: list[Literal["ai", "image", "clip"]]


class EditLineRequest(BaseModel):
    """카드 B 줄 텍스트 편집 sync."""
    line_index: int = Field(..., ge=0)
    text: str = Field(..., min_length=0, max_length=5000)


class MergeLineRequest(BaseModel):
    """카드 B 병합 요청 — line_index 카드를 line_index-1과 합친다."""
    line_index: int = Field(..., ge=1)


class DeleteLineRequest(BaseModel):
    """카드 B 줄 삭제 요청 (× 버튼)."""
    line_index: int = Field(..., ge=0)
    line_id: Optional[str] = None


class TtsPreviewBuildRequest(BaseModel):
    """음성 설정 단계에서 TTS를 미리 생성해 세션에 저장하는 요청."""
    sentences: list[str]
    voice_id: str
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    emotion: Optional[str] = None
    # 카드 B incremental rebuild — sentences와 1:1 대응하는 line_id 목록.
    # 제공되면 기존 세션의 line_id별 text_hash와 비교해 변경 줄만 재생성.
    line_ids: Optional[list[str]] = None
    # 같은 세션으로 재빌드할 때 전달. 음성 시그니처가 같으면 incremental, 다르면 전체 재생성.
    existing_session_id: Optional[str] = None
    # 6초 초과 분리는 첫 빌드에만 적용. incremental 모드(existing_session_id 제공 시)에서는 건너뜀.
    content_type: Optional[str] = None
    topic: Optional[str] = None
    style: Optional[str] = None


# ── 제품 이미지 ──

class UserProductResponse(BaseModel):
    id: str
    name: str
    filename: str
    created_at: datetime.datetime

    class Config:
        from_attributes = True


# ── 응답 ──

class ScriptResponse(BaseModel):
    title: str
    lines: list[ScriptLine]


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float
    current_step: str
    created_at: str
    completed_at: Optional[str] = None
    video_url: Optional[str] = None
    error: Optional[str] = None
    files_expired: bool = False
    days_remaining: Optional[int] = None
    topic: Optional[str] = None
    owner_nickname: Optional[str] = None
    owner_email: Optional[str] = None
    task_id: Optional[str] = None
    task_kind: Optional[str] = None
    task_status: Optional[str] = None
    task_error: Optional[str] = None
    # 작업이력 UI의 "편집 계속" 버튼 노출 여부.
    # reopen_job 라우트 검증 조건과 일치하게 _job_to_response에서 계산. 활성 task는 N+1 회피 위해 생략.
    can_reopen: bool = False


class PreviewResponse(BaseModel):
    title: str
    lines: list[ScriptLine]
    image_urls: list[str]


class ClipPreviewResponse(BaseModel):
    title: str
    lines: list[ScriptLine]
    clip_urls: list[str]
    image_urls: list[str]


# ── 인증 ──

class LoginRequest(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    nickname: Optional[str]
    role: str
    provider: str
    approved: bool = False
    has_gemini_key: bool = False
    has_typecast_key: bool = False
    has_fal_key: bool = False


# ── API 키 설정 ──

class ApiKeysUpdateRequest(BaseModel):
    gemini_api_key: Optional[str] = None
    typecast_api_key: Optional[str] = None
    fal_key: Optional[str] = None


class ApiKeysResponse(BaseModel):
    gemini: Optional[str] = None
    typecast: Optional[str] = None
    fal: Optional[str] = None
