"""Pydantic 요청/응답 스키마"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class StylePreset(str, Enum):
    REALISTIC = "realistic"
    ANIME = "anime"
    ILLUSTRATION = "illustration"


class TTSEngine(str, Enum):
    EDGE = "edge"
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


# AI 영상 모드 (Ken Burns가 아닌 모든 모드)
AI_VIDEO_MODES = frozenset({"hailuo", "hailuo23", "wan", "kling", "kling26", "veo", "veo_lite"})


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
    mention_type: Optional[str] = None
    product_name: Optional[str] = None


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
    mention_type: Optional[str] = None
    product_name: Optional[str] = None


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


class ScriptLine(BaseModel):
    text: str
    image_prompt: str
    motion: MotionType


class ImagePromptResponse(BaseModel):
    lines: list[ScriptLine]


class JobCreateRequest(BaseModel):
    topic: str
    style: StylePreset
    video_mode: VideoMode = VideoMode.KENBURNS
    tts_engine: TTSEngine = TTSEngine.EDGE
    tts_speed: float = Field(default=1.0, ge=0.5, le=2.0)
    voice_id: Optional[str] = None
    emotion: Optional[str] = None
    title: str
    title_line1: Optional[str] = None
    title_line2: Optional[str] = None
    lines: list[ScriptLine]
    bgm_volume: float = Field(default=0.12, ge=0.0, le=0.5)
    bgm_filename: Optional[str] = None
    bgm_start_sec: float = Field(default=0.0, ge=0.0)


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
