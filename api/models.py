"""Pydantic 요청/응답 스키마"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class StylePreset(str, Enum):
    REALISTIC = "realistic"
    ANIME = "anime"
    THREE_D = "3d"
    ILLUSTRATION = "illustration"
    CINEMATIC = "cinematic"


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


class NarrationLine(BaseModel):
    text: str
    role: str


class NarrationResponse(BaseModel):
    lines: list[NarrationLine]


# ── Step 4: 이미지 프롬프트 생성 ──

class ImagePromptRequest(BaseModel):
    narration_lines: list[str]
    style: StylePreset = StylePreset.CINEMATIC


class ImagePromptResponse(BaseModel):
    lines: list[ScriptLine]


# ── 요청 (기존) ──


class ScriptLine(BaseModel):
    text: str
    image_prompt: str
    motion: MotionType


class JobCreateRequest(BaseModel):
    topic: str
    style: StylePreset
    video_mode: VideoMode = VideoMode.KENBURNS
    tts_engine: TTSEngine = TTSEngine.EDGE
    tts_speed: float = Field(default=1.0, ge=0.5, le=2.0)
    voice_id: Optional[str] = None
    emotion: Optional[str] = None
    title: str
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


class PreviewResponse(BaseModel):
    title: str
    lines: list[ScriptLine]
    image_urls: list[str]


class ClipPreviewResponse(BaseModel):
    title: str
    lines: list[ScriptLine]
    clip_urls: list[str]
    image_urls: list[str]
