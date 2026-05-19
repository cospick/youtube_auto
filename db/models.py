"""SQLAlchemy ORM 모델"""

from sqlalchemy import Column, String, Float, Text, DateTime, Index, Boolean, Integer
from sqlalchemy.orm import declarative_base
import uuid
from core.time_utils import utc_now_naive

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    email = Column(String, unique=True, nullable=False, index=True)
    nickname = Column(String, nullable=True)
    hashed_password = Column(String, nullable=True)
    role = Column(String, default="user")
    provider = Column(String, default="email")
    provider_id = Column(String, nullable=True)
    approved = Column(Boolean, default=False)
    reset_token = Column(String, nullable=True)
    reset_token_expires = Column(DateTime, nullable=True)
    gemini_api_key_enc = Column(String, nullable=True)
    typecast_api_key_enc = Column(String, nullable=True)
    fal_key_enc = Column(String, nullable=True)
    created_at = Column(DateTime, default=utc_now_naive)


class UserBgm(Base):
    __tablename__ = "user_bgms"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex[:12])
    user_id = Column(String, nullable=False, index=True)
    filename = Column(String, nullable=False)
    duration = Column(Float, default=0.0)
    r2_key = Column(String, nullable=False)
    created_at = Column(DateTime, default=utc_now_naive)


class UserProduct(Base):
    __tablename__ = "user_products"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex[:12])
    user_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    r2_key = Column(String, default="")
    created_at = Column(DateTime, default=utc_now_naive)


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex[:12])
    user_id = Column(String, nullable=True, index=True)
    status = Column(String, default="pending")
    progress = Column(Float, default=0.0)
    current_step = Column(String, default="")

    # 입력 파라미터
    topic = Column(Text, default="")
    style = Column(String, default="realistic")
    video_mode = Column(String, default="kenburns")
    tts_engine = Column(String, default="typecast")
    tts_speed = Column(Float, default=1.1)
    voice_id = Column(String, nullable=True)
    emotion = Column(String, nullable=True)
    title = Column(Text, default="")
    title_line1 = Column(String, nullable=True)
    title_line2 = Column(String, nullable=True)
    script_json = Column(Text, default="[]")
    # 카드 A("AI가 모두 생성")는 "ai_full", 카드 B("사용자 직접 제공")는 "user_assets"
    generation_mode = Column(String, default="ai_full")
    # 줄별 자산 출처: ["ai"|"image"|"clip", ...] (길이 == 줄 개수). 카드 B에서만 사용.
    line_sources_json = Column(Text, default="[]")
    # 카드 B 전용: 전체 대본에서 추론한 visual bible + line_id 기반 shot plan.
    visual_plan_json = Column(Text, default="")
    product_image_id = Column(String, nullable=True)
    bgm_volume = Column(Float, default=0.12)
    bgm_filename = Column(String, nullable=True)
    bgm_start_sec = Column(Float, default=0.0)

    # 음성 단계에서 사전 생성한 TTS 세션 ID (있으면 영상 조립 시 재사용)
    tts_session_id = Column(String, nullable=True)

    # 출력
    video_path = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)

    # R2 동기화
    r2_synced = Column(String, default="none")
    files_expired_at = Column(DateTime, nullable=True)

    # 카드 B: 완료 후 사용자가 "다운로드/새 영상"을 누르기 전까지 중간 산출물(images/clips/tts/temp) 보존 여부.
    # 기본 True → 기존 완료 job은 backfill로 "편집 불가" 상태. 신규 완료 job만 worker가 False로 SET.
    intermediates_purged = Column(Boolean, default=True, nullable=False)

    # 카드 B 재제작 시 변경 감지용: 직전 render의 voice 시그니처 + line_id 순서 + line_id별 text_hash JSON.
    last_render_signature = Column(Text, nullable=True)

    # 시간
    created_at = Column(DateTime, default=utc_now_naive)
    completed_at = Column(DateTime, nullable=True)


class JobTask(Base):
    __tablename__ = "job_tasks"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex[:12])
    job_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=True, index=True)
    kind = Column(String, nullable=False, index=True)
    dedupe_key = Column(String, nullable=True, index=True)
    status = Column(String, default="queued", index=True)
    payload_json = Column(Text, default="{}")
    attempt_count = Column(Integer, default=0)
    max_attempts = Column(Integer, default=80)
    next_run_at = Column(DateTime, nullable=True, index=True)
    locked_by = Column(String, nullable=True, index=True)
    locked_until = Column(DateTime, nullable=True, index=True)
    heartbeat_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now_naive)
    updated_at = Column(DateTime, default=utc_now_naive)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
