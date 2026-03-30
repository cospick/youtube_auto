"""SQLAlchemy ORM 모델"""

from sqlalchemy import Column, String, Float, Text, DateTime
from sqlalchemy.orm import declarative_base
import uuid
import datetime

Base = declarative_base()


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex[:12])
    status = Column(String, default="pending")
    progress = Column(Float, default=0.0)
    current_step = Column(String, default="")

    # 입력 파라미터
    topic = Column(Text, default="")
    style = Column(String, default="realistic")
    video_mode = Column(String, default="kenburns")
    tts_engine = Column(String, default="edge")
    tts_speed = Column(Float, default=1.1)
    voice_id = Column(String, nullable=True)
    emotion = Column(String, nullable=True)
    title = Column(Text, default="")
    title_line1 = Column(String, nullable=True)
    title_line2 = Column(String, nullable=True)
    script_json = Column(Text, default="[]")
    bgm_volume = Column(Float, default=0.12)
    bgm_filename = Column(String, nullable=True)
    bgm_start_sec = Column(Float, default=0.0)

    # 출력
    video_path = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)

    # 시간
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
