"""SQLite 데이터베이스 연결 및 세션 관리"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from db.models import Base
from config import settings
import os

DATABASE_PATH = os.path.join(settings.STORAGE_DIR, "shorts.db")
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


def init_db():
    """테이블 자동 생성 + 마이그레이션"""
    os.makedirs(settings.STORAGE_DIR, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    # 기존 DB에 새 컬럼이 없으면 추가
    with engine.connect() as conn:
        columns = [row[1] for row in conn.execute(text("PRAGMA table_info(jobs)"))]
        if "bgm_filename" not in columns:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN bgm_filename VARCHAR"))
        if "bgm_start_sec" not in columns:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN bgm_start_sec FLOAT DEFAULT 0.0"))
        if "voice_id" not in columns:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN voice_id VARCHAR"))
        if "emotion" not in columns:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN emotion VARCHAR"))
        if "video_mode" not in columns:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN video_mode VARCHAR DEFAULT 'kenburns'"))
        conn.commit()


def get_db():
    """FastAPI Depends용 DB 세션 제너레이터"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
