"""데이터베이스 연결 및 세션 관리 — SQLite / PostgreSQL 양쪽 지원"""

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
from db.models import Base
from config import settings
import os

# ── DATABASE_URL 결정 ──

_raw_url = settings.DATABASE_URL

if _raw_url:
    # Railway는 postgres:// 를 줄 수 있음 → SQLAlchemy는 postgresql:// 필요
    if _raw_url.startswith("postgres://"):
        _raw_url = _raw_url.replace("postgres://", "postgresql://", 1)
    DATABASE_URL = _raw_url
    _is_sqlite = False
else:
    # 로컬 개발: SQLite
    DATABASE_PATH = os.path.join(settings.STORAGE_DIR, "shorts.db")
    DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
    _is_sqlite = True

# ── 엔진 생성 ──

if _is_sqlite:
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
    )
else:
    engine = create_engine(
        DATABASE_URL,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )

SessionLocal = sessionmaker(bind=engine)


# ── 마이그레이션: 누락 컬럼 추가 ──

# 테이블별 추가될 수 있는 컬럼 정의 {테이블: {컬럼명: SQL 타입}}
_MIGRATIONS = {
    "jobs": {
        "bgm_filename": "VARCHAR",
        "bgm_start_sec": "FLOAT DEFAULT 0.0",
        "voice_id": "VARCHAR",
        "emotion": "VARCHAR",
        "video_mode": "VARCHAR DEFAULT 'kenburns'",
        "title_line1": "VARCHAR",
        "title_line2": "VARCHAR",
        "user_id": "VARCHAR",
        "r2_synced": "VARCHAR DEFAULT 'none'",
        "files_expired_at": "TIMESTAMP",
    },
    "users": {
        "gemini_api_key_enc": "VARCHAR",
        "typecast_api_key_enc": "VARCHAR",
        "fal_key_enc": "VARCHAR",
        "approved": "BOOLEAN DEFAULT 0",
    },
}


def _run_migrations():
    """DB 호환 마이그레이션 — 누락된 컬럼을 안전하게 추가"""
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    with engine.connect() as conn:
        for table_name, columns in _MIGRATIONS.items():
            if table_name not in existing_tables:
                continue

            existing_cols = {col["name"] for col in inspector.get_columns(table_name)}

            for col_name, col_type in columns.items():
                if col_name in existing_cols:
                    continue

                if _is_sqlite:
                    # SQLite는 IF NOT EXISTS 미지원 — inspect로 이미 확인함
                    conn.execute(text(
                        f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"
                    ))
                else:
                    # PostgreSQL: IF NOT EXISTS로 동시 실행 안전
                    conn.execute(text(
                        f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {col_name} {col_type}"
                    ))

                # 기존 사용자 자동 승인 (approved 컬럼 최초 추가 시 1회)
                if table_name == "users" and col_name == "approved":
                    conn.execute(text("UPDATE users SET approved = 1"))

        conn.commit()


def init_db():
    """테이블 생성 + 마이그레이션 (서버 시작 시 호출)"""
    if _is_sqlite:
        os.makedirs(settings.STORAGE_DIR, exist_ok=True)

    # DB 연결 테스트
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        db_type = "PostgreSQL" if not _is_sqlite else "SQLite"
        raise RuntimeError(
            f"\n\n  [오류] {db_type} 데이터베이스에 연결할 수 없습니다!\n"
            f"  DATABASE_URL: {DATABASE_URL[:50]}...\n"
            f"  상세: {e}\n"
        ) from e

    # 테이블 생성 + 마이그레이션
    Base.metadata.create_all(bind=engine)
    _run_migrations()


def get_db():
    """FastAPI Depends용 DB 세션 제너레이터"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
