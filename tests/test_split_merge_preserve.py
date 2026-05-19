"""split-line / merge-line의 AI 이미지 보존 동작 단위 테스트.

코덱스 리뷰 반영 — Scope A:
- split 가운데 분기는 first 줄의 이미지/자산을 보존 (edit-line 정책과 일관)
- merge 분기는 prev 줄의 이미지/자산을 보존, 사라지는 L 줄 자산만 정리

이전엔 _is_ai_owned_asset이 True면 _discard_line_assets + clear_line_visual_fields로
이미지를 자동 삭제했지만, 사용자 의도에 반하는 자산 손실이라 제거.
"""

import json
import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.routes import preview as preview_module
from api.routes.preview import split_line, merge_line
from api.models import SplitLineRequest, MergeLineRequest
from db.database import Base
from db.models import Job, User
from core.user_assets_visual import new_line_id


# ── fixtures ──────────────────────────────────────────────


@pytest.fixture
def env(tmp_path, monkeypatch):
    """임시 storage + 임시 SQLite + 사용자 1명."""
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    # api.routes.preview는 settings.STORAGE_DIR를 os.path.join에 직접 쓴다.
    monkeypatch.setattr(preview_module.settings, "STORAGE_DIR", str(storage_root))

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = Session()

    user = User(
        id="testuser0001",
        email="test@example.com",
        nickname="tester",
        approved=True,
    )
    db.add(user)
    db.commit()

    yield storage_root, db, user
    db.close()


def _make_line(line_id: str, text: str, *, status: str = "ready") -> dict:
    return {
        "line_id": line_id,
        "text": text,
        "image_prompt": "test prompt for " + text,
        "motion": "zoom_in",
        "asset_version": 1,
        "status": status,
        "visual_text_hash": "abc123",
        "visual_anchor": "anchor",
        "visual_intent": "intent",
    }


def _make_job(db, user, lines, sources, job_id="testjob01234") -> Job:
    job = Job(
        id=job_id,
        user_id=user.id,
        status="preview_ready",
        generation_mode="user_assets",
        script_json=json.dumps(lines, ensure_ascii=False),
        line_sources_json=json.dumps(sources),
    )
    db.add(job)
    db.commit()
    return job


def _write_image_for_line(storage_root: Path, job_id: str, line_id: str) -> Path:
    """images/line_{line_id}.png 파일 생성. line_asset_rel("image", line) 경로와 일치."""
    p = storage_root / job_id / "images" / f"line_{line_id}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"fake image data")
    return p


# ── tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_split_middle_preserves_ai_image(env):
    """AI 이미지 ready 줄을 가운데 split → first 보존, second 빈 줄."""
    storage_root, db, user = env

    lid_a = new_line_id()
    lid_b = new_line_id()
    lines = [
        _make_line(lid_a, "맥북 사면 못 돌아가는 이유"),
        _make_line(lid_b, "두 번째 줄 텍스트"),
    ]
    sources = ["ai", "ai"]
    job = _make_job(db, user, lines, sources)
    img_a = _write_image_for_line(storage_root, job.id, lid_a)
    img_b = _write_image_for_line(storage_root, job.id, lid_b)
    assert img_a.exists() and img_b.exists()

    body = SplitLineRequest(line_index=0, before="맥북 사면", after="못 돌아가는 이유")
    await split_line(body, job.id, db, user)

    db.refresh(job)
    new_lines = json.loads(job.script_json)
    new_sources = json.loads(job.line_sources_json)

    # first 줄: line_id·status·source·이미지 모두 보존 (핵심 검증)
    assert new_lines[0]["line_id"] == lid_a
    assert new_lines[0]["text"] == "맥북 사면"
    assert new_lines[0]["status"] == "ready", "AI 이미지 ready 상태가 보존돼야 함"
    assert new_sources[0] == "ai"
    assert img_a.exists(), "first 줄 이미지 파일이 보존돼야 함 (이게 핵심 버그 수정)"

    # second 줄: 새 line_id, pending, ai source
    assert new_lines[1]["text"] == "못 돌아가는 이유"
    assert new_lines[1]["line_id"] != lid_a
    assert new_lines[1]["status"] == "pending"
    assert new_sources[1] == "ai"

    # 원래 다음 줄 (이제 인덱스 2)도 그대로
    assert new_lines[2]["line_id"] == lid_b
    assert new_sources[2] == "ai"
    assert img_b.exists()


@pytest.mark.asyncio
async def test_split_middle_preserves_user_image(env):
    """사용자 업로드(source='image') 줄을 가운데 split → 동일하게 보존 (회귀)."""
    storage_root, db, user = env

    lid_a = new_line_id()
    lines = [_make_line(lid_a, "사용자 업로드 줄 텍스트")]
    sources = ["image"]
    job = _make_job(db, user, lines, sources)
    img_a = _write_image_for_line(storage_root, job.id, lid_a)

    body = SplitLineRequest(line_index=0, before="사용자 업로드", after="줄 텍스트")
    await split_line(body, job.id, db, user)

    db.refresh(job)
    new_lines = json.loads(job.script_json)
    new_sources = json.loads(job.line_sources_json)

    # source='image'는 _is_ai_owned_asset이 False라 원래도 보존됐던 케이스 — 회귀 검증
    assert new_lines[0]["line_id"] == lid_a
    assert new_lines[0]["status"] == "ready"
    assert new_sources[0] == "image", "사용자 업로드 source가 그대로 유지돼야 함"
    assert img_a.exists()


@pytest.mark.asyncio
async def test_split_edge_preserves_existing(env):
    """끝/시작 엔터 케이스 회귀: 코드 주석 line 335-336의 기존 동작 유지."""
    storage_root, db, user = env

    lid_a = new_line_id()
    lines = [_make_line(lid_a, "원본 텍스트")]
    sources = ["ai"]
    job = _make_job(db, user, lines, sources)
    img_a = _write_image_for_line(storage_root, job.id, lid_a)

    # 끝에서 엔터: before==old_text, after==""
    body = SplitLineRequest(line_index=0, before="원본 텍스트", after="")
    await split_line(body, job.id, db, user)

    db.refresh(job)
    new_lines = json.loads(job.script_json)
    new_sources = json.loads(job.line_sources_json)

    # 기존 줄은 line_id·텍스트·이미지 모두 그대로
    assert new_lines[0]["line_id"] == lid_a
    assert new_lines[0]["text"] == "원본 텍스트"
    assert new_lines[0]["status"] == "ready"
    assert new_sources[0] == "ai"
    assert img_a.exists()

    # 빈 카드 추가됨
    assert new_lines[1]["text"] == ""
    assert new_lines[1]["status"] == "pending"
    assert new_sources[1] == "ai"


@pytest.mark.asyncio
async def test_merge_preserves_prev_image(env):
    """prev 줄이 AI 이미지인 상태로 merge → prev 보존, L만 자산 정리."""
    storage_root, db, user = env

    lid_a = new_line_id()
    lid_b = new_line_id()
    lines = [
        _make_line(lid_a, "첫번째 줄"),
        _make_line(lid_b, "두번째 줄"),
    ]
    sources = ["ai", "ai"]
    job = _make_job(db, user, lines, sources)
    img_a = _write_image_for_line(storage_root, job.id, lid_a)
    img_b = _write_image_for_line(storage_root, job.id, lid_b)

    # line_index=1을 line_index=0과 합침
    body = MergeLineRequest(line_index=1)
    await merge_line(body, job.id, db, user)

    db.refresh(job)
    new_lines = json.loads(job.script_json)
    new_sources = json.loads(job.line_sources_json)

    # prev(L-1=0) 줄 보존: 텍스트는 합쳐졌지만 line_id·status·source·이미지는 그대로 (핵심)
    assert len(new_lines) == 1
    assert new_lines[0]["line_id"] == lid_a
    assert new_lines[0]["text"] == "첫번째 줄두번째 줄"
    assert new_lines[0]["status"] == "ready", "prev 줄의 AI 이미지 ready 상태가 보존돼야 함"
    assert new_sources[0] == "ai"
    assert img_a.exists(), "prev 줄 이미지가 보존돼야 함 (코덱스 지적 1번 수정)"


@pytest.mark.asyncio
async def test_merge_drops_target_assets(env):
    """merge 회귀: 사라지는 L 줄(병합되는 줄)의 자산은 여전히 정리됨."""
    storage_root, db, user = env

    lid_a = new_line_id()
    lid_b = new_line_id()
    lines = [
        _make_line(lid_a, "첫번째 줄"),
        _make_line(lid_b, "두번째 줄"),
    ]
    sources = ["ai", "ai"]
    job = _make_job(db, user, lines, sources)
    img_a = _write_image_for_line(storage_root, job.id, lid_a)
    img_b = _write_image_for_line(storage_root, job.id, lid_b)

    body = MergeLineRequest(line_index=1)
    await merge_line(body, job.id, db, user)

    # L 줄(사라지는 줄)의 이미지 파일은 삭제돼야 함 (기존 동작 유지)
    assert not img_b.exists(), "merge로 사라진 L 줄의 자산은 정리돼야 함"
    # prev 줄은 그대로
    assert img_a.exists()
