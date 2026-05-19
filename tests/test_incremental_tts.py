"""incremental TTS preview-build 의 diff/rename 로직 단위 테스트.

Typecast 호출은 모킹해 wav 파일 생성만 흉내낸다. 실제 검증 대상은:
- 어떤 인덱스가 재생성 대상으로 선정되는가
- 변경되지 않은 wav가 올바르게 새 인덱스로 옮겨지는가
- signature.json이 올바르게 갱신되는가
"""

import asyncio
import json
import os
import struct
import wave

import pytest

# tests/conftest.py가 JWT_SECRET을 주입한 뒤에 모듈 import
from api.routes import tts_preview as tp
from core.user_assets_visual import line_text_hash


def _write_dummy_wav(path: str, duration_sec: float = 0.5) -> None:
    """1채널 16kHz silence wav. duration 측정은 soundfile/wave 모두 동일."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sr = 16000
    n = int(sr * duration_sec)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack("<" + "h" * n, *([0] * n)))


@pytest.fixture
def temp_session(tmp_path, monkeypatch):
    """TTS_SESSIONS_DIR를 임시 디렉토리로 패치하고 session_id 1개 반환."""
    sessions_root = tmp_path / "tts_sessions"
    sessions_root.mkdir()
    monkeypatch.setattr(tp, "TTS_SESSIONS_DIR", str(sessions_root))
    session_id = "abcdef012345"
    return session_id, sessions_root / session_id


def _seed_session(session_dir, voice, lines):
    """기존 빌드 결과를 임시 세션에 주입 (sent_*.wav + signature.json + metadata)."""
    session_dir.mkdir(parents=True, exist_ok=True)
    for i, (lid, text) in enumerate(lines):
        _write_dummy_wav(str(session_dir / f"sent_{i:02d}.wav"), duration_sec=0.3 + 0.1 * i)
    signature = {
        "voice": voice,
        "line_order": [lid for lid, _ in lines],
        "line_hashes": {lid: line_text_hash(text) for lid, text in lines},
    }
    (session_dir / "signature.json").write_text(json.dumps(signature), encoding="utf-8")


def _make_req(sentences, line_ids, voice_id="voice-A", speed=1.0, emotion=None, existing_session_id=None):
    """incremental 경로의 Pydantic 요청 객체. preview-build는 TtsPreviewBuildRequest를 받음."""
    from api.models import TtsPreviewBuildRequest
    return TtsPreviewBuildRequest(
        sentences=sentences,
        voice_id=voice_id,
        speed=speed,
        emotion=emotion,
        line_ids=line_ids,
        existing_session_id=existing_session_id,
    )


@pytest.fixture
def stub_generate_for_indices(monkeypatch):
    """generate_tts_for_indices 호출을 캡처해 wav 파일만 만든다. (실제 Typecast 호출 차단)"""
    calls = {"indices": None, "sentences": None}

    async def fake(tts_dir, sentences, indices, voice_id=None, speed=None, emotion=None, api_key=None):
        calls["indices"] = list(indices)
        calls["sentences"] = list(sentences)
        result = {}
        for idx in indices:
            path = os.path.join(tts_dir, f"sent_{idx:02d}.wav")
            _write_dummy_wav(path, duration_sec=0.42)
            result[idx] = {"text": sentences[idx], "duration": 0.42}
        return result

    monkeypatch.setattr(tp, "generate_tts_for_indices", fake)
    return calls


def _run_incremental(session_dir, req, prev_sig):
    """동기 래퍼 — _rebuild_incremental은 비동기."""
    return asyncio.run(
        tp._rebuild_incremental(str(session_dir), req, typecast_key="fake", prev_sig=prev_sig)
    )


# ────────────────────────────────────────────────────────────────────
# 1. voice 동일 + 전체 텍스트 동일 → indices_to_regen == []
# ────────────────────────────────────────────────────────────────────
def test_no_change_skips_all_regen(temp_session, stub_generate_for_indices):
    session_id, session_dir = temp_session
    lines = [("l1", "안녕"), ("l2", "반가워"), ("l3", "잘 가")]
    voice = ["voice-A", 1.0, None, "typecast"]
    _seed_session(session_dir, voice, lines)

    req = _make_req(
        sentences=[t for _, t in lines],
        line_ids=[lid for lid, _ in lines],
        existing_session_id=session_id,
    )
    prev_sig = json.loads((session_dir / "signature.json").read_text(encoding="utf-8"))
    sentences, durations, regen = _run_incremental(session_dir, req, prev_sig)

    assert regen == []
    assert stub_generate_for_indices["indices"] is None  # 호출 자체가 없어야 함
    assert len(sentences) == 3
    assert len(durations) == 3
    for i in range(3):
        assert (session_dir / f"sent_{i:02d}.wav").exists()


# ────────────────────────────────────────────────────────────────────
# 2. 중간 줄 텍스트 변경 → 해당 인덱스만 재생성
# ────────────────────────────────────────────────────────────────────
def test_middle_line_text_change_regens_only_that(temp_session, stub_generate_for_indices):
    session_id, session_dir = temp_session
    seed_lines = [("l1", "안녕"), ("l2", "반가워"), ("l3", "잘 가")]
    voice = ["voice-A", 1.0, None, "typecast"]
    _seed_session(session_dir, voice, seed_lines)

    new_sentences = ["안녕", "반갑습니다", "잘 가"]  # 가운데 줄만 변경
    req = _make_req(
        sentences=new_sentences,
        line_ids=["l1", "l2", "l3"],
        existing_session_id=session_id,
    )
    prev_sig = json.loads((session_dir / "signature.json").read_text(encoding="utf-8"))
    _, _, regen = _run_incremental(session_dir, req, prev_sig)

    assert regen == [1]
    assert stub_generate_for_indices["indices"] == [1]


# ────────────────────────────────────────────────────────────────────
# 3. 줄 삽입 (line_id 새로 추가) → 새 인덱스만 regen, 기존 wav는 rename
# ────────────────────────────────────────────────────────────────────
def test_line_insertion_keeps_existing(temp_session, stub_generate_for_indices):
    session_id, session_dir = temp_session
    seed_lines = [("l1", "안녕"), ("l3", "잘 가")]
    voice = ["voice-A", 1.0, None, "typecast"]
    _seed_session(session_dir, voice, seed_lines)

    # 가운데에 l2 삽입
    new_lines = [("l1", "안녕"), ("l2", "반가워"), ("l3", "잘 가")]
    req = _make_req(
        sentences=[t for _, t in new_lines],
        line_ids=[lid for lid, _ in new_lines],
        existing_session_id=session_id,
    )
    prev_sig = json.loads((session_dir / "signature.json").read_text(encoding="utf-8"))
    _, _, regen = _run_incremental(session_dir, req, prev_sig)

    # 새 인덱스 1만 재생성 — 기존 l3는 새 인덱스 2로 rename되어야 함
    assert regen == [1]
    assert (session_dir / "sent_00.wav").exists()
    assert (session_dir / "sent_01.wav").exists()
    assert (session_dir / "sent_02.wav").exists()


# ────────────────────────────────────────────────────────────────────
# 4. 줄 삭제 → 사라진 line_id의 wav 정리, 뒤 인덱스 당김
# ────────────────────────────────────────────────────────────────────
def test_line_deletion_drops_wav(temp_session, stub_generate_for_indices):
    session_id, session_dir = temp_session
    seed_lines = [("l1", "안녕"), ("l2", "반가워"), ("l3", "잘 가")]
    voice = ["voice-A", 1.0, None, "typecast"]
    _seed_session(session_dir, voice, seed_lines)

    # l2 삭제
    new_lines = [("l1", "안녕"), ("l3", "잘 가")]
    req = _make_req(
        sentences=[t for _, t in new_lines],
        line_ids=[lid for lid, _ in new_lines],
        existing_session_id=session_id,
    )
    prev_sig = json.loads((session_dir / "signature.json").read_text(encoding="utf-8"))
    _, _, regen = _run_incremental(session_dir, req, prev_sig)

    assert regen == []  # 둘 다 재사용 가능
    assert (session_dir / "sent_00.wav").exists()
    assert (session_dir / "sent_01.wav").exists()
    assert not (session_dir / "sent_02.wav").exists()
    # _swap 디렉토리는 정리됐어야 함
    assert not (session_dir / "_swap").exists()


# ────────────────────────────────────────────────────────────────────
# 5. signature가 있는데 wav 파일 누락 → 누락된 인덱스 재생성
# ────────────────────────────────────────────────────────────────────
def test_missing_wav_triggers_regen(temp_session, stub_generate_for_indices):
    session_id, session_dir = temp_session
    seed_lines = [("l1", "안녕"), ("l2", "반가워"), ("l3", "잘 가")]
    voice = ["voice-A", 1.0, None, "typecast"]
    _seed_session(session_dir, voice, seed_lines)
    # 가운데 wav 강제 삭제 (R2 복구 실패 시뮬레이션)
    os.remove(str(session_dir / "sent_01.wav"))

    req = _make_req(
        sentences=[t for _, t in seed_lines],
        line_ids=[lid for lid, _ in seed_lines],
        existing_session_id=session_id,
    )
    prev_sig = json.loads((session_dir / "signature.json").read_text(encoding="utf-8"))
    _, _, regen = _run_incremental(session_dir, req, prev_sig)

    assert regen == [1]


# ────────────────────────────────────────────────────────────────────
# 6. signature가 None (구버전 / 첫 빌드)일 때 incremental 진입 차단
#    → preview_build의 분기 조건 검증
# ────────────────────────────────────────────────────────────────────
def test_signature_missing_falls_back_to_full(temp_session, stub_generate_for_indices):
    session_id, session_dir = temp_session
    # signature.json 없는 상태
    session_dir.mkdir(parents=True, exist_ok=True)

    # _load_signature가 None을 반환해야 함
    assert tp._load_signature(str(session_dir)) is None


# ────────────────────────────────────────────────────────────────────
# 7. 구버전 signature (필드 누락) → _load_signature가 None 반환
# ────────────────────────────────────────────────────────────────────
def test_corrupt_signature_returns_none(temp_session):
    session_id, session_dir = temp_session
    session_dir.mkdir(parents=True, exist_ok=True)
    # voice 필드만 있고 line_order/line_hashes 누락
    (session_dir / "signature.json").write_text(
        json.dumps({"voice": ["voice-A", 1.0, None, "typecast"]}), encoding="utf-8"
    )
    assert tp._load_signature(str(session_dir)) is None

    # 손상된 JSON
    (session_dir / "signature.json").write_text("{not json", encoding="utf-8")
    assert tp._load_signature(str(session_dir)) is None


# ────────────────────────────────────────────────────────────────────
# 8. voice 변경 → preview_build의 incremental 조건이 False가 되어
#    full rebuild로 진입해야 함 — _voice_signature가 다르게 나옴을 검증
# ────────────────────────────────────────────────────────────────────
def test_voice_change_signature_mismatch():
    from api.models import TtsPreviewBuildRequest

    sig_a = tp._voice_signature(TtsPreviewBuildRequest(
        sentences=["x"], voice_id="A", speed=1.0, emotion=None
    ))
    sig_b = tp._voice_signature(TtsPreviewBuildRequest(
        sentences=["x"], voice_id="B", speed=1.0, emotion=None
    ))
    assert sig_a != sig_b

    sig_c = tp._voice_signature(TtsPreviewBuildRequest(
        sentences=["x"], voice_id="A", speed=1.2, emotion=None
    ))
    assert sig_a != sig_c

    sig_d = tp._voice_signature(TtsPreviewBuildRequest(
        sentences=["x"], voice_id="A", speed=1.0, emotion="happy"
    ))
    assert sig_a != sig_d
