"""TTS 엔진 통합 (Typecast)"""

import asyncio
import json
import os
import time


V21_ONLY_VOICES = {
    "tc_61659c5818732016a95fe763",
    "tc_6059dad0b83880769a50502f",
    "tc_61de29497924994f5abd68db",
}

# 동시 요청 개수. 디버깅 중: 1로 낮춰 순차 처리 (병렬 처리 때 sent_XX.wav
# 파일이 간헐적으로 손상되는 현상 격리). 원인 확정 후 다시 2~3으로 복원.
_TYPECAST_MAX_CONCURRENCY = 1


def _generate_one_sentence_typecast(tts_dir, index, sent, headers, vid, model, speed, emotion):
    """한 문장만 Typecast로 합성하고 sent_XX.wav 저장. duration 반환.

    실패 지점(HTTP 에러·polling 타임아웃·파일 누락)마다 명확한 RuntimeError를 던져
    병렬 처리 중 어느 줄이 왜 실패했는지 즉시 추적 가능하게 한다.
    """
    import requests
    import soundfile as sf

    prefix = f"[Typecast sent_{index:02d}]"
    payload = {
        "text": sent,
        "voice_id": vid,
        "model": model,
        "output": {"format": "wav", "sample_rate": 44100, "audio_tempo": speed or 1.0},
    }
    if emotion and emotion != "normal":
        payload["prompt"] = {"emotion_type": "preset", "emotion_preset": emotion}
    resp = requests.post(
        "https://api.typecast.ai/v1/text-to-speech",
        headers=headers,
        json=payload,
        timeout=60,
    )
    out_path = os.path.join(tts_dir, f"sent_{index:02d}.wav")

    if resp.status_code == 429:
        raise RuntimeError(f"{prefix} Typecast rate limit (429)")
    if resp.status_code >= 400:
        raise RuntimeError(f"{prefix} HTTP {resp.status_code}: {resp.text[:200]}")

    content_type = resp.headers.get("Content-Type", "")
    if "audio" in content_type or "octet-stream" in content_type:
        with open(out_path, "wb") as f:
            f.write(resp.content)
    else:
        try:
            result = resp.json()
        except Exception as e:
            raise RuntimeError(f"{prefix} 응답 JSON 파싱 실패: {e} / body={resp.text[:200]}")
        speak_url = result.get("result", {}).get("speak_v2_url")
        if not speak_url:
            raise RuntimeError(f"{prefix} speak_v2_url 없음 / response={json.dumps(result)[:300]}")

        done = False
        last_status = None
        for _ in range(30):
            time.sleep(2)
            poll = requests.get(speak_url, headers=headers, timeout=30)
            if poll.status_code != 200:
                last_status = f"polling HTTP {poll.status_code}"
                continue
            data = poll.json()
            status = data.get("result", {}).get("status")
            last_status = status
            if status == "done":
                audio_url = data["result"].get("audio_download_url") or data["result"].get("audio_url")
                if not audio_url:
                    raise RuntimeError(f"{prefix} status=done인데 audio_url 없음")
                audio_resp = requests.get(audio_url, timeout=60)
                if audio_resp.status_code != 200:
                    raise RuntimeError(f"{prefix} audio 다운로드 HTTP {audio_resp.status_code}")
                with open(out_path, "wb") as f:
                    f.write(audio_resp.content)
                done = True
                break
            if status in ("failed", "error"):
                raise RuntimeError(f"{prefix} Typecast polling status={status} / data={json.dumps(data)[:300]}")
        if not done:
            raise RuntimeError(f"{prefix} 60초 polling 타임아웃 (마지막 상태: {last_status})")

    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"{prefix} wav 파일이 생성되지 않음: {out_path}")

    wav, sr = sf.read(out_path)
    duration = len(wav) / sr
    return {"text": sent, "duration": round(duration, 2)}


async def generate_tts_typecast(tts_dir, sentences, voice_id=None, speed=None, emotion=None, api_key=None):
    """
    Typecast API TTS (고품질 한국어). 5줄 병렬 처리.
    반환: raw_timings (문장별 duration 목록, sentences 순서 보존)
    """
    from config import settings

    key = api_key or settings.TYPECAST_API_KEY
    if not key:
        raise RuntimeError("Typecast API 키가 설정되지 않았습니다")

    vid = voice_id or "tc_62e8f21e979b3860fe2f6a24"
    model = "ssfm-v21" if vid in V21_ONLY_VOICES else "ssfm-v30"
    headers = {"X-API-KEY": key, "Content-Type": "application/json"}

    sem = asyncio.Semaphore(_TYPECAST_MAX_CONCURRENCY)

    async def _one(i, sent):
        async with sem:
            return await asyncio.to_thread(
                _generate_one_sentence_typecast,
                tts_dir, i, sent, headers, vid, model, speed, emotion,
            )

    tasks = [_one(i, s) for i, s in enumerate(sentences)]
    raw_timings = await asyncio.gather(*tasks)

    with open(os.path.join(tts_dir, "timings_raw.json"), "w", encoding="utf-8") as f:
        json.dump(raw_timings, f, ensure_ascii=False, indent=2)

    return raw_timings
