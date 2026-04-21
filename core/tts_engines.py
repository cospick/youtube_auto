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

# 병렬 요청 시 rate limit 보수 대응 — 세마포어 동시 3개
_TYPECAST_MAX_CONCURRENCY = 3


def _generate_one_sentence_typecast(tts_dir, index, sent, headers, vid, model, speed, emotion):
    """한 문장만 Typecast로 합성하고 sent_XX.wav 저장. duration 반환."""
    import requests
    import soundfile as sf

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
    )
    out_path = os.path.join(tts_dir, f"sent_{index:02d}.wav")

    content_type = resp.headers.get("Content-Type", "")
    if "audio" in content_type or "octet-stream" in content_type:
        with open(out_path, "wb") as f:
            f.write(resp.content)
    else:
        result = resp.json()
        speak_url = result.get("result", {}).get("speak_v2_url")
        if speak_url:
            for _ in range(30):
                time.sleep(2)
                poll = requests.get(speak_url, headers=headers)
                if poll.status_code == 200:
                    data = poll.json()
                    if data.get("result", {}).get("status") == "done":
                        audio_url = data["result"].get("audio_download_url") or data["result"].get("audio_url")
                        if audio_url:
                            with open(out_path, "wb") as f:
                                f.write(requests.get(audio_url).content)
                        break

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
