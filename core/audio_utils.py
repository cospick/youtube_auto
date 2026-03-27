"""오디오 처리 유틸리티 (build_shorts.py에서 복사)"""

import shlex
import subprocess
import sys
import json
import os
import numpy as np
import soundfile as sf


def run(cmd, desc=""):
    """ffmpeg 명령 실행"""
    if sys.platform == "win32":
        args = cmd
    else:
        args = shlex.split(cmd)
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 에러: {result.stderr[-1000:]}")
    return result


def extract_sentence_from_warmup(wav, sr):
    """
    '음. [문장]' TTS 출력에서 실제 문장 부분만 추출.
    워밍업 '음.' 이후 무음 구간을 찾고 실제 음성 시작점부터 추출.
    """
    abs_wav = np.abs(wav)
    window = int(sr * 0.02)
    step = window // 2

    energies = []
    for i in range(0, len(abs_wav) - window, step):
        energies.append((i, np.mean(abs_wav[i : i + window])))

    first_speech = False
    pause_found = False
    pause_pos = 0

    for pos, eng in energies:
        t_ms = pos / sr * 1000
        if eng > 0.03 and not first_speech:
            first_speech = True
        elif eng < 0.01 and first_speech and t_ms > 200:
            pause_pos = pos
            pause_found = True
            break

    if not pause_found:
        pause_pos = int(sr * 0.5)

    sentence_start = pause_pos
    for pos, eng in energies:
        if pos > pause_pos and eng > 0.02:
            sentence_start = max(0, pos - int(sr * 0.005))
            break

    return wav[sentence_start:]


def trim_trailing_silence(wav, sr, threshold=0.005):
    """끝부분 무음만 제거 (시작은 보존)"""
    abs_wav = np.abs(wav)
    window = int(sr * 0.02)

    end = len(abs_wav)
    for i in range(len(abs_wav) - window, 0, -(window // 2)):
        if np.mean(abs_wav[i : i + window]) > threshold:
            end = min(len(abs_wav), i + window + int(sr * 0.1))
            break

    if end < len(abs_wav) * 0.7:
        return wav

    return wav[:end]


def apply_fade(wav, sr, fade_in_ms=15, fade_out_ms=10):
    """부드러운 페이드 인/아웃"""
    fade_in_samples = int(sr * fade_in_ms / 1000)
    fade_out_samples = int(sr * fade_out_ms / 1000)
    wav = wav.copy()
    if len(wav) > fade_in_samples:
        wav[:fade_in_samples] *= np.linspace(0, 1, fade_in_samples)
    if len(wav) > fade_out_samples:
        wav[-fade_out_samples:] *= np.linspace(1, 0, fade_out_samples)
    return wav


def speed_up_sentences(temp_dir, sentences, tts_speed=1.0):
    """각 문장 WAV에 atempo 적용, 최종 듀레이션 반환"""
    final_durations = []

    for i in range(len(sentences)):
        sent_path = os.path.join(temp_dir, f"sent_{i:02d}.wav")
        sent_fast_path = os.path.join(temp_dir, f"sent_{i:02d}_fast.wav")

        if tts_speed > 1.0:
            run(
                f'ffmpeg -y -i "{sent_path}" -filter:a "atempo={tts_speed}" "{sent_fast_path}"',
                f"문장 {i + 1} → {tts_speed}x",
            )
            wav, sr = sf.read(sent_fast_path)
        else:
            wav, sr = sf.read(sent_path)
            sf.write(sent_fast_path, wav, sr)

        dur = len(wav) / sr
        final_durations.append(round(dur, 2))

    return final_durations


def build_aligned_narration(temp_dir, sentences, clip_starts, total_dur):
    """문장별 WAV를 클립 시작에 맞춰 배치"""
    wav0, sr = sf.read(os.path.join(temp_dir, "sent_00_fast.wav"))
    total_samples = int(total_dur * sr) + sr
    aligned = np.zeros(total_samples)
    aligned_timings = []

    for i in range(len(sentences)):
        if i >= len(clip_starts):
            break
        fast_path = os.path.join(temp_dir, f"sent_{i:02d}_fast.wav")
        wav, _ = sf.read(fast_path)
        sent_dur = len(wav) / sr

        offset = clip_starts[i]
        start_sample = int(offset * sr)
        end_sample = start_sample + len(wav)

        if end_sample <= len(aligned):
            aligned[start_sample:end_sample] += wav
        else:
            avail = len(aligned) - start_sample
            if avail > 0:
                aligned[start_sample : start_sample + avail] += wav[:avail]
            sent_dur = avail / sr

        aligned_timings.append(
            {
                "text": sentences[i],
                "offset": round(offset, 2),
                "duration": round(sent_dur, 2),
                "end": round(offset + sent_dur, 2),
            }
        )

    aligned = aligned[: int(total_dur * sr)]

    aligned_wav_path = os.path.join(temp_dir, "narration_aligned.wav")
    mp3_path = os.path.join(temp_dir, "narration.mp3")
    sf.write(aligned_wav_path, aligned, sr)

    run(
        f'ffmpeg -y -i "{aligned_wav_path}" -codec:a libmp3lame -b:a 192k "{mp3_path}"',
    )

    return mp3_path, aligned_timings
