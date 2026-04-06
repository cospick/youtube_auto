"""이미지 기반 YouTube Shorts 영상 조립 파이프라인"""

import asyncio
import subprocess
import json
import os

from core.audio_utils import (
    run,
    speed_up_sentences,
    build_aligned_narration,
)
from core.subtitle_utils import split_subtitle_natural, split_title
from core.tts_engines import generate_tts_edge, generate_tts_typecast
from core.image_pipeline import apply_ken_burns, process_ai_clip
from api.models import AI_VIDEO_MODES
from config import settings


def get_duration(filepath):
    """ffprobe로 미디어 길이 조회"""
    probe = subprocess.run(
        f'ffprobe -v quiet -print_format json -show_format "{filepath}"',
        shell=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return float(json.loads(probe.stdout)["format"]["duration"])


def calculate_dynamic_clips_image(sentence_durations, buffer=0.0):
    """이미지 기반 클립 길이 계산 (소스 영상 제약 없음)"""
    clip_durations = []
    clip_starts = []
    t = 0.0
    for dur in sentence_durations:
        actual_dur = dur + buffer
        clip_durations.append(round(actual_dur, 2))
        clip_starts.append(round(t, 2))
        t += actual_dur
    return clip_durations, clip_starts, round(t, 2)


async def assemble_shorts(job_id: str, config: dict, progress_callback=None):
    """
    메인 파이프라인: 이미지 + TTS → 최종 9:16 쇼츠 영상.

    config 키:
        job_dir, images, lines, title,
        tts_engine, tts_speed, bgm_path, bgm_volume,
        font_title, font_sub
    """
    job_dir = config["job_dir"]
    temp_dir = os.path.join(job_dir, "temp")
    tts_dir = os.path.join(job_dir, "tts")
    output_dir = os.path.join(job_dir, "output")

    sentences = [line["text"] for line in config["lines"]]
    images = config["images"]
    motions = [line["motion"] for line in config["lines"]]

    # ── Step 1: TTS 생성 ──
    _update(progress_callback, job_id, "generating_tts", 0.4, "TTS 나레이션 생성 중...")

    engine = config.get("tts_engine", "edge")
    tts_speed = config.get("tts_speed", 1.1)
    voice_id = config.get("voice_id")
    emotion = config.get("emotion")

    if engine == "edge":
        narration_path, timings = await generate_tts_edge(tts_dir, sentences, voice=voice_id, speed=tts_speed)
        clip_durations = [t["duration"] + 0.5 for t in timings]
        clip_starts = []
        t_acc = 0.0
        for d in clip_durations:
            clip_starts.append(round(t_acc, 2))
            t_acc += d
        total_dur = round(t_acc, 2)
    elif engine == "typecast":
        tc_api_key = config.get("typecast_api_key")
        await asyncio.to_thread(
            generate_tts_typecast, tts_dir, sentences, voice_id=voice_id, speed=tts_speed, emotion=emotion, api_key=tc_api_key
        )
        sentence_durations = [
            t["duration"] for t in json.loads(
                open(os.path.join(tts_dir, "timings_raw.json"), encoding="utf-8").read()
            )
        ]
        clip_durations, clip_starts, total_dur = calculate_dynamic_clips_image(
            sentence_durations
        )
        # Typecast API가 이미 속도를 처리하므로 1.0으로 호출 → _fast.wav 파일 생성
        await asyncio.to_thread(
            speed_up_sentences, tts_dir, sentences, tts_speed=1.0
        )
        narration_path, timings = await asyncio.to_thread(
            build_aligned_narration, tts_dir, sentences, clip_starts, total_dur
        )
    else:
        raise ValueError(f"알 수 없는 TTS 엔진: {engine}")

    # ── Step 2: 영상 클립 생성 (Ken Burns 또는 AI 클립 trim+zoom) ──
    video_mode = config.get("video_mode", "kenburns")
    ai_clips = config.get("ai_clips")

    clip_files = []

    if video_mode in AI_VIDEO_MODES and ai_clips and len(ai_clips) == len(images):
        # AI 영상 모드: AI 클립에 trim + 서서히 줌인 적용
        _update(
            progress_callback, job_id, "assembling_video", 0.55, "AI 클립 trim + 줌인 적용 중..."
        )
        for i, (raw_clip, dur) in enumerate(zip(ai_clips, clip_durations)):
            clip_path = os.path.join(temp_dir, f"clip_{i:02d}.mp4")
            await asyncio.to_thread(
                process_ai_clip,
                clip_path=raw_clip,
                output_path=clip_path,
                duration=dur,
                width=settings.TARGET_WIDTH,
                height=settings.TARGET_HEIGHT,
                fps=settings.FPS,
            )
            clip_files.append(clip_path)
            _update(
                progress_callback,
                job_id,
                "assembling_video",
                0.55 + (i + 1) / len(images) * 0.15,
                f"AI 클립 처리 ({i + 1}/{len(images)})",
            )
    else:
        # Ken Burns 모드: 이미지에 줌/팬 효과 적용
        _update(
            progress_callback, job_id, "assembling_video", 0.55, "Ken Burns 모션 적용 중..."
        )
        for i, (img_path, motion, dur) in enumerate(zip(images, motions, clip_durations)):
            clip_path = os.path.join(temp_dir, f"clip_{i:02d}.mp4")
            await asyncio.to_thread(
                apply_ken_burns,
                image_path=img_path,
                output_path=clip_path,
                motion_type=motion,
                duration=dur,
                width=settings.TARGET_WIDTH,
                height=settings.TARGET_HEIGHT,
                fps=settings.FPS,
            )
            clip_files.append(clip_path)
            _update(
                progress_callback,
                job_id,
                "assembling_video",
                0.55 + (i + 1) / len(images) * 0.15,
                f"Ken Burns 적용 ({i + 1}/{len(images)})",
            )

    # ── Step 3: 클립 연결 ──
    _update(progress_callback, job_id, "assembling_video", 0.72, "클립 연결 중...")

    concat_list = os.path.join(temp_dir, "concat_list.txt")
    with open(concat_list, "w") as f:
        for clip in clip_files:
            f.write(f"file '{clip}'\n")

    concat_out = os.path.join(temp_dir, "concat_raw.mp4")
    await asyncio.to_thread(
        run,
        f'ffmpeg -y -f concat -safe 0 -i "{concat_list}" '
        f'-c:v libx264 -preset fast -crf 18 "{concat_out}"',
    )

    # ── Step 4: 오디오 믹싱 ──
    _update(progress_callback, job_id, "assembling_video", 0.80, "오디오 믹싱 중...")

    audio_out = os.path.join(temp_dir, "mixed_audio.mp4")
    vid_duration = await asyncio.to_thread(get_duration, concat_out)

    bgm_path = config.get("bgm_path")
    bgm_vol = config.get("bgm_volume", 0.12)
    bgm_start = config.get("bgm_start_sec", 0.0)
    has_bgm = bgm_path and os.path.exists(bgm_path)

    if has_bgm:
        await asyncio.to_thread(
            run,
            f'ffmpeg -y -i "{concat_out}" -i "{narration_path}" -i "{bgm_path}" '
            f'-filter_complex "'
            f"[1:a]volume=1.0[narr];"
            f"[2:a]atrim={bgm_start}:{bgm_start + vid_duration},asetpts=PTS-STARTPTS,volume={bgm_vol}[bgm];"
            f'[narr][bgm]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[aout]'
            f'" '
            f'-map 0:v -map "[aout]" -c:v copy -c:a aac -b:a 192k '
            f'-shortest "{audio_out}"',
        )
    else:
        await asyncio.to_thread(
            run,
            f'ffmpeg -y -i "{concat_out}" -i "{narration_path}" '
            f"-map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k "
            f'-shortest "{audio_out}"',
        )

    # ── Step 4.5: 오디오 라우드니스 노멀라이즈 (-14 LUFS, YouTube 기준) ──
    _update(progress_callback, job_id, "assembling_video", 0.85, "오디오 노멀라이즈 중...")

    normalized_out = os.path.join(temp_dir, "normalized.mp4")
    await asyncio.to_thread(
        run,
        f'ffmpeg -y -i "{audio_out}" '
        f'-af loudnorm=I=-14:LRA=11:TP=-1.0 '
        f'-c:v copy -c:a aac -b:a 192k '
        f'"{normalized_out}"',
    )
    audio_out = normalized_out

    # ── Step 5: 자막 + 타이틀 오버레이 ──
    _update(progress_callback, job_id, "assembling_video", 0.90, "자막/타이틀 합성 중...")

    subtitles = split_subtitle_natural(timings)
    font_title = config.get("font_title", settings.FONT_TITLE)
    font_sub = config.get("font_sub", settings.FONT_SUB)
    title_text = config.get("title", "")
    title_color = "#00CED1"
    sq = settings.TARGET_WIDTH  # 1080
    h = settings.TARGET_HEIGHT  # 1920
    sq_y = (h - sq) // 2  # 420

    sub_y = sq_y + sq - 200

    def _escape_filter(text):
        return text.replace("'", "'\\''").replace(",", "\\,").replace(":", "\\:")

    def _escape_fontpath(path):
        """Windows 드라이브 콜론(C:)을 ffmpeg 필터용으로 이스케이프"""
        return path.replace(":", "\\:")

    sub_filters = []
    for start, end, text in subtitles:
        escaped = _escape_filter(text)
        sub_filters.append(
            f"drawtext=fontfile='{_escape_fontpath(font_sub)}':text='{escaped}':"
            f"fontsize=55:fontcolor=white:borderw=3:bordercolor=black:"
            f"x=(w-text_w)/2:y={sub_y}:"
            f"enable='between(t,{start},{end})'"
        )

    title_filters = []
    if title_text and font_title:
        tl1 = config.get("title_line1")
        tl2 = config.get("title_line2")
        if tl1:
            title_lines = [tl1, tl2] if tl2 else [tl1]
        else:
            title_lines = split_title(title_text, max_chars=8)
        title_fontsize = 120
        title_line_gap = 130
        title_colors = ["white", "#E8D44D"]  # 윗줄 흰색, 아랫줄 톤다운 노란색
        font_path_escaped = _escape_fontpath(font_title)
        for j, line in enumerate(title_lines):
            escaped = _escape_filter(line)
            if len(title_lines) == 1:
                ty = sq_y - title_fontsize - 30
            else:
                base_y = sq_y - (len(title_lines) * title_line_gap) - 10
                ty = base_y + (j * title_line_gap)
            line_color = title_colors[min(j, len(title_colors) - 1)]
            # 그림자 레이어 (검정, 살짝 오프셋)
            title_filters.append(
                f"drawtext=fontfile='{font_path_escaped}':text='{escaped}':"
                f"fontsize={title_fontsize}:fontcolor=black@0.5:"
                f"x=(w-text_w)/2+6:y={ty}+6"
            )
            # 본문 레이어 (테두리 + 색상)
            title_filters.append(
                f"drawtext=fontfile='{font_path_escaped}':text='{escaped}':"
                f"fontsize={title_fontsize}:fontcolor={line_color}:"
                f"borderw=4:bordercolor=black@0.8:"
                f"x=(w-text_w)/2:y={ty}"
            )

    all_filters = title_filters + sub_filters
    output_path = os.path.join(output_dir, "shorts_final.mp4")

    if all_filters:
        # Windows에서 인라인 -vf는 경로/한글 이스케이핑 문제가 있으므로
        # filter_script 파일로 전달
        filter_str = ",".join(all_filters)
        filter_script = os.path.join(temp_dir, "subtitle_filter.txt")
        with open(filter_script, "w", encoding="utf-8") as f:
            f.write(filter_str)
        await asyncio.to_thread(
            run,
            f'ffmpeg -y -i "{audio_out}" '
            f'-filter_script:v "{filter_script}" '
            f'-c:v libx264 -preset fast -crf 18 -c:a copy "{output_path}"',
        )
    else:
        await asyncio.to_thread(
            run,
            f'ffmpeg -y -i "{audio_out}" -c copy "{output_path}"',
        )

    # ── 완료 ──
    _update(progress_callback, job_id, "completed", 1.0, "완료!")
    return output_path


def _update(callback, job_id, status, progress, step):
    if callback:
        callback(job_id=job_id, status=status, progress=progress, step=step)
