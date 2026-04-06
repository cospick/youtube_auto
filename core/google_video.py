"""Google Veo 3.1 Lite Image-to-Video 클라이언트"""

import asyncio
import os
import pathlib
import time

import httpx
from google.genai import types

from core.gemini_client import get_client

# fal_video.py와 동일한 프롬프트 공유
DEFAULT_VIDEO_PROMPT = (
    "Subtle natural movement, the scene gently comes alive. No camera movement."
)

MODEL_ID = "veo-3.1-lite-generate-preview"

MAX_RETRIES = 3


async def _save_video(video, output_path: str) -> None:
    """영상 저장 — video_bytes 우선, 없으면 URI에서 다운로드"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if video.video_bytes:
        await asyncio.to_thread(pathlib.Path(output_path).write_bytes, video.video_bytes)
    elif video.uri:
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as http:
            resp = await http.get(video.uri)
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(resp.content)
    else:
        raise RuntimeError("Google Veo: 영상 데이터(bytes/uri) 없음")


async def _submit_with_retry(client, image_bytes: bytes, api_key: str = None):
    """영상 생성 요청 제출 (429/503 재시도 포함)"""
    image = types.Image(image_bytes=image_bytes, mime_type="image/png")
    config = types.GenerateVideosConfig(
        aspect_ratio="9:16",
        duration_seconds=6,
        resolution="1080p",
        generate_audio=False,
        person_generation="allow_all",
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            operation = await client.aio.models.generate_videos(
                model=MODEL_ID,
                prompt=DEFAULT_VIDEO_PROMPT,
                image=image,
                config=config,
            )
            return operation
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                if attempt < MAX_RETRIES:
                    print(f"[Google Veo] 429 Rate Limit — {30}초 대기 후 재시도 ({attempt}/{MAX_RETRIES})")
                    await asyncio.sleep(30)
                    continue
            elif "503" in err_str or "500" in err_str:
                if attempt < MAX_RETRIES:
                    wait = 5 * attempt
                    print(f"[Google Veo] 서버 에러 — {wait}초 대기 후 재시도 ({attempt}/{MAX_RETRIES})")
                    await asyncio.sleep(wait)
                    continue
            raise


async def _poll_operation(client, operation, timeout: int = 600, interval: int = 10):
    """operation 완료까지 폴링"""
    start = time.time()

    while time.time() - start < timeout:
        operation = await client.aio.operations.get(operation)
        if operation.done:
            break
        await asyncio.sleep(interval)
    else:
        raise RuntimeError(f"Google Veo 타임아웃: {timeout}초 초과")

    if operation.error:
        raise RuntimeError(f"Google Veo 영상 생성 실패: {operation.error}")

    result = operation.result or operation.response
    if not result or not result.generated_videos:
        raise RuntimeError("Google Veo: 생성된 영상이 없습니다")

    return result.generated_videos[0].video


async def generate_video_clip(
    image_path: str,
    output_path: str,
    api_key: str = None,
    timeout: int = 600,
    interval: int = 10,
) -> str:
    """
    이미지 → Google Veo 3.1 Lite 영상 클립 생성.

    Returns: 저장된 영상 파일 경로
    """
    client = get_client(api_key)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    operation = await _submit_with_retry(client, image_bytes, api_key)
    video = await _poll_operation(client, operation, timeout, interval)
    await _save_video(video, output_path)

    return output_path


async def generate_clips_batch(
    images: list[str],
    output_dir: str,
    progress_callback=None,
    job_id: str = None,
    api_key: str = None,
) -> list[str]:
    """
    여러 이미지를 Google Veo 3.1 Lite 영상 클립으로 변환.
    fal_video.generate_clips_batch()와 동일한 시그니처.

    ★ 순차 제출 + 동시 폴링 전략 (Rate Limit 대응)
    """
    client = get_client(api_key)

    if progress_callback and job_id:
        progress_callback(
            job_id=job_id,
            status="generating_clips",
            progress=0.26,
            step="Veo 3.1 Lite 영상 생성 요청 준비 중...",
        )

    # 1단계: 순차 제출 (2초 간격으로 Rate Limit 방지)
    operations = []
    for i, img_path in enumerate(images):
        with open(img_path, "rb") as f:
            image_bytes = f.read()

        operation = await _submit_with_retry(client, image_bytes, api_key)
        operations.append(operation)

        if progress_callback and job_id:
            progress_callback(
                job_id=job_id,
                status="generating_clips",
                progress=0.26 + (i + 1) / len(images) * 0.04,
                step=f"Veo 3.1 Lite 요청 제출 ({i + 1}/{len(images)})",
            )

        # 마지막 요청 후에는 대기 불필요
        if i < len(images) - 1:
            await asyncio.sleep(2)

    if progress_callback and job_id:
        progress_callback(
            job_id=job_id,
            status="generating_clips",
            progress=0.30,
            step=f"Veo 3.1 Lite 영상 생성 대기 중 (0/{len(images)})",
        )

    # 2단계: 동시 폴링 + 저장
    completed = 0

    async def _poll_and_save(i, op):
        nonlocal completed
        output_path = os.path.join(output_dir, f"clip_raw_{i:02d}.mp4")

        video = await _poll_operation(client, op)
        await _save_video(video, output_path)

        completed += 1
        if progress_callback and job_id:
            p = 0.30 + (completed / len(images)) * 0.20
            progress_callback(
                job_id=job_id,
                status="generating_clips",
                progress=round(p, 2),
                step=f"Veo 3.1 Lite 영상 생성 중 ({completed}/{len(images)})",
            )
        return output_path

    results = await asyncio.gather(
        *[_poll_and_save(i, op) for i, op in enumerate(operations)],
        return_exceptions=True,
    )

    clip_paths = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            raise RuntimeError(f"클립 {i} 생성 실패: {r}")
        clip_paths.append(r)

    return clip_paths
