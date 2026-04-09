"""fal.ai Image-to-Video 클라이언트 — 다중 모델 지원"""

import asyncio
import os
import httpx

from config import settings

# ── fal.ai 모델 설정 ──

# AI 영상 생성용 범용 프롬프트 (카메라 움직임 없이 자연스러운 미세 모션만)
# 줌인 효과는 ffmpeg process_ai_clip에서 별도 적용
DEFAULT_VIDEO_PROMPT = "Subtle natural movement, the scene gently comes alive. No camera movement."

MODELS = {
    "hailuo": {
        "model_id": "fal-ai/minimax/hailuo-02/standard/image-to-video",
        "label": "Hailuo 02 Standard 512P",
        "build_args": lambda image_url: {
            "prompt": DEFAULT_VIDEO_PROMPT,
            "image_url": image_url,
            "duration": "6",
            "resolution": "512P",
            "prompt_optimizer": True,
        },
    },
    "hailuo23": {
        "model_id": "fal-ai/minimax/hailuo-2.3-fast/standard/image-to-video",
        "label": "Hailuo 2.3 Fast 768P",
        "build_args": lambda image_url: {
            "prompt": DEFAULT_VIDEO_PROMPT,
            "image_url": image_url,
            "duration": "6",
            "prompt_optimizer": True,
        },
    },
    "wan": {
        "model_id": "fal-ai/wan-i2v",
        "label": "Wan 2.1 I2V",
        "build_args": lambda image_url: {
            "prompt": DEFAULT_VIDEO_PROMPT,
            "image_url": image_url,
            "resolution": "480p",
            "aspect_ratio": "9:16",
            "num_frames": 81,
            "frames_per_second": 16,
            "negative_prompt": "slow motion, timelapse, blurry, distorted, low quality, watermark, static, overexposed, camera pan, camera zoom, camera movement",
        },
    },
    "kling": {
        "model_id": "fal-ai/kling-video/v2.1/standard/image-to-video",
        "label": "Kling 2.1 Standard",
        "build_args": lambda image_url: {
            "prompt": DEFAULT_VIDEO_PROMPT,
            "image_url": image_url,
            "duration": "5",
            "aspect_ratio": "9:16",
        },
    },
    "kling26": {
        "model_id": "fal-ai/kling-video/v2.6/pro/image-to-video",
        "label": "Kling 2.6 Pro",
        "build_args": lambda image_url: {
            "prompt": DEFAULT_VIDEO_PROMPT,
            "image_url": image_url,
            "duration": "5",
            "aspect_ratio": "9:16",
        },
    },
    "veo": {
        "model_id": "fal-ai/veo3.1/fast/image-to-video",
        "label": "Veo 3.1 Fast 1080p",
        "build_args": lambda image_url: {
            "prompt": DEFAULT_VIDEO_PROMPT,
            "image_url": image_url,
            "duration": "6s",
            "aspect_ratio": "9:16",
            "resolution": "1080p",
            "generate_audio": False,
        },
    },
    "veo_lite": {
        "model_id": "fal-ai/veo3.1/lite/image-to-video",
        "label": "Veo 3.1 Lite 720p",
        "build_args": lambda image_url: {
            "prompt": DEFAULT_VIDEO_PROMPT,
            "image_url": image_url,
            "duration": "6s",
            "aspect_ratio": "9:16",
            "resolution": "720p",
            "generate_audio": False,
        },
    },
}

FAL_QUEUE_URL = "https://queue.fal.run"


def _headers(api_key: str = None) -> dict:
    key = api_key or settings.FAL_KEY
    if not key:
        raise RuntimeError("fal.ai API 키가 설정되지 않았습니다")
    return {
        "Authorization": f"Key {key}",
        "Content-Type": "application/json",
    }


async def submit_task(model_key: str, image_url: str, api_key: str = None) -> dict:
    """
    fal.ai 큐에 Image-to-Video 태스크 제출.

    Returns: {"request_id": "...", "status_url": "...", "response_url": "..."}
    """
    model = MODELS[model_key]
    model_id = model["model_id"]
    body = model["build_args"](image_url)

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{FAL_QUEUE_URL}/{model_id}",
            json=body,
            headers=_headers(api_key),
        )
        resp.raise_for_status()
        data = resp.json()

    return {
        "request_id": data["request_id"],
        "status_url": data["status_url"],
        "response_url": data["response_url"],
    }


async def poll_task(status_url: str, response_url: str, timeout: int = 600, interval: int = 5, api_key: str = None) -> str:
    """
    fal.ai 태스크 완료까지 폴링 (submit 응답의 URL을 그대로 사용).

    Returns: 완성된 영상 URL
    """
    import time

    start = time.time()

    async with httpx.AsyncClient(timeout=30) as client:
        while time.time() - start < timeout:
            resp = await client.get(status_url, headers=_headers(api_key))
            resp.raise_for_status()
            status_data = resp.json()
            status = status_data.get("status")

            if status == "COMPLETED":
                result_resp = await client.get(response_url, headers=_headers(api_key))
                result_resp.raise_for_status()
                result = result_resp.json()
                video = result.get("video", {})
                video_url = video.get("url")
                if not video_url:
                    raise RuntimeError("fal.ai: 영상 URL이 없습니다")
                return video_url

            if status == "FAILED":
                error = status_data.get("error", "알 수 없는 에러")
                raise RuntimeError(f"fal.ai 영상 생성 실패: {error}")

            await asyncio.sleep(interval)

    raise RuntimeError(f"fal.ai 타임아웃: {timeout}초 초과")


async def download_video(video_url: str, output_path: str) -> str:
    """생성된 영상을 로컬 파일로 다운로드"""
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        resp = await client.get(video_url)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(resp.content)
    return output_path


async def upload_image_to_fal(image_path: str, api_key: str = None) -> str:
    """
    로컬 이미지를 fal.ai CDN에 업로드하여 URL 획득.
    fal.ai API는 image_url을 요구하므로 로컬 파일을 먼저 업로드해야 함.
    """
    import mimetypes

    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/png"
    filename = os.path.basename(image_path)

    async with httpx.AsyncClient(timeout=60) as client:
        # 1단계: 업로드 URL 요청
        initiate_resp = await client.post(
            "https://rest.alpha.fal.ai/storage/upload/initiate",
            json={"file_name": filename, "content_type": mime},
            headers=_headers(api_key),
        )
        initiate_resp.raise_for_status()
        upload_data = initiate_resp.json()
        upload_url = upload_data["upload_url"]
        file_url = upload_data["file_url"]

        # 2단계: 파일 업로드
        with open(image_path, "rb") as f:
            file_bytes = f.read()

        upload_resp = await client.put(
            upload_url,
            content=file_bytes,
            headers={"Content-Type": mime},
        )
        upload_resp.raise_for_status()

    return file_url


async def generate_video_clip(
    image_path: str,
    output_path: str,
    model_key: str = "hailuo",
    api_key: str = None,
) -> str:
    """
    이미지 → AI 영상 클립 생성 (업로드 → 제출 → 폴링 → 다운로드).

    Returns: 저장된 영상 파일 경로
    """
    image_url = await upload_image_to_fal(image_path, api_key=api_key)
    task_info = await submit_task(model_key, image_url, api_key=api_key)
    video_url = await poll_task(task_info["status_url"], task_info["response_url"], api_key=api_key)
    await download_video(video_url, output_path)
    return output_path


async def generate_clips_batch(
    images: list[str],
    output_dir: str,
    model_key: str = "hailuo",
    progress_callback=None,
    job_id: str = None,
    api_key: str = None,
) -> list[str]:
    """
    여러 이미지를 동시에 AI 영상 클립으로 변환.

    Returns: 생성된 클립 파일 경로 리스트
    """
    # 1단계: 모든 이미지를 fal CDN에 업로드
    if progress_callback and job_id:
        progress_callback(
            job_id=job_id,
            status="generating_clips",
            progress=0.26,
            step=f"이미지 업로드 중 (0/{len(images)})",
        )

    upload_tasks = [upload_image_to_fal(img, api_key=api_key) for img in images]
    image_urls = await asyncio.gather(*upload_tasks, return_exceptions=True)

    for i, url in enumerate(image_urls):
        if isinstance(url, Exception):
            raise RuntimeError(f"이미지 {i} 업로드 실패: {url}")

    if progress_callback and job_id:
        progress_callback(
            job_id=job_id,
            status="generating_clips",
            progress=0.28,
            step=f"AI 영상 생성 요청 중...",
        )

    # 2단계: 모든 태스크 동시 제출
    submit_tasks_list = [submit_task(model_key, url, api_key=api_key) for url in image_urls]
    submissions = await asyncio.gather(*submit_tasks_list, return_exceptions=True)

    for i, sub in enumerate(submissions):
        if isinstance(sub, Exception):
            raise RuntimeError(f"클립 {i} 제출 실패: {sub}")

    if progress_callback and job_id:
        model_label = MODELS[model_key]["label"]
        progress_callback(
            job_id=job_id,
            status="generating_clips",
            progress=0.30,
            step=f"{model_label} 영상 생성 대기 중 (0/{len(images)})",
        )

    # 3단계: 모든 태스크 동시 폴링 + 다운로드
    completed = 0

    async def _poll_and_download(i, task_info):
        nonlocal completed
        output_path = os.path.join(output_dir, f"clip_raw_{i:02d}.mp4")
        video_url = await poll_task(task_info["status_url"], task_info["response_url"], api_key=api_key)
        await download_video(video_url, output_path)
        completed += 1
        if progress_callback and job_id:
            p = 0.30 + (completed / len(images)) * 0.20
            progress_callback(
                job_id=job_id,
                status="generating_clips",
                progress=round(p, 2),
                step=f"AI 영상 생성 중 ({completed}/{len(images)})",
            )
        return output_path

    results = await asyncio.gather(
        *[_poll_and_download(i, info) for i, info in enumerate(submissions)],
        return_exceptions=True,
    )

    clip_paths = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            raise RuntimeError(f"클립 {i} 생성 실패: {r}")
        clip_paths.append(r)

    return clip_paths
