"""Cloudflare R2 스토리지 — 업로드, 스트리밍, presigned URL"""

import os
import glob
import asyncio
import logging
from config import settings

logger = logging.getLogger(__name__)

_r2_client = None


def is_r2_enabled() -> bool:
    return bool(
        settings.R2_BUCKET_NAME
        and settings.R2_ENDPOINT_URL
        and settings.R2_ACCESS_KEY_ID
        and settings.R2_SECRET_ACCESS_KEY
    )


def is_r2_required_for_generation() -> bool:
    """Railway/PostgreSQL 운영 모드는 로컬 디스크를 영구 저장소로 신뢰하지 않는다."""
    return bool(settings.DATABASE_URL)


def require_r2_for_generation() -> None:
    if is_r2_required_for_generation() and not is_r2_enabled():
        raise RuntimeError(
            "Railway/PostgreSQL 환경에서는 R2 설정이 필요합니다. "
            "R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME을 모두 설정해주세요."
        )


def get_r2_client():
    global _r2_client
    if _r2_client is None:
        import boto3
        _r2_client = boto3.client(
            "s3",
            endpoint_url=settings.R2_ENDPOINT_URL,
            aws_access_key_id=settings.R2_ACCESS_KEY_ID,
            aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
            region_name="auto",
        )
    return _r2_client


# ── 업로드 ──

def _upload_sync(local_path: str, r2_key: str) -> bool:
    """동기 업로드 (asyncio.to_thread에서 호출)"""
    try:
        get_r2_client().upload_file(local_path, settings.R2_BUCKET_NAME, r2_key)
        return True
    except Exception as e:
        logger.warning(f"R2 upload failed: {r2_key} — {e}")
        return False


async def upload_file(local_path: str, r2_key: str, max_retries: int = 2) -> bool:
    """로컬 → R2 업로드 (비동기 + 재시도)"""
    if not is_r2_enabled() or not os.path.exists(local_path):
        return False

    for attempt in range(max_retries + 1):
        success = await asyncio.to_thread(_upload_sync, local_path, r2_key)
        if success:
            return True
        if attempt < max_retries:
            logger.info(f"R2 upload retry {attempt + 1}/{max_retries}: {r2_key}")
            await asyncio.sleep(1)

    logger.error(f"R2 upload failed after {max_retries + 1} attempts: {r2_key}")
    return False


async def upload_job_files(job_id: str, file_type: str) -> bool:
    """job의 파일 일괄 업로드. file_type: 'images', 'clips', 'output'"""
    if not is_r2_enabled():
        return False

    job_dir = os.path.join(settings.STORAGE_DIR, job_id)
    all_ok = True

    if file_type == "images":
        files = sorted(set(
            glob.glob(os.path.join(job_dir, "images", "img_*.png"))
            + glob.glob(os.path.join(job_dir, "images", "line_*.png"))
        ))
        for f in files:
            r2_key = f"jobs/{job_id}/images/{os.path.basename(f)}"
            if not await upload_file(f, r2_key):
                all_ok = False

    elif file_type == "clips":
        files = sorted(set(
            glob.glob(os.path.join(job_dir, "clips", "clip_raw_*.mp4"))
            + glob.glob(os.path.join(job_dir, "clips", "clip_*.mp4"))
        ))
        for f in files:
            r2_key = f"jobs/{job_id}/clips/{os.path.basename(f)}"
            if not await upload_file(f, r2_key):
                all_ok = False

    elif file_type == "output":
        output = os.path.join(job_dir, "output", "shorts_final.mp4")
        if os.path.exists(output):
            r2_key = f"jobs/{job_id}/output/shorts_final.mp4"
            if not await upload_file(output, r2_key):
                all_ok = False

    elif file_type == "tts":
        tts_dir = os.path.join(job_dir, "tts")
        if os.path.isdir(tts_dir):
            for name in sorted(os.listdir(tts_dir)):
                if name.startswith("_") or name.startswith("."):
                    continue
                src = os.path.join(tts_dir, name)
                if not os.path.isfile(src):
                    continue
                r2_key = f"jobs/{job_id}/tts/{name}"
                if not await upload_file(src, r2_key):
                    all_ok = False

    return all_ok


async def download_job_tts_to_local(job_id: str) -> bool:
    """R2의 jobs/{job_id}/tts/* 를 로컬 storage/{job_id}/tts/ 로 복구.
    카드 B reopen 시 컨테이너 재시작 등으로 로컬 tts/가 비어 있을 때 사용."""
    if not is_r2_enabled():
        return False

    prefix = f"jobs/{job_id}/tts/"
    local_dir = os.path.join(settings.STORAGE_DIR, job_id, "tts")
    os.makedirs(local_dir, exist_ok=True)

    def _sync() -> int:
        client = get_r2_client()
        token = None
        downloaded = 0
        while True:
            kwargs = {"Bucket": settings.R2_BUCKET_NAME, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            resp = client.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                key = obj["Key"]
                name = key.rsplit("/", 1)[-1]
                if not name:
                    continue
                local_path = os.path.join(local_dir, name)
                if os.path.exists(local_path):
                    continue
                try:
                    client.download_file(settings.R2_BUCKET_NAME, key, local_path)
                    downloaded += 1
                except Exception as e:
                    logger.warning(f"R2 download tts failed {key}: {e}")
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return downloaded

    try:
        n = await asyncio.to_thread(_sync)
        logger.info(f"R2 → 로컬 tts 복구 ({job_id}): {n}개")
        return True
    except Exception as e:
        logger.error(f"R2 download_job_tts_to_local 실패 {job_id}: {e}")
        return False


# ── 스트리밍 / 다운로드 ──

def download_file_sync(r2_key: str, local_path: str) -> bool:
    """R2 → 로컬 동기 다운로드 (워커에서 asyncio.to_thread로 호출)"""
    if not is_r2_enabled():
        return False
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        get_r2_client().download_file(settings.R2_BUCKET_NAME, r2_key, local_path)
        return True
    except Exception as e:
        logger.error(f"R2 download failed: {r2_key} — {e}")
        return False


def stream_from_r2(r2_key: str):
    """R2에서 직접 스트리밍 (StreamingResponse용 generator)"""
    try:
        resp = get_r2_client().get_object(Bucket=settings.R2_BUCKET_NAME, Key=r2_key)
        body = resp["Body"]
        while True:
            chunk = body.read(64 * 1024)  # 64KB chunks
            if not chunk:
                break
            yield chunk
    except Exception as e:
        logger.error(f"R2 stream failed: {r2_key} — {e}")
        return


def generate_presigned_url(r2_key: str, expires: int = None) -> str:
    """다운로드용 presigned URL 생성"""
    if expires is None:
        expires = settings.R2_PRESIGN_EXPIRE_SECONDS
    return get_r2_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.R2_BUCKET_NAME, "Key": r2_key},
        ExpiresIn=expires,
    )


def r2_file_exists(r2_key: str) -> bool:
    """R2에 파일 존재 확인"""
    try:
        get_r2_client().head_object(Bucket=settings.R2_BUCKET_NAME, Key=r2_key)
        return True
    except Exception:
        return False


# ── 객체 단위 복사/삭제 (카드 분할 시 인덱스 시프트용) ──

async def copy_object(src_key: str, dst_key: str) -> bool:
    """R2 객체 복사. is_r2_enabled() 거짓이면 True 반환 후 no-op."""
    if not is_r2_enabled():
        return True

    def _sync():
        client = get_r2_client()
        client.copy_object(
            Bucket=settings.R2_BUCKET_NAME,
            CopySource={"Bucket": settings.R2_BUCKET_NAME, "Key": src_key},
            Key=dst_key,
        )
        return True

    try:
        return await asyncio.to_thread(_sync)
    except Exception as e:
        logger.warning(f"R2 copy_object 실패 {src_key} → {dst_key}: {e}")
        return False


async def delete_object(r2_key: str) -> bool:
    """R2 객체 단일 삭제. is_r2_enabled() 거짓이면 True 반환 후 no-op."""
    if not is_r2_enabled():
        return True

    def _sync():
        client = get_r2_client()
        client.delete_object(Bucket=settings.R2_BUCKET_NAME, Key=r2_key)
        return True

    try:
        return await asyncio.to_thread(_sync)
    except Exception as e:
        logger.warning(f"R2 delete_object 실패 {r2_key}: {e}")
        return False


# ── 삭제 ──

async def delete_job_files(job_id: str):
    """특정 Job의 R2 파일 전체 삭제"""
    if not is_r2_enabled():
        return

    def _delete_sync():
        client = get_r2_client()
        prefix = f"jobs/{job_id}/"
        try:
            resp = client.list_objects_v2(Bucket=settings.R2_BUCKET_NAME, Prefix=prefix)
            objects = resp.get("Contents", [])
            if objects:
                client.delete_objects(
                    Bucket=settings.R2_BUCKET_NAME,
                    Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
                )
        except Exception as e:
            logger.error(f"R2 delete failed for job {job_id}: {e}")

    await asyncio.to_thread(_delete_sync)


def _delete_local_subdirs(job_id: str, subdirs: list[str]) -> None:
    """로컬 storage/{job_id}/<sub>/ 디렉토리 통째 삭제."""
    import shutil
    job_dir = os.path.join(settings.STORAGE_DIR, job_id)
    for sub in subdirs:
        path = os.path.join(job_dir, sub)
        if os.path.isdir(path):
            try:
                shutil.rmtree(path)
            except Exception as e:
                logger.warning(f"local subdir delete 실패 {path}: {e}")


async def delete_job_intermediate_files(job_id: str) -> None:
    """완료 후 R2 + 로컬에서 최종 output을 제외한 중간 산출물(images/clips/tts/temp)을 삭제."""
    intermediates = ["images", "clips", "tts", "temp"]

    if is_r2_enabled():
        prefixes = [f"jobs/{job_id}/{sub}/" for sub in intermediates]

        def _delete_sync():
            client = get_r2_client()
            for prefix in prefixes:
                token = None
                while True:
                    kwargs = {"Bucket": settings.R2_BUCKET_NAME, "Prefix": prefix}
                    if token:
                        kwargs["ContinuationToken"] = token
                    try:
                        resp = client.list_objects_v2(**kwargs)
                        objects = resp.get("Contents", [])
                        if objects:
                            client.delete_objects(
                                Bucket=settings.R2_BUCKET_NAME,
                                Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
                            )
                        if not resp.get("IsTruncated"):
                            break
                        token = resp.get("NextContinuationToken")
                    except Exception as e:
                        logger.warning(f"R2 intermediate delete failed for {prefix}: {e}")
                        break

        await asyncio.to_thread(_delete_sync)

    # 로컬 동반 삭제 — R2 활성 여부와 무관하게 항상 실행 (R2 비활성 환경 대응)
    _delete_local_subdirs(job_id, intermediates)


async def delete_job_all_files(job_id: str) -> None:
    """discard용 — R2 + 로컬에서 해당 job의 모든 산출물(intermediates + output + product 등) 삭제."""
    if is_r2_enabled():
        await delete_job_files(job_id)  # 기존 함수: jobs/{job_id}/* 전부 삭제

    # 로컬 job 디렉토리 전체 삭제
    import shutil
    job_dir = os.path.join(settings.STORAGE_DIR, job_id)
    if os.path.isdir(job_dir):
        try:
            shutil.rmtree(job_dir)
        except Exception as e:
            logger.warning(f"local job dir delete 실패 {job_dir}: {e}")
