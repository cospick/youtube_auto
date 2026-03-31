"""앱 설정 - 환경 변수 및 경로 관리"""

from pydantic_settings import BaseSettings
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Settings(BaseSettings):
    # API 키
    GEMINI_API_KEY: str = ""
    TYPECAST_API_KEY: str = ""
    FAL_KEY: str = ""

    # 데이터베이스 (빈 값이면 로컬 SQLite, Railway 배포 시 자동 설정)
    DATABASE_URL: str = ""

    # 경로
    BASE_DIR: str = BASE_DIR
    STORAGE_DIR: str = os.path.join(BASE_DIR, "storage")
    BGM_DIR: str = os.path.join(BASE_DIR, "bgm")

    # 폰트 (우선순위별 탐색)
    FONT_TITLE: str = ""
    FONT_SUB: str = ""

    # 영상 기본값
    TARGET_WIDTH: int = 1080
    TARGET_HEIGHT: int = 1920
    FPS: int = 30

    # Gemini 모델
    GEMINI_TEXT_MODEL: str = "gemini-2.5-pro"
    GEMINI_IMAGE_MODEL: str = "gemini-3.1-flash-image-preview"

    # JWT 인증
    JWT_SECRET: str = ""
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # 회원가입 초대 코드 (빈 값이면 초대 코드 없이 가입 가능)
    INVITE_CODE: str = ""

    # 서비스 기본 URL (비밀번호 재설정 이메일 링크 등에 사용)
    BASE_URL: str = "http://localhost:8000"

    # OAuth - Google
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/auth/google/callback"

    # OAuth - Kakao
    KAKAO_CLIENT_ID: str = ""
    KAKAO_CLIENT_SECRET: str = ""
    KAKAO_REDIRECT_URI: str = "http://localhost:8000/api/auth/kakao/callback"

    # SMTP (비밀번호 재설정 이메일)
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""

    # Cloudflare R2 (빈 값이면 로컬 전용)
    R2_ENDPOINT_URL: str = ""
    R2_ACCESS_KEY_ID: str = ""
    R2_SECRET_ACCESS_KEY: str = ""
    R2_BUCKET_NAME: str = ""
    R2_PRESIGN_EXPIRE_SECONDS: int = 3600

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()


def find_font(bold=True):
    """시스템에서 한국어 폰트 탐색 (Windows / macOS / Linux)"""
    import glob as _glob
    home = os.path.expanduser("~")

    if sys.platform == "win32":
        if bold:
            candidates = [
                "C:/Windows/Fonts/Pretendard-ExtraBold.otf",
                "C:/Windows/Fonts/Pretendard-Bold.otf",
                "C:/Windows/Fonts/malgunbd.ttf",
                "C:/Windows/Fonts/NanumGothicBold.ttf",
            ]
        else:
            candidates = [
                "C:/Windows/Fonts/Pretendard-SemiBold.otf",
                "C:/Windows/Fonts/Pretendard-Regular.otf",
                "C:/Windows/Fonts/malgun.ttf",
                "C:/Windows/Fonts/NanumGothic.ttf",
            ]
    elif sys.platform == "darwin":
        if bold:
            candidates = [
                f"{home}/Library/Fonts/GmarketSansTTFBold.ttf",
                f"{home}/Library/Fonts/NanumSquareEB.ttf",
                f"{home}/Library/Fonts/Pretendard-Bold.ttf",
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            ]
        else:
            candidates = [
                f"{home}/Library/Fonts/NanumSquareR.ttf",
                f"{home}/Library/Fonts/Pretendard-Regular.ttf",
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            ]
    else:
        # Linux (Docker) — Pretendard 우선, Noto CJK 폴백
        if bold:
            candidates = [
                "/usr/share/fonts/pretendard/Pretendard-ExtraBold.otf",
                "/usr/share/fonts/pretendard/Pretendard-Bold.otf",
            ]
        else:
            candidates = [
                "/usr/share/fonts/pretendard/Pretendard-SemiBold.otf",
                "/usr/share/fonts/pretendard/Pretendard-Regular.otf",
            ]
        for path in candidates:
            if os.path.exists(path):
                return path
        # 폴백: Noto Sans CJK
        pattern = "**/NotoSansCJK*Bold*" if bold else "**/NotoSansCJK*Regular*"
        found = _glob.glob(f"/usr/share/fonts/{pattern}", recursive=True)
        if found:
            return found[0]
        fallback = "NotoSansCJK-Bold.ttc" if bold else "NotoSansCJK-Regular.ttc"
        candidates = [
            f"/usr/share/fonts/opentype/noto/{fallback}",
            f"/usr/share/fonts/truetype/noto/{fallback}",
        ]

    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[-1]  # 최종 폴백


# 폰트 자동 설정
if not settings.FONT_TITLE:
    settings.FONT_TITLE = find_font(bold=True)
if not settings.FONT_SUB:
    settings.FONT_SUB = find_font(bold=False)
