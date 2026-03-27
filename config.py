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
    GEMINI_TEXT_MODEL: str = "gemini-3-flash-preview"
    GEMINI_IMAGE_MODEL: str = "gemini-3.1-flash-image-preview"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()


def find_font(bold=True):
    """시스템에서 한국어 폰트 탐색"""
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
    else:
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
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[-1]  # 최종 폴백


# 폰트 자동 설정
if not settings.FONT_TITLE:
    settings.FONT_TITLE = find_font(bold=True)
if not settings.FONT_SUB:
    settings.FONT_SUB = find_font(bold=False)
