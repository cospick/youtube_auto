"""YouTube Shorts 자동 제작 웹앱 - FastAPI 진입점"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from db.database import init_db
from api.routes import generate, jobs, preview, assets, tts_preview
from api.routes.assets import bgm_router
from config import settings
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 초기화
    init_db()
    os.makedirs(settings.STORAGE_DIR, exist_ok=True)
    os.makedirs(settings.BGM_DIR, exist_ok=True)
    print(f"\n  AI 쇼츠 자동 제작 웹앱 시작!")
    print(f"  http://localhost:8000\n")
    yield


app = FastAPI(title="AI 쇼츠 자동 제작", lifespan=lifespan)

STATIC_DIR = os.path.join(settings.BASE_DIR, "static")

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# API 라우트 등록
app.include_router(generate.router)
app.include_router(jobs.router)
app.include_router(preview.router)
app.include_router(assets.router)
app.include_router(bgm_router)
app.include_router(tts_preview.router)


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
