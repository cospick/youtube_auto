"""인증 API - Google OAuth + 관리자 비상 로그인 + API 키 관리"""

from fastapi import APIRouter, HTTPException, Depends, Response, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from db.database import get_db
from db.models import User, PreApprovedEmail
from api.models import LoginRequest, ApiKeysUpdateRequest, ApiKeysResponse
from api.deps import get_current_user, get_approved_user
from core.security import (
    verify_password,
    create_access_token, create_refresh_token, decode_token,
    encrypt_api_key, decrypt_api_key,
)
from config import settings
import jwt
import uuid
import time

router = APIRouter(prefix="/api/auth", tags=["auth"])

# OAuth CSRF state 저장소 (메모리, {state: expire_timestamp})
_oauth_states: dict[str, float] = {}

def _create_oauth_state() -> str:
    """state 생성 + 저장 (5분 유효)"""
    now = time.time()
    expired = [k for k, v in _oauth_states.items() if v < now]
    for k in expired:
        del _oauth_states[k]
    state = uuid.uuid4().hex
    _oauth_states[state] = now + 300
    return state

def _verify_oauth_state(state: str) -> bool:
    """state 검증 + 소비 (1회용)"""
    expire = _oauth_states.pop(state, None)
    if expire is None:
        return False
    return time.time() < expire


def _set_auth_cookies(response: Response, user: User):
    """JWT 쿠키 설정"""
    token_data = {"sub": user.id, "email": user.email, "role": user.role}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    is_https = settings.BASE_URL.startswith("https")
    response.set_cookie(
        key="access_token", value=access_token,
        httponly=True, samesite="lax", secure=is_https,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
    response.set_cookie(
        key="refresh_token", value=refresh_token,
        httponly=True, samesite="lax", secure=is_https,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
    )
    return access_token


def _user_response(user: User) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "nickname": user.nickname,
        "role": user.role,
        "provider": user.provider,
        "approved": user.approved,
        "has_gemini_key": bool(user.gemini_api_key_enc),
        "has_typecast_key": bool(user.typecast_api_key_enc),
        "has_fal_key": bool(user.fal_key_enc),
    }


# ── 공개 설정 ──

@router.get("/settings")
async def get_auth_settings():
    """프론트엔드에서 로그인 방식 확인용"""
    return {"google_only": True}


# ── 관리자 비상 로그인 (Google OAuth 없는 환경용) ──

@router.post("/login")
async def login(req: LoginRequest, response: Response, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not user.hashed_password:
        raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 올바르지 않습니다")

    if user.role != "admin":
        raise HTTPException(status_code=403, detail="관리자만 이메일 로그인이 가능합니다")

    if not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 올바르지 않습니다")

    _set_auth_cookies(response, user)
    return {"message": "로그인 성공", "user": _user_response(user)}


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie("access_token")
    response.delete_cookie("refresh_token")
    return {"message": "로그아웃 완료"}


@router.get("/me")
async def get_me(user: User = Depends(get_current_user)):
    return {"user": _user_response(user)}


@router.post("/refresh")
async def refresh_token(request: Request, response: Response, db: Session = Depends(get_db)):
    token = request.cookies.get("refresh_token")
    if not token:
        raise HTTPException(status_code=401, detail="리프레시 토큰이 없습니다")

    try:
        payload = decode_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="리프레시 토큰이 만료되었습니다")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다")

    user = db.query(User).filter(User.id == payload.get("sub")).first()
    if not user:
        raise HTTPException(status_code=401, detail="유효하지 않은 사용자입니다")

    _set_auth_cookies(response, user)
    return {"message": "토큰 갱신 완료", "user": _user_response(user)}


# ── OAuth: Google ──

@router.get("/google/login")
async def google_login():
    if not settings.GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Google OAuth가 설정되지 않았습니다")

    from core.oauth import get_google_auth_url
    state = _create_oauth_state()
    url = get_google_auth_url(state)
    return RedirectResponse(url)


@router.get("/google/callback")
async def google_callback(code: str, state: str = "", db: Session = Depends(get_db)):
    if not _verify_oauth_state(state):
        return RedirectResponse("/static/login.html?error=invalid_state")

    from core.oauth import exchange_google_code

    try:
        user_info = await exchange_google_code(code)
    except Exception:
        return RedirectResponse("/static/login.html?error=google_failed")

    email = user_info.get("email")
    if not email:
        return RedirectResponse("/static/login.html?error=no_email")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        email_lc = email.lower()
        pre = db.query(PreApprovedEmail).filter(PreApprovedEmail.email == email_lc).first()
        user = User(
            id=uuid.uuid4().hex,
            email=email,
            nickname=user_info.get("name", email.split("@")[0]),
            provider="google",
            provider_id=user_info.get("sub"),
            approved=bool(pre),
        )
        db.add(user)
        if pre:
            db.delete(pre)
        db.commit()
        db.refresh(user)

    if user.approved:
        response = RedirectResponse("/")
    else:
        response = RedirectResponse("/static/pending.html")
    _set_auth_cookies(response, user)
    return response


# ── API 키 관리 ──

def _mask_key(key: str) -> str:
    """API 키 마스킹: 앞4자 + *** + 뒤3자"""
    if not key or len(key) < 8:
        return "***"
    return key[:4] + "***" + key[-3:]


@router.get("/api-keys")
async def get_api_keys(user: User = Depends(get_approved_user)):
    """현재 사용자의 API 키 상태 조회 (마스킹)"""
    result = {"gemini": None, "typecast": None, "fal": None}
    if user.gemini_api_key_enc:
        result["gemini"] = _mask_key(decrypt_api_key(user.gemini_api_key_enc))
    if user.typecast_api_key_enc:
        result["typecast"] = _mask_key(decrypt_api_key(user.typecast_api_key_enc))
    if user.fal_key_enc:
        result["fal"] = _mask_key(decrypt_api_key(user.fal_key_enc))
    return result


@router.put("/api-keys")
async def update_api_keys(
    req: ApiKeysUpdateRequest,
    user: User = Depends(get_approved_user),
    db: Session = Depends(get_db),
):
    """API 키 저장 (검증 후 암호화)"""
    import httpx

    # Gemini 키 검증 + 저장
    if req.gemini_api_key is not None:
        if req.gemini_api_key == "":
            user.gemini_api_key_enc = None
        else:
            try:
                from google import genai
                client = genai.Client(api_key=req.gemini_api_key)
                client.models.list(config={"page_size": 1})
            except Exception:
                raise HTTPException(status_code=400, detail="Gemini API 키가 유효하지 않습니다")
            user.gemini_api_key_enc = encrypt_api_key(req.gemini_api_key)

    # Typecast 키 검증 + 저장
    if req.typecast_api_key is not None:
        if req.typecast_api_key == "":
            user.typecast_api_key_enc = None
        else:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        "https://api.typecast.ai/v1/voices",
                        headers={"X-API-KEY": req.typecast_api_key},
                    )
                    if resp.status_code == 401:
                        raise ValueError("Unauthorized")
            except Exception:
                raise HTTPException(status_code=400, detail="Typecast API 키가 유효하지 않습니다")
            user.typecast_api_key_enc = encrypt_api_key(req.typecast_api_key)

    # FAL 키 검증 + 저장
    if req.fal_key is not None:
        if req.fal_key == "":
            user.fal_key_enc = None
        else:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        "https://queue.fal.run/fal-ai/fast-sdxl",
                        headers={"Authorization": f"Key {req.fal_key}"},
                    )
                    if resp.status_code == 401:
                        raise ValueError("Unauthorized")
            except Exception:
                raise HTTPException(status_code=400, detail="FAL API 키가 유효하지 않습니다")
            user.fal_key_enc = encrypt_api_key(req.fal_key)

    db.commit()
    return {"message": "API 키가 저장되었습니다"}
