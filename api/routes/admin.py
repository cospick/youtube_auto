"""관리자 API - 사용자 관리, 작업 이력"""

import re
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from db.database import get_db
from db.models import User, Job, PreApprovedEmail
from api.deps import get_current_admin
from core.time_utils import utc_isoformat

router = APIRouter(prefix="/api/admin", tags=["admin"])


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class PreApprovedEmailRequest(BaseModel):
    emails: str


@router.get("/users")
async def list_all_users(db: Session = Depends(get_db), _admin: User = Depends(get_current_admin)):
    """전체 사용자 목록"""
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [
        {
            "id": u.id,
            "email": u.email,
            "nickname": u.nickname,
            "role": u.role,
            "provider": u.provider,
            "approved": u.approved,
            "created_at": utc_isoformat(u.created_at),
        }
        for u in users
    ]


@router.get("/pending-users")
async def list_pending_users(db: Session = Depends(get_db), _admin: User = Depends(get_current_admin)):
    """승인 대기 사용자 목록"""
    users = db.query(User).filter(User.approved == False).order_by(User.created_at.desc()).all()
    return [
        {
            "id": u.id,
            "email": u.email,
            "nickname": u.nickname,
            "provider": u.provider,
            "created_at": utc_isoformat(u.created_at),
        }
        for u in users
    ]


@router.post("/users/{user_id}/approve")
async def approve_user(user_id: str, db: Session = Depends(get_db), _admin: User = Depends(get_current_admin)):
    """사용자 승인"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")
    user.approved = True
    db.commit()
    return {"message": f"{user.email} 승인 완료"}


@router.post("/users/{user_id}/reject")
async def reject_user(user_id: str, db: Session = Depends(get_db), _admin: User = Depends(get_current_admin)):
    """사용자 거절 (삭제)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")
    if user.role == "admin":
        raise HTTPException(status_code=400, detail="관리자는 거절할 수 없습니다")
    db.delete(user)
    db.commit()
    return {"message": f"{user.email} 거절(삭제) 완료"}


@router.post("/users/{user_id}/role")
async def toggle_user_role(user_id: str, db: Session = Depends(get_db), _admin: User = Depends(get_current_admin)):
    """사용자 역할 변경 (user ↔ admin)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")
    if user.id == _admin.id and user.role == "admin":
        admin_count = db.query(User).filter(User.role == "admin").count()
        if admin_count <= 1:
            raise HTTPException(status_code=400, detail="관리자가 1명일 때는 자신의 역할을 변경할 수 없습니다")
    user.role = "admin" if user.role == "user" else "user"
    db.commit()
    return {"message": f"{user.email} → {user.role}로 변경 완료"}


@router.get("/pre-approved-emails")
async def list_pre_approved_emails(
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    """사전 승인 이메일 목록 (아직 가입하지 않은 이메일만 보관)"""
    rows = db.query(PreApprovedEmail).order_by(PreApprovedEmail.created_at.desc()).all()
    return [
        {"email": r.email, "created_at": utc_isoformat(r.created_at)}
        for r in rows
    ]


@router.post("/pre-approved-emails")
async def add_pre_approved_emails(
    req: PreApprovedEmailRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    """사전 승인 이메일 추가 (단일/다중, 쉼표·줄바꿈으로 구분).

    응답: 입력한 이메일을 처리 결과별로 분류.
      - added: 새로 사전 승인 목록에 등록한 이메일
      - already_pre_approved: 이미 사전 승인 목록에 있던 이메일
      - auto_approved_existing: 이미 가입했지만 미승인 → 즉시 approved=True 처리
      - already_user_approved: 이미 가입+승인 완료된 이메일 (변경 없음)
      - invalid: 형식 오류
    """
    tokens = [t.strip() for t in re.split(r"[,\s]+", req.emails or "") if t.strip()]
    # 입력 순서 유지하면서 중복 제거 (사용자가 같은 이메일을 여러 번 적어도 1번만 처리)
    seen: set[str] = set()
    unique: list[str] = []
    for t in tokens:
        lc = t.lower()
        if lc in seen:
            continue
        seen.add(lc)
        unique.append(lc)

    added: list[str] = []
    already_pre_approved: list[str] = []
    auto_approved_existing: list[str] = []
    already_user_approved: list[str] = []
    invalid: list[str] = []

    for email_lc in unique:
        if not _EMAIL_RE.match(email_lc):
            invalid.append(email_lc)
            continue

        existing_user = db.query(User).filter(User.email.ilike(email_lc)).first()
        if existing_user:
            if existing_user.approved:
                already_user_approved.append(email_lc)
            else:
                existing_user.approved = True
                auto_approved_existing.append(email_lc)
            continue

        existing_pre = db.query(PreApprovedEmail).filter(PreApprovedEmail.email == email_lc).first()
        if existing_pre:
            already_pre_approved.append(email_lc)
            continue

        db.add(PreApprovedEmail(email=email_lc, created_by_user_id=admin.id))
        added.append(email_lc)

    db.commit()
    return {
        "added": added,
        "already_pre_approved": already_pre_approved,
        "auto_approved_existing": auto_approved_existing,
        "already_user_approved": already_user_approved,
        "invalid": invalid,
    }


@router.delete("/pre-approved-emails/{email}")
async def delete_pre_approved_email(
    email: str,
    db: Session = Depends(get_db),
    _admin: User = Depends(get_current_admin),
):
    """사전 승인 이메일 삭제"""
    row = db.query(PreApprovedEmail).filter(PreApprovedEmail.email == email.lower()).first()
    if not row:
        raise HTTPException(status_code=404, detail="등록되지 않은 이메일입니다")
    db.delete(row)
    db.commit()
    return {"message": f"{email} 삭제 완료"}


@router.get("/jobs")
async def list_all_jobs(limit: int = 50, db: Session = Depends(get_db), _admin: User = Depends(get_current_admin)):
    """전체 작업 이력 (관리자용, 작성자 정보 포함)"""
    jobs = db.query(Job).order_by(Job.created_at.desc()).limit(limit).all()

    user_ids = list({j.user_id for j in jobs if j.user_id})
    user_map = {}
    if user_ids:
        users = db.query(User).filter(User.id.in_(user_ids)).all()
        user_map = {u.id: u for u in users}

    from api.routes.jobs import _job_to_response
    return [_job_to_response(j, user_map) for j in jobs]
