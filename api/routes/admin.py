"""관리자 API - 사용자 관리, 작업 이력"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from db.database import get_db
from db.models import User, Job
from api.deps import get_current_admin

router = APIRouter(prefix="/api/admin", tags=["admin"])


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
            "created_at": u.created_at.isoformat() if u.created_at else None,
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
            "created_at": u.created_at.isoformat() if u.created_at else None,
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
