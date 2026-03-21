import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import CurrentUser
from app.auth.security import (
    create_access_token, create_refresh_token,
    decode_refresh_token, hash_password, hash_token, verify_password,
)
from app.database.models import RefreshToken, User, UserRole
from app.database.session import get_db

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: EmailStr
    full_name: str
    password: str

    @field_validator("password")
    @classmethod
    def pw_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    full_name: str
    role: UserRole


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(
    payload: RegisterRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    if (await db.execute(select(User).where(User.email == payload.email))).scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        email=payload.email,
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password),
        role=UserRole.USER,
    )
    db.add(user)
    await db.flush()

    access = create_access_token(str(user.id), user.role.value)
    refresh, expires = create_refresh_token(str(user.id))
    db.add(RefreshToken(user_id=user.id, token_hash=hash_token(refresh), expires_at=expires))
    await db.commit()
    return TokenResponse(access_token=access, refresh_token=refresh)


@router.post("/login", response_model=TokenResponse)
async def login(
    payload: LoginRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    result = await db.execute(select(User).where(User.email == payload.email))
    user = result.scalar_one_or_none()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    access = create_access_token(str(user.id), user.role.value)
    refresh, expires = create_refresh_token(str(user.id))
    db.add(RefreshToken(user_id=user.id, token_hash=hash_token(refresh), expires_at=expires))
    await db.commit()
    return TokenResponse(access_token=access, refresh_token=refresh)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    payload: RefreshRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    try:
        data = decode_refresh_token(payload.refresh_token)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    stored = (await db.execute(
        select(RefreshToken).where(
            RefreshToken.token_hash == hash_token(payload.refresh_token),
            RefreshToken.is_revoked == False,
        )
    )).scalar_one_or_none()

    if not stored or stored.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Refresh token expired")

    stored.is_revoked = True
    user = (await db.execute(
        select(User).where(User.id == uuid.UUID(data["sub"]), User.is_active == True)
    )).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    new_access = create_access_token(str(user.id), user.role.value)
    new_refresh, new_exp = create_refresh_token(str(user.id))
    db.add(RefreshToken(user_id=user.id, token_hash=hash_token(new_refresh), expires_at=new_exp))
    await db.commit()
    return TokenResponse(access_token=new_access, refresh_token=new_refresh)


@router.post("/logout", status_code=204)
async def logout(
    payload: RefreshRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    stored = (await db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == hash_token(payload.refresh_token))
    )).scalar_one_or_none()
    if stored:
        stored.is_revoked = True
        await db.commit()


@router.get("/me", response_model=UserResponse)
async def me(current_user: CurrentUser):
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
    )
