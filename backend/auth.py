"""
Authentication and Authorization
=================================

JWT-based authentication system with password hashing.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import os
from dotenv import load_dotenv
import logging

from backend.models import UserSignup, UserLogin, UserInDB, UserResponse, Token, TokenData
from backend.database import get_users_collection

load_dotenv()

logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-CHANGE-IN-PRODUCTION-use-openssl-rand-hex-32")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token
security = HTTPBearer()

# Router
router = APIRouter(prefix="/api/auth", tags=["Authentication"])


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserInDB:
    """Get current authenticated user from JWT token."""
    token = credentials.credentials
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("user_id")
        email: str = payload.get("email")
        
        if user_id is None or email is None:
            raise credentials_exception
            
        token_data = TokenData(
            user_id=user_id,
            email=email,
            username=payload.get("username", "")
        )
        
    except JWTError as e:
        logger.error(f"JWT decode error: {e}")
        raise credentials_exception
    
    # Get user from database
    users_collection = get_users_collection()
    user = await users_collection.find_one({"email": token_data.email})
    
    if user is None:
        raise credentials_exception
    
    return UserInDB(**user)


# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@router.post("/signup", response_model=Token, status_code=status.HTTP_201_CREATED)
async def signup(user_data: UserSignup):
    """
    Register a new user.
    
    - **username**: Unique username (3-50 chars, alphanumeric)
    - **email**: Valid email address
    - **password**: Strong password (min 8 chars, uppercase, lowercase, digit)
    - **full_name**: Optional full name
    """
    users_collection = get_users_collection()
    
    # Check if user already exists
    existing_user = await users_collection.find_one({
        "$or": [
            {"email": user_data.email},
            {"username": user_data.username.lower()}
        ]
    })
    
    if existing_user:
        if existing_user.get("email") == user_data.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    
    new_user = UserInDB(
        username=user_data.username.lower(),
        email=user_data.email,
        full_name=user_data.full_name,
        hashed_password=hashed_password,
        role="user",
        is_active=True,
        created_at=datetime.utcnow(),
        last_login=datetime.utcnow()
    )
    
    # Insert into database
    result = await users_collection.insert_one(new_user.dict(by_alias=True, exclude={"id"}))
    user_id = str(result.inserted_id)
    
    logger.info(f"✅ New user registered: {user_data.email}")
    
    # Create access token
    access_token = create_access_token(
        data={
            "user_id": user_id,
            "email": new_user.email,
            "username": new_user.username
        }
    )
    
    # Prepare user response
    user_response = UserResponse(
        _id=user_id,
        username=new_user.username,
        email=new_user.email,
        full_name=new_user.full_name,
        role=new_user.role,
        created_at=new_user.created_at,
        last_login=new_user.last_login
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=user_response
    )


@router.post("/login", response_model=Token)
async def login(credentials: UserLogin):
    """
    Authenticate user and return JWT token.
    
    - **email**: User email
    - **password**: User password
    """
    users_collection = get_users_collection()
    
    # Find user by email
    user = await users_collection.find_one({"email": credentials.email})
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    # Verify password
    if not verify_password(credentials.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    # Check if user is active
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled"
        )
    
    # Update last login
    await users_collection.update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login": datetime.utcnow()}}
    )
    
    logger.info(f"✅ User logged in: {credentials.email}")
    
    # Create access token
    access_token = create_access_token(
        data={
            "user_id": str(user["_id"]),
            "email": user["email"],
            "username": user["username"]
        }
    )
    
    # Prepare user response
    user_response = UserResponse(
        _id=str(user["_id"]),
        username=user["username"],
        email=user["email"],
        full_name=user.get("full_name"),
        role=user.get("role", "user"),
        created_at=user["created_at"],
        last_login=datetime.utcnow()
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=user_response
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: UserInDB = Depends(get_current_user)):
    """
    Get current authenticated user information.
    
    Requires valid JWT token in Authorization header.
    """
    return UserResponse(
        _id=str(current_user.id),
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )


@router.post("/logout")
async def logout(current_user: UserInDB = Depends(get_current_user)):
    """
    Logout current user (client should delete token).
    """
    logger.info(f"User logged out: {current_user.email}")
    
    return {
        "status": "success",
        "message": "Successfully logged out"
    }
