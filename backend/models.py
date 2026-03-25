"""
Data Models for Authentication and User Management
===================================================

Pydantic models for MongoDB documents and API request/response schemas.
"""

from pydantic import BaseModel, Field, EmailStr, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic."""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)
    
    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


class UserSignup(BaseModel):
    """User registration request."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    full_name: Optional[str] = Field(None, max_length=100)
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must be alphanumeric (with _ or -)')
        return v.lower()
    
    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class UserLogin(BaseModel):
    """User login request."""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """User information response (public data)."""
    id: str = Field(alias="_id")
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    role: str = "user"
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        populate_by_name = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }


class UserInDB(BaseModel):
    """User document in MongoDB (includes password hash)."""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    hashed_password: str
    role: str = "user"
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    
    class Config:
        populate_by_name = True
        json_encoders = {ObjectId: str}


class Token(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class TokenData(BaseModel):
    """JWT token payload."""
    user_id: str
    email: str
    username: str


class PredictionHistory(BaseModel):
    """Prediction history document."""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    disease: str
    patient_data: Dict[str, Any]
    prediction: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
        json_encoders = {ObjectId: str}
