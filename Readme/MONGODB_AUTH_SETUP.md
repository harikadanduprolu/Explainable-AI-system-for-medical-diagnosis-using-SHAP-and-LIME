# 🔐 MongoDB Authentication Setup Guide

**Created:** March 12, 2026  
**Application:** Explainable Medical AI System

---

## 📋 Overview

This guide explains how to set up and use the MongoDB-based user authentication system with JWT tokens.

### ✅ What's Been Added

**Backend Components:**
- 📁 `backend/database.py` - MongoDB Atlas connection with Motor (async driver)
- 📁 `backend/models.py` - Pydantic models for users, auth, predictions
- 📁 `backend/auth.py` - JWT authentication with bcrypt password hashing
- 🔄 `backend/main.py` - Updated with auth routes and lifecycle events

**Frontend Pages:**
- 🌐 `backend/static/login.html` - Beautiful login page with validation
- 🌐 `backend/static/signup.html` - Registration page with password strength checker

**Configuration:**
- 📝 `.env.example` - Updated with MongoDB and JWT settings
- 📦 `backend/requirements.txt` - Added auth dependencies

---

## 🚀 Quick Setup (3 Steps)

### 1. Install Dependencies

```bash
# Activate virtual environment
.\.venv\Scripts\activate

# Install new dependencies
pip install motor pymongo passlib[bcrypt] python-jose[cryptography] bcrypt email-validator
```

Or install from updated requirements:

```bash
pip install -r backend/requirements.txt
```

### 2. Configure Environment

Create `.env` file:

```bash
copy .env.example .env
```

Edit `.env` and update MongoDB settings:

```dotenv
# MongoDB Atlas Connection
MONGODB_URL=mongodb+srv://harikadanduprolu740_db_user:Vdo8kp6VVaEtCd3e@cluster0.2jis5r9.mongodb.net/?appName=Cluster0
DATABASE_NAME=medical_ai_db

# JWT Secret (Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))")
SECRET_KEY=your-super-secret-key-here

# JWT Token Expiration (minutes)
ACCESS_TOKEN_EXPIRE_MINUTES=1440
```

**Generate SECRET_KEY:**
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 3. Start the Application

```bash
python start_web_application.py
```

Or:

```bash
uvicorn backend.main:app --reload --port 8000
```

---

## 🌐 Access the Application

### User Interface
- 🔐 **Login:** http://localhost:8000/login
- 📝 **Signup:** http://localhost:8000/signup
- 🏥 **App:** http://localhost:8000/app (requires login)

### API Endpoints
- 📚 **API Docs:** http://localhost:8000/docs
- ❤️ **Health:** http://localhost:8000/health

---

## 🔑 Authentication API Endpoints

### 1. Signup (Create Account)

**Endpoint:** `POST /api/auth/signup`

**Request:**
```json
{
  "username": "johndoe",
  "email": "john@example.com",
  "password": "SecurePass123",
  "full_name": "John Doe"
}
```

**Password Requirements:**
- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one digit

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "_id": "65f8...",
    "username": "johndoe",
    "email": "john@example.com",
    "full_name": "John Doe",
    "role": "user",
    "created_at": "2026-03-12T10:30:00",
    "last_login": "2026-03-12T10:30:00"
  }
}
```

### 2. Login (Authenticate)

**Endpoint:** `POST /api/auth/login`

**Request:**
```json
{
  "email": "john@example.com",
  "password": "SecurePass123"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "_id": "65f8...",
    "username": "johndoe",
    "email": "john@example.com",
    "full_name": "John Doe",
    "role": "user",
    "created_at": "2026-03-12T10:30:00",
    "last_login": "2026-03-12T10:32:00"
  }
}
```

### 3. Get Current User

**Endpoint:** `GET /api/auth/me`

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "_id": "65f8...",
  "username": "johndoe",
  "email": "john@example.com",
  "full_name": "John Doe",
  "role": "user",
  "created_at": "2026-03-12T10:30:00",
  "last_login": "2026-03-12T10:32:00"
}
```

### 4. Logout

**Endpoint:** `POST /api/auth/logout`

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "status": "success",
  "message": "Successfully logged out"
}
```

---

## 💻 Using Authentication in Code

### Frontend JavaScript Example

```javascript
// Signup
async function signup(username, email, password, fullName) {
    const response = await fetch('/api/auth/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password, full_name: fullName })
    });
    
    const data = await response.json();
    
    if (response.ok) {
        // Store token
        localStorage.setItem('access_token', data.access_token);
        localStorage.setItem('user', JSON.stringify(data.user));
        return data;
    } else {
        throw new Error(data.detail);
    }
}

// Login
async function login(email, password) {
    const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
    });
    
    const data = await response.json();
    
    if (response.ok) {
        localStorage.setItem('access_token', data.access_token);
        localStorage.setItem('user', JSON.stringify(data.user));
        return data;
    } else {
        throw new Error(data.detail);
    }
}

// Make authenticated request
async function getPrediction(features) {
    const token = localStorage.getItem('access_token');
    
    const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ features })
    });
    
    return await response.json();
}

// Logout
function logout() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user');
    window.location.href = '/login';
}
```

### Python Example (Testing)

```python
import requests

BASE_URL = "http://localhost:8000"

# Signup
response = requests.post(f"{BASE_URL}/api/auth/signup", json={
    "username": "testuser",
    "email": "test@example.com",
    "password": "TestPass123",
    "full_name": "Test User"
})
data = response.json()
token = data["access_token"]

# Login
response = requests.post(f"{BASE_URL}/api/auth/login", json={
    "email": "test@example.com",
    "password": "TestPass123"
})
data = response.json()
token = data["access_token"]

# Get current user
headers = {"Authorization": f"Bearer {token}"}
response = requests.get(f"{BASE_URL}/api/auth/me", headers=headers)
user = response.json()
print(f"Logged in as: {user['username']}")

# Make prediction (with auth)
response = requests.post(
    f"{BASE_URL}/api/predict",
    headers=headers,
    json={
        "features": {
            "age": 45, "gender": 0, "heart_rate": 72,
            "systolic_bp": 120, "diastolic_bp": 80,
            "temperature": 98.6, "respiratory_rate": 16,
            "wbc_count": 7.5, "hemoglobin": 14.0,
            "platelet_count": 250, "creatinine": 1.0,
            "bun": 15, "glucose": 95, "lactate": 1.2
        }
    }
)
```

---

## 🗄️ MongoDB Database Structure

### Collections

**1. users**
```javascript
{
  "_id": ObjectId("..."),
  "username": "johndoe",
  "email": "john@example.com",
  "full_name": "John Doe",
  "hashed_password": "$2b$12$...",
  "role": "user",
  "is_active": true,
  "created_at": ISODate("2026-03-12T10:30:00Z"),
  "updated_at": ISODate("2026-03-12T10:30:00Z"),
  "last_login": ISODate("2026-03-12T10:32:00Z")
}
```

**Indexes:**
- `email` (unique)
- `username` (unique)
- `created_at`

**2. predictions** (optional - for history)
```javascript
{
  "_id": ObjectId("..."),
  "user_id": "65f8...",
  "disease": "sepsis",
  "patient_data": { ... },
  "prediction": { ... },
  "created_at": ISODate("2026-03-12T10:35:00Z")
}
```

**Indexes:**
- `user_id`
- `created_at`
- `(user_id, created_at)` compound

---

## 🔒 Security Features

### Password Security
- ✅ **Bcrypt hashing** with automatic salt generation
- ✅ **Password strength validation** (uppercase, lowercase, digit, min 8 chars)
- ✅ **No plain text storage** - only bcrypt hashes stored

### JWT Tokens
- ✅ **HS256 algorithm** (HMAC with SHA-256)
- ✅ **Configurable expiration** (default 24 hours)
- ✅ **Signed with SECRET_KEY** from environment

### Input Validation
- ✅ **Email validation** with email-validator
- ✅ **Username sanitization** (alphanumeric + _ or -)
- ✅ **Pydantic models** for type safety

### Database Security
- ✅ **Unique constraints** on email and username
- ✅ **Indexed queries** for performance
- ✅ **Connection pooling** with Motor

---

## 🧪 Testing Authentication

### Test Signup
```bash
curl -X POST "http://localhost:8000/api/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "TestPass123",
    "full_name": "Test User"
  }'
```

### Test Login
```bash
curl -X POST "http://localhost:8000/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "TestPass123"
  }'
```

### Test Protected Route
```bash
# Replace <TOKEN> with actual token from login response
curl -X GET "http://localhost:8000/api/auth/me" \
  -H "Authorization: Bearer <TOKEN>"
```

---

## 📊 Monitoring MongoDB

### View Logs
Check application logs for MongoDB connection status:
```
INFO:__main__:🚀 Application startup...
INFO:backend.database:Connecting to MongoDB Atlas...
INFO:backend.database:✅ Successfully connected to MongoDB database: medical_ai_db
INFO:backend.database:✅ Database indexes created
```

### MongoDB Atlas Dashboard
1. Login to https://cloud.mongodb.com
2. Navigate to your cluster
3. Click "Browse Collections" to view data
4. Check "Metrics" for connection stats

---

## 🐛 Troubleshooting

### Error: "Failed to connect to MongoDB"

**Check:**
- MongoDB URL is correct in `.env`
- Network access is configured in MongoDB Atlas (allow your IP)
- Database user credentials are correct

### Error: "Authentication disabled"

**Cause:** Import error when loading auth modules

**Solution:**
```bash
pip install motor pymongo passlib[bcrypt] python-jose[cryptography]
```

### Error: "Could not validate credentials"

**Cause:** Invalid or expired JWT token

**Solution:**
- Check token is being sent in `Authorization: Bearer <token>` header
- Token may have expired (check `ACCESS_TOKEN_EXPIRE_MINUTES`)
- Login again to get new token

### Error: "Email already registered"

**Cause:** User with that email already exists

**Solution:**
- Use different email
- Or login with existing account

---

## 🎯 Next Steps

### Optional Enhancements

1. **Password Reset:**
   - Add forgot password endpoint
   - Email verification with tokens

2. **Email Verification:**
   - Send verification email on signup
   - Verify email before activation

3. **Role-Based Access Control:**
   - Add admin role
   - Restrict certain endpoints

4. **Session Management:**
   - Store sessions in MongoDB
   - Invalidate tokens on logout

5. **Rate Limiting:**
   - Limit login attempts
   - Prevent brute force attacks

6. **Two-Factor Authentication:**
   - TOTP (Google Authenticator)
   - SMS verification

---

## 📚 Dependencies Added

```txt
# Authentication & Security
passlib[bcrypt]==1.7.4       # Password hashing
python-jose[cryptography]==3.3.0   # JWT tokens
bcrypt==4.1.2                 # Bcrypt algorithm
email-validator==2.1.0        # Email validation

# MongoDB
motor==3.3.2                  # Async MongoDB driver
pymongo==4.6.1               # MongoDB Python driver
```

---

## ✅ Summary

You now have a complete authentication system with:

- ✅ **MongoDB Atlas** database connection
- ✅ **User signup** with password validation
- ✅ **User login** with JWT tokens
- ✅ **Protected routes** with bearer token authentication
- ✅ **Beautiful UI** for login/signup
- ✅ **Secure password storage** with bcrypt
- ✅ **Email validation** and username constraints

**To use:**
1. Install dependencies: `pip install -r backend/requirements.txt`
2. Update `.env` with MongoDB URL and SECRET_KEY
3. Start app: `python start_web_application.py`
4. Visit: http://localhost:8000/signup

**Your Database:** `mongodb+srv://harikadanduprolu740_db_user:Vdo8kp6VVaEtCd3e@cluster0.2jis5r9.mongodb.net/?appName=Cluster0`

🎉 **Ready to use!**
