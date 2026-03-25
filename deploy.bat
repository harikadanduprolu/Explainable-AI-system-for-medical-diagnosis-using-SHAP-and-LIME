@echo off
REM Production deployment script for Windows
REM Explainable Medical AI System

echo ========================================================================
echo 🚀 Deploying Explainable Medical AI System
echo ========================================================================
echo.

REM Check Docker
where docker >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

where docker-compose >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Docker Compose is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

echo ✅ Docker and Docker Compose are installed
echo.

REM Check for .env file
if not exist .env (
    echo ⚠️  .env file not found. Copying from .env.example...
    copy .env.example .env
    echo ⚠️  Please edit .env file with your configuration!
    echo    Update: SECRET_KEY, CORS_ORIGINS, DATABASE_URL
    pause
)

REM Check for trained models
echo 📦 Checking for trained models...
set MODEL_COUNT=0
for %%f in (trained_models\*.pkl) do set /a MODEL_COUNT+=1

if %MODEL_COUNT% LSS 9 (
    echo ⚠️  Warning: Only %MODEL_COUNT% models found. Expected at least 9.
    echo    Run: python train_advanced_models.py
    set /p CONTINUE=Continue anyway? (y/N): 
    if /i not "%CONTINUE%"=="y" exit /b 1
) else (
    echo ✅ Found %MODEL_COUNT% trained models
)

REM Build Docker images
echo.
echo 🏗️  Building Docker images...
docker-compose build

REM Start services
echo.
echo 🚀 Starting services...
docker-compose up -d

REM Wait for services
echo.
echo ⏳ Waiting for services to be ready...
timeout /t 10 /nobreak >nul

REM Check health
echo.
echo 🏥 Checking service health...
curl -s http://localhost:8000/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Application is healthy!
) else (
    echo ❌ Application health check failed
    echo Checking logs...
    docker-compose logs --tail=50 web
    exit /b 1
)

REM Display information
echo.
echo ========================================================================
echo ✅ Deployment Complete!
echo ========================================================================
echo.
echo 📊 Services Running:
echo   🌐 Web Application: http://localhost:8000/app
echo   📚 API Documentation: http://localhost:8000/docs
echo   🔍 ReDoc: http://localhost:8000/redoc
echo   ❤️  Health Check: http://localhost:8000/health
echo.
echo 🛠️  Management Commands:
echo   View logs:        docker-compose logs -f web
echo   Stop services:    docker-compose down
echo   Restart:          docker-compose restart
echo   View status:      docker-compose ps
echo.
echo ⚠️  Security Reminders:
echo   1. Update SECRET_KEY in .env
echo   2. Configure CORS_ORIGINS for your domain
echo   3. Set up SSL/HTTPS for production
echo   4. Enable authentication if needed
echo.
echo ========================================================================
pause
