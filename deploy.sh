#!/bin/bash
# Production deployment script for Explainable Medical AI System

set -e  # Exit on error

echo "========================================================================"
echo "🚀 Deploying Explainable Medical AI System"
echo "========================================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker and Docker Compose are installed${NC}"

# Check for .env file
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Copying from .env.example...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}⚠️  Please edit .env file with your configuration before deploying!${NC}"
    echo -e "${YELLOW}   Especially update: SECRET_KEY, CORS_ORIGINS, DATABASE_URL${NC}"
    read -p "Press Enter to continue or Ctrl+C to abort..."
fi

# Check for trained models
echo ""
echo "📦 Checking for trained models..."
MODEL_COUNT=$(find trained_models -name "*.pkl" 2>/dev/null | wc -l || echo "0")

if [ "$MODEL_COUNT" -lt 9 ]; then
    echo -e "${YELLOW}⚠️  Warning: Only $MODEL_COUNT models found. Expected at least 9.${NC}"
    echo -e "${YELLOW}   Run: python train_advanced_models.py${NC}"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✅ Found $MODEL_COUNT trained models${NC}"
fi

# Build Docker images
echo ""
echo "🏗️  Building Docker images..."
docker-compose build

# Start services
echo ""
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo ""
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check health
echo ""
echo "🏥 Checking service health..."
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health || echo "000")

if [ "$HEALTH_STATUS" = "200" ]; then
    echo -e "${GREEN}✅ Application is healthy!${NC}"
else
    echo -e "${RED}❌ Application health check failed (HTTP $HEALTH_STATUS)${NC}"
    echo "Checking logs..."
    docker-compose logs --tail=50 web
    exit 1
fi

# Display service information
echo ""
echo "========================================================================"
echo -e "${GREEN}✅ Deployment Complete!${NC}"
echo "========================================================================"
echo ""
echo "📊 Services Running:"
echo "  🌐 Web Application: http://localhost:8000/app"
echo "  📚 API Documentation: http://localhost:8000/docs"
echo "  🔍 ReDoc: http://localhost:8000/redoc"
echo "  ❤️  Health Check: http://localhost:8000/health"
echo ""
echo "🛠️  Management Commands:"
echo "  View logs:        docker-compose logs -f web"
echo "  Stop services:    docker-compose down"
echo "  Restart:          docker-compose restart"
echo "  View status:      docker-compose ps"
echo ""
echo "📁 Important Directories:"
echo "  Models:    ./trained_models"
echo "  Logs:      ./logs"
echo "  Audit:     ./audit_logs"
echo ""
echo "⚠️  Security Reminders:"
echo "  1. Update SECRET_KEY in .env"
echo "  2. Configure CORS_ORIGINS for your domain"
echo "  3. Set up SSL/HTTPS for production"
echo "  4. Enable authentication if needed"
echo "  5. Review security settings in backend/main.py"
echo ""
echo "========================================================================"
