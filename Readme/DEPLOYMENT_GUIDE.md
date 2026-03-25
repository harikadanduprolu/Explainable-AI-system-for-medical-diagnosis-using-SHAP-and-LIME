# 🌐 Production Deployment Guide
## Explainable Medical AI System

**Last Updated:** March 12, 2026  
**Version:** 1.0.0

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Docker Deployment](#docker-deployment)
4. [Manual Deployment](#manual-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Security Configuration](#security-configuration)
7. [Monitoring & Maintenance](#monitoring--maintenance)
8. [Troubleshooting](#troubleshooting)

---

## 📦 Prerequisites

### Required Software
- **Docker** 20.10+ and **Docker Compose** 2.0+
- **Python** 3.8+ (for manual deployment)
- **Nginx** (optional, for reverse proxy)
- **SSL Certificate** (for HTTPS)

### Required Files
- ✅ Trained models in `trained_models/` (23 .pkl files)
- ✅ `.env` configuration file
- ✅ Valid SSL certificates (for production)

---

## 🚀 Quick Start

### Option 1: Docker Deployment (Recommended)

**Linux/Mac:**
```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit configuration
nano .env  # Update SECRET_KEY, CORS_ORIGINS, etc.

# 3. Deploy
chmod +x deploy.sh
./deploy.sh
```

**Windows:**
```bash
# 1. Copy environment template
copy .env.example .env

# 2. Edit .env file with your settings

# 3. Deploy
deploy.bat
```

**Access:**
- Web App: http://localhost:8000/app
- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

---

## 🐳 Docker Deployment

### File Structure

```
.
├── Dockerfile              # Application container
├── docker-compose.yml      # Multi-service orchestration
├── nginx.conf             # Reverse proxy config
├── .env                   # Environment variables
└── deploy.sh / deploy.bat # Deployment scripts
```

### Step-by-Step Deployment

#### 1. Configure Environment

Edit `.env` file:

```bash
# Critical settings
SECRET_KEY=your-secret-key-here-generate-with-python-secrets
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
APP_ENV=production
DEBUG=false

# Database (optional)
DATABASE_URL=postgresql://user:pass@postgres:5432/medical_ai

# Security
ENABLE_AUDIT_LOGGING=true
RATE_LIMIT_ENABLED=true
```

#### 2. Build Images

```bash
docker-compose build
```

#### 3. Start Services

```bash
# Start in background
docker-compose up -d

# View logs
docker-compose logs -f web
```

#### 4. Verify Deployment

```bash
# Check health
curl http://localhost:8000/health

# Check service status
docker-compose ps
```

### Docker Commands Reference

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart services
docker-compose restart

# View logs
docker-compose logs -f [service]

# Scale workers
docker-compose up -d --scale web=4

# Execute command in container
docker-compose exec web python verify_trained_models.py

# Update and restart
docker-compose pull
docker-compose up -d
```

---

## 🔧 Manual Deployment

### Using Gunicorn (Linux/Mac)

```bash
# 1. Install production dependencies
pip install -r backend/requirements.txt
pip install gunicorn

# 2. Set environment variables
export APP_ENV=production
export SECRET_KEY=your-secret-key

# 3. Start with Gunicorn
gunicorn backend.main:app \
    --bind 0.0.0.0:8000 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --access-logfile logs/access.log \
    --error-logfile logs/error.log
```

### Using systemd (Linux)

Create `/etc/systemd/system/medical-ai.service`:

```ini
[Unit]
Description=Explainable Medical AI System
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/opt/medical-ai
Environment="PATH=/opt/medical-ai/.venv/bin"
EnvironmentFile=/opt/medical-ai/.env
ExecStart=/opt/medical-ai/.venv/bin/gunicorn \
    backend.main:app \
    --bind 0.0.0.0:8000 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker

Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable medical-ai
sudo systemctl start medical-ai
sudo systemctl status medical-ai
```

---

## ☁️ Cloud Deployment

### AWS Deployment

#### Using EC2

```bash
# 1. Launch EC2 instance (Ubuntu 22.04, t3.medium)
# 2. Install Docker
sudo apt update
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker ubuntu

# 3. Clone repository
git clone <your-repo>
cd medical-ai

# 4. Deploy
./deploy.sh
```

#### Using ECS (Elastic Container Service)

```bash
# 1. Build and push to ECR
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

docker build -t medical-ai .
docker tag medical-ai:latest <account>.dkr.ecr.us-east-1.amazonaws.com/medical-ai:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/medical-ai:latest

# 2. Create ECS task definition and service
# Use AWS Console or CLI
```

### Google Cloud Platform

```bash
# 1. Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT-ID/medical-ai

# 2. Deploy to Cloud Run
gcloud run deploy medical-ai \
    --image gcr.io/PROJECT-ID/medical-ai \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

### Azure

```bash
# 1. Create container registry
az acr create --name medicalairegistry --resource-group mygroup --sku Basic

# 2. Build and push
az acr build --registry medicalairegistry --image medical-ai:latest .

# 3. Deploy to Container Instances
az container create \
    --resource-group mygroup \
    --name medical-ai \
    --image medicalairegistry.azurecr.io/medical-ai:latest \
    --ports 8000
```

### Heroku

```bash
# 1. Create app
heroku create medical-ai-app

# 2. Add buildpack
heroku buildpacks:set heroku/python

# 3. Deploy
git push heroku main

# 4. Scale
heroku ps:scale web=2
```

---

## 🔐 Security Configuration

### 1. Generate Secret Key

```python
# Generate secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 2. SSL/HTTPS Setup

#### Using Let's Encrypt (Free)

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal
sudo certbot renew --dry-run
```

#### Update nginx.conf

Uncomment HTTPS section and update:

```nginx
ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
```

### 3. CORS Configuration

Update `.env`:

```bash
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

### 4. Rate Limiting

Enabled by default in `nginx.conf`:
- API: 10 requests/second
- General: 100 requests/second

Adjust in nginx.conf:

```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=20r/s;
```

### 5. Firewall Rules

```bash
# Ubuntu/Debian
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

---

## 📊 Monitoring & Maintenance

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Expected response
{"status":"healthy","version":"1.0.0","models_loaded":9}
```

### Log Management

```bash
# View application logs
docker-compose logs -f web

# View nginx logs
docker-compose logs -f nginx

# Tail access logs
tail -f logs/access.log

# Search error logs
grep ERROR logs/error.log
```

### Performance Monitoring

#### Install monitoring tools

```yaml
# Add to docker-compose.yml
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - " 3000:3000"
```

### Backup Strategy

```bash
# Backup trained models
tar -czf models-backup-$(date +%Y%m%d).tar.gz trained_models/

# Backup audit logs
tar -czf audit-backup-$(date +%Y%m%d).tar.gz audit_logs/

# Backup database (if used)
docker-compose exec postgres pg_dump -U user medical_ai > backup.sql
```

### Updates & Maintenance

```bash
# Update code
git pull origin main

# Rebuild and restart
docker-compose build
docker-compose up -d

# Zero-downtime update
docker-compose up -d --no-deps --build web
```

---

## 🐛 Troubleshooting

### Application Won't Start

```bash
# Check logs
docker-compose logs web

# Common issues:
# 1. Models not loaded
docker-compose exec web ls -la trained_models/

# 2. Port already in use
lsof -i :8000
# Kill process: kill -9 <PID>

# 3. Environment variables
docker-compose exec web env | grep APP_
```

### High Memory Usage

```bash
# Check container stats
docker stats

# Reduce workers in .env
WORKERS=2

# Restart with new config
docker-compose restart web
```

### Slow Predictions

```bash
# Enable model caching
ENABLE_MODEL_CACHING=true

# Increase cache size
MODEL_CACHE_SIZE=200

# Check Redis
docker-compose exec redis redis-cli ping
```

### SSL/HTTPS Issues

```bash
# Test SSL certificate
openssl s_client -connect yourdomain.com:443

# Renew Let's Encrypt
sudo certbot renew

# Check nginx config
docker-compose exec nginx nginx -t
```

---

## 📈 Performance Tuning

### Optimal Worker Count

```bash
# Formula: (2 × CPUs) + 1
# For 4 CPUs: WORKERS=9
```

### Database Connection Pooling

```bash
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40
```

### Redis Caching

```bash
REDIS_URL=redis://redis:6379/0
ENABLE_MODEL_CACHING=true
```

---

## ✅ Production Checklist

- [ ] `.env` file configured with production settings
- [ ] `SECRET_KEY` generated and set
- [ ] `CORS_ORIGINS` set to your domain(s)
- [ ] SSL/HTTPS certificate installed
- [ ] Firewall rules configured
- [ ] Rate limiting enabled
- [ ] Audit logging enabled
- [ ] Models trained and loaded (9/9)
- [ ] Health check endpoint responding
- [ ] Backup strategy in place
- [ ] Monitoring configured
- [ ] Domain DNS configured
- [ ] Load testing completed

---

## 📞 Support

**Issues:** Create an issue in the repository  
**Docs:** See `README.md` and `README_WEB_APP.md`  
**Health:** http://yourdomain.com/health  
**API Docs:** http://yourdomain.com/docs

---

**Deployment Status:** ✅ Production Ready  
**Last Tested:** March 12, 2026
