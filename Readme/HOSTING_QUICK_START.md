# 🚀 Production Hosting - Quick Start

**Last Updated:** March 12, 2026  
**Status:** ✅ Production Ready

---

## ⚡ Quick Deploy (5 Minutes)

### Prerequisites
✅ Docker & Docker Compose installed  
✅ 9 trained models in `trained_models/`  
✅ Port 8000 available

### Deploy Now

**Linux/Mac:**
```bash
# 1. Setup environment
cp .env.example .env

# 2. Edit .env (REQUIRED - set SECRET_KEY)
nano .env

# 3. Deploy
chmod +x deploy.sh
./deploy.sh
```

**Windows:**
```bash
# 1. Setup environment
copy .env.example .env

# 2. Edit .env file

# 3. Deploy
deploy.bat
```

**Access:**
- 🌐 App: http://localhost:8000/app
- 📚 Docs: http://localhost:8000/docs
- ❤️ Health: http://localhost:8000/health

---

## 📦 What Gets Deployed

```
Services:
  ├── web (FastAPI)     - Port 8000
  ├── nginx             - Port 80/443
  └── redis             - Port 6379

Volumes:
  ├── trained_models/   - ML models (read-only)
  ├── logs/             - Application logs
  └── audit_logs/       - Audit trails
```

---

## 🔐 Critical Security Steps

### 1. Generate Secret Key
```python
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Add to `.env`:
```bash
SECRET_KEY=<generated-key-here>
```

### 2. Configure CORS
```bash
# .env
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

### 3. Disable Debug Mode
```bash
# .env
APP_ENV=production
DEBUG=false
```

---

## 🌐 Domain & SSL Setup

### Point Domain to Server
```bash
# DNS A Record
yourdomain.com  →  Your Server IP
```

### Install SSL Certificate (Let's Encrypt)
```bash
# Install certbot
sudo apt install certbot

# Get certificate
sudo certbot certonly --standalone -d yourdomain.com

# Certificates will be in:
# /etc/letsencrypt/live/yourdomain.com/
```

### Update nginx.conf
Uncomment HTTPS section and update paths:
```nginx
ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
```

Restart:
```bash
docker-compose restart nginx
```

---

## 📊 Verify Deployment

### Health Check
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "environment": "production",
  "version": "2.0.0",
  "models_loaded": 9
}
```

### Service Status
```bash
docker-compose ps
```

All services should show "Up" and "healthy".

### View Logs
```bash
# Application logs
docker-compose logs -f web

# All services
docker-compose logs -f
```

---

## 🛠️ Common Commands

```bash
# Stop services
docker-compose down

# Restart specific service
docker-compose restart web

# View running containers
docker-compose ps

# Check resource usage
docker stats

# Execute command in container
docker-compose exec web python verify_trained_models.py

# View logs (last 100 lines)
docker-compose logs --tail=100 web
```

---

## 🔥 Troubleshooting

### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>
```

### Models Not Loading
```bash
# Check models directory
docker-compose exec web ls -la trained_models/

# Should show 23 .pkl files
```

### CORS Errors
Update `.env`:
```bash
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

Restart:
```bash
docker-compose restart web
```

### Out of Memory
Reduce workers in `.env`:
```bash
WORKERS=2
```

---

## 📈 Performance Optimization

### Recommended Settings

**For 2 CPU Cores:**
```bash
WORKERS=4
```

**For 4 CPU Cores:**
```bash
WORKERS=9
```

**For 8 CPU Cores:**
```bash
WORKERS=17
```

Formula: `(2 × CPU Cores) + 1`

---

## 🔄 Updates & Maintenance

### Update Application
```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose build
docker-compose up -d
```

### Backup Data
```bash
# Backup models
tar -czf models-backup-$(date +%Y%m%d).tar.gz trained_models/

# Backup audit logs
tar -czf audit-backup-$(date +%Y%m%d).tar.gz audit_logs/
```

### Monitor Logs
```bash
# Real-time logs
docker-compose logs -f

# Last 50 lines
docker-compose logs --tail=50 web

# Since timestamp
docker-compose logs --since 2024-03-12T10:00:00
```

---

## 💾 Database Setup (Optional)

Uncomment PostgreSQL in `docker-compose.yml`:

```yaml
postgres:
  image: postgres:15-alpine
  environment:
    POSTGRES_DB: medical_ai
    POSTGRES_USER: medicalai
    POSTGRES_PASSWORD: changeme
```

Update `.env`:
```bash
DATABASE_URL=postgresql://medicalai:changeme@postgres:5432/medical_ai
```

Restart:
```bash
docker-compose up -d
```

---

## 📞 Support

**Quick Help:**
- Health: http://localhost:8000/health
- API Docs: http://localhost:8000/docs
- Logs: `docker-compose logs -f`

**Full Documentation:**
- See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- See [README_WEB_APP.md](README_WEB_APP.md)

---

## ✅ Production Checklist

Before going live:

- [ ] `.env` configured with production values
- [ ] `SECRET_KEY` generated and set
- [ ] `CORS_ORIGINS` set to your domain
- [ ] `DEBUG=false` in .env
- [ ] SSL certificate installed
- [ ] Domain DNS configured
- [ ] Firewall rules set (ports 80, 443)
- [ ] All 9 models loaded (`/health` endpoint)
- [ ] Health check returns 200 OK
- [ ] Logs directory writable
- [ ] Backup strategy in place
- [ ] Monitoring configured
- [ ] Load testing completed

---

**Deployment Time:** ~5 minutes  
**Production Status:** ✅ Ready  
**Support:** Create issue on GitHub
