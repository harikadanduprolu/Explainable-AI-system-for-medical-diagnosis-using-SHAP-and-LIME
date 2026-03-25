# Multi-stage Docker build for Explainable Medical AI System
# Optimized for production deployment

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt backend/requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r backend/requirements.txt && \
    pip install --no-cache-dir gunicorn uvicorn[standard]

# Stage 2: Frontend build
FROM node:20-alpine as frontend-builder

WORKDIR /frontend
ENV NODE_ENV=production \
    VITE_OUTPUT_DIR=/frontend/dist

COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund

COPY frontend/ .
RUN npm run build

# Stage 3: Runtime
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_HOME=/app \
    USER=appuser \
    UID=1000

# Create non-root user
RUN groupadd -g $UID $USER && \
    useradd -m -u $UID -g $USER $USER

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR $APP_HOME

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=$USER:$USER backend/ ./backend/
COPY --chown=$USER:$USER *.py ./
COPY --chown=$USER:$USER requirements.txt ./
COPY --from=frontend-builder --chown=$USER:$USER /frontend/dist/ ./backend/static/

# Create necessary directories
RUN mkdir -p logs audit_logs trained_models && \
    chown -R $USER:$USER logs audit_logs trained_models

# Copy trained models (if included in image)
COPY --chown=$USER:$USER trained_models/*.pkl ./trained_models/ 2>/dev/null || true

# Switch to non-root user
USER $USER

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application with Gunicorn
CMD ["gunicorn", "backend.main:app", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--access-logfile", "logs/access.log", \
     "--error-logfile", "logs/error.log", \
     "--log-level", "info"]
