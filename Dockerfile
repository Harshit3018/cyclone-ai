# ==============================================================
# CYCLONE-AI Dockerfile — Multi-stage build
# Stage 1: Build React/Vite frontend
# Stage 2: Python runtime with FastAPI + ML
# ==============================================================

# --- Stage 1: Frontend Build ---
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend

# Install dependencies first (layer caching)
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit

# Copy source and build
COPY frontend/ ./
RUN npm run build


# --- Stage 2: Python Runtime ---
FROM python:3.11-slim AS runtime

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY ml/ ./ml/
COPY configs/ ./configs/
COPY data/sample/ ./data/sample/
COPY models/ ./models/
COPY scripts/ ./scripts/

# Copy built frontend from Stage 1
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Copy env defaults
COPY .env.example ./.env

# Create necessary directories
RUN mkdir -p models/checkpoints models/metadata data/raw data/processed logs

# Environment
ENV APP_ENV=demo \
    APP_PORT=8000 \
    APP_HOST=0.0.0.0 \
    SERVE_FRONTEND=true \
    FRONTEND_DIR=frontend/dist \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Start
CMD ["python", "backend/run.py"]
