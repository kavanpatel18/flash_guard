# ─────────────────────────────────────────────────────────────────────────────
# FlashGuard v4 — Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Build:  docker build -t flashguard:latest .
# Run:    docker run --env-file .env -p 5000:5000 flashguard:latest
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Base image ─────────────────────────────────────────────────────────────
# python:3.11-slim keeps the image lean while retaining pip and standard libs.
FROM python:3.11-slim

# ── 2. Environment hygiene ────────────────────────────────────────────────────
# Prevent Python from writing .pyc files and buffer stdout/stderr immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── 3. System dependencies ────────────────────────────────────────────────────
# libgomp1 is required by some TensorFlow builds on slim images.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        curl && \
    rm -rf /var/lib/apt/lists/*

# ── 4. Working directory ──────────────────────────────────────────────────────
WORKDIR /app

# ── 5. Install Python dependencies ────────────────────────────────────────────
# Copy only requirements first so Docker layer-caching works efficiently:
# dependency layers are only rebuilt when requirements.txt changes.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── 6. Copy application code ──────────────────────────────────────────────────
COPY api_server.py .
COPY frontend/ ./frontend/

# ── 7. Copy ML models ─────────────────────────────────────────────────────────
# Models are large but required at runtime.
COPY improved_minute_model.keras .
COPY improved_flash_crash_model.keras .

# ── 8. Copy configuration template ───────────────────────────────────────────
# .env is NOT baked in; pass secrets at runtime via --env-file or -e flags.
COPY .env.example .

# ── 9. Expose port ────────────────────────────────────────────────────────────
EXPOSE 5000

# ── 10. Healthcheck ───────────────────────────────────────────────────────────
# Docker will mark the container unhealthy if /api/health stops responding.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# ── 11. Run with Gunicorn ─────────────────────────────────────────────────────
# 4 workers, 2 threads each — safe for a TF model (models are loaded once per
# worker via the module-level _load cache).
# Timeout set to 120s to allow model cold-start on first inference.
CMD ["gunicorn", \
     "--workers", "4", \
     "--threads", "2", \
     "--bind", "0.0.0.0:5000", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "api_server:app"]
