# Methanex EPSSC dashboard — deploys to Google Cloud Run
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# System deps (only what scikit-learn / numpy need at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Cloud Run injects $PORT (default 8080). Bind 0.0.0.0 so the container is reachable.
EXPOSE 8080
CMD exec gunicorn app:server \
    --bind 0.0.0.0:${PORT} \
    --workers 1 \
    --threads 8 \
    --timeout 120 \
    --graceful-timeout 30
