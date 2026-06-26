# MindGraph++ Celery Worker
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY backend/ ./backend/
COPY configs/ ./configs/

CMD ["celery", "-A", "app.workers.celery_app", "worker", "--loglevel=INFO", "--concurrency=2"]
