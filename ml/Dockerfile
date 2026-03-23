FROM python:3.11-slim

WORKDIR /app

# Install system deps for OpenCV (libxcb for cv2 in headless container)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libxcb1 \
    libxcb-xfixes0 \
    libxcb-shm0 \
    && rm -rf /var/lib/apt/lists/*

# Install ML requirements from PyPI (faster than download.pytorch.org for many regions)
# PyPI torch/torchvision for Linux are CPU-only by default
COPY ml/requirements.txt ./
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["celery", "-A", "backend.app.tasks", "worker", "--loglevel=info"]
