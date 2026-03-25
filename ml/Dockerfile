FROM python:3.11-slim

WORKDIR /app

# Системные libs для cv2: ultralytics тянет opencv-python (не headless) → нужен libGL;
# libxcb — на случай headless-сборки / части кода OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libxcb-xfixes0 \
    libxcb-shm0 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch с pypi.org для Linux — CUDA-сборка и пакеты nvidia-*; для CPU только индекс PyTorch.
COPY ml/requirements.txt ./
# Версии torch/torchvision совпадают со строками в ml/requirements.txt (их sed вырезает перед вторым pip).
RUN pip install --no-cache-dir  \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu
RUN sed '/^torch==/d;/^torchvision==/d' requirements.txt > /tmp/req-no-torch.txt \
    && pip install --no-cache-dir  -r /tmp/req-no-torch.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["celery", "-A", "backend.app.tasks", "worker", "--loglevel=info"]
