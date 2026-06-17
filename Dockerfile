FROM python:3.10-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libgl1 \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PaddleDetection deploy/python for PicoDet + TinyPose inference.
RUN git clone --depth 1 --branch release/2.6 https://github.com/PaddlePaddle/PaddleDetection.git /opt/PaddleDetection

COPY app ./app
COPY run_server.py ./run_server.py
COPY scripts ./scripts

RUN chmod +x /app/scripts/download_portrait_models.sh \
    && /app/scripts/download_portrait_models.sh

# Pre-download PaddleOCR models at build time to avoid cold-start delay.
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)"

ENV OCR_PORT=8088 \
    OCR_LOG_DIR=/data/ai/apps/ocr-service/logs \
    PADDLE_DET_DEPLOY_DIR=/opt/PaddleDetection/deploy/python \
    PORTRAIT_DET_MODEL_DIR=/app/models/picodet_s_320_pedestrian \
    PORTRAIT_KPT_MODEL_DIR=/app/models/tinypose_256x192 \
    PORTRAIT_ENABLED=true \
    PORTRAIT_DEVICE=CPU \
    PORTRAIT_WARMUP=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8088

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${OCR_PORT}/health" || exit 1

CMD ["python", "run_server.py"]
