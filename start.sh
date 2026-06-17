#!/bin/bash
set -e

APP_DIR=/data/ai/apps/ocr-service
LOG_DIR="${OCR_LOG_DIR:-${APP_DIR}/logs}"

mkdir -p "${LOG_DIR}"
export PYTHONPATH="${APP_DIR}"
export OCR_LOG_DIR="${LOG_DIR}"
export OCR_PORT="${OCR_PORT:-8088}"

# 头肩照校验（PicoDet + TinyPose）：模型与 PaddleDetection 放在 /data/ai 下，与代码一同挂载
export PADDLE_DET_DEPLOY_DIR="${PADDLE_DET_DEPLOY_DIR:-/data/ai/apps/PaddleDetection/deploy/python}"
export PORTRAIT_DET_MODEL_DIR="${PORTRAIT_DET_MODEL_DIR:-${APP_DIR}/models/picodet_s_320_pedestrian}"
export PORTRAIT_KPT_MODEL_DIR="${PORTRAIT_KPT_MODEL_DIR:-${APP_DIR}/models/tinypose_256x192}"
export PORTRAIT_ENABLED="${PORTRAIT_ENABLED:-true}"
export PORTRAIT_DEVICE="${PORTRAIT_DEVICE:-CPU}"
export PORTRAIT_WARMUP="${PORTRAIT_WARMUP:-false}"

# 降低 Paddle 底层日志噪音
export GLOG_minloglevel=2

cd "${APP_DIR}"

echo "Python: $(python3 --version 2>&1)"
echo "Preflight import check..."
python3 - <<'PY'
import sys
import traceback

try:
    import app.main  # noqa: F401
    print("import app.main OK")
except Exception:
    traceback.print_exc()
    sys.exit(1)
PY

exec python3 "${APP_DIR}/run_server.py"
