#!/usr/bin/env bash
# 现网首次启用头肩照校验时，在服务器上执行一次（需 curl、git、bash）。
set -euo pipefail

BASE_DIR="${BASE_DIR:-/data/ai/apps}"
OCR_DIR="${OCR_DIR:-${BASE_DIR}/ocr-service}"
PD_DIR="${PD_DIR:-${BASE_DIR}/PaddleDetection}"
MODEL_ROOT="${MODEL_ROOT:-${OCR_DIR}/models}"

echo "===== 头肩照校验环境初始化 ====="
echo "OCR_DIR=${OCR_DIR}"
echo "PD_DIR=${PD_DIR}"
echo "MODEL_ROOT=${MODEL_ROOT}"

if [[ ! -d "${PD_DIR}/deploy/python" ]]; then
  echo "1. 克隆 PaddleDetection release/2.6 ..."
  git clone --depth 1 --branch release/2.6 \
    https://github.com/PaddlePaddle/PaddleDetection.git "${PD_DIR}"
else
  echo "1. PaddleDetection 已存在，跳过"
fi

echo "2. 下载 PicoDet + TinyPose 推理模型 ..."
export PORTRAIT_MODEL_ROOT="${MODEL_ROOT}"
bash "${OCR_DIR}/scripts/download_portrait_models.sh"

echo "3. 校验目录 ..."
test -f "${PD_DIR}/deploy/python/infer.py"
test -f "${MODEL_ROOT}/picodet_s_320_pedestrian/infer_cfg.yml"
test -f "${MODEL_ROOT}/tinypose_256x192/infer_cfg.yml"

echo "===== 初始化完成 ====="
echo "接下来请："
echo "  1) 容器内: pip install PyYAML shapely scipy（勿装 paddledet）"
echo "  2) 上传最新 ocr-service 代码到 ${OCR_DIR}"
echo "  3) 执行 deploy.sh 重启容器"
echo "  4) 重启 Java system（nh.ocr.enabled=true）"
