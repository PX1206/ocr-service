#!/usr/bin/env bash
# 在已有 ocr-server:v2 的机器上执行，生成 ocr-server:v3
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if ! docker image inspect ocr-server:v2 >/dev/null 2>&1; then
  echo "错误：未找到 ocr-server:v2，请先确认 docker images" >&2
  exit 1
fi

echo "===== 基于 ocr-server:v2 构建 v3（仅增量依赖）====="
docker build -f Dockerfile.v3 -t ocr-server:v3 .

echo "===== 构建完成 ====="
docker images ocr-server:v2 ocr-server:v3
echo
echo "下一步："
echo "  1) bash scripts/setup_portrait_on_host.sh   # 首次：模型 + PaddleDetection"
echo "  2) 上传最新 app/ scripts/ start.sh 等到 /data/ai/apps/ocr-service"
echo "  3) bash deploy.sh"
