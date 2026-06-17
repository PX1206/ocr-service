#!/bin/bash
set -e

CONTAINER_NAME=ocr-server
IMAGE_NAME=ocr-server:v3
APP_DIR=/data/ai/apps/ocr-service
LOG_DIR="${APP_DIR}/logs"

echo "===== $(date '+%F %T') 开始发布 ====="

mkdir -p "${LOG_DIR}"

echo "1. 停止旧容器..."
docker stop "${CONTAINER_NAME}" 2>/dev/null || true

echo "2. 删除旧容器..."
docker rm "${CONTAINER_NAME}" 2>/dev/null || true

echo "3. 启动新容器..."
docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart=always \
  --memory=6g \
  -v /data/ai:/data/ai \
  -p 127.0.0.1:8088:8088 \
  -e OCR_LOG_DIR="${LOG_DIR}" \
  "${IMAGE_NAME}" \
  bash "${APP_DIR}/start.sh"

echo "4. 等待服务启动..."
for i in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8088/health >/dev/null 2>&1; then
    echo "5. 健康检查通过："
    curl -s http://127.0.0.1:8088/health
    echo
    echo "日志目录：${LOG_DIR}"
    echo "  - ocr-service.log  服务/错误日志"
    echo "  - access.log       访问日志"
    echo "===== 发布完成 ====="
    exit 0
  fi
  sleep 2
done

echo "===== 发布失败 ====="
docker logs --tail 80 "${CONTAINER_NAME}" || true
exit 1
