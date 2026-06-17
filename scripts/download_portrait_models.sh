#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${PORTRAIT_MODEL_ROOT:-/app/models}"
DET_URL="${PORTRAIT_DET_MODEL_URL:-https://bj.bcebos.com/v1/paddledet/models/keypoint/picodet_s_320_pedestrian.tar}"
KPT_URL="${PORTRAIT_KPT_MODEL_URL:-https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_256x192.tar}"

mkdir -p "${MODEL_ROOT}"

download_and_extract() {
  local url="$1"
  local dest_name="$2"
  local dest_dir="${MODEL_ROOT}/${dest_name}"

  if [[ -f "${dest_dir}/infer_cfg.yml" ]]; then
    echo "model already exists: ${dest_dir}"
    return 0
  fi

  local tmp_tar
  tmp_tar="$(mktemp /tmp/portrait-model.XXXXXX.tar)"
  echo "downloading ${url}"
  curl -fsSL "${url}" -o "${tmp_tar}"
  mkdir -p "${dest_dir}"
  tar -xf "${tmp_tar}" -C "${MODEL_ROOT}"
  rm -f "${tmp_tar}"

  if [[ ! -f "${dest_dir}/infer_cfg.yml" ]]; then
    echo "expected infer_cfg.yml missing under ${dest_dir}" >&2
    ls -la "${MODEL_ROOT}" >&2 || true
    exit 1
  fi
  echo "ready: ${dest_dir}"
}

download_and_extract "${DET_URL}" "picodet_s_320_pedestrian"
download_and_extract "${KPT_URL}" "tinypose_256x192"

echo "portrait models ready under ${MODEL_ROOT}"
