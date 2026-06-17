# 现网 OCR 容器升级指南（CentOS 7 + 手工 v2 环境）

本文档是 [README.md](../README.md) 的补充，面向**已在 CentOS 7 上用 Ubuntu 22.04 容器手工搭建 v2** 的现网环境。

> **不要用仓库根目录 `Dockerfile` 全量 build 替换现网 v2。**

---

## 核心原则

| 包 / 资源 | 用途 | 是否安装 |
|-----------|------|----------|
| `paddleocr` + `paddlepaddle` | 证件 OCR（v2 已有） | ✅ 保持不动 |
| **`paddledet`** | 训练框架 | ❌ **禁止 pip install** |
| PaddleDetection `deploy/python` | 头肩推理脚本 | ✅ 宿主机 git/zip |
| `PyYAML` / `shapely` / `scipy` | 推理脚本依赖 | ✅ 容器内 pip |
| `models/picodet_*` + `tinypose_*` | 推理权重 | ✅ 宿主机挂载 |

`paddledet` 会拖入 visualdl、sklearn、**降级 numpy**、替换 opencv，破坏 v2 OCR。

`pose_engine.py` 只调用 `deploy/python/infer.py`，**不 import ppdet**。

---

## 架构

```
ocr-server 容器（ocr-server:v3）
├── paddleocr / numpy 1.24.4 / opencv-contrib     ← 证件 OCR
├── PyYAML + shapely + scipy                        ← 头肩轻量依赖
├── /data/ai/apps/PaddleDetection/deploy/python     ← 推理脚本
└── /data/ai/apps/ocr-service/models/               ← 推理模型

业务代码：/data/ai/apps/ocr-service/app/（挂载，可随时更新）
```

---

## 升级步骤（v2 已恢复 OCR 正常）

### 步骤 1：容器内安装头肩依赖

```bash
docker exec -it ocr-server bash

pip install "PyYAML>=5.1,<6.1" shapely "scipy>=1.7,<1.12" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

python3 -c "import yaml, shapely, scipy; from paddleocr import PaddleOCR; print('OK')"
```

### 步骤 2：宿主机准备 PaddleDetection

**在线（可能慢/失败）：**

```bash
cd /data/ai/apps/ocr-service
bash scripts/setup_portrait_on_host.sh
```

**离线（推荐）：**

1. 本地下载 [PaddleDetection-release-2.6.zip](https://github.com/PaddlePaddle/PaddleDetection/archive/refs/heads/release/2.6.zip)（约 36MB，正常）
2. 上传解压：

```bash
cd /data/ai/apps
unzip PaddleDetection-release-2.6.zip
mv PaddleDetection-release-2.6 PaddleDetection
ls PaddleDetection/deploy/python/infer.py
```

### 步骤 3：宿主机准备模型

**在线：**

```bash
bash /data/ai/apps/ocr-service/scripts/download_portrait_models.sh
```

**离线下载地址：**

- https://bj.bcebos.com/v1/paddledet/models/keypoint/picodet_s_320_pedestrian.tar
- https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_256x192.tar

解压到 `/data/ai/apps/ocr-service/models/`。

校验：

```bash
ls models/picodet_s_320_pedestrian/infer_cfg.yml
ls models/tinypose_256x192/infer_cfg.yml
```

### 步骤 4：上传代码并发布

```bash
bash /data/ai/apps/ocr-service/deploy.sh
```

### 步骤 5：验证

```bash
curl http://127.0.0.1:8088/health

curl -X POST http://127.0.0.1:8088/ocr/bankcard -F "file=@/data/ai/upload/test/card.png"
curl -X POST http://127.0.0.1:8088/validate/portrait -F "file=@/data/ai/upload/test/photo.jpg"
```

### 步骤 6：固化镜像

```bash
docker commit ocr-server ocr-server:v3
```

---

## 误装 paddledet 后的恢复

```bash
docker exec -it ocr-server bash

pip uninstall -y paddledet visualdl sklearn scikit-learn motmetrics lap pycocotools \
  opencv-python opencv-python-headless 2>/dev/null || true

pip install numpy==1.24.4 opencv-contrib-python==4.6.0.66 \
  --force-reinstall --no-cache-dir \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install "PyYAML>=5.1,<6.1" shapely "scipy>=1.7,<1.12" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

python3 -c "import numpy, cv2; from paddleocr import PaddleOCR; print('OCR OK')"
```

---

## portrait 500：`strides` 参数冲突

日志示例：

```
TypeError: PicoDetPostProcess.__init__() got multiple values for argument 'strides'
```

原因：PaddleDetection `deploy/python` 内 `infer.py` 与 `picodet_postprocess.py` 版本签名不一致。

**解决：** 上传最新 `app/common/pose_engine.py`（内含自动 patch），重启容器。

---

## 日常运维

| 操作 | 命令 |
|------|------|
| 更新代码 | 上传 `app/` → `bash deploy.sh` |
| 看日志 | `tail -f logs/ocr-service.log` |
| 命令行测 portrait | `python3 -m app.parsers.portrait_validator photo.jpg` |
| 新增 pip 包 | 容器内 pip install → `docker commit` |

---

## 灾难恢复

1. 按 v2 文档重建 Ubuntu 22.04 容器 + PaddleOCR
2. `pip install PyYAML shapely scipy`（见 `requirements-portrait-inference.txt`）
3. 准备 PaddleDetection + 模型（可离线上传）
4. 上传 ocr-service 代码
5. `docker commit ocr-server ocr-server:v3`
6. `bash deploy.sh`

---

## 可选方案对比

| 方案 | 说明 |
|------|------|
| **A. deploy/python（当前）** | 单容器，不装 paddledet，推荐 |
| B. 双容器 | OCR / portrait 分离，运维更重 |
| C. 双 venv + subprocess | 进程隔离 |
| D. ONNX | 依赖最干净，前期工作多 |

---

## 相关脚本

| 脚本 | 用途 |
|------|------|
| `deploy.sh` | 停止旧容器、启动新容器、健康检查 |
| `start.sh` | 容器内设置环境变量并启动 FastAPI |
| `build-v3.sh` | 可选，基于 v2 镜像增量 pip（`Dockerfile.v3`） |
