# NH OCR Service

独立 Python 服务，提供：

- **身份证 OCR**（正/反面字段提取）
- **银行卡 OCR**（卡号、银行名等）
- **头肩照构图校验**（人脸 + 双肩，PicoDet + PP-TinyPose）

部署为 Docker 容器，由 Java `system` 模块 HTTP 代理，**小程序不直连 Python**。

---

## 架构概览

```
CentOS 7 宿主机
├── Docker 容器 ocr-server（Ubuntu 22.04，镜像 ocr-server:v3）
│     ├── Python 3.10 + PaddleOCR（证件 OCR，v2 环境）
│     ├── PyYAML / shapely / scipy（头肩推理轻量依赖）
│     └── start.sh → run_server.py（FastAPI :8088）
│
├── 挂载 /data/ai
│     ├── apps/ocr-service/          ← 业务代码、模型、日志
│     └── apps/PaddleDetection/      ← 推理脚本 deploy/python（非 pip）
│
└── Java system :3168
      └── /mini/ocr/*  →  http://127.0.0.1:8088
```

| 能力 | 技术栈 | 依赖来源 |
|------|--------|----------|
| 证件 OCR | PaddleOCR 2.7.3 + PaddlePaddle 2.6.2 | 容器 pip（v2 手工搭建） |
| 头肩校验 | PicoDet + PP-TinyPose | PaddleDetection `deploy/python` + 宿主机模型 |
| **不要装** | `paddledet` pip 包 | 会污染 numpy/opencv，破坏 OCR |

---

## 环境版本（现网）

| 组件 | 版本 |
|------|------|
| 宿主机 | CentOS 7 |
| 容器 OS | Ubuntu 22.04 |
| Python | 3.10 |
| PaddleOCR | 2.7.3 |
| PaddlePaddle | 2.6.2 |
| numpy | **1.24.4**（勿被 paddledet 降级） |
| OpenCV | **opencv-contrib-python 4.6.0.66** |
| 容器内存 | 建议 **≥6GB**（`deploy.sh` 已设 `--memory=6g`） |

---

## 服务器目录结构

```
/data/ai/
├── apps/
│   ├── ocr-service/                    # 本仓库代码（挂载进容器）
│   │   ├── app/
│   │   │   ├── main.py                 # FastAPI 入口
│   │   │   ├── common/
│   │   │   │   ├── ocr_engine.py       # PaddleOCR 单例
│   │   │   │   ├── pose_engine.py      # PicoDet + TinyPose
│   │   │   │   └── portrait_rules.py   # 头肩构图规则
│   │   │   └── parsers/
│   │   │       ├── idcard_parser.py
│   │   │       ├── bankcard_parser.py
│   │   │       └── portrait_validator.py
│   │   ├── models/                     # 头肩推理模型（宿主机）
│   │   │   ├── picodet_s_320_pedestrian/
│   │   │   └── tinypose_256x192/
│   │   ├── scripts/
│   │   ├── logs/
│   │   ├── deploy.sh
│   │   ├── start.sh
│   │   └── run_server.py
│   │
│   └── PaddleDetection/                # release/2.6，仅需 deploy/python
│       └── deploy/python/
│           ├── infer.py
│           ├── keypoint_infer.py
│           └── picodet_postprocess.py
│
└── upload/test/                        # 测试图片等
```

---

## 现网部署（CentOS 7，推荐）

> **不要用根目录 `Dockerfile` 全量 build 替换已跑通的 v2 环境。**  
> 详见 [docs/PRODUCTION_UPGRADE.md](docs/PRODUCTION_UPGRADE.md)。

### 首次启用头肩校验

#### 1. 容器内安装轻量依赖（不装 paddledet）

```bash
docker exec -it ocr-server bash

pip install "PyYAML>=5.1,<6.1" shapely "scipy>=1.7,<1.12" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

python3 -c "import yaml, shapely, scipy; from paddleocr import PaddleOCR; print('OK')"
```

#### 2. 宿主机准备 PaddleDetection + 模型

**方式 A：脚本下载（网络好时用）**

```bash
cd /data/ai/apps/ocr-service
chmod +x scripts/*.sh
bash scripts/setup_portrait_on_host.sh   # 克隆 PaddleDetection + 下模型
bash scripts/download_portrait_models.sh # 若上一步模型失败，可单独跑
```

**方式 B：本地下载后上传（网络差时推荐）**

| 资源 | 下载地址 | 上传到服务器 |
|------|----------|--------------|
| PaddleDetection release/2.6 | [GitHub zip](https://github.com/PaddlePaddle/PaddleDetection/archive/refs/heads/release/2.6.zip)（约 36MB） | 解压为 `/data/ai/apps/PaddleDetection` |
| PicoDet 人体检测 | https://bj.bcebos.com/v1/paddledet/models/keypoint/picodet_s_320_pedestrian.tar | 解压到 `ocr-service/models/` |
| TinyPose 关键点 | https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_256x192.tar | 解压到 `ocr-service/models/` |

校验：

```bash
ls /data/ai/apps/PaddleDetection/deploy/python/infer.py
ls /data/ai/apps/ocr-service/models/picodet_s_320_pedestrian/infer_cfg.yml
ls /data/ai/apps/ocr-service/models/tinypose_256x192/infer_cfg.yml
```

#### 3. 上传业务代码

确保 `app/common/pose_engine.py`、`app/parsers/portrait_validator.py` 等已更新到 `/data/ai/apps/ocr-service/`。

#### 4. 发布并验证

```bash
bash /data/ai/apps/ocr-service/deploy.sh

curl http://127.0.0.1:8088/health
curl -X POST http://127.0.0.1:8088/validate/portrait -F "file=@/data/ai/upload/test/photo.jpg"
curl -X POST http://127.0.0.1:8088/ocr/idcard -F "file=@/data/ai/upload/test/id.jpg"
```

#### 5. 固化镜像（避免 rm 容器后丢 pip 包）

```bash
docker commit ocr-server ocr-server:v3
```

`deploy.sh` 默认使用 `ocr-server:v3` 镜像。

---

### 日常更新（只改代码）

```bash
# 上传 app/、start.sh 等到 /data/ai/apps/ocr-service
bash /data/ai/apps/ocr-service/deploy.sh
```

**不必**重建镜像，除非 `requirements` 新增 pip 包。

---

### v2 初始搭建（参考）

v2 是在 CentOS 7 上用 `ubuntu:22.04` 容器手工安装 PaddleOCR 调通的（详见内部 CSDN 文档）。核心要点：

```bash
# 宿主机
docker run -itd --name ocr-server --restart=always \
  -v /data/ai:/data/ai ubuntu:22.04 sleep infinity

# 容器内
apt install -y python3 python3-pip libgl1 libglib2.0-0 ...
pip install paddlepaddle==2.6.2 paddleocr==2.7.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy==1.24.4 opencv-contrib-python==4.6.0.66 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 本地 / 新机器（docker compose，非现网 v2 替换）

```bash
cd ocr-service
docker compose up -d --build
curl http://127.0.0.1:8088/health
```

---

## HTTP 接口

统一响应：

```json
{ "code": 200, "success": true, "data": { } }
```

失败时 `success: false`，`message` 为错误说明。

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/ocr/idcard?side=auto` | 身份证 OCR，`file` 字段 |
| POST | `/ocr/bankcard` | 银行卡 OCR |
| POST | `/validate/portrait` | 头肩照构图校验 |

`side` 可选：`auto` | `front` | `back`。

### 身份证 `data` 字段

`side`, `name`, `gender`, `nation`, `birth`, `address`, `id_no`, `issue_authority`, `valid_start`, `valid_end`

### 银行卡 `data` 字段

`card_no`, `card_no_partial`, `bank_name`, `holder_name`

### 头肩照 `data` 字段

| 字段 | 说明 |
|------|------|
| `passed` | 是否通过构图校验 |
| `reason` | 未通过原因码（见下表） |
| `message` | 前端可直接展示的提示文案 |
| `person_count` | 检测到的人数 |

**`reason` 原因码：**

| reason | 含义 |
|--------|------|
| `NO_PERSON` | 未检测到人物 |
| `MULTIPLE_PERSONS` | 检测到多人 |
| `FACE_NOT_VISIBLE` | 面部不清晰 |
| `SHOULDER_NOT_VISIBLE` | 双肩未检测到 |
| `SHOULDER_CROPPED` | 肩膀被裁切 |
| `FACE_TOO_SMALL` | 人脸过小 |
| `PERSON_TOO_SMALL` | 人物过小 |

> 头肩校验为**构图检测**，不含活体 / 防翻拍。首次请求会懒加载模型（约 3～5 秒）。

---

## Java system 对接

### 配置（`application.yml` / `application-prod.yml`）

```yaml
nh:
  ocr:
    enabled: true
    base-url: http://127.0.0.1:8088
    idcard-path: /ocr/idcard
    bankcard-path: /ocr/bankcard
    portrait-path: /validate/portrait
    connect-timeout-ms: 5000
    read-timeout-ms: 120000
```

### 小程序 API（需登录）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/mini/ocr/idcard` | 身份证 OCR |
| POST | `/mini/ocr/bankcard` | 银行卡 OCR |
| POST | `/mini/ocr/portrait` | 头肩照校验（仅校验，不存图） |

请求头：`Authorization: <token>`

### 小程序示例

```javascript
// 身份证
wx.uploadFile({
  url: `${API_BASE}/mini/ocr/idcard`,
  filePath: tempFilePath,
  name: 'file',
  formData: { side: 'front' },
  header: { Authorization: token },
})

// 头肩照（passed 后再上传存图）
wx.uploadFile({
  url: `${API_BASE}/mini/ocr/portrait`,
  filePath: tempFilePath,
  name: 'file',
  header: { Authorization: token },
  success(res) {
    const body = JSON.parse(res.data)
    if (body.code === 200 && body.data.passed) {
      // 校验通过
    } else {
      wx.showToast({ title: body.data?.message || '请重新拍摄', icon: 'none' })
    }
  },
})
```

---

## 环境变量

由 `start.sh` 设置，一般无需改。

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OCR_PORT` | `8088` | 服务端口 |
| `OCR_LOG_DIR` | `/data/ai/apps/ocr-service/logs` | 日志目录 |
| `PADDLE_DET_DEPLOY_DIR` | `/data/ai/apps/PaddleDetection/deploy/python` | 推理脚本 |
| `PORTRAIT_DET_MODEL_DIR` | `{APP_DIR}/models/picodet_s_320_pedestrian` | 人体检测模型 |
| `PORTRAIT_KPT_MODEL_DIR` | `{APP_DIR}/models/tinypose_256x192` | 关键点模型 |
| `PORTRAIT_ENABLED` | `true` | 关闭则 portrait 接口返回 503 |
| `PORTRAIT_DEVICE` | `CPU` | 推理设备 |
| `PORTRAIT_WARMUP` | `false` | `true` 则启动时预加载头肩模型 |

---

## 命令行调试

```bash
docker exec -it ocr-server bash
cd /data/ai/apps/ocr-service
export PYTHONPATH=/data/ai/apps/ocr-service

python3 -m app.parsers.idcard_parser /data/ai/upload/test/id.jpg
python3 -m app.parsers.bankcard_parser /data/ai/upload/test/card.jpg
python3 -m app.parsers.portrait_validator /data/ai/upload/test/photo.jpg
```

---

## 日志

```
/data/ai/apps/ocr-service/logs/
├── ocr-service.log   # 服务 / 错误（含 portrait Traceback）
└── access.log        # HTTP 访问
```

```bash
tail -f /data/ai/apps/ocr-service/logs/ocr-service.log
docker logs --tail 100 ocr-server
```

日志按天滚动，保留 14 天。禁止打印完整 `id_no` / `card_no`。

---

## 常见问题

### 1. `pip install paddledet` 后 OCR 挂了 / numpy 被降级

**不要装 paddledet。** 卸载并恢复 v2：

```bash
pip uninstall -y paddledet visualdl sklearn scikit-learn motmetrics lap pycocotools opencv-python
pip install numpy==1.24.4 opencv-contrib-python==4.6.0.66 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install PyYAML shapely scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. `No module named 'cv2'`

opencv 被误删，重装：

```bash
pip install opencv-contrib-python==4.6.0.66 --force-reinstall --no-cache-dir \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. portrait 500：`got multiple values for argument 'strides'`

PaddleDetection `infer.py` 与 `picodet_postprocess` 版本不一致。  
**已在新版 `pose_engine.py` 中自动 patch**；上传最新 `app/common/pose_engine.py` 后重启即可。

### 4. portrait 503：`deploy dir not found`

检查 PaddleDetection 路径、`infer.py` 是否存在。

### 5. git clone PaddleDetection 超时

本地下载 [release/2.6 zip](https://github.com/PaddlePaddle/PaddleDetection/archive/refs/heads/release/2.6.zip) 上传解压到 `/data/ai/apps/PaddleDetection`。

### 6. 仅改代码后接口没变化

确认已 `bash deploy.sh` 重启容器，且代码在挂载目录 `/data/ai/apps/ocr-service`。

---

## 安全与运维

1. OCR 容器 **仅绑定 127.0.0.1:8088**，勿暴露公网。
2. 仅 Java system 访问 OCR；Nginx 只代理 `/api` 到 Java。
3. 容器建议 **6GB** 内存。
4. 灾难恢复：按 v2 文档重建 PaddleOCR → 装 PyYAML/shapely/scipy → 准备 PaddleDetection + 模型 → `docker commit`。

---

## 限制说明

- OCR 准确率依赖拍摄质量（反光、模糊、裁切）。
- 银行卡复杂背景可能只返回 `card_no_partial`。
- 身份证为 OCR + 规则提取，关键场景需人工复核。
- 头肩校验仅判断构图，**非活体检测**。

---

## 相关文件

| 文件 | 说明 |
|------|------|
| [docs/PRODUCTION_UPGRADE.md](docs/PRODUCTION_UPGRADE.md) | 现网 v2→v3 升级详细步骤 |
| `requirements.txt` | 完整依赖（开发参考） |
| `requirements-portrait-inference.txt` | 头肩推理增量依赖（勿装 paddledet） |
| `deploy.sh` | 现网发布脚本 |
| `start.sh` | 容器内启动脚本 |
| `scripts/setup_portrait_on_host.sh` | 首次初始化 PaddleDetection + 模型 |
| `scripts/download_portrait_models.sh` | 仅下载头肩模型 |
| `Dockerfile.v3` | 可选：基于 v2 镜像增量 pip（`build-v3.sh`） |
