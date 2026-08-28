# BM1684X 多路 RTSP 采样与火烟确认接口对接

面向 SE7-32（BM1684X）SoC 盒子的轻量服务。原生运行，不依赖 Docker，支持最多 16 路 RTSP，并通过 `mode` 切换两种模式：

| 模式 | 何时上传 | 盒子上的 YOLO |
|---|---|---|
| `yolo_trigger` | 周期采样，YOLO 命中指定目标后上传这张原图 | 加载一份共享模型，逐路推理 |
| `periodic` | 每路按周期直接上传原图 | 不加载、不推理，也不要求模型文件 |

两种模式均调用 [ascend-sam3-llm-pipeline](https://github.com/dongxuecheng/ascend-sam3-llm-pipeline) 的 `POST /v1/frames`。**运行服务不画框、不保存本地图片**，仅在内存编码 JPEG。远端负责 SAM3、LLM 二次确认和证据保存。历史图片不会自动删除。

本项目内置 [sophgo/sophon-demo](https://github.com/sophgo/sophon-demo) 已验证提交中的最小 YOLO26 BMCV 运行文件，不需要另外克隆完整仓库。保留 VPU 硬解码、BMCV 转换/前处理和 TPU 推理，JPEG 使用 SAIL / Sophon OpenCV 内存编码接口。

## 1. 架构

```mermaid
flowchart LR
    R["1～16路RTSP"] --> D["每路独立MultiDecoder<br/>VPU硬解码+最新帧队列"]
    D --> S["固定延时调度器<br/>非阻塞读取最新帧"]
    S --> M{"mode"}
    M -->|yolo_trigger| Y["YOLO26 TPU推理"]
    Y -->|命中| E["原图 JPEG 内存编码"]
    Y -->|未命中| X["丢弃采样帧"]
    M -->|periodic| E
    E --> Q["每路一个最新待传帧"]
    Q --> H["独立 HTTP 线程<br/>有界并发 + 总超时"]
    H --> A["POST /v1/frames<br/>远端 SAM3 + LLM"]
```

关键行为：

- 每路摄像头拥有独立的解码器和生命周期线程。一条流离线、读取异常或正在重连时，其余摄像头继续取帧和检测。
- 解码队列只保留少量最新帧，调度器使用非阻塞读取，不积压历史画面。
- 单路持续无有效帧达到 `read_timeout_sec` 后，仅重建该路解码器；摄像头重新上线后自动恢复。
- YOLO 推理和图片编码由一个硬件线程串行执行，避免并发调用同一模型。HTTP 在独立线程的异步事件循环中执行，不等待远端模型结果。
- 每路最多一张正在上传、一张最新待传图片；新图片替换同路待传旧图，保持队列顺序，防止某一路长期占用队列。
- 总并发由 `pipeline.workers` 限制（默认 2）。上传慢或失败不会阻塞解码/推理，但可能替换未上传的图片；所有上传名额都被占用时其他路也需等待。
- 超时、HTTP 错误（包括 429/503）、无效响应均丢弃本次请求，不自动重试。下一次采样继续，网络恢复后自然恢复上传。避免响应丢失时重复提交远端已接收的图片。
- 待传图片超过 `max_pending_age_sec` 则丢弃；进程退出取消请求、清空内存队列，不落盘补发。这不是可靠投递系统，短暂事件可能遗漏。

### 硬件加速覆盖

| 环节 | 实现 |
|---|---|
| RTSP H.264/H.265解码 | SAIL MultiDecoder / Sophon FFmpeg，VPU |
| 色彩转换、缩放和归一化 | 官方YOLO26 BMCV实现 |
| YOLO26推理 | BM1684X TPU，建议INT8 1 batch |
| JPEG 内存编码 | BMCV 转 BGR_PACKED → `Bmcv.imencode(".jpg", image)` |
| 调度、过滤、HTTP multipart 上传 | Python CPU；HTTPX 0.27.2（兼容 Python 3.8） |

SAIL 3.11.1 的 `imencode` 内部调用 Sophon OpenCV；是否实际启用 JPU 与 SDK/图片格式相关，需在盒子观察 `/proc/jpuinfo` 验证，不把调用 BMCV 等同于全程 TPU。接口参考：[SAIL Bmcv 文档](https://doc.sophgo.com/sdk-docs/v23.09.01-lts-sp5/docs_latest_release/docs/sophon-sail/docs/en/html/module/sail.Bmcv.html#imencode)、[v3.11.1 实现](https://github.com/sophgo/sophon-sail/blob/v3.11.1/src/cvwrapper.cpp)。

## 2. 目录结构

```text
config/
  config.example.yaml       YOLO 命中上传示例（修改 mode 即可切换）
  config.periodic.example.yaml  无模型定时直传示例
  pipeline.env.example      systemd 密钥环境变量示例
  fire_smoke.names          YOLO类别顺序示例
deploy/
  fire-smoke-monitor.service
docs/
  BM1684X_SDK_26.03.01_OTA_UPGRADE.md
  SAIL_3.11.1_BM1684X_SDK_26.03.01_CROSS_COMPILE.md
src/fire_smoke_monitor/
  config.py                 配置解析和运行文件校验
  streams.py                独立RTSP解码与自动重连
  models.py                 官方YOLO26适配器
  scheduler.py              多路固定延时调度
  encoding.py               原图 JPEG 内存编码，不画框/落盘
  uploader.py               有界队列、异步 HTTP、鉴权与超时
  app.py / cli.py           应用生命周期和命令行
  vendor/yolo26/            固定版本的官方YOLO26 BMCV运行文件和许可证
tests/                       不依赖算能硬件的单元测试
```

## 3. SoC运行环境

原 YOLO/取流链路已经在以下环境验证；新增上传模式仍需在真实盒子与远端服务联调：

- SE7-32 / BM1684X SoC；
- Ubuntu 20.04，aarch64；
- SophonSDK v26.03.01；
- libsophon 0.5.3；
- sophon-ffmpeg、sophon-opencv 0.15.0；
- SAIL 3.11.1；
- Python 3.8；
- NumPy 1.24+、PyYAML 6.x；新增 HTTPX 0.27.2。

旧系统升级参见[SDK v26.03.01 OTA升级记录](docs/BM1684X_SDK_26.03.01_OTA_UPGRADE.md)，安装匹配SAIL参见[SAIL 3.11.1交叉编译记录](docs/SAIL_3.11.1_BM1684X_SDK_26.03.01_CROSS_COMPILE.md)。

不要用普通的 `opencv-python` 覆盖SoC自带Sophon OpenCV，也不要从PyPI安装来源不明的SAIL。驱动、运行时、FFmpeg、OpenCV和SAIL应匹配同一套SDK。

SDK镜像中的OpenCV Python绑定通常需要加入模块路径：

```bash
export PYTHONPATH=/opt/sophon/sophon-opencv-latest/opencv-python${PYTHONPATH:+:$PYTHONPATH}
```

验证环境：

```bash
python3 - <<'PY'
import cv2
import numpy
import sophon.sail as sail

handle = sail.Handle(0)
print("target:", handle.get_target())
print("opencv:", cv2.__version__, cv2.__file__)
print("numpy:", numpy.__version__)
print("sail:", sail.__file__)
print("MultiDecoder:", hasattr(sail, "MultiDecoder"))
print("Bmcv.imencode:", hasattr(sail.Bmcv, "imencode"))
PY
```

`target` 应包含 `BM1684X`。

## 4. 准备项目

假设本仓库已经位于盒子的 `/data` 分区：

```bash
export PROJECT_ROOT=/data/sophgo-rtsp-yolo-sam3
cd "${PROJECT_ROOT}"

mkdir -p logs
```

项目已内置 `sophon-demo release@dfe0dff31c21984b681536df9cb2b1ba6c96a8a4` 的 `yolo26_bmcv.py`、`utils.py` 和BSD-2-Clause许可证。运行盒子不再需要 `vendor/sophon-demo` 目录。

创建复用系统SDK包的虚拟环境：

```bash
python3 -m venv --system-site-packages .venv  # 已有可用环境时不必重新创建
"${PROJECT_ROOT}/.venv/bin/python" -m pip install -r requirements.txt
cp -n config/config.example.yaml config/config.yaml
chmod 600 config/config.yaml
```

Python 3.8和旧版pip若不能执行editable安装，可以直接使用源码目录运行，无需安装项目包：

```bash
export PYTHONPATH="${PROJECT_ROOT}/src:/opt/sophon/sophon-opencv-latest/opencv-python"
```

已有部署只需在原虚拟环境安装新增的 `httpx==0.27.2`，并按第 7 节迁移配置。`cp -n` 不覆盖已有配置，旧配置不能直接启动新版。若选定时直传，可以从 `config/config.periodic.example.yaml` 另存配置，跳过下面模型下载和训练步骤。

## 5. 先用官方COCO模型验证链路

仅 `yolo_trigger` 需要模型。使用算能 `dfss` 直接下载测试数据、COCO类别和BM1684X模型：

```bash
mkdir -p "${PROJECT_ROOT}/models/yolo26" "${PROJECT_ROOT}/datasets" \
  "${PROJECT_ROOT}/downloads" "${PROJECT_ROOT}/output/official-yolo26"
cd "${PROJECT_ROOT}/downloads"

"${PROJECT_ROOT}/.venv/bin/python" -m pip install dfss
"${PROJECT_ROOT}/.venv/bin/python" -m dfss \
  --url=open@sophgo.com:sophon-demo/YOLO26_det/BM1684X.tar.gz
tar xf BM1684X.tar.gz
cp BM1684X/yolo26s_fp16_1b.bmodel \
  "${PROJECT_ROOT}/models/yolo26/"

"${PROJECT_ROOT}/.venv/bin/python" -m dfss \
  --url=open@sophgo.com:sophon-demo/common/coco.names
cp coco.names "${PROJECT_ROOT}/models/yolo26/"

"${PROJECT_ROOT}/.venv/bin/python" -m dfss \
  --url=open@sophgo.com:sophon-demo/common/test.tar.gz
tar xf test.tar.gz -C "${PROJECT_ROOT}/datasets"
```

将下载得到的模型、COCO类别文件和测试图片放入本项目对应目录，例如：

```text
models/yolo26/yolo26s_fp16_1b.bmodel
models/yolo26/coco.names
datasets/test/dog.jpg
```

可选：先执行官方静态图片程序。它是独立验证工具，会在本地保存测试结果，不属于两种上传模式的运行链路：

```bash
cd "${PROJECT_ROOT}/output/official-yolo26"
PYTHONPATH="${PROJECT_ROOT}/src:/opt/sophon/sophon-opencv-latest/opencv-python" \
"${PROJECT_ROOT}/.venv/bin/python" -m \
  fire_smoke_monitor.vendor.yolo26.yolo26_bmcv \
  --input "${PROJECT_ROOT}/datasets/test" \
  --bmodel "${PROJECT_ROOT}/models/yolo26/yolo26s_fp16_1b.bmodel" \
  --dev_id 0 \
  --conf_thresh 0.25
```

COCO模型不包含 `fire`、`smoke` 类别。验证RTSP阶段可临时将 `target_classes` 设置为 `person`，并让 `class_names` 指向 `coco.names`；此时只能验证“人触发上传”，远端仍只确认火烟，可能不保存任何证据。

## 6. 火焰/烟雾模型训练与转换

官方COCO模型只能验证部署链路。`yolo_trigger` 用于火烟生产检测时，需训练火焰/烟雾专用模型；YOLO 漏检的帧不会进入远端复核。`periodic` 不依赖这一步。训练和模型转换应在有GPU、大内存和TPU-MLIR环境的x86工作站完成。

### 6.1 数据集与训练

Ultralytics detect数据集示例：

```yaml
# fire_smoke.yaml
path: /data/fire_smoke
train: images/train
val: images/val
names:
  0: fire
  1: smoke
```

训练示例：

```bash
python3 -m venv .venv-yolo
. .venv-yolo/bin/activate
pip install 'ultralytics==8.4.9' onnx
yolo detect train model=yolo26s.pt data=fire_smoke.yaml imgsz=640 epochs=100 batch=16
```

训练集和验证集应覆盖真实摄像机画质、昼夜、远近尺度，并加入蒸汽、云雾、灰尘、灯光、反光、焊接和夕阳等困难负样本。

### 6.2 导出ONNX

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
model.export(format="onnx", opset=13, imgsz=640, batch=1)
```

### 6.3 转换BM1684X BModel

准备128～1000张具有代表性的量化图片到 `datasets/fire_smoke_cali/`，然后在已激活TPU-MLIR的环境执行：

```bash
cd /opt/sophon-demo/sample/YOLO26/scripts

model_transform.py \
  --model_name fire_smoke_yolo26s \
  --model_def ../models/onnx/yolo26s.onnx \
  --input_shapes '[[1,3,640,640]]' \
  --mean 0.0,0.0,0.0 \
  --scale 0.0039216,0.0039216,0.0039216 \
  --keep_aspect_ratio \
  --pixel_format rgb \
  --mlir fire_smoke_yolo26s_1b.mlir

run_calibration.py fire_smoke_yolo26s_1b.mlir \
  --dataset ../datasets/fire_smoke_cali/ \
  --input_num 128 \
  --part_asymmetric \
  --cali_method percentile9999 \
  --fp_type F32 \
  -o fire_smoke_yolo26s_cali_table

model_deploy.py \
  --mlir fire_smoke_yolo26s_1b.mlir \
  --quantize INT8 \
  --chip bm1684x \
  --quantize_table ../models/onnx/yolo26s_qtable_int8 \
  --calibration_table fire_smoke_yolo26s_cali_table \
  --model fire_smoke_yolo26s_int8_1b.bmodel
```

如果INT8精度下降明显，应重新生成适合自定义权重的qtable，或先使用FP16建立准确率基线。转换完成后执行：

```bash
bmrt_test --bmodel fire_smoke_yolo26s_int8_1b.bmodel --devid 0
```

类别文件必须与训练class id顺序完全一致：

```text
fire
smoke
```

## 7. 配置

核心配置示例：

```yaml
mode: yolo_trigger  # 改成 periodic 并重启，即不再加载 YOLO

pipeline:
  endpoint: http://192.168.1.200:18080/v1/frames
  machine_id: se7-01
  api_key_env: ""  # 若远端启用鉴权，填 PIPELINE_API_KEY
  workers: 2
  timeout_sec: 10
  max_pending_age_sec: 30

runtime:
  dev_id: 0
  decoder_queue_size: 2
  read_timeout_sec: 2
  reconnect_cooldown_sec: 10
  log_level: INFO

cameras:
  - id: camera_01
    name: 仓库东门
    url: rtsp://user:password@192.168.1.101:554/stream1
  - id: camera_02
    name: 仓库西门
    url: rtsp://user:password@192.168.1.102:554/stream1

    # interval_sec: 20  # 可选，覆盖当前模式默认值

periodic:
  interval_sec: 10

models:
  yolo26:
    bmodel: ../models/yolo26/fire_smoke_yolo26s_int8_1b.bmodel
    class_names: ./fire_smoke.names
    target_classes: [fire, smoke]
    conf_threshold: 0.25
    interval_sec: 3
```

配置说明：

- 最多16路摄像头，`id` 必须唯一，并直接作为接口的 `stream_id`。`machine_id` 与 `id` 均只允许 1–64 位英文字母、数字、下划线或横线，不能带点。多盒部署时 `machine_id` 不可重复。
- `mode` 是唯一模式开关，不需要再设置 `models.yolo26.enabled`。旧配置里若有 `enabled: false`，切回 `yolo_trigger` 时请删除或改为 `true`；`periodic` 完全忽略 `models`。
- `yolo_trigger` 默认间隔取 `models.yolo26.interval_sec`；`periodic` 取 `periodic.interval_sec`。摄像头自己的 `interval_sec` 可覆盖两者。
- 间隔是本路取帧/检测/编码/入队完成到下次采样的等待时间，不含等待 HTTP 的时间。单硬件线程忙时会晚于配置周期，不追赶、不保证严格每 N 秒；刚出现目标也可能等到下一次采样。
- YOLO 一张图命中多个目标也只上传一次整张、未标注图，不上传裁剪图或检测框字段。`captured_at` 为盒子读到该帧时的 UTC 时间，不是摄像头硬件曝光时间；请校准盒子系统时钟。
- 不改变原图分辨率；默认最大 8 MiB JPEG、1600 万像素，与远端默认限制一致。超限跳过并记录日志，可降低摄像头子码流分辨率；调大限制必须同时调整远端。
- 配置文件包含RTSP账号密码，应设置为 `chmod 600`，不要提交到Git。

### 从原“检测后保存图片”版本迁移

1. 备份旧配置，安装 `httpx==0.27.2`。
2. 增加顶层 `mode`、`pipeline`；直传模式增加 `periodic.interval_sec`。
3. 删除 `runtime.output_root`、`runtime.min_free_disk_gb`、`models.yolo26.save_cooldown_sec`。遗留字段会告警并忽略，**不会把旧保存冷却转换成上传冷却**。
4. 保留摄像头 URL；YOLO 模式保留原 BModel、类别和目标配置；定时直传可删除整个 `models`。
5. 检查配置后重启。不是热加载，运行中编辑 YAML 不会即时切换。

### 远端接口与鉴权

已核对 [远端 main@0ba7bfb 的接口实现](https://github.com/dongxuecheng/ascend-sam3-llm-pipeline/blob/0ba7bfb15f033579b28ce547012a2b38d077145c/app/main.py)：

- 完整地址为 `http://远端服务器IP:18080/v1/frames`，不要填 GitHub 地址或 `SAM3 /predict/file`。
- multipart 字段为 `image`（JPEG）、`machine_id`、`stream_id`、可选 `stream_name` 和带时区的 `captured_at`。不发送 RTSP 地址/密码。
- 远端设置 `PIPELINE_API_KEY` 后，将 YAML 的 `api_key_env` 设置为 `PIPELINE_API_KEY`，并给盒子进程设置同名环境变量，值与远端一致。程序用 `X-API-Key` 请求头发送，不把密钥写进 YAML。
- 只将 `202` 且 JSON `accepted: true` 记为接收成功；不等待 SAM3/LLM，不查询任务。`202` 不代表确认火烟，更不保证保存。
- HTTPX 禁用环境代理和重定向，HTTPS 保留证书校验。HTTP 明文只用于可信隔离内网；跨不可信网络请部署 HTTPS/VPN。

接口无幂等键，因此网络超时后即使远端可能已经接收，也不重传旧图。远端队列满返回 `429` 时下一次采样照常进行；频繁 `429` 应增大采样间隔，增大并发不能解决长期推理吞吐不足。

## 8. 启动

每次重新打开 SSH 终端，先设置路径。远端若启用鉴权，可交互输入密钥，避免写进 shell 历史：

```bash
export PROJECT_ROOT=/data/sophgo-rtsp-yolo-sam3
cd "${PROJECT_ROOT}"
mkdir -p logs
export PYTHONPATH="${PROJECT_ROOT}/src:/opt/sophon/sophon-opencv-latest/opencv-python"
# 仅在 YAML api_key_env: PIPELINE_API_KEY 时执行以下三行：
read -r -s -p "Pipeline API key: " PIPELINE_API_KEY
printf '\n'
export PIPELINE_API_KEY
```

检查配置、密钥环境变量以及该模式所需的模型文件，不初始化硬件、不访问远端：

```bash
cd "${PROJECT_ROOT}"
PYTHONPATH="${PROJECT_ROOT}/src:/opt/sophon/sophon-opencv-latest/opencv-python" \
"${PROJECT_ROOT}/.venv/bin/python" -m fire_smoke_monitor \
  --config config/config.yaml \
  --check-config
```

前台运行：

```bash
umask 077
set -o pipefail
PYTHONUNBUFFERED=1 \
PYTHONPATH="${PROJECT_ROOT}/src:/opt/sophon/sophon-opencv-latest/opencv-python" \
"${PROJECT_ROOT}/.venv/bin/python" -m fire_smoke_monitor \
  --config config/config.yaml \
  2>&1 | tee "logs/pipeline-$(date +%Y%m%d-%H%M%S).log"
```

正常日志示例（采样入队不等于 HTTP 成功）：

```text
取帧调度已启动 mode=periodic cameras=10 local_save=disabled
图片待上传 mode=periodic camera=camera_01 targets=0 queue=queued ...
远端已接收 camera=camera_01 status=202 ...
```

`periodic` 的 `targets=0` 表示没有做本地检测，不代表远端未检出。`queue=replaced` 表示替换了同路尚未上传的旧图，应关注远端负载/网络。

可以从盒子检查 `curl -fsS http://远端服务器IP:18080/health`；它只反映远端服务健康和累计计数，不检查真实模型效果。最终证据由远端的存储配置决定，盒子 `output/` 不再新增检测图片。原生解码日志仍可能含 RTSP 账号密码，分享日志前请脱敏。

## 9. 24×7运行

`deploy/fire-smoke-monitor.service` 已按 `/data/sophgo-rtsp-yolo-sam3`、用户 `linaro` 配好。若实际位置不同，请修改：

- `WorkingDirectory`；
- `ExecStart`；
- `Environment=PYTHONPATH=...`。

服务端启用鉴权时，复制 `config/pipeline.env.example` 为 `config/pipeline.env`，用编辑器填入真实密钥，执行 `chmod 600 config/pipeline.env`。服务会读取该文件；YAML 中仍需设 `api_key_env: PIPELINE_API_KEY`。前台运行不会自动读取这个文件，使用前述交互环境变量方式即可。

安装服务：

```bash
sudo cp deploy/fire-smoke-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now fire-smoke-monitor
sudo systemctl status fire-smoke-monitor --no-pager -l
journalctl -u fire-smoke-monitor -f
```

服务使用 `Restart=always`。SIGTERM/SIGINT 会取消未完成上传、丢弃待传帧、关闭解码通道并释放模型。已被远端接收的帧可能继续处理，取消 HTTP 不等于取消远端任务。修改配置/密钥后执行 `sudo systemctl restart fire-smoke-monitor`；前台与 systemd 不要同时启动两份程序。

## 10. 性能监控

```bash
pgrep -af '[f]ire_smoke_monitor'
PID=$(pgrep -n -f '[f]ire_smoke_monitor')
top -p "${PID}"
free -h
vmstat 1
df -h /data
/opt/sophon/libsophon-current/bin/bm-smi
watch -n 1 cat /proc/vpuinfo
watch -n 1 cat /proc/jpuinfo
```

项目按低频采样设计。摄像头数量增加时通常只需添加配置项，并根据网络和远端吞吐增大 `interval_sec`。例如 10 路每 10 秒一张，直传约 1 张/秒（实际略低）；若 JPEG 平均 500 KiB，仅图片流量约 4.1 Mbit/s。YOLO 模式仅上传命中帧，但本地漏检会挡住后续复核。

## 11. 测试和故障定位

开发机测试：

```bash
python -m pip install -r requirements-dev.txt
PYTHONPATH=src pytest -q
```

测试覆盖两种模式、每路间隔、离线恢复、内存编码调用、multipart 字段与鉴权、真实本机 HTTP 上传、总超时、429、不重试、最新帧替换及 16 路队列上限。模型和图片编码测试使用替身；它们不代表已完成 BM1684X 硬件或真实 SAM3/LLM 联调。

常见问题：

- `No module named sophon.sail`：虚拟环境未继承系统SDK包，或SAIL没有安装到当前Python 3.8。
- `No module named cv2`：将Sophon OpenCV Python目录加入 `PYTHONPATH`。
- YOLO一直没有火烟框：先用官方静态图片程序验证BModel、类别文件顺序和阈值。
- 单路RTSP反复重连：先用Sophon FFmpeg验证URL、认证、网络和编码格式。
- `缺少配置项 root.mode` / `root.pipeline`：使用了旧保存模式配置，按第 7 节迁移。
- `No module named httpx`：在项目的 `.venv/bin/python` 中安装 `httpx==0.27.2`。
- `HTTP 401`：检查两端密钥和 YAML `api_key_env`，不要在日志中打印密钥。
- `HTTP 400/422`：检查图片、机器/摄像头 ID、时间；`HTTP 413`：图片字节或像素超限。
- `HTTP 429/503` 或频繁 `queue=replaced`：远端积压/繁忙，先增大采样间隔并检查远端健康。
- `上传超时` / `ConnectError`：检查盒子到远端的地址、路由、防火墙和监听端口。
- 有 `202` 但没有证据：查看远端 SAM3/LLM 日志及 `/health` 计数；未确认、模型失败等都可能不保存。

## 12. 第三方代码边界

本仓库包含多路流调度、检测适配、内存编码和 HTTP 上传，以及运行所需的两个 `sophon-demo` YOLO26 Python文件。内置文件的来源提交、改动说明和BSD-2-Clause许可证位于 `src/fire_smoke_monitor/vendor/yolo26/`。Ultralytics、模型权重和数据集分别受其许可证约束，商用部署前应单独核对。
