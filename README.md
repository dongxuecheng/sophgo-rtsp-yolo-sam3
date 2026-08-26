# BM1684X 多路 RTSP YOLO26 火焰/烟雾检测

面向 SE7-32（BM1684X）SoC 盒子的轻量监控服务。程序原生运行，不依赖 Docker，支持最多 16 路 RTSP；各摄像头按可配置周期抽取最新帧，通过 YOLO26 BModel 检测指定类别，命中后保存带框 JPEG。

本项目内置 [sophgo/sophon-demo](https://github.com/sophgo/sophon-demo) 已验证提交中的最小YOLO26 BMCV运行文件，不需要另外克隆完整仓库。RTSP解码、YOLO前处理、TPU推理、画框和JPEG编码尽量使用BM1684X硬件加速。

## 1. 架构

```mermaid
flowchart LR
    R["1～16路RTSP"] --> D["每路独立MultiDecoder<br/>VPU硬解码+最新帧队列"]
    D --> S["固定延时调度器<br/>非阻塞读取最新帧"]
    S --> Y["YOLO26<br/>BMCV前处理+TPU推理"]
    Y --> H{"命中目标类别?"}
    H -->|是| W["BMCV画框+JPEG编码"]
    H -->|否| X["丢弃当前采样帧"]
    W --> O["output/yolo26/摄像头/日期"]
```

关键行为：

- 每路摄像头拥有独立的解码器和生命周期线程。一条流离线、读取异常或正在重连时，其余摄像头继续取帧和检测。
- 解码队列只保留少量最新帧，调度器使用非阻塞读取，不积压历史画面。
- 单路持续无有效帧达到 `read_timeout_sec` 后，仅重建该路解码器；摄像头重新上线后自动恢复。
- TPU推理串行执行，避免多个Python线程并发调用同一个模型。每路检测间隔通过 `interval_sec` 配置。
- 只有检测到 `target_classes` 中的目标才保存图片；保存冷却和剩余磁盘保护适合24×7运行。

### 硬件加速覆盖

| 环节 | 实现 |
|---|---|
| RTSP H.264/H.265解码 | SAIL MultiDecoder / Sophon FFmpeg，VPU |
| 色彩转换、缩放和归一化 | 官方YOLO26 BMCV实现 |
| YOLO26推理 | BM1684X TPU，建议INT8 1 batch |
| 检测框绘制、JPEG编码 | BMCV |
| 调度、过滤和文件管理 | Python CPU |

## 2. 目录结构

```text
config/
  config.example.yaml       配置示例
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
  writer.py                 BMCV画框与JPEG保存
  app.py / cli.py           应用生命周期和命令行
  vendor/yolo26/            固定版本的官方YOLO26 BMCV运行文件和许可证
tests/                       不依赖算能硬件的单元测试
```

## 3. SoC运行环境

已经验证的环境：

- SE7-32 / BM1684X SoC；
- Ubuntu 20.04，aarch64；
- SophonSDK v26.03.01；
- libsophon 0.5.3；
- sophon-ffmpeg、sophon-opencv 0.15.0；
- SAIL 3.11.1；
- Python 3.8；
- NumPy 1.24+、PyYAML 6.x。

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
PY
```

`target` 应包含 `BM1684X`。

## 4. 准备项目

假设本仓库已经位于盒子的 `/data` 分区，进入仓库后：

```bash
export PROJECT_ROOT="$(pwd)"
cd "${PROJECT_ROOT}"

mkdir -p models/yolo26 datasets downloads output/detections logs
```

项目已内置 `sophon-demo release@dfe0dff31c21984b681536df9cb2b1ba6c96a8a4` 的 `yolo26_bmcv.py`、`utils.py` 和BSD-2-Clause许可证。运行盒子不再需要 `vendor/sophon-demo` 目录。

创建复用系统SDK包的虚拟环境：

```bash
python3 -m venv --system-site-packages .venv
"${PROJECT_ROOT}/.venv/bin/python" -m pip install 'PyYAML>=6,<7'
cp config/config.example.yaml config/config.yaml
chmod 600 config/config.yaml
```

Python 3.8和旧版pip若不能执行editable安装，可以直接使用源码目录运行，无需安装项目包：

```bash
export PYTHONPATH="${PROJECT_ROOT}/src:/opt/sophon/sophon-opencv-latest/opencv-python"
```

## 5. 先用官方COCO模型验证链路

使用算能 `dfss` 直接下载测试数据、COCO类别和BM1684X模型：

```bash
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

先执行官方静态图片程序：

```bash
cd "${PROJECT_ROOT}/output"
PYTHONPATH="${PROJECT_ROOT}/src:/opt/sophon/sophon-opencv-latest/opencv-python" \
"${PROJECT_ROOT}/.venv/bin/python" -m \
  fire_smoke_monitor.vendor.yolo26.yolo26_bmcv \
  --input "${PROJECT_ROOT}/datasets/test" \
  --bmodel "${PROJECT_ROOT}/models/yolo26/yolo26s_fp16_1b.bmodel" \
  --dev_id 0 \
  --conf_thresh 0.25
```

COCO模型不包含 `fire`、`smoke` 类别。验证RTSP阶段可临时将配置中的 `target_classes` 设置为 `person`，并让 `class_names` 指向 `coco.names`。

## 6. 火焰/烟雾模型训练与转换

官方COCO模型只能验证部署链路，生产使用必须训练火焰/烟雾专用模型。训练和模型转换应在有GPU、大内存和TPU-MLIR环境的x86工作站完成。

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
runtime:
  dev_id: 0
  output_root: ../output/detections
  decoder_queue_size: 2
  read_timeout_sec: 2
  reconnect_cooldown_sec: 10
  min_free_disk_gb: 2
  log_level: INFO

cameras:
  - id: camera_01
    url: rtsp://user:password@192.168.1.101:554/stream1
  - id: camera_02
    url: rtsp://user:password@192.168.1.102:554/stream1

models:
  yolo26:
    enabled: true
    bmodel: ../models/yolo26/fire_smoke_yolo26s_int8_1b.bmodel
    class_names: ./fire_smoke.names
    target_classes: [fire, smoke]
    conf_threshold: 0.25
    interval_sec: 3
    save_cooldown_sec: 30
```

配置说明：

- 最多配置16路摄像头，`id` 必须唯一。
- `interval_sec` 是某路检测完成后到下次检测的等待时间；不会追赶历史任务。
- `save_cooldown_sec: 0` 表示每次命中都保存。持续火情建议至少30秒，减少eMMC写入。
- `min_free_disk_gb` 达到阈值后停止保存，但继续取流、推理和记录日志。
- 配置文件包含RTSP账号密码，应设置为 `chmod 600`，不要提交到Git。

## 8. 启动

检查配置及模型文件，不初始化硬件推理：

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
PYTHONUNBUFFERED=1 \
PYTHONPATH="${PROJECT_ROOT}/src:/opt/sophon/sophon-opencv-latest/opencv-python" \
"${PROJECT_ROOT}/.venv/bin/python" -m fire_smoke_monitor \
  --config config/config.yaml \
  2>&1 | tee "logs/yolo26-$(date +%Y%m%d-%H%M%S).log"
```

检测图片保存在：

```text
output/detections/yolo26/<camera_id>/YYYY-MM-DD/<timestamp>_<labels>_<id>.jpg
```

## 9. 24×7运行

根据项目实际位置修改 `deploy/fire-smoke-monitor.service` 中的：

- `WorkingDirectory`；
- `ExecStart`；
- `Environment=PYTHONPATH=...`。

安装服务：

```bash
sudo cp deploy/fire-smoke-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now fire-smoke-monitor
sudo systemctl status fire-smoke-monitor --no-pager -l
journalctl -u fire-smoke-monitor -f
```

服务使用 `Restart=always`。SIGTERM/SIGINT会关闭所有解码通道并释放模型。

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

项目按低频采样设计。摄像头数量增加时通常只需继续添加配置项，并根据总体检测延迟适当增大 `interval_sec`；不需要修改代码。

## 11. 测试和故障定位

开发机测试：

```bash
python -m pip install -r requirements-dev.txt
PYTHONPATH=src pytest -q
```

常见问题：

- `No module named sophon.sail`：虚拟环境未继承系统SDK包，或SAIL没有安装到当前Python 3.8。
- `No module named cv2`：将Sophon OpenCV Python目录加入 `PYTHONPATH`。
- YOLO一直没有火烟框：先用官方静态图片程序验证BModel、类别文件顺序和阈值。
- 单路RTSP反复重连：先用Sophon FFmpeg验证URL、认证、网络和编码格式。
- 保存图片停止：检查 `/data` 剩余空间和 `min_free_disk_gb`。

## 12. 第三方代码边界

本仓库包含多路流调度、检测适配、保存逻辑，以及运行所需的两个 `sophon-demo` YOLO26 Python文件。内置文件的来源提交、改动说明和BSD-2-Clause许可证位于 `src/fire_smoke_monitor/vendor/yolo26/`。Ultralytics、模型权重和数据集分别受其许可证约束，商用部署前应单独核对。
