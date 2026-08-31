# BM1684X 多路 RTSP 采样与火烟确认

面向 SE7-32（BM1684X）SoC，原生运行，无需 Docker，最多 16 路 RTSP。
两种模式通过 YAML 的 `mode` 切换，重启后生效：

- `periodic`：每路周期取帧，直接上传；不加载 YOLO，不要求模型文件。
- `yolo_trigger`：共享一份 YOLO26 模型，周期检测，命中指定目标才上传。

两种模式都上传整张未标注 JPEG，**不在盒子保存图片**。
远端 [ascend-sam3-llm-pipeline](https://github.com/dongxuecheng/ascend-sam3-llm-pipeline)
负责 SAM3、LLM 二次确认和证据保存。

## 快速启动

代码位于 `/data/sophgo-rtsp-yolo-sam3`，沿用已有虚拟环境和实际配置时：

```bash
cd /data/sophgo-rtsp-yolo-sam3
bash start.sh config/config.periodic.yaml --check-config
bash start.sh config/config.periodic.yaml
```

不需要手动 export、激活虚拟环境、source /etc/profile 或 tee。
脚本自动定位项目、设置 Sophon OpenCV / 源码路径、开启无缓冲输出，并防止同一项目通过脚本重复运行。
从其他工作目录也可用脚本绝对路径启动；传入的相对配置路径以项目目录为基准。
`Ctrl+C` 停止前台程序。首次换用脚本前先停止旧的直接 Python/systemd 实例，因为旧实例没有脚本锁。

```bash
bash start.sh                 # 默认 config/config.yaml
bash start.sh --help
```

脚本不自动安装依赖、不覆盖配置、不终止其他进程；失败时保留 Python 的退出码。
仅检查配置时，不初始化硬件、不访问远端、不创建/清理日志文件。

### 首次部署

已验证基础环境：Ubuntu 20.04 aarch64、Python 3.8、SophonSDK 26.03.01、
libsophon 0.5.3、Sophon FFmpeg/OpenCV 0.15.0、完整 SAIL 3.11.1。
系统须提供 `flock`（util-linux）；`stdbuf`（coreutils）用于及时输出 C/C++ 日志。

```bash
cd /data/sophgo-rtsp-yolo-sam3
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
cp -n config/config.periodic.example.yaml config/config.periodic.yaml
chmod 600 config/config.periodic.yaml
```

编辑配置中的实际服务器地址、设备标识和摄像头 URL，再执行快速启动命令。
YOLO 模式改用 `config/config.example.yaml`，模型准备见 [YOLO26 教程](docs/YOLO26_MODELS.md)。
不要从 PyPI 安装普通 OpenCV 或替换算能 SDK/SAIL。

## 配置

下面是最小 periodic 配置；完整示例位于 `config/`：

```yaml
mode: periodic

pipeline:
  endpoint: http://192.168.1.200:18080/v1/frames  # 替换为实际地址
  machine_id: se7-01                           # 每台盒子不同
  api_key_env: ""                              # 有鉴权时填 PIPELINE_API_KEY
  workers: 2
  timeout_sec: 10
  max_pending_age_sec: 30

runtime:
  log_dir: ../logs
  log_retention_days: 30
  log_max_file_mb: 100
  log_max_total_mb: 1024
  min_free_disk_gb: 5
  disk_check_interval_sec: 60
  log_level: INFO

health:
  max_consecutive_processing_failures: 10
  scheduler_stall_sec: 30
  status_log_interval_sec: 300

periodic:
  interval_sec: 10

cameras:
  - id: camera_01
    name: 仓库东门
    url: "rtsp://user:password@192.168.1.101:554/stream1"
  - id: camera_02
    name: 仓库西门
    url: "rtsp://user:password@192.168.1.102:554/stream1"
    interval_sec: 20
```

- `machine_id`、摄像头 `id` 只允许 1–64 位字母、数字、下划线和横线，摄像头 ID 必须唯一。
- 每路 `interval_sec` 覆盖模式默认值；YOLO 默认值取 `models.yolo26.interval_sec`，periodic 取 `periodic.interval_sec`。
- 间隔指本路取帧/检测/编码/入队完成后到下次采样的等待时间，不等待 HTTP；硬件忙时会延后，不追赶旧任务。
- `runtime.dev_id` 默认 0，`decoder_queue_size` 默认 2；持续无帧超过 `read_timeout_sec`（默认 2）后，仅后台重建该路，重试间隔 `reconnect_cooldown_sec` 默认 10 秒。
- `periodic` 忽略整个 `models`。YOLO 命中多个框也只上传一张原图，本地漏检会挡住后续复核。
- 原图不缩放；默认最大 8 MiB、1600 万像素。超限时跳过并记日志，可降低摄像头子码流分辨率。调整 `max_image_bytes/max_image_pixels` 须同步远端限制。
- 相对模型/日志路径以 YAML 所在目录为基准。旧配置不添加新增字段时使用安全默认值：30 天、单文件 100 MiB、日志总量 1 GiB、日志所在文件系统至少保留 5 GiB。
- 旧 `output_root/save_cooldown_sec` 已停用；不会转换成上传冷却。`min_free_disk_gb` 现在用于日志磁盘保护，两种模式仍不写检测图片。

### 密钥

远端启用鉴权时，设置 YAML `pipeline.api_key_env: PIPELINE_API_KEY`，
复制 `config/pipeline.env.example` 为 **实际 YAML 同目录下的** `pipeline.env`，编辑其值并限制权限：

```text
PIPELINE_API_KEY=与远端相同的密钥
```

```bash
chmod 600 config/pipeline.env
```

前台脚本和 systemd 都自动读取；已设置的同名环境变量优先。
只读取指定密钥的字面量 `KEY=value`（可带成对引号），不执行 shell、不展开 $变量，不支持同行注释。
远端没有鉴权则保持 `api_key_env: ""`，不需要该文件。

## 日志与磁盘保护

默认自动写入 `logs/monitor-YYYY-MM-DD.log`（UTF-8），新文件权限为 0600。
包含 Python 业务日志、异常堆栈及 SAIL/FFmpeg 的 stdout/stderr，前台同时显示，无需 `tee`。

- 按盒子本地日期分文件，同一天重启继续追加；达到 `log_max_file_mb` 后轮转为同日分段文件，不覆盖。
- 保留当天及之前 29 天；`log_retention_days` 可设 1–30，不允许超过 30。
- `log_max_total_mb` 限制所有受管日志的合计容量，默认 1024 MiB；超过时优先删除最旧分段。
- `min_free_disk_gb` 检查 `log_dir` 所在文件系统。空间不足时先删最旧受管日志；仍不足则进入紧急模式，暂停 DEBUG/INFO 文件日志，ERROR 仍尝试落盘；空间恢复并留出迟滞余量后自动恢复。
- 启动立即执行清理；运行中跨天轮转，并按 `disk_check_interval_sec` 检查。停机期间不执行清理，下次启动时补做。
- 只管理本日志目录内名称严格匹配 `monitor-YYYY-MM-DD.log` 或 `monitor-YYYY-MM-DD.NNN.log` 的普通文件，不递归、不跟随符号链接，不误删模型、配置及其他日志。
- 旧手工 `tee` 日志、历史图片和模型不在自动清理范围；已过期的受管日志会永久删除，没有回收站。
- 日期依赖盒子系统时间，请校准时钟。容量与余量保护只约束本项目日志，无法清理由其他程序占用的空间，仍应监控整个 `/data`。
- RTSP URL 和已配置 API 密钥会在输出前脱敏；分享日志前仍建议检查其他业务信息。
- INFO 保留启动/停止、每次远端接收和异常；抑制重复 HTTPX 请求日志。每帧入队/编码详情在 DEBUG；替换待传帧仍为 WARNING。
- 连续相同的 SAIL/FFmpeg 原生日志会合并，避免故障时刷爆磁盘。写盘/清理异常向终端或 journal 限频报告，不递归刷屏；不能把无日志视作服务正常。

```bash
tail -F "logs/monitor-$(date +%F).log"
```

跨天后重新运行上述命令以查看新日期文件。正常业务日志示例：

```text
取帧调度已启动 mode=periodic cameras=11 local_save=disabled
远端已接收 camera=camera_01 status=202 bytes=768576 queue_wait=0.003s elapsed=0.058s
```

`queue_wait` 为采样后到开始发送的时间（含本地处理/等待）；`elapsed` 为此次 HTTP 请求耗时，不是模型推理时间。

## 调用行为与故障隔离

每路独立 MultiDecoder 硬件解码、少量最新帧队列、非阻塞读取和后台自动重连。
YOLO/BMCV 处理在一个硬件线程串行执行；压缩 JPEG 才进入独立 HTTP 线程，不跨线程共享 BMImage。
JPEG 通过 `Bmcv.imencode` 调用 Sophon OpenCV，JPU 是否实际启用需在盒子验证。

每路最多一张正在上传和一张最新待传图，新图替换同路待传旧图，保持队列位置。
HTTP 并发默认 2（可设 1–8），超时覆盖整个请求；所有上传名额占满时其他路需排队，但不阻塞取帧/推理。
超时、429、503 等不补发旧图；下一次采样继续，网络恢复后自动恢复。待传图过期或程序退出时丢弃，没有磁盘补发队列。

### 健康检查与自动恢复边界

进程持续检查主调度器心跳、HTTP 上传事件循环、所有 RTSP 监督线程、日志维护线程和原生日志采集线程。

- 摄像头离线/无帧：只把该路标记为降级，后台继续重连，不影响其他路，不重启进程。
- 远端接口超时、拒绝或网络中断：标记远端降级，丢弃当前帧，后续周期继续尝试，不重启进程。
- 取帧以外的本地处理（YOLO、JPEG、提交队列）连续异常达到 `max_consecutive_processing_failures`：进程以失败状态退出，由 systemd 重启；任意一次完整处理成功会把连续计数清零。
- 调度器/上传循环超过 `scheduler_stall_sec` 无心跳，或 RTSP、上传、日志内部线程意外退出：判定致命故障，停止 watchdog 心跳并请求退出。
- 磁盘余量不足：健康状态显示 `degraded: disk`，启用日志紧急模式；本身不重启服务，避免重启风暴。

健康状态会写入项目日志，同时通过 systemd `STATUS` 暴露；状态只含组件名，不包含 RTSP 地址、接口地址或密钥。

接口为 `POST /v1/frames`，multipart 字段：`image`、`machine_id`、`stream_id`、`stream_name`、`captured_at`。
不发送 RTSP 地址；`captured_at` 是盒子读取帧时的带时区 UTC 时间，不是摄像头曝光时间。
`202 + accepted=true` 只表示远端入队，不表示确认火烟或保存证据。没有任务查询，也不等待 SAM3/LLM。
HTTPX 禁用环境代理和重定向，保留 HTTPS 证书校验；HTTP 明文仅适用于可信隔离内网。

## 24×7 后台运行

`deploy/fire-smoke-monitor.service` 默认读取 `config/config.yaml`。
若使用 `config/config.periodic.yaml`，先把服务文件 `ExecStart` 中的配置参数改为该路径，不必覆盖原配置。
服务使用相同启动脚本，并通过 `--no-console` 避免把正常运行日志再复制到 journal。
单元采用 `Type=notify` 和 `WatchdogSec=60`：程序完成硬件/线程初始化后才报告 READY，只有内部组件健康时才发送 watchdog 心跳；进程卡死或停止喂狗后 systemd 会终止并重新启动。
启动前的脚本/配置错误和日志写盘错误仍可查看 journal（其保留期由系统管理）。

```bash
# 先停止之前的前台程序；已有 systemd 服务时先停止它
sudo systemctl stop fire-smoke-monitor
sudo cp deploy/fire-smoke-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now fire-smoke-monitor
sudo systemctl status fire-smoke-monitor --no-pager -l
journalctl -u fire-smoke-monitor -n 50 --no-pager
systemctl show fire-smoke-monitor -p ActiveState -p SubState -p StatusText -p NRestarts
```

修改代码、配置、密钥或 service 文件后，先执行配置检查；service 文件有变化时重新复制并 `daemon-reload`，然后 `sudo systemctl restart fire-smoke-monitor`。
服务异常会重启；停止时关闭解码、取消上传并释放模型，不补发待传帧。远端已接收的图仍可能继续处理。

## 排查与开发

- 启动报缺依赖：使用 `.venv/bin/python -m pip install -r requirements.txt`。
- 缺 `sophon.sail` / `cv2`：检查完整 SAIL 以及 SDK OpenCV，禁止普通 pip OpenCV 覆盖。
- `already running`：先停止已有脚本/systemd 实例；锁会随进程退出释放，不要删除锁文件绕过保护。
- `401`：核对密钥；`400/422`：核对图片/ID/参数；`413`：图片过大。
- 频繁 `429/503`、超时、待传替换：检查远端负载/网络，必要时增大采样间隔。
- 有 202 无证据：查远端模型日志；`GET /health` 只检查流水线服务及累计计数，不验证模型效果。
- 观察盒子：`free -h`、`vmstat 1`、`bm-smi`、`cat /proc/vpuinfo`、`cat /proc/jpuinfo`、`df -h /data`。

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
PYTHONPATH=src .venv/bin/python -m pytest -q
```

测试覆盖上传、调度、离线恢复、日志保留/脱敏、原生 fd 捕获和启动脚本；硬件使用替身，不等于完成真实 BM1684X 联调。

保留的维护资料：

- [YOLO26 模型准备与转换](docs/YOLO26_MODELS.md)
- [SDK 26.03.01 升级记录](docs/BM1684X_SDK_26.03.01_OTA_UPGRADE.md)
- [SAIL 3.11.1 交叉编译记录](docs/SAIL_3.11.1_BM1684X_SDK_26.03.01_CROSS_COMPILE.md)
- [内置 YOLO26 来源与裁剪说明](src/fire_smoke_monitor/vendor/yolo26/README.md)

运行不依赖完整 sophon-demo。只保留其推理类及 BSD-2-Clause 许可证，已移除独立图片/视频 demo、画框保存和 COCO 配色工具。
配置、模型、旧测试数据和历史结果不会被启动脚本清理；测试和升级教程是维护资料，予以保留。
