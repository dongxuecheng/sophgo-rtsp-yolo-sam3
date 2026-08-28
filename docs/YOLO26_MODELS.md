# YOLO26 模型准备、训练与 BM1684X 转换

只有 `yolo_trigger` 模式需要本教程；`periodic` 不需要本地模型。
运行环境和启动方法见主 README。已有可用模型时，不必重新下载或转换。

## 1. 用官方 COCO 模型验证 YOLO 上传链路

以下操作在盒子上执行，文件仍放在本项目内：

```bash
export PROJECT_ROOT=/data/sophgo-rtsp-yolo-sam3
cd "${PROJECT_ROOT}"
mkdir -p models/yolo26 downloads

"${PROJECT_ROOT}/.venv/bin/python" -m pip install dfss
cd "${PROJECT_ROOT}/downloads"
"${PROJECT_ROOT}/.venv/bin/python" -m dfss \
  --url=open@sophgo.com:sophon-demo/YOLO26_det/BM1684X.tar.gz
tar xf BM1684X.tar.gz
cp BM1684X/yolo26s_fp16_1b.bmodel "${PROJECT_ROOT}/models/yolo26/"

"${PROJECT_ROOT}/.venv/bin/python" -m dfss \
  --url=open@sophgo.com:sophon-demo/common/coco.names
cp coco.names "${PROJECT_ROOT}/models/yolo26/"
cd "${PROJECT_ROOT}"
cp -n config/config.example.yaml config/config.yolo.yaml
```

编辑 `config/config.yolo.yaml`，填写实际接口地址和 RTSP，同时修改：

```yaml
mode: yolo_trigger
models:
  yolo26:
    bmodel: ../models/yolo26/yolo26s_fp16_1b.bmodel
    class_names: ../models/yolo26/coco.names
    target_classes: [person]
    conf_threshold: 0.25
    interval_sec: 3
```

```bash
bash start.sh config/config.yolo.yaml --check-config
bash start.sh config/config.yolo.yaml
```

COCO 不包含火焰/烟雾类别；这里仅验证“人触发上传”。
远端仍只确认火烟，所以接收成功也可能不保存证据。
已移除内置运行文件中的独立图片 demo、画框和保存入口，不再运行其 `python -m` 命令。
原有测试图片、下载归档和旧结果是用户数据，不由启动脚本删除。

## 2. 火焰/烟雾模型训练与转换

官方COCO模型只能验证部署链路。`yolo_trigger` 用于火烟生产检测时，需训练火焰/烟雾专用模型；YOLO 漏检的帧不会进入远端复核。`periodic` 不依赖这一步。训练和模型转换应在有GPU、大内存和TPU-MLIR环境的x86工作站完成。

### 2.1 数据集与训练

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

### 2.2 导出ONNX

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
model.export(format="onnx", opset=13, imgsz=640, batch=1)
```

### 2.3 转换BM1684X BModel

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


## 3. 第三方来源

盒子运行只使用内置的 YOLO26 推理类，不依赖完整 sophon-demo 仓库。
参考实现固定于 [sophon-demo dfe0dff](https://github.com/sophgo/sophon-demo/tree/dfe0dff31c21984b681536df9cb2b1ba6c96a8a4/sample/YOLO26)。
模型转换章节中的工作站路径是示例，需替换为你的实际工作目录。
Ultralytics、模型权重、数据集与内置代码分别遵循各自许可证。
