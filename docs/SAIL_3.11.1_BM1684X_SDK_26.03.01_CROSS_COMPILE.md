# SAIL v3.11.1 面向 BM1684X SDK v26.03.01 的交叉编译记录

本文记录一次已经在 BM1684X SoC 盒子上验证通过的 SAIL Python wheel 交叉编译过程，供重装系统、部署第二台设备或以后重新构建时复用。

最终产物为：

```text
sophon_arm-3.11.1-py3-none-any.whl
└── sophon/sail.cpython-38-aarch64-linux-gnu.so
```

该 wheel 已在目标机完成以下验证：

- SAIL 包版本为 3.11.1；
- 原生扩展为 CPython 3.8、AArch64；
- `Handle`、`Engine`、`BMImage`、`Bmcv`、`MultiDecoder` 均存在；
- `BMImage.asmat` 存在；
- `sail.Handle(0).get_target()` 返回 `BM1684X`；
- `MultiDecoder` 可以正常创建和销毁。

> 本文只适用于 BM1684X **SoC 模式**。PCIe 卡、BM1688、不同 Python 版本或不同 SDK 版本不能直接照搬产物。

## 1. 为什么选择 SAIL v3.11.1

算能官方 [SAIL v3.11.1 Release Notes](https://github.com/sophgo/sophon-sail/releases/tag/v3.11.1) 明确列出：

- 新增 BM1684X SDK v26.03.01 支持；
- 修复 `MultiDecoder` 内存泄漏；
- 修复 `BMImage.asnumpy` 内存泄漏；
- 修复 `cvwrapper convert_format` 内存泄漏。

本项目需要 24×7 读取多路 RTSP，并使用 `MultiDecoder`、`BMImage`、`Bmcv` 和 `Engine`，因此 3.11.1 比 SDK 压缩包内附带的 3.11.0 更适合。

SDK v26.03.01 自带的 `sophon-sail_3.11.0.tar.gz` 虽然含有许多预编译 wheel，但没有发现同时匹配下列组合的现成 wheel：

```text
BM1684X SoC + SDK v26.03.01 + libsophon 0.5.3
+ sophon-mw 0.15.0 + Python 3.8 + AArch64
```

因此本次从 SAIL v3.11.1 源码交叉编译。

## 2. 已验证的环境矩阵

| 项目 | 已验证版本 |
|---|---|
| 构建宿主机 | x86_64，Ubuntu 24.04.2 LTS |
| Docker | 29.2.1 |
| 构建容器 | Ubuntu 20.04 LTS |
| 容器 Python | 3.8.10 |
| 容器 CMake | 3.16.3 |
| AArch64 GCC/G++ | 9.4.0 |
| SAIL 源码 | v3.11.1 |
| 目标芯片/模式 | BM1684X SoC，AArch64 |
| 目标系统 | Ubuntu 20.04 LTS |
| 目标 Python | 3.8.10 |
| 目标 SDK | v26.03.01 |
| libsophon | 0.5.3 |
| sophon-ffmpeg | 0.15.0 |
| sophon-opencv | 0.15.0 |

选择 Ubuntu 20.04 构建容器的主要目的是获得与目标机一致的 Python 3.8 和较接近的 glibc/C++ 用户态环境。宿主机继续使用 Ubuntu 24.04 没有问题，因为真正的编译工具运行在容器内。

## 3. 必须遵守的兼容原则

重新编译前，先确认以下五项：

1. SAIL 版本明确支持目标 SDK；
2. sysroot 中的 libsophon 和 sophon-mw 与盒子上的版本完全一致；
3. Python 主次版本一致，本次必须都是 CPython 3.8；
4. BM1684X 盒子使用 `BUILD_TYPE=soc`，不能误用 PCIe 构建；
5. 本项目需要视频解码和图像处理，不能使用 `ONLY_RUNTIME=ON`。

`ONLY_RUNTIME=ON` 会排除项目需要的 Decoder、MultiDecoder、Bmcv、FFmpeg 和 OpenCV 等多媒体能力。

## 4. SDK 中使用的交叉编译依赖

完整 SDK v26.03.01 解压在构建服务器的：

```text
/mnt/dxc/sophon/release
```

本次使用的两个 AArch64 SoC 依赖包是：

```text
/mnt/dxc/sophon/release/sophon-img_20260414_154148/
  libsophon_soc_0.5.3_aarch64.tar.gz

/mnt/dxc/sophon/release/sophon-mw_20260414_035450/
  sophon-mw-soc_0.15.0_aarch64.tar.gz
```

注意必须使用带 `soc` 的包。不要换成 x86_64、普通 PCIe AArch64、LoongArch64 或 RISC-V 包。

## 5. 在 x86 宿主机准备工作目录

下面命令在 x86 构建服务器执行。变量名可以修改，但后续命令应保持一致。

```bash
export SOPHON_RELEASE_ROOT=/mnt/dxc/sophon/release
export SAIL_BUILD_ROOT=/mnt/dxc/sophon/build/sail-3.11.1-sdk26

mkdir -p "${SAIL_BUILD_ROOT}/sysroot"
```

获取固定版本源码：

```bash
git clone --depth 1 --branch v3.11.1 \
  https://github.com/sophgo/sophon-sail.git \
  "${SAIL_BUILD_ROOT}/sophon-sail"

git -C "${SAIL_BUILD_ROOT}/sophon-sail" describe \
  --tags --always --dirty
```

预期版本为 `v3.11.1`。正式留档时还可以记录提交：

```bash
git -C "${SAIL_BUILD_ROOT}/sophon-sail" rev-parse HEAD
```

解压匹配 SDK v26.03.01 的 sysroot：

```bash
tar xzf \
  "${SOPHON_RELEASE_ROOT}/sophon-img_20260414_154148/libsophon_soc_0.5.3_aarch64.tar.gz" \
  -C "${SAIL_BUILD_ROOT}/sysroot"

tar xzf \
  "${SOPHON_RELEASE_ROOT}/sophon-mw_20260414_035450/sophon-mw-soc_0.15.0_aarch64.tar.gz" \
  -C "${SAIL_BUILD_ROOT}/sysroot"
```

检查关键目录：

```bash
test -d "${SAIL_BUILD_ROOT}/sysroot/libsophon_soc_0.5.3_aarch64/opt/sophon/libsophon-0.5.3"
test -d "${SAIL_BUILD_ROOT}/sysroot/sophon-mw-soc_0.15.0_aarch64/opt/sophon/sophon-ffmpeg_0.15.0"
test -d "${SAIL_BUILD_ROOT}/sysroot/sophon-mw-soc_0.15.0_aarch64/opt/sophon/sophon-opencv_0.15.0"
```

三个命令都应返回退出码 0。

## 6. 启动 Ubuntu 20.04 构建容器

```bash
docker run --rm -it \
  --name sail-3.11.1-sdk26-build \
  -v "${SAIL_BUILD_ROOT}:/work" \
  -w /work \
  ubuntu:20.04 bash
```

`--rm` 只删除退出后的容器，源码、sysroot、build 目录和最终 wheel 都位于宿主机绑定的 `SAIL_BUILD_ROOT` 中，不会丢失。

## 7. 在容器内安装构建工具

以下命令均在容器内执行：

```bash
apt-get update

DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  cmake \
  git \
  pkg-config \
  unzip \
  python3 \
  python3-dev \
  python3-pip \
  python3-setuptools \
  python3-wheel \
  gcc-aarch64-linux-gnu \
  g++-aarch64-linux-gnu
```

如果软件源使用了不可用的 HTTP 镜像，可以先改为可访问的 HTTPS 镜像再重新执行 `apt-get update`。这只是网络源调整，不影响编译参数。

检查工具版本：

```bash
python3 -V
cmake --version
aarch64-linux-gnu-gcc --version
aarch64-linux-gnu-g++ --version
```

本次输出的核心版本为：

```text
Python 3.8.10
cmake version 3.16.3
aarch64-linux-gnu-gcc/g++ 9.4.0
```

## 8. 配置 SAIL

进入源码目录：

```bash
cd /work/sophon-sail
```

为了使下面命令同时适用于全新构建和曾经配置失败的构建，使用 `-U CUSTOM_PY_LIBDIR` 主动删除 CMake 缓存中的错误值：

```bash
cmake -S . -B build \
  -U CUSTOM_PY_LIBDIR \
  -DBUILD_TYPE=soc \
  -DONLY_RUNTIME=OFF \
  -DBUILD_PYSAIL=ON \
  -DCMAKE_TOOLCHAIN_FILE=/work/sophon-sail/cmake/BM168x_SOC/ToolChain_aarch64_linux.cmake \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DLIBSOPHON_BASIC_PATH=/work/sysroot/libsophon_soc_0.5.3_aarch64/opt/sophon/libsophon-0.5.3 \
  -DFFMPEG_BASIC_PATH=/work/sysroot/sophon-mw-soc_0.15.0_aarch64/opt/sophon/sophon-ffmpeg_0.15.0 \
  -DOPENCV_BASIC_PATH=/work/sysroot/sophon-mw-soc_0.15.0_aarch64/opt/sophon/sophon-opencv_0.15.0
```

检查配置结果：

```bash
grep '^CUSTOM_PY_LIBDIR' build/CMakeCache.txt \
  || echo 'CUSTOM_PY_LIBDIR cleared'
```

本次应输出：

```text
CUSTOM_PY_LIBDIR cleared
```

同时应在 CMake 输出中确认：

```text
CMAKE_C_COMPILER: aarch64-linux-gnu-gcc
CMAKE_CXX_COMPILER: aarch64-linux-gnu-g++
CROSS_COMPILE: ON
WITH_OPENCV: ON
WITH_FFMPEG: ON
WITH_BMCV: ON
SUFFIX: .cpython-38-aarch64-linux-gnu.so
```

### 为什么不能设置本机 x86 Python 库目录

第一次配置曾错误加入：

```bash
-DCUSTOM_PY_LIBDIR=/usr/lib/x86_64-linux-gnu
```

这是 x86_64 容器的 Python 动态库目录，不是 AArch64 Python sysroot。随后编译出现：

```text
/usr/include/python3.8/pyconfig.h:9:12: fatal error:
aarch64-linux-gnu/python3.8/pyconfig.h: No such file or directory
```

处理方法不是复制或伪造头文件，而是清除错误的 `CUSTOM_PY_LIBDIR` 并重新配置。对于本文这套 SAIL v3.11.1/Python 3.8 构建环境，直接省略该参数即可。

只有在另外准备了完整且匹配目标架构的 AArch64 Python sysroot 时，才可以把 `CUSTOM_PY_LIBDIR` 指向那个 **AArch64** Python 的库目录；绝不能指向 x86_64 目录。

## 9. 编译 Python SAIL 扩展

```bash
cmake --build build \
  --target pysail \
  --parallel "$(nproc)"
```

成功标志：

```text
[100%] Linking CXX shared library ../lib/sail.cpython-38-aarch64-linux-gnu.so
[100%] Built target pysail
```

链接阶段可能反复出现以下警告：

```text
.dynsym local symbol at index ... (>= sh_info of 3)
```

这些警告来自 SDK 中预编译的 `libbmcv.so` 或 Sophon OpenCV 动态库。本次构建中它们没有阻止链接。判断成功与否应以以下条件为准：

- 命令退出码为 0；
- 最终出现 `[100%] Built target pysail`；
- 生成的 `.so` 经 `file` 检查为 AArch64；
- wheel 在目标 BM1684X 上可以导入并完成硬件验证。

若没有出现最终成功标志，则不能把警告当作可忽略错误。

## 10. 打包 wheel

```bash
cd /work/sophon-sail/python
chmod +x sophon_whl.sh
./sophon_whl.sh
```

本次生成：

```text
/work/sophon-sail/python/dist/sophon_arm-3.11.1-py3-none-any.whl
```

如果打包最后出现：

```text
cp: cannot stat '../hooks/*': No such file or directory
```

这是脚本尝试向 Git 仓库安装开发 hooks，而浅克隆/发布源码中没有该目录。本次 wheel 已在此消息之前成功创建，因此该消息不影响运行产物。仍需继续执行下一节的结构检查，不能只看文件名判断成功。

## 11. 在构建服务器检查产物

检查原生扩展架构：

```bash
file /work/sophon-sail/build/lib/sail*.so
```

预期包含：

```text
ELF 64-bit LSB shared object, ARM aarch64
```

检查 wheel 内部文件：

```bash
unzip -l \
  /work/sophon-sail/python/dist/sophon_arm-3.11.1-py3-none-any.whl \
  | grep 'sophon/sail'
```

本次实际得到：

```text
5511664  sophon/sail.cpython-38-aarch64-linux-gnu.so
117431   sophon/sail.pyi
```

虽然外层 wheel 文件名是 `py3-none-any`，其中实际包含 AArch64/CPython 3.8 原生模块。该 wheel 不能安装到 x86，也不能用于 Python 3.9、3.10、3.11 或 3.12。

在宿主机保存校验值和副本：

```bash
mkdir -p "${SAIL_BUILD_ROOT}/artifacts"

cp "${SAIL_BUILD_ROOT}/sophon-sail/python/dist/sophon_arm-3.11.1-py3-none-any.whl" \
  "${SAIL_BUILD_ROOT}/artifacts/"

sha256sum \
  "${SAIL_BUILD_ROOT}/artifacts/sophon_arm-3.11.1-py3-none-any.whl" \
  | tee "${SAIL_BUILD_ROOT}/artifacts/sophon_arm-3.11.1-py3-none-any.whl.sha256"
```

建议把 `artifacts` 目录单独备份。以后给相同 SDK、相同 Python 版本的第二台盒子安装时，不需要重复编译。

## 12. 复制并安装到 BM1684X 盒子

在构建服务器执行，替换设备 IP：

```bash
scp \
  "${SAIL_BUILD_ROOT}/artifacts/sophon_arm-3.11.1-py3-none-any.whl" \
  linaro@BM1684X_IP:/data/
```

登录盒子后先核对运行环境：

```bash
bm_version
python3 -V
uname -m
```

必须确认：

```text
SophonSDK version: v26.03.01
Python 3.8.x
aarch64
```

安装 wheel：

```bash
sudo python3 -m pip install \
  --no-deps \
  --force-reinstall \
  /data/sophon_arm-3.11.1-py3-none-any.whl
```

`--no-deps` 用于避免 pip 下载普通 OpenCV、NumPy 或其他包覆盖 SDK 已验证的系统组件。

Sophon OpenCV 0.15.0 的 Python 模块实测位于：

```text
/opt/sophon/sophon-opencv-latest/opencv-python/cv2.so
```

当前 shell 先设置：

```bash
export PYTHONPATH=/opt/sophon/sophon-opencv-latest/opencv-python${PYTHONPATH:+:$PYTHONPATH}
```

## 13. 目标机最终验证

```bash
python3 - <<'PY'
from importlib.metadata import version
import cv2
import sophon.sail as sail

print("package:", version("sophon-arm"))
print("sail file:", sail.__file__)
print("opencv:", cv2.__version__, cv2.__file__)

for name in ("Handle", "Engine", "BMImage", "Bmcv", "MultiDecoder"):
    print(name, hasattr(sail, name))

print("BMImage.asmat:", hasattr(sail.BMImage, "asmat"))

handle = sail.Handle(0)
print("target:", handle.get_target())

decoder = sail.MultiDecoder(2, 0, 1)
decoder.set_read_timeout(3)
print("MultiDecoder init: OK")

del decoder
print("MultiDecoder destroy: OK")
PY
```

本次实测输出：

```text
package: 3.11.1
sail file: /usr/local/lib/python3.8/dist-packages/sophon/sail.cpython-38-aarch64-linux-gnu.so
opencv: 4.1.0 /opt/sophon/sophon-opencv-latest/opencv-python/cv2.so
Handle True
Engine True
BMImage True
Bmcv True
MultiDecoder True
BMImage.asmat: True
target: BM1684X
MultiDecoder init: OK
MultiDecoder destroy: OK
```

这里的 Sophon OpenCV 软件包版本是 0.15.0，而 Python `cv2.__version__` 显示其 OpenCV 上游版本 4.1.0，两者并不矛盾。

## 14. 常见问题

### 14.1 找不到 AArch64 `pyconfig.h`

症状：

```text
fatal error: aarch64-linux-gnu/python3.8/pyconfig.h: No such file or directory
```

检查：

```bash
grep '^CUSTOM_PY_LIBDIR' build/CMakeCache.txt
```

若仍是 `/usr/lib/x86_64-linux-gnu`，重新执行第 8 节带 `-U CUSTOM_PY_LIBDIR` 的完整 CMake 命令。

### 14.2 wheel 能安装但不能导入

依次检查：

```bash
python3 -V
uname -m
python3 -m pip show sophon-arm
file /usr/local/lib/python3.8/dist-packages/sophon/sail*.so
ldd /usr/local/lib/python3.8/dist-packages/sophon/sail*.so
```

常见原因包括 Python 版本不是 3.8、目标不是 AArch64、SDK 运行库版本不匹配，或登录环境没有加载 `/opt/sophon` 的动态库配置。

### 14.3 `import cv2` 失败

SAIL wheel 不负责安装 Sophon OpenCV Python 模块。设置：

```bash
export PYTHONPATH=/opt/sophon/sophon-opencv-latest/opencv-python${PYTHONPATH:+:$PYTHONPATH}
```

不要安装 PyPI 的普通 `opencv-python` 来规避该问题。

### 14.4 `ONLY_RUNTIME=ON` 导致接口缺失

如果 `MultiDecoder`、`Bmcv` 或视频相关接口不存在，检查 CMake 配置。为本项目重新编译时必须使用：

```text
BUILD_TYPE=soc
ONLY_RUNTIME=OFF
WITH_OPENCV=ON
WITH_FFMPEG=ON
WITH_BMCV=ON
```

后三项由匹配的 `OPENCV_BASIC_PATH`、`FFMPEG_BASIC_PATH` 和 `LIBSOPHON_BASIC_PATH` 启用。

## 15. 重新编译检查清单

- [ ] 目标机仍是 BM1684X SoC、AArch64
- [ ] 目标机仍是 Python 3.8
- [ ] `bm_version` 仍显示 SDK v26.03.01
- [ ] libsophon 仍是 0.5.3
- [ ] sophon-ffmpeg 和 sophon-opencv 仍是 0.15.0
- [ ] SAIL 源码固定到 v3.11.1
- [ ] 使用 `libsophon_soc` 和 `sophon-mw-soc` 包
- [ ] Ubuntu 20.04 容器内具备 AArch64 GCC/G++ 9.4
- [ ] `BUILD_TYPE=soc`
- [ ] `ONLY_RUNTIME=OFF`
- [ ] 没有把 x86_64 路径传给 `CUSTOM_PY_LIBDIR`
- [ ] 产物内部是 `cpython-38-aarch64-linux-gnu.so`
- [ ] 保存 wheel 的 SHA-256
- [ ] 在目标机验证 `Handle`、`Engine`、`BMImage`、`Bmcv`、`MultiDecoder` 和真实 RTSP

## 16. 官方参考

- [SAIL v3.11.1 Release Notes](https://github.com/sophgo/sophon-sail/releases/tag/v3.11.1)
- [SAIL v3.11.1 Python 编译与安装文档](https://github.com/sophgo/sophon-sail/blob/v3.11.1/docs/source_zh/3_build/3.1_build_python.rst)
- [SAIL v3.11.1 源码](https://github.com/sophgo/sophon-sail/tree/v3.11.1)

