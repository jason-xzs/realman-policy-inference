# Client 环境配置与恢复

本文把“可从 Python 包管理器安装的依赖”和“必须另外准备的硬件运行时”分开。后者不能由本仓库自动下载：RealMan SDK 可能受供应商分发条款约束，而且当前 active runner 依赖的 `deltawam`/`r3kit` 代码包含已验证工作区补丁。

## 1. 已验证环境快照

`environment/known-good-client.toml` 是 2026-08-30 的 x86_64 Client 快照。关键基线为：

| 项目 | 已验证值 |
| --- | --- |
| OS | Ubuntu 24.04.4 LTS |
| Kernel | Linux 6.8.0-138-generic |
| Python | 3.11.15（conda-forge build） |
| NumPy | 1.26.4 |
| OpenCV | 4.10.0.84 |
| Pillow | 12.3.0 |
| PyYAML | 6.0.3 |
| pyrealsense2 | 2.58.1.10581 |
| openpi-client | 0.1.0，来自匹配的本地 OpenPI source snapshot |
| websockets | 17.0.1 |
| FFmpeg | 6.1.1-3ubuntu5，包含 `h264_vaapi` |
| VAAPI render node | `/dev/dri/renderD128`，Client 用户可读写 |

lock 还记录所有相关 Python distribution、`deltawam`/`r3kit` revision 和关键外部文件 SHA-256。它表示“已验证相等”，不是对更新版本兼容性的推断。

## 2. 推荐目录布局

以下布局与仓库默认配置一致；换位置时同步修改 `sources.deltawam_src`、录像路径和启动命令即可：

```text
/home/robot2/
├── deltawam/
│   └── src/realman_deployment/
│       ├── client.py
│       ├── hardware.py
│       ├── protocol.py
│       ├── Robotic_Arm/
│       └── r3kit/
└── jhk/phywam-mope-real/
    ├── envs/robot-client-py311/
    ├── real-records/
    └── robot-client-runtime/
```

克隆本仓库：

```bash
mkdir -p /home/robot2/jhk/phywam-mope-real
cd /home/robot2/jhk/phywam-mope-real
git clone git@github.com:Ginger-Kay/Dual-Realman-Client-Runtime.git \
  robot-client-runtime
```

如果使用了自定义 GitHub SSH host alias，可把 `github.com` 换为本机 alias。

## 3. 恢复外部硬件源码

### 3.1 deltawam 基线和补丁

先恢复 lock 中的基线 revision，再应用仓库保存的 active runtime 补丁：

```bash
CLIENT_ROOT=/home/robot2/jhk/phywam-mope-real/robot-client-runtime
DELTAWAM_ROOT=/home/robot2/deltawam

git clone --branch realman git@github.com:coldyan/deltawam.git "$DELTAWAM_ROOT"
git -C "$DELTAWAM_ROOT" checkout 9a7389356c83e0da90af88cc7d0e1c706dc16baa
git -C "$DELTAWAM_ROOT" apply --check \
  "$CLIENT_ROOT/environment/patches/deltawam-runtime.patch"
git -C "$DELTAWAM_ROOT" apply \
  "$CLIENT_ROOT/environment/patches/deltawam-runtime.patch"
```

补丁只覆盖 active runner 使用的 `realman_deployment/client.py` 与 `hardware.py`。不要跳过 `--check`，也不要在来源不明或已有冲突修改的 checkout 上强行应用。

### 3.2 r3kit 基线和补丁

当前 `deltawam` revision 不跟踪嵌套的 `r3kit` 目录，需要单独恢复：

```bash
R3KIT_ROOT=/home/robot2/deltawam/src/realman_deployment/r3kit

git clone https://gitee.com/LiJiahang617/r3kit "$R3KIT_ROOT"
git -C "$R3KIT_ROOT" checkout 7fad4962c65c353694c1d213a07a9852fd9e43f0
git -C "$R3KIT_ROOT" apply --check \
  "$CLIENT_ROOT/environment/patches/r3kit-realsense.patch"
git -C "$R3KIT_ROOT" apply \
  "$CLIENT_ROOT/environment/patches/r3kit-realsense.patch"
```

这组补丁固定 D435 的 `640x480 BGR8 @ 30 Hz` 路径及调用接口。不要用高分辨率录制分支替换它后仍宣称输入分布相同。

### 3.3 RealMan SDK

从实验室批准的 RealMan Python SDK bundle 或当前已验证 Client 备份恢复：

```text
/home/robot2/deltawam/src/realman_deployment/Robotic_Arm/
├── __init__.py
├── rm_ctypes_wrap.py
├── rm_robot_interface.py
└── libs/
    ├── linux_arm/libapi_c.so
    └── linux_x86/libapi_c.so
```

本仓库不重新分发这些供应商文件。x86_64 已验证运行使用 `libs/linux_x86/libapi_c.so`；ARM 文件存在不代表 ARM Client 已验证。离线预检会对当前 x86_64 所需的 wrapper 和 `.so` 做 SHA-256 比对。

## 4. Python 环境

### 4.1 继续使用当前环境

当前已验证解释器为：

```bash
/home/robot2/jhk/phywam-mope-real/envs/robot-client-py311/bin/python -VV
```

这次仓库整理不安装、不升级也不重装任何包。以后在同一台 Client 上优先直接复用该解释器，并先运行离线预检。

### 4.2 新机器创建环境

只有在新机器或原环境确实丢失时才执行以下步骤。`.[client]` 安装 PyPI 可获得的已验证直接依赖：

```bash
python3.11 -m venv /home/robot2/jhk/phywam-mope-real/envs/robot-client-py311
source /home/robot2/jhk/phywam-mope-real/envs/robot-client-py311/bin/activate
python -m pip install --upgrade pip setuptools wheel
cd /home/robot2/jhk/phywam-mope-real/robot-client-runtime
python -m pip install -e '.[client]'
```

`openpi-client==0.1.0` 的当前安装元数据指向本地 source snapshot，不能把同名的新版本自动视为等价。应从与服务端匹配的 OpenPI/pi05-service source 安装其 `packages/openpi-client`（实际目录以服务端仓库为准）：

```bash
python -m pip install /path/to/matching-openpi-source/packages/openpi-client
python -m pip show openpi-client
```

不要为消除版本告警直接执行无约束升级。若明确验证了新版本，应同时更新测试、lock 和说明。

## 5. 系统与设备前置条件

### FFmpeg/VAAPI

启用录像的 preset 需要：

```bash
ffmpeg -version
ffmpeg -hide_banner -encoders | grep h264_vaapi
ls -l /dev/dri/renderD128
test -r /dev/dri/renderD128 && test -w /dev/dri/renderD128
```

如果 render node 不可访问，应按本机发行版配置 `render`/ACL 权限，然后重新登录。不要用 `sudo` 启动整个机器人 Client 来绕过权限。

### RealSense

设备枚举是硬件读取步骤，不属于离线预检。操作员准备好三路相机后手工执行：

```bash
rs-enumerate-devices -s
```

确认三个 serial 与 preset 一一对应，并确认每路可提供 `640x480 BGR8 @ 30 Hz`。换相机后必须更新 local config，不要只按 USB 插口猜测 role。

### 双臂网络

Client 主机需要同时具备配置中的本地 UDP 地址，并能访问左右臂控制器地址。先做只读网络检查：

```bash
ip -br address
ip route
```

不要因为 ping 不通就自动修改机械臂、主机网卡或现有路由；先核对实验室网络方案。

## 6. 配置本机路径

复制 tracked baseline 为本地文件：

```bash
cd /home/robot2/jhk/phywam-mope-real/robot-client-runtime
cp configs/pi05_realman_presets.toml configs/pi05_realman_presets.local.toml
```

至少检查：

- `[default.sources].deltawam_src`；
- `[default.recording].root_dir` 与 `.cache`；
- 三个 camera serial；
- 左右臂 IP、port、local IP 和 UDP port；
- policy tunnel endpoint；
- 每个 preset 的服务 identity、prompt、录像目录和 `execution.mode`。

local config 已被 `.gitignore` 排除。详细字段见 [configuration.md](configuration.md)。

## 7. 离线预检

```bash
cd /home/robot2/jhk/phywam-mope-real/robot-client-runtime
PYTHONDONTWRITEBYTECODE=1 \
/home/robot2/jhk/phywam-mope-real/envs/robot-client-py311/bin/python -B \
scripts/check_pi05_environment.py \
--config configs/pi05_realman_presets.local.toml \
--preset pi05-organize-15000
```

它会检查：

- Python 与已安装 distribution 版本；
- 必需 import 是否可解析；
- config schema；
- `deltawam`/`r3kit` revision；
- active 外部文件和 RealMan SDK SHA-256；
- 录像启用时 FFmpeg、`h264_vaapi`、render node 与目标父目录权限。

它明确不会连接网络、探测 policy 服务、枚举/打开相机、加载/连接 RealMan SDK或发送运动命令。`passed` 只表示本地静态环境与 lock 一致，不表示硬件或部署 ready。

版本或指纹有意更新后，可临时用 `--allow-version-drift` 将 drift 降为 warning；这个选项不证明兼容，也不应作为 execute 的自动放行条件。

## 8. 环境升级流程

升级任一 Python 包、SDK、`deltawam`、`r3kit` 或 FFmpeg 时：

1. 在独立环境/分支完成离线测试。
2. 更新并验证外部 patch，不能只改 lock 让检查变绿。
3. 运行完整 unittest、compile 和 `git diff --check`。
4. 先做 offline preflight，再由操作员单独批准设备只读/no-motion 验证。
5. 只有服务 identity、相机 profile、双臂状态和安全边界都重新确认后，才另行批准 execute。
