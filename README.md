# RealMan Policy Inference

在 RealMan RM75 工作站运行 FastWAM、π0.5 和 LingBot-VA 策略。机器人端读取相机和关节状态，调用 GPU 服务器推理，再执行返回的动作。

| 策略 | 机器人 | 相机 | 服务连接 |
| --- | --- | --- | --- |
| FastWAM | 双臂 | 顶部 + 左腕 + 右腕 | HTTP；本仓库提供服务端 |
| π0.5 / OpenPI | 双臂 | 顶部 + 左腕 + 右腕 | OpenPI WebSocket |
| LingBot-VA | 左臂 | 顶部 + 左腕 | LingBot RealMan WebSocket |

支持动作块对齐、异步预取、后台相机采集和录像。VR 遥操作与训练数据采集见 [realman-vr-teleop](https://github.com/jason-xzs/realman-vr-teleop)。

## 安装机器人端

需要 Ubuntu x86_64、Python 3.11 或 3.12。模型推理在 GPU 服务器运行，机器人端不需要安装 CUDA。

```bash
git clone https://github.com/jason-xzs/realman-policy-inference.git
cd realman-policy-inference
bash scripts/setup_client.sh
```

安装脚本创建 `.venv`，安装 SDK 配套依赖与 OpenPI 客户端。后续启动脚本自动使用这个环境。

已有 Conda 环境时可以指定其中的 Python，例如：

```bash
conda create -n realman python=3.11 -y
conda activate realman
PYTHON_BIN=python bash scripts/setup_client.sh
```

## 连接工作站

默认设备配置与 VR 采集仓库一致：

| 设备 | 地址 / 序列号 |
| --- | --- |
| 左臂 | `192.168.1.18:8080`，主机网口 `enx6c1ff7eccd8e` |
| 右臂 | `192.168.1.19:8080`，主机网口 `enp89s0` |
| 顶部相机 | `420122071413` |
| 左腕相机 | `323622272614` |
| 右腕相机 | `230422272673` |

机械臂上电、相机接 USB 3 后配置网络：

```bash
bash scripts/setup_network.sh
ping -c 2 192.168.1.18
ping -c 2 192.168.1.19
```

更换网卡时，使用 `LEFT_IFACE`、`RIGHT_IFACE` 指定网口。开始推理前停止 VR 采集和其他占用相机、机械臂的程序。执行动作前确认工作空间和初始位姿，保持急停可触达。

## FastWAM

### GPU 服务器

准备 RealMan 微调权重、训练导出的 `config.yaml`、`dataset_stats.json`，以及配置引用的模型基础权重。默认查找结构为：

```text
run/
├── config.yaml
├── dataset_stats.json
└── checkpoints/weights/step_015000.pt
```

在 GPU 服务器克隆本仓库，激活模型环境。仓库依赖版本使用 CUDA 12.8 / PyTorch 2.7.1：

```bash
pip install -e fastwam --extra-index-url https://download.pytorch.org/whl/cu128
CHECKPOINT=/absolute/path/run/checkpoints/weights/step_015000.pt \
  HOST=127.0.0.1 bash fastwam/scripts/tasks/serve_realman_fastwam.sh
```

在机器人主机另开终端，保持 SSH 转发连接。将 `GPU_HOST` 替换为自己的 SSH 主机名或 `user@address`：

```bash
ssh -N -L 18000:127.0.0.1:8000 GPU_HOST
```

### 机器人主机

先读取真实设备并检查模型输出：

```bash
bash scripts/run_fastwam_client.sh --max-replans 2
```

确认相机和服务输出正常后，执行任务：

```bash
EXECUTE=1 bash scripts/run_fastwam_client.sh \
  --instruction 'Place the three objects into the blue container.' \
  --align-action-chunk
```

默认连接 `http://127.0.0.1:18000`，不录像。`Ctrl+C` 停止。使用其他服务地址时设置 `SERVER_URL`。

## π0.5 / OpenPI

GPU 服务器需运行经过 RealMan 数据适配的 OpenPI 模型服务，监听 `8020`。服务端需与任务配置中的 checkpoint、输入输出维度匹配；服务启动与接口要求见 [模型服务配置](docs/model_services.md#π05--openpi)。

机器人主机建立 SSH 转发：

```bash
ssh -N -L 18200:127.0.0.1:8020 GPU_HOST
```

运行任务：

```bash
bash scripts/run_pi05_client.sh --preset pi05-organize-15000
```

首次运行自动生成 `pi05/configs/pi05_realman_presets.local.toml`，并填入当前仓库路径。默认只采集观测、请求一次推理，**不发送运动命令**。

| 任务 preset | 任务 |
| --- | --- |
| `pi05-organize-15000` | 将物品放入蓝色容器 |
| `pi05-sugar-15000` | 将小方块放入杯子 |
| `pi05-bowls-15000` | 叠放纸碗 |

执行任务时，编辑本地 TOML：

```toml
[default.execution]
mode = "execute"
```

保留该段的其他配置项，再运行相同命令。按终端提示确认启动和归位；`Ctrl+C` 停止。更换权重时，更新对应任务段的 `expected_config`、`expected_checkpoint` 和 `prompt`，让它们与服务器实际返回的 metadata 一致。

## LingBot-VA

使用左臂和顶部、左腕两路相机。GPU 服务器需提供 `realman_va_socket_v1` 协议的 LingBot-VA 服务，接口见 [模型服务配置](docs/model_services.md#lingbot-va)。

```bash
ssh -N -L 18080:127.0.0.1:18080 GPU_HOST
```

先检查相机和服务输出：

```bash
bash scripts/run_lingbot_client.sh \
  --task-text 'Place the brown cup in the white basin.' \
  --dry-run --max-steps 2
```

执行任务，并让下一次推理与当前动作块执行重叠：

```bash
bash scripts/run_lingbot_client.sh \
  --task-text 'Place the brown cup in the white basin.' \
  --enable-robot --use-canfd \
  --mode async --disable-kv-cache --async-prefetch-after-actions 8
```

执行模式默认先归位并打开夹爪。任务结束按 `Ctrl+C`。启动脚本已绑定当前工作站的顶部、左腕相机，并加载 [动作统计](configs/lingbot/action_norm_stats.json)；使用其他训练权重时，通过 `REAL_ACTION_STATS_PATH` 指向匹配的统计文件。

## 常用选项

| 场景 | 选项 |
| --- | --- |
| FastWAM 换相机 | `HIGH_CAMERA_ID`、`LEFT_CAMERA_ID`、`RIGHT_CAMERA_ID` |
| FastWAM 跳过过时动作 | `--align-action-chunk --action-align-max-skip 8` |
| FastWAM 等待实际关节到位 | `--tracking-gate --tracking-error-rad 0.05` |
| FastWAM 录像 | `--record-dir recordings/task_001 --record-codec MJPG`；Intel 硬件编码可用 `h264_vaapi` |
| π0.5 配置相机、速度、服务或录像 | 编辑自动生成的 `pi05_realman_presets.local.toml` |
| LingBot 换相机 | `REAL_HIGH_CAMERA_SERIAL`、`REAL_WRIST_CAMERA_SERIAL` |
| LingBot 选择预取时机 | `--async-prefetch-after-actions 0` 越早预取；`8` / `16` 使用更晚的观测 |
| 指定其他 Python 环境 | `REALMAN_PYTHON=/path/to/python` |

各类推理优化的行为区别见 [推理调度与延迟](docs/inference_optimizations.md)。异步预取不包含完整 RTC 的延迟前缀约束或动作 inpainting。

## 排障

| 问题 | 检查 |
| --- | --- |
| 连接模型服务失败 | GPU 服务是否启动，SSH 转发终端是否仍在运行，端口是否一致 |
| π0.5 报 metadata mismatch | 本地任务段中的服务名称、checkpoint、状态和动作维度是否与服务一致 |
| 相机无法启动 | 停止采集程序；检查 USB 3 连接、序列号、设备权限和相机占用 |
| LingBot 报统计文件错误 | 使用匹配该 checkpoint 的 `action_norm_stats.json`，字段格式见模型服务文档 |
| VAAPI 编码失败 | FastWAM 使用 `--record-codec MJPG` 或关闭录像；π0.5 在本地配置关闭录像 |

模型协议和动作顺序见 [模型服务配置](docs/model_services.md)，第三方组件许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
