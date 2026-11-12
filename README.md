# RealMan 真机策略推理脚本

整理本机 **FastWAM、π0.5 / OpenPI、LingBot-VA** 的 RealMan 客户端、FastWAM 服务端和推理优化逻辑。保留已有行为及任务配置，原目录保持不变。模型权重、采集数据、录像、登录信息和运行缓存不入库。

## 目录与原始位置

| 目录 | 内容 | 本机来源 |
| --- | --- | --- |
| [`fastwam/`](fastwam) | FastWAM HTTP 服务、双臂客户端、硬件适配、模型源码和配置 | `/home/robot2/deltawam` |
| [`pi05/`](pi05) | 当前双臂 OpenPI 客户端、任务 preset、历史低延迟实现和离线测试 | `/home/robot2/jhk/phywam-mope-real/robot-client-runtime` |
| [`legacy/scripts/evaluation/`](legacy/scripts/evaluation) | LingBot-VA 单臂客户端、早期 π0.5 单臂客户端、相机封装 | `/home/robot2/PhyWAM_old_useless/scripts/evaluation` |
| [`legacy/openpi/openpi-client/`](legacy/openpi/openpi-client) | 本机配套 OpenPI WebSocket 客户端 | `/home/robot2/PhyWAM_old_useless/openpi/openpi-client` |

## RTC 或类似推理优化

已找到 **LingBot 异步预取并与执行重叠、FastWAM 动作块对齐与后台采集、π0.5 低延迟无录像路径**。详细函数、参数、使用条件与局限见 **[`docs/inference_optimizations.md`](docs/inference_optimizations.md)**。

本机这些 RealMan 脚本中未发现带延迟前缀约束/动作 inpainting 的完整 RTC 算法。LingBot 的异步预取最接近“推理与执行重叠”；π0.5 当前默认逐块执行并等待末端反馈稳定，再发下一次推理请求。仓库保留原有优化，没有新增未经真机验证的调度算法。

## 机器人侧安装

使用独立的 Python 3.11 环境，避免与 ROS 采集环境混用：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-client.txt
pip install -e legacy/openpi/openpi-client
pip install --no-deps -e pi05
```

真机运行还需要 RealSense USB 权限、独占的机械臂控制连接和 FFmpeg；录像默认使用 Intel VAAPI 编码，需要 `/dev/dri/renderD128`。仓库保留机器人 IP 和相机序列号的本机配置。

## FastWAM

GPU 服务端需要 FastWAM 的模型依赖与对应权重。`fastwam/pyproject.toml` 保留原始 CUDA/PyTorch 依赖清单，安装时按 GPU 环境匹配 wheel；客户端安装不需要完整 GPU 训练依赖。

```bash
# GPU 机器：权重旁需要匹配的 config.yaml 和 dataset_stats.json。
CHECKPOINT=/path/to/checkpoints/weights/step_015000.pt \
  bash fastwam/scripts/tasks/serve_realman_fastwam.sh

# 机器人侧：读取真实设备并验证动作；不下发动作。
SERVER_URL=http://127.0.0.1:18000 bash scripts/run_fastwam_client.sh --max-replans 2

# 已核对现场配置后，下发动作并开启动作块对齐。
EXECUTE=1 SERVER_URL=http://127.0.0.1:18000 \
  bash scripts/run_fastwam_client.sh --align-action-chunk --action-align-max-skip 8
```

SSH 转发示例：`ssh -N -L 18000:127.0.0.1:8000 YOUR_GPU_HOST`。参数、夹爪迟滞、tracking gate、录像见 [`fastwam/src/realman_deployment/README.md`](fastwam/src/realman_deployment/README.md)。

## π0.5 双臂客户端

先生成适用于本仓库路径的本地配置，默认 `no_motion`、关闭录像。该配置被 Git 忽略，可编辑相机、服务身份、checkpoint、任务和运动参数：

```bash
python scripts/prepare_pi05_config.py
python pi05/scripts/run_pi05_realman.py \
  --config pi05/configs/pi05_realman_presets.local.toml \
  --preset pi05-organize-15000
```

`no_motion` 会打开相机和机器人连接、读取状态并请求模型，但不发运动命令。完全离线检查使用 `pi05/scripts/check_pi05_environment.py`；其历史环境指纹需要按本仓库路径更新，详见 `pi05/docs/environment.md`。

需要执行时，在本地 TOML 中将 `[default.execution] mode` 改为 `"execute"`，运行时保留原有终端确认流程。**原始受跟踪 preset 的默认值是 execute**，直接运行原始配置会进入执行路径。模型服务必须提供匹配的 OpenPI metadata、动作维度和 checkpoint 身份。这个主机仅找到 π0.5 客户端，GPU 服务端工程位于历史文档中的远端路径，未从远端取回。

双臂策略输入是 left-first 16D，底层执行转换为 right-first 16D；与 FastWAM 的原始 left-first 28D 状态输入有区别。详见 [`pi05/docs/interface-contract.md`](pi05/docs/interface-contract.md)。

## LingBot-VA 单臂异步版本

```bash
# 只验证 CLI，不连接设备或服务。
bash scripts/run_lingbot_client.sh --help

# 模拟观测 + 不下发动作；仍需匹配的 LingBot WebSocket 服务。
bash scripts/run_lingbot_client.sh \
  --server-host 127.0.0.1 --server-port 18080 \
  --mock-observation --dry-run \
  --mode async --disable-kv-cache --async-prefetch-after-actions 8
```

连接真实相机时去掉 `--mock-observation`，提供匹配的训练数据统计 `REAL_DATASET_PATH` 或 CLI 统计参数。执行动作需要显式 `--enable-robot`；首次会归位，确认关节初始位姿适合现场再使用。异步预取必须关闭 KV cache，否则默认退回 `safe_sync`。早期单臂 π0.5 脚本保留于 `legacy/scripts/evaluation/client_pi05_real.py`，其规范化统计需另行提供。

LingBot 的 GPU 服务端未在当前主机源码中找到，仓库不伪造服务实现。其单臂 WebSocket 协议与双臂 OpenPI/FastWAM 协议不能互换。

## 离线验证

```bash
PYTHONPATH=fastwam/src python -m unittest discover -s fastwam/src/realman_deployment/tests -v
PYTHONPATH=fastwam/src python -m realman_deployment.simulate --replans 3 --actions-per-replan 4
PYTHONPATH=pi05/src python -m unittest discover -s pi05/tests -v
```

以上验证不启动真机或相机。整理时执行的检查结果见 [`docs/validation.md`](docs/validation.md)。来源和修改说明见 [`docs/source_manifest.json`](docs/source_manifest.json) 与 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
