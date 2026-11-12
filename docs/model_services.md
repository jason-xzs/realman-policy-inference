# 模型服务配置

机器人客户端与 GPU 模型服务通过 SSH 转发或局域网连接。模型必须使用与当前任务一致的 RealMan 训练数据、相机顺序、动作单位和归一化统计。

## FastWAM

仓库内的 `fastwam/scripts/tasks/serve_realman_fastwam.sh` 提供服务启动入口。默认使用权重文件的上三级目录查找 `config.yaml`、`dataset_stats.json`：

```text
run/config.yaml
run/dataset_stats.json
run/checkpoints/weights/step_015000.pt
```

其他目录结构可显式指定：

```bash
CHECKPOINT=/models/step_015000.pt bash fastwam/scripts/tasks/serve_realman_fastwam.sh \
  --config /models/config.yaml --stats /models/dataset_stats.json
```

`config.yaml` 内的基础模型路径也必须有效，或可从模型源下载。GPU 环境安装命令：

```bash
pip install -e fastwam --extra-index-url https://download.pytorch.org/whl/cu128
```

默认端口为 `8000`，客户端地址为 `http://127.0.0.1:18000`，对应 SSH 转发 `18000 → 8000`。HTTP 接口包括 `/health`、`/v1/infer` 和 `/v1/reset`。

原始状态为左臂 14D + 右臂 14D。返回动作是右臂 7 个关节 + 夹爪、左臂 7 个关节 + 夹爪，共 16D；关节为弧度，FastWAM 夹爪为 0–1000。

## π0.5 / OpenPI

本仓库提供机器人客户端。GPU 侧使用 [OpenPI](https://github.com/Physical-Intelligence/openpi) 的策略服务及 RealMan 的数据适配配置。已有匹配的服务时，只需建立转发并选择对应任务 preset。

在 OpenPI 项目中，checkpoint 服务的常规启动形式为：

```bash
uv run scripts/serve_policy.py --port=8020 policy:checkpoint \
  --policy.config=YOUR_REALMAN_CONFIG \
  --policy.dir=/path/to/checkpoint
```

`YOUR_REALMAN_CONFIG` 需要在 GPU 侧 OpenPI 注册；它定义 RealMan 相机键、状态和动作映射，不能用其他机器人配置代替。本仓库不包含 GPU 侧的模型训练工程和微调权重。

客户端使用 OpenPI WebSocket / MessagePack，发送：

- `state`：左臂关节与夹爪、右臂关节与夹爪，共 16D。
- `base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`：顶部、左腕和右腕的 `uint8 RGB [224,224,3]` 图像；腕部图像已旋转 180°。
- 任务 `prompt`。

服务返回 `actions`，形状为 `[H,16]` 的 finite `float32`；关节单位为弧度，夹爪为 0–1。双臂顺序为左臂、右臂，客户端下发前转换成 SDK 的执行顺序。

本地任务配置中的 `expected_config`、`expected_checkpoint` 与服务 metadata 的同名字段精确匹配。更换 GPU 上的权重目录时，要同步修改本地配置。metadata 如果提供 `state_dimension` / `action_dimension`，均应为 16。

更详细的输入格式见 [`realman_pi05.py`](../pi05/src/robot_client_runtime/realman_pi05.py) 的 `capture_observation()` 和 [`interface-contract.md`](../pi05/docs/interface-contract.md)。

## LingBot-VA

本仓库提供机器人客户端。GPU 侧使用 LingBot-VA 的 RealMan 服务适配器，默认监听 `18080`，需支持以下协议标识：

```text
version: realman_va_socket_v1
action_space: realman8_abs_joint_deg_gripper
```

输入为顶部和左腕图像、7 关节与夹爪状态以及任务文本。返回的每条动作为 7 个绝对关节目标（度）和夹爪（0–1），共 8D。消息封装、图像编码及 KV cache 命令以 [`client_realman_on_machine.py`](../legacy/scripts/evaluation/client_realman_on_machine.py) 的 `request_action()` 为准；它与 OpenPI 的 WebSocket 协议不同。

启动脚本默认加载 `configs/lingbot/action_norm_stats.json`。这是工作站原单臂任务的动作统计，用于关节范围校验。更换 checkpoint 时设置：

```bash
REAL_ACTION_STATS_PATH=/path/to/action_norm_stats.json \
  bash scripts/run_lingbot_client.sh --task-text 'Your task' --dry-run
```

统计文件的 `q01`、`q99` 至少包含 7 个关节的分位数，单位为度；其余归一化信息由模型服务加载。

## 录像

FastWAM 启动脚本默认关闭录像。软件编码使用 `--record-codec MJPG`，输出 AVI；Intel 硬件编码使用 `--record-codec h264_vaapi`，输出 MP4，需要 FFmpeg 和 `/dev/dri/renderD128` 的访问权限。

π0.5 录像在本地 TOML 中设置。开启前安装 FFmpeg，并确保配置的编码器可用。视频保留 `640×480@30` 的原始相机帧，策略输入另外 resize 到模型所需尺寸。
