# Action 与 Observation Interface Contract

本文同时保留历史 RR-DEP-001 adapter 边界，并描述 active production runner。历史的 unresolved 项不能覆盖后文已经固定的 active 控制语义。

## 历史 RR-DEP-001 Action

`realman.dual_arm.absolute.v1` 的 canonical shape 为 `[H,16]`：

| Index | Meaning |
| --- | --- |
| `0..6` | left `q1..q7`，radians，next-step absolute joint-position target |
| `7` | left gripper，next-step absolute opening target |
| `8..14` | right `q1..q7`，radians，next-step absolute joint-position target |
| `15` | right gripper，next-step absolute opening target |

28-D 映射固定为：

```text
action16 = action28[0:8] + action28[14:22]
```

这项 deployment interpretation 是批准的 adapter design；historical dataset 的 `action[t] = observation.state[t+1]` 是 measured next state，不因此变成 historical robot command。

16-D mode 只做 exact shape 与 finite validation。32-D mode 默认关闭；RR-DEP-002 必须先确认 server response 确实是 first 28 native values 加 last 4 padding，之后才能显式启用 first-28 extraction。

在 RR-DEP-001 fixture 范围内，gripper scalar 原样保留，未定义 range、normalization、clamp 或 SDK conversion，也未定义 joint/workspace/velocity/acceleration/torque limits。

## 历史 RR-DEP-001 Observation

输入 state 是 left-first 28-D。image role 映射如下：

| Client role | OpenPI key |
| --- | --- |
| `high_rgb` | `base_0_rgb` |
| `left_wrist_rgb` | `left_wrist_0_rgb` |
| `right_wrist_rgb` | `right_wrist_0_rgb` |

adapter 还输出 `state` 和 `prompt`。本任务不绑定 camera serial，不 resize 或打开任何真实 image/device。

## Historical RR-DEP-001 no-live-access gate

`Pi05PolicyClient` 仅描述 OpenPI `openpi_client.WebsocketClientPolicy` 的 lazy wrapper；disabled 状态在 optional import 之前拒绝。CLI 仅允许 fixture/mock dry-run。`RealManDualArmAdapter` 只接受注入 backend；无 backend 时拒绝。RR-DEP-001 不提供 vendor SDK backend。

## Active real-robot runner

`scripts/run_pi05_realman.py` 当前调用 `robot_client_runtime.production_pi05`，CLI 只保留 `--config` 与 `--preset`，省略 `--preset` 时选择 `pi05-sugar-15000`。运行模式、replan 数量、prompt、初始位姿和 log level 全部来自 `configs/pi05_realman_presets.toml`；`mode = "execute"` 在进入真机确认后增加归位选择，空回车归位、`n/N` 跳过归位并继续，`mode = "no_motion"` 不归位且只执行一次 observation/inference validation。

主进程监督一个通过 `spawn` 创建的硬件 worker；worker 独占 OpenPI、`RealManHardware`、相机和 RealMan Python SDK。选择归位后，worker 用 `rm_movej` 向双臂各提交一次 TOML 中的 `q1..q7` degree 目标，左、右臂分别使用 `left_speed_percent`、`right_speed_percent`，并将双夹爪打开到 `1000`；随后同时验证关节误差、关节稳定窗口和夹爪实际位置。任何命令拒绝或反馈超时都会 fail closed，且不会推理或启动录像。相机会在 startup 预热，但录像 submit gate 只在归位通过或操作员明确跳过后打开，因此归位前帧不会进入 MP4。主进程保持信号响应并对 startup/initial-pose/observe/inference/action/cleanup 设置 deadline；worker 卡死时执行有界终止，并在 motion 已开始时启动独立 slow-stop 进程。

policy 侧 canonical contract 是 native left-first `[H,16]`；每 row 在 worker 内转换为 FastWAM hardware 的 right-first joint/gripper action。每个 chunk 只发送配置的 `actions_per_chunk` 行；发送完最后一行后，worker 读取双臂实际关节反馈，等待最后目标在 tolerance 内保持稳定，再开始下一轮 observation/inference。`action_done` 仅表示 SDK 调用返回，不表示物理到位；chunk settle 超时或反馈异常会停止并阻断后续推理。`RealManHardware` 负责同步 observation、CANFD、step gate、双臂锁和 gripper one-shot。历史 RR-DEP backend/poller/RMPlus 路径不再被 active 入口调用。

active runner 进一步固定 gripper scalar 为归一化语义：`0` 表示关闭、`1` 表示打开；小幅越界只在 `value_tolerance` 范围内饱和，超出即拒绝。`<= close_threshold` 关闭、`>= open_threshold` 打开、中间区域保持上一目标，最终分别转换为 preset 的 SDK `close_position`/`open_position`。该语义只适用于 active production preset，不能倒推为历史 RR-DEP-001 fixture 已经验证。
