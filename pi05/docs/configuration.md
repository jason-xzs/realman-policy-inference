# Preset 配置说明

active runner 默认读取 `configs/pi05_realman_presets.toml`。文件由一个 `[default]` 和多个 `[presets."<name>"]` 组成；preset 对 default 做递归覆盖，因此只需写任务差异。

## 1. tracked baseline 与 local config

tracked 文件用于保存已审查的部署基线。机器路径、临时实验参数和每次录像名放到被忽略的本地副本：

```bash
cp configs/pi05_realman_presets.toml configs/pi05_realman_presets.local.toml
```

runner 不支持跨 TOML include，所以 local 文件必须是完整 registry。启动时同时显式给出 config 与 preset：

```bash
python -B scripts/run_pi05_realman.py \
  --config configs/pi05_realman_presets.local.toml \
  --preset pi05-organize-15000
```

## 2. 公共 section

| Section | 作用 | 每台 Client 必查 |
| --- | --- | --- |
| `cameras` | 三个 camera role、serial 和采集 profile | serial 唯一且 role 正确；固定 `640x480/30/bgr8` |
| `robots.left/right` | 控制器 TCP 与本机 UDP capture 地址 | 左右不能互换；本机确实拥有 `local_ip` |
| `sources` | 外部 `deltawam/src` 路径 | 指向包含已验证补丁、r3kit 和 SDK 的目录 |
| `policy` | WebSocket host/port、prompt、timeout、服务 identity | tunnel 端口和 server preset 一致 |
| `execution` | `execute`/`no_motion`、replan 上限、确认方式、日志级别 | 真机前明确 mode；不要依赖记忆中的默认值 |
| `initial_pose` | 双臂归位目标、速度、夹爪打开目标和稳定门 | 换装夹具/场景后重新审查 |
| `observation` | 28D capture 到 16D policy state 与图像尺寸 | 不改变顺序和 mapping，除非模型 contract 同步更新 |
| `action` | response shape/order、执行 row 窗口、频率和 command order | 与服务端输出一致；窗口不能超过 horizon 50 |
| `runtime` | freshness、同步、phase deadline、chunk settle 和清理 | timeout 不是物理完成证明 |
| `motion` | CANFD API、速度、trajectory/smoothing 和 step gate | `software_joint_step_limit_enabled` 必须为 true |
| `gripper` | 双阈值滞回、SDK position 与容差 | 模型标量语义必须仍是 0 关、1 开 |
| `recording` | 开关、目录、30 Hz VAAPI writer 与清理期限 | 每次运行使用新的 direct-child `subdirectory` |

config loader 会 fail closed：缺 section、非法 shape/order、非 finite 数值、错误 camera profile、无效路径或不安全 recording subdirectory 都会在 worker 启动前拒绝。

## 3. Policy 与模型 identity

任务 preset 至少应显式覆盖：

```toml
[presets."pi05-organize-15000".policy]
expected_config = "pi05_realman_joint_16d_step15000_ema"
expected_checkpoint = "/share/jhk/phywam-mope-real/checkpoints/finetuned/pi05/RR-BL-008/RR-BL-008-RUN-001/15000"
prompt = "Place the three objects into the blue container."
```

- `expected_config` 和 `expected_checkpoint` 非空时必须与 server metadata 完全相等。
- 留空表示跳过该 identity 字段，不表示自动验证成功；生产 preset 建议填写。
- server 的 `maximum_unique_matching_Client_requests` 若存在，必须覆盖本次 `max_replans`。
- tunnel 只改变传输路径，不改变模型 identity；不要只因为端口可连接就认为 checkpoint 正确。

## 4. Observation/action contract

原始双臂 observation 为 left-first 28D：

```text
left(q1..q7, gripper, eef6), right(q1..q7, gripper, eef6)
```

当前 native policy state/action 是 left-first 16D：

```text
left(q1..q7, gripper), right(q1..q7, gripper)
```

固定 mapping：

```text
value16 = value28[0:8] + value28[14:22]
```

底层 `RealManHardware` command 是 right-first，转换只在 worker 内进行。不要在 server、preset 和 hardware 三处重复交换左右顺序。详见 [interface-contract.md](interface-contract.md)。

默认 action window：

```toml
row_index = 0
actions_per_chunk = 20
action_hz = 30.0
```

服务返回 `[50,16]` 时只执行 row `0..19`。发送完最后一行后必须通过实际关节反馈 settle gate，才会发下一次 inference；SDK 接受不等于物理到位。

## 5. Execute、no-motion 与离线检查

三种边界不要混淆：

| 方式 | 网络 | 相机/机械臂读取 | 推理 | 运动命令 |
| --- | --- | --- | --- | --- |
| `check_pi05_environment.py` | 否 | 否 | 否 | 否 |
| `execution.mode = "no_motion"` | 是 | 是 | 一次 | 否 |
| `execution.mode = "execute"` | 是 | 是 | 最多 `max_replans` | 操作员确认后是 |

建议在 local config 添加或覆盖：

```toml
[presets."pi05-organize-15000".execution]
mode = "no_motion"
max_replans = 1

[presets."pi05-organize-15000".recording]
enabled = false
subdirectory = ""
```

no-motion 仍会打开真实相机和双臂 read-only 状态通道，并连接服务；它需要单独的现场批准，不能当作纯单元测试。

## 6. 初始位姿

`mode = "execute"` 通过只读硬件门和服务门后有两次交互：

1. `请回车确认，Ctrl+C 退出：`：允许进入真机操作阶段。
2. `是否回到初始位姿？回车同意，n/N拒绝：`：空回车发送双臂 `rm_movej` 并打开夹爪；`n/N` 跳过归位但继续录像与推理。

归位目标单位是 degree，policy/执行 joint contract 单位是 radian。归位命令返回后，还必须同时满足关节误差、反馈稳定窗口与夹爪实际位置门。

跳过归位不是安全确认。只有机械臂已经处于与任务、模型和场景一致的起始状态时才可选择 `n/N`。

## 7. 夹爪

当前控制规则：

- 模型值在 `[-value_tolerance, 1 + value_tolerance]` 外直接拒绝；容差内先饱和到 `[0,1]`。
- `value <= close_threshold` 请求关闭。
- `value >= open_threshold` 请求打开。
- 中间区域保持上一次目标；首次遇到中间值时由实际 observation 初始化。
- SDK position 当前为 `close_position = 1`、`open_position = 1000`。
- 左右侧只在 position 改变时发送 one-shot command。

日志的 `gripper: left & right` 显示服务端 canonical row 的原始标量，显示层裁剪并保留两位小数；它不是 SDK position。

## 8. 录像

录像关闭时不创建 target/cache。启用时必须配置绝对 root 与固定的 `.cache`：

```toml
[default.recording]
enabled = false
root_dir = "/home/robot2/jhk/phywam-mope-real/real-records"
cache_dir = "/home/robot2/jhk/phywam-mope-real/real-records/.cache"
fps = 30.0
codec = "h264_vaapi"
queue_size = 256
shutdown_timeout_sec = 120.0

[presets."pi05-organize-15000".recording]
enabled = true
subdirectory = "pi05-organize-15000-003"
```

`subdirectory` 只能是 root 的单层直接子目录名，不能是绝对路径、`.`、`..`、`.cache` 或带 `/` 的路径。目标非空时，在创建 worker、打开设备或推理前显示：

```text
目录非空，覆盖[y/N]:
```

只有精确输入 `y` 才清空该目标目录；其他输入都安全退出。更推荐每次使用新的 episode 名，避免覆盖已有证据。

最终目录只发布：

```text
<episode>/<episode>_high.mp4
<episode>/<episode>_left.mp4
<episode>/<episode>_right.mp4
```

相机线程保留三路原始 `640x480@30` 帧；模型的 `224x224` resize 是独立路径。`Ctrl+C` 后 recorder 排空队列、关闭 ffmpeg stdin、写 MP4 trailer，再发布文件。任一路失败都会返回 `blocked`。

## 9. 新增任务 preset

复制最接近的任务段，只覆盖任务差异：

```toml
[presets."pi05-new-task-15000".policy]
expected_config = "pi05_realman_joint_16d_step15000_ema"
expected_checkpoint = "/share/.../15000"
prompt = "Task instruction in the same language/style used for training."

[presets."pi05-new-task-15000".gripper]
close_threshold = 0.4
open_threshold = 0.7

[presets."pi05-new-task-15000".recording]
enabled = false
subdirectory = ""
```

先让离线预检通过，再做一次 no-motion response schema 验证。是否改为 execute 必须是另一个明确决定。
