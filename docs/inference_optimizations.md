# RTC 与推理端优化代码定位

这里按实际实现区分：推理/执行重叠、重规划动作对齐、传感器延迟，以及录像开销。搜索覆盖本机当前 `deltawam`、`realman_teleop_v2`、`PhyWAM_old_useless`、双臂 π0.5 runtime 及历史脚本。

| 模型/路径 | 实现 | 启用方式/默认值 |
| --- | --- | --- |
| LingBot-VA | `ThreadPoolExecutor(max_workers=1)` 在执行当前 action chunk 时请求下一 chunk；支持执行 N 步后再读取观测并预取 | `--mode async --disable-kv-cache --async-prefetch-after-actions 8`；默认 `safe_sync` |
| FastWAM | 推理返回后重新读取关节，寻找与当前状态更接近的动作，跳过过时的 chunk 前缀 | `--align-action-chunk --action-align-max-skip 8`；默认关闭 |
| FastWAM | `BufferedCameraRig` 为每个相机独立后台采集，按机器人状态时间选取同步帧 | 客户端默认启用；`--max-sync-skew-ms 50 --sync-timeout 8` |
| FastWAM | `BackgroundCameraVideoRecorder` 独立 writer 线程编码，避免视频写盘阻塞控制循环 | `--record-dir`；空字符串可关闭录像 |
| FastWAM | tracking gate 重复目标直到反馈到位，避免机器人跟不上模型动作时跳过路径 | `--tracking-gate`；这是跟踪约束，会拉长执行时间 |
| π0.5 | RR-DEP-006 每相机只保留最新一帧，不编码、不录像，模型输入转为 224×224 | `pi05/scripts/run_rr_dep_006.py`；历史实验入口，需要匹配原实验协议和参数 |
| π0.5 | 当前 production runner 使用独立硬件 worker 和 watchdog；动作按 30 Hz 下发，慢调用后不突发补发 | 当前任务 preset；每个 chunk 后等待到位并稳定，再请求下一次推理 |

## LingBot 异步预取：最接近 RTC 的已找到实现

源码：[`client_realman_on_machine.py`](../legacy/scripts/evaluation/client_realman_on_machine.py)。

- `RealManVAClient.run_async()`：线程池管理当前 response 与下一次 `prefetch_future`。
- `execute_chunk()` 下的执行函数：通过 `prefetch_hook` 在执行到配置步数后读取当前图像与状态并发起下一次推理。
- `run()` / 参数处理：当 KV cache 开启时默认回退 `safe_sync`；也可用 `--async-kv-policy disable_kv` 显式关闭 cache。
- `CameraCaptureWorker`：后台相机线程，避免每次请求临时打开相机。

`--async-prefetch-after-actions 0` 尽早发起下一次请求，最大化重叠；设为 8/16 会使用稍晚的观测，代价是可重叠时间更短。异步预测使用预取时刻的观测，没有完整的延迟补偿前缀约束或动作 inpainting；新旧 chunk 之间仍可能出现状态滞后。

## FastWAM 对齐与后台工作

[`client.py`](../fastwam/src/realman_deployment/client.py) 的 `align_action_chunk()` 在有限的动作索引内比较 14 个关节的最大绝对误差，仅在误差改善达到 `--action-align-min-improvement-rad` 后跳过前缀，夹爪随被选中的动作一起执行。它补偿重规划后的状态错位，当前 HTTP 推理请求与 chunk 执行仍串行。

[`hardware.py`](../fastwam/src/realman_deployment/hardware.py) 的 `BufferedCameraRig` 使用各相机缓存与 `select_synchronized_frames()` 选择匹配机器人采样时间的三路图像。`client.py` 的 `BackgroundCameraVideoRecorder` 为录像提供独立 writer 队列；队列满时失败，不静默丢帧。

## π0.5 低延迟路径与当前执行语义

[`rr_dep_006.py`](../pi05/src/robot_client_runtime/rr_dep_006.py)、[`rr_dep_005.py`](../pi05/src/robot_client_runtime/rr_dep_005.py) 包含无录像、最新帧采样和低开销输入路径。历史入口涉及原任务 ID、服务身份和现场确认，保留用于定位与复现，不作为新的通用启动命令。

当前入口是 [`production_pi05.py`](../pi05/src/robot_client_runtime/production_pi05.py)。后台 worker 和录像线程能减少主进程阻塞，但 `wait_for_chunk_completion()` 要求双臂最终反馈到位并稳定，然后才允许下一次 policy 请求；它没有异步预测重叠。直接修改这项约束属于新的控制算法，整理版本保留原逻辑。

## 额外发现

`/home/robot2/vla_class_pub/lerobot/src/lerobot/policies/rtc` 有通用 LeRobot RTC 实现，以及 pi0/pi05 的 RTC 测试。未找到它接到这台 RealMan 上的启动入口，因此没有把整套教学/第三方代码混入本仓库。若以后移植正式 RTC，可将其作为上游参考；本次不声称已完成真机集成。
