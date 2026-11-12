# π0.5 真机运行手册

本手册按 Server、tunnel、离线检查、no-motion、execute 分阶段。每个阶段的成功只证明该阶段；不要把“服务可连”“返回 finite action”或“SDK 接受命令”写成物理执行完成。

## 0. 运行前确认

在任何设备或运动步骤前确认：

- Server preset、checkpoint 与 Client preset 是同一个任务版本。
- 三路相机 role/serial 与训练时一致，原生 profile 是 `640x480 BGR8 @ 30 Hz`。
- 左右臂 IP、网络接口、工具/夹具、初始位姿和工作空间已人工复核。
- 实体急停可触达；现场没有人员或障碍物进入双臂运动范围。
- 录像 episode 名没有复用，磁盘空间足够。
- 没有计划外的旧 Client、tunnel 或同端口 listener。

## 1. 启动 Server

在 Compute 服务器：

```bash
cd /share/jhk/phywam-mope-real/pi05-service-runtime && \
./scripts/run_pi05_service.py serve --preset pi05-organize-15000
```

记录 server 输出的 config、checkpoint、session/commit 和最大请求数。Client 会校验 metadata 中可用的 identity 字段，但操作员仍要核对任务名和 checkpoint 路径。

Server 路径是 Compute 侧路径，Client 本机不存在是正常现象。

## 2. 建立 SSH tunnel

先确认 Client 本地 18200 未被其他进程占用：

```bash
ss -ltnp | grep ':18200'
```

有输出时先确认 owner；不要自动 kill 或复用一个用途未知的 listener。端口空闲后，在独立 Client 终端运行并持续保持：

```bash
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -L 127.0.0.1:18200:127.0.0.1:8020 xzs_server
```

健康 tunnel 通常没有输出。`ExitOnForwardFailure` 只保证 SSH 成功绑定转发端口，不证明远端 policy 已加载正确 checkpoint。

## 3. 离线环境预检

这个步骤不依赖 tunnel，也不会访问设备：

```bash
cd ~/jhk/phywam-mope-real/robot-client-runtime && \
PYTHONDONTWRITEBYTECODE=1 \
/home/robot2/jhk/phywam-mope-real/envs/robot-client-py311/bin/python -B \
scripts/check_pi05_environment.py \
--preset pi05-organize-15000
```

只有输出 `"status": "passed"` 才表示本地环境与已验证 lock 一致。该状态的 scope 会明确显示 service、camera、robot SDK 和 motion 均未运行。

## 4. 可选的 no-motion 集成验证

先在 ignored local config 中把目标 preset 改为：

```toml
[presets."pi05-organize-15000".execution]
mode = "no_motion"
max_replans = 1

[presets."pi05-organize-15000".recording]
enabled = false
subdirectory = ""
```

然后显式使用 local config：

```bash
cd ~/jhk/phywam-mope-real/robot-client-runtime && \
PYTHONDONTWRITEBYTECODE=1 \
/home/robot2/jhk/phywam-mope-real/envs/robot-client-py311/bin/python -B \
scripts/run_pi05_realman.py \
--config configs/pi05_realman_presets.local.toml \
--preset pi05-organize-15000
```

这个阶段会连接 tunnel、打开三路相机、连接双臂并读取状态、发送一次 inference request，但不会发送归位、关节或夹爪 command。预期 terminal result：

```json
{
  "status": "no_motion_pass",
  "policy_requests": 1,
  "robot_commands": 0,
  "gripper_commands": 0
}
```

它只证明一次集成 schema/identity 路径，不证明 execute 安全，也不消耗之前 execute 的运动授权。

## 5. Execute

确保所选 config 中目标 preset 的最终合并结果为 `mode = "execute"`，并重新确认录像 episode 名。运行：

```bash
cd ~/jhk/phywam-mope-real/robot-client-runtime && \
PYTHONDONTWRITEBYTECODE=1 \
/home/robot2/jhk/phywam-mope-real/envs/robot-client-py311/bin/python -B \
scripts/run_pi05_realman.py \
--preset pi05-organize-15000
```

若使用 local config，必须增加对应的 `--config`；不要以为刚编辑的 local 文件会自动加载。

启动后的顺序：

1. worker 打开 OpenPI、相机和双臂 SDK。
2. 读取服务 metadata，并执行双臂 alarm/controller/joint-enable gate。
3. 显示 `请回车确认，Ctrl+C 退出：`。
4. 回车后显示 `是否回到初始位姿？回车同意，n/N拒绝：`。
5. 第二次空回车发送双臂 `rm_movej` 和开夹爪，等待实际反馈稳定；`n/N` 跳过这些命令。
6. 归位完成或明确跳过后才开始录像和第一次推理。
7. 每个 chunk 发送完成并实际 settle 后才开始下一次 inference。

任一提示出现时，`Ctrl+C` 都表示退出，不要通过管道自动喂回车。跳过归位只适用于人工确认当前姿态已经是正确起始状态的情况。

## 6. 停止与结果判读

正常停止使用一次 `Ctrl+C`。父进程会：

- 请求 worker 停止；
- 如果动作已经开始，先调用 worker 内 slow-stop；
- 停止三路相机并排空录像队列；
- 等待 FFmpeg 写完 trailer 和发布三个 MP4；
- worker/SDK 卡死或 phase timeout 时有界终止，并尝试独立 slow-stop 进程。

常见 terminal status：

| Status | 含义 |
| --- | --- |
| `completed` | 达到配置的 replan 上限且每个 chunk settle gate 通过 |
| `stopped` | 操作员停止或拒绝覆盖录像目录；查看 `reason` |
| `no_motion_pass` | 一次真实 observation/inference 验证通过，未发运动命令 |
| `blocked` | 配置、服务、设备、反馈、action、timeout、录像或清理失败；禁止继续自动请求 |

即使 status 是 `completed`，也只说明 runner 的反馈门通过；仍应结合现场观察和任务验收判断物理任务是否完成。

## 7. Fail-closed 处理

出现以下任一情况时，停止本次 run，不要盲目重试同一 request 或运动：

- 已收到 inference response，但不知道 action 是否已经开始发送；
- SDK 返回、timeout 或进程退出让 command acceptance 状态不明确；
- 任一关节方向、幅度或双臂顺序与预期不一致；
- chunk settle timeout、反馈异常或 camera/state freshness 失败；
- tunnel 中断后无法证明服务端没有处理请求；
- 三路录像缺失、为空或 cleanup blocked。

保留 terminal 输出、server session/commit、preset/config 快照和已生成视频。确认实际机械臂状态后，再建立新的、范围明确的诊断或运行计划。

## 8. 常见问题

### `preset_not_found`

确认 `--config` 指向包含目标 preset 的完整 registry。local TOML 不是增量 include 文件。

### 服务连接失败或走代理

确认 tunnel 仍在运行、Client port 与 `[policy].port` 一致。runner 对 loopback 自动设置进程内 `NO_PROXY/no_proxy`，不会改系统代理。不要把 policy host 改成外网地址来绕过 tunnel。

### `service_*_mismatch`

Server 与 Client 选择了不同 config/checkpoint，或服务 metadata 不完整。修正 preset/服务启动，不要删除 identity 校验字段来强行继续。

### camera profile/serial 失败

用 `rs-enumerate-devices -s` 重新核对 serial，再检查 USB/udev；不要把左右 wrist role 对调。active runner要求三路都是真实 `640x480@30 bgr8`。

### `recording_overwrite_declined`

这是安全退出。给 local config 换一个新的 `subdirectory`，通常比覆盖更合适。

### `h264_vaapi` 或 render node 失败

重新运行离线预检并核对 FFmpeg encoder、`/dev/dri/renderD128` 权限和磁盘路径。不要临时改成另一编码器后仍声称使用了已验证录像路径。

### 外部 source/hash drift

对照 [environment.md](environment.md) 恢复 revision 和 patch。`--allow-version-drift` 只用于查看其他检查项，不是 execute 放行开关。
