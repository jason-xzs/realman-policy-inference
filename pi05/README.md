# RealMan π0.5 Client Runtime

这是双 RealMan 真机侧的 π0.5 推理客户端。它负责同步采集三路 RealSense 与双臂状态、调用远端 OpenPI WebSocket 服务、校验 action chunk，并在操作员确认后执行关节与夹爪命令。主进程监督独立硬件 worker；异常、超时或 `Ctrl+C` 会停止后续推理，并在已开始动作时尝试双臂 slow-stop。

> 这是实验室真机运行工具，不是安全认证控制器。SDK 返回成功只表示命令被接受，不表示机械臂已经物理到位；软件 slow-stop 也不等同于实体急停。每次 execute 前都要确认工作空间清空、初始位姿适用并且急停可触达。

## 仓库范围

active 入口只有：

- `scripts/run_pi05_realman.py`：真机 runner。
- `scripts/check_pi05_environment.py`：严格离线的环境与依赖指纹检查。
- `configs/pi05_realman_presets.toml`：当前部署基线和任务 preset。
- `src/robot_client_runtime/production_pi05.py`：进程监督、录像和执行循环。
- `src/robot_client_runtime/realman_pi05.py`：配置、schema、服务 metadata 和 action/gripper 转换。

`rr_dep_00x.py`、对应 scripts/configs/tests 是历史实验复现代码，不在 active 入口调用链中。接口顺序与执行语义见 [docs/interface-contract.md](docs/interface-contract.md)。

本仓库不是单独克隆后就能连接硬件的完整二进制发行包。它还需要：

- 匹配服务端的 `openpi-client`；
- `deltawam/src/realman_deployment` 硬件实现；
- 嵌套的 `r3kit` RealSense 实现；
- RealMan Python/C SDK；
- FFmpeg VAAPI 和三个 RealSense 设备。

已验证环境的版本、外部补丁和 SHA-256 指纹放在 `environment/`；完整恢复步骤见 [docs/environment.md](docs/environment.md)。

## 本次使用示范

在 Compute 服务器启动服务：

```bash
cd /share/jhk/phywam-mope-real/pi05-service-runtime && \
./scripts/run_pi05_service.py serve --preset pi05-organize-15000
```

在 Client 单独终端保持 SSH tunnel：

```bash
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -L 127.0.0.1:18200:127.0.0.1:8020 xzs_server
```

先在 Client 做离线检查。它不连接网络、不枚举/打开相机、不加载或连接 RealMan SDK，也不会运动：

```bash
cd ~/jhk/phywam-mope-real/robot-client-runtime && \
PYTHONDONTWRITEBYTECODE=1 \
/home/robot2/jhk/phywam-mope-real/envs/robot-client-py311/bin/python -B \
scripts/check_pi05_environment.py \
--preset pi05-organize-15000
```

确认服务、模型、设备绑定、初始位姿和现场安全条件后，再运行 Client：

```bash
cd ~/jhk/phywam-mope-real/robot-client-runtime && \
PYTHONDONTWRITEBYTECODE=1 \
/home/robot2/jhk/phywam-mope-real/envs/robot-client-py311/bin/python -B \
scripts/run_pi05_realman.py \
--preset pi05-organize-15000
```

runner 会为 loopback WebSocket 进程内补上 `NO_PROXY=127.0.0.1,localhost`。`pi05-organize-15000` 当前继承 `mode = "execute"` 且启用录像，因此最后一条命令不是 dry-run。完整的分阶段运行、终端提示、停止和故障处理见 [docs/operations.md](docs/operations.md)。

## 配置

registry 的 `[default]` 是公共配置，`[presets."<name>"]` 对嵌套字段做递归覆盖。切换机器或临时修改录像名时，建议复制一份被 `.gitignore` 排除的本地配置，避免每次运行污染 Git：

```bash
cp configs/pi05_realman_presets.toml configs/pi05_realman_presets.local.toml
```

运行时显式指定：

```bash
python -B scripts/run_pi05_realman.py \
  --config configs/pi05_realman_presets.local.toml \
  --preset pi05-organize-15000
```

相机序列号、机械臂地址、外部源码路径、服务 identity、初始位姿、动作窗口、夹爪滞回和录像目录的逐项说明见 [docs/configuration.md](docs/configuration.md)。

## 运行语义摘要

- 原始 observation 是 left-first 28D；发给 policy 前按 `[0..7,14..21]` 映射为 left-first 16D。
- policy 只接受 finite `float32 [H,16]`；worker 再转换为底层硬件的 right-first 16D command。
- 默认每个 50-row response 执行 `row 0..19`，按 30 Hz 逐条提交；慢 SDK 调用后不会追赶式 burst。
- chunk 最后一条命令必须从双臂实际反馈达到 tolerance 并稳定，才允许下一次推理。settle timeout 会阻断后续请求。
- 夹爪使用双阈值滞回，只在目标开/关状态变化时发送 one-shot position command。
- 三路录像保存原始 `640x480 BGR8 @ 30 Hz` 帧；模型输入另行 resize 到 `224x224`，两者不混同。
- `mode = "no_motion"` 会打开设备、读取 observation 并请求一次推理，但不会发送归位、关节或夹爪命令；完全离线检查只能使用 `check_pi05_environment.py`。

## 测试

测试不应打开相机、连接服务或发送真机命令：

```bash
cd /home/robot2/jhk/phywam-mope-real/robot-client-runtime
PYTHONDONTWRITEBYTECODE=1 \
  /home/robot2/jhk/phywam-mope-real/envs/robot-client-py311/bin/python -B \
  -m unittest discover -s tests -v
```

提交前还应运行：

```bash
git diff --check
PYTHONDONTWRITEBYTECODE=1 \
  /home/robot2/jhk/phywam-mope-real/envs/robot-client-py311/bin/python -B \
  -m compileall -q src scripts tests
```
