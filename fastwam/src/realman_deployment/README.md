# FastWAM RealMan Deployment

FastWAM 的 RealMan RM75 双臂部署：GPU 服务器运行 HTTP policy 服务，机器人客户端读取三路 RGB 和双臂状态，按 30 Hz 执行动作块。

安装、模型准备与启动命令见 [项目快速开始](../../../README.md#fastwam)。

| 文件 | 功能 |
| --- | --- |
| `server.py` / `policy.py` | HTTP 服务、模型加载和预处理 |
| `client.py` | 观测上传、动作块对齐、执行和录像 |
| `hardware.py` | 双臂 SDK、相机同步、关节步长限制和夹爪控制 |
| `protocol.py` | 请求格式和动作校验 |
| `simulate.py` | 不连接真机的本地模拟 |

接口与权重目录见 [模型服务配置](../../../docs/model_services.md#fastwam)，调度选项见 [推理调度与延迟](../../../docs/inference_optimizations.md)。
