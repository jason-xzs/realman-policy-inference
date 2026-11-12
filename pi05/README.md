# π0.5 RealMan Client

RealMan RM75 双臂的 OpenPI 客户端，提供三路 RealSense 采集、双臂状态同步、动作校验、30 Hz 关节执行、夹爪控制和录像。

从仓库根目录安装并运行：

```bash
bash scripts/setup_client.sh
bash scripts/run_pi05_client.sh --preset pi05-organize-15000
```

首次运行会生成本地配置，默认不发送运动命令。连接 GPU 服务、切换任务和启用执行见 [项目快速开始](../README.md#π05--openpi)。

- [任务与设备配置](configs/pi05_realman_presets.toml)
- [服务协议与权重配置](../docs/model_services.md#π05--openpi)
- [状态与动作映射](docs/interface-contract.md)

`production_pi05.py` 为日常运行入口；`rr_dep_00x.py` 和对应脚本用于历史实验复现。
