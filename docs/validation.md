# 整理验证记录

日期：2026-11-12。只运行离线或模拟设备测试，未启动真机、相机、模型权重加载或真实推理服务。

| 检查 | 结果 |
| --- | --- |
| Python 语法解析 | 199 个 `.py` 文件通过 |
| Shell 语法检查 | 6 个 `.sh` 文件通过 `bash -n` |
| π0.5 原有离线测试 | 130/130 通过；其中端口选择测试需本机回环 socket 权限，单独在沙箱外复核通过 |
| FastWAM 原有离线测试 | 20 通过，1 项可选本地 checkpoint 测试跳过 |
| LingBot CLI | `scripts/run_lingbot_client.sh --help` 通过，未访问设备 |
| π0.5 本地配置生成器 | 通过；路径指向导出仓库，`no_motion`，各任务录像关闭 |
| 凭据扫描 | 未发现 GitHub token、云密钥、私钥或实际口令；删除历史 SSH 登录注释 |

π0.5 使用本机已存在的 Python 3.11 客户端环境测试。FastWAM 测试需要 Torch/Hydra/OmegaConf，使用本机已存在的 Python 3.12 教学环境测试，不要求机器人侧最小安装包含 GPU 依赖。

FastWAM 的原测试遗漏 `RealManHardware` 导入，导致三个 mock 测试报 `NameError`；导出副本补上导入。另一个测试依赖未发布的 `step_005000.pt`，改为文件缺失时显式跳过，权重可用时仍执行完整检查。未修改推理控制算法。

FastWAM 完整模拟回环已经包含在 `test_full_control_loop` 中通过。未重复构建 CUDA 环境或 ROS Docker 镜像；部署到不同设备仍需核对本地配置和实际依赖。
