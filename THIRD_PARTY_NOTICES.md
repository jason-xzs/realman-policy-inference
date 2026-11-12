# 来源与第三方组件

- `fastwam/` 从本机 `deltawam` 的 RealMan 分支导出；原始许可保留在 `fastwam/LICENSE`，上游 https://github.com/coldyan/deltawam 。未导出训练数据、权重和 RoboTwin 仿真资源。
- `fastwam/src/realman_deployment/Robotic_Arm/` 为 RealMan 官方 Python/C SDK，本机版本原样保留；上游 https://github.com/RealManRobot/RM_API2 ，使用与分发遵循上游条款。
- 嵌套 `r3kit/` 为本机设备封装及修改，原文件声明保留。
- `pi05/` 为本机双臂客户端工程；其已有版权声明、配置和原始环境文档保留。
- `legacy/openpi/openpi-client/` 来自本机 OpenPI 配套客户端，保留文件级版权/许可声明；上游 https://github.com/Physical-Intelligence/openpi 。
- `legacy/scripts/evaluation/` 为本机 LingBot-VA 和早期 π0.5 适配脚本。原 LingBot launcher 的历史服务器登录注释已从导出版本删除。

没有给不同来源的全部文件重新授予统一许可。模型权重另行取得并遵循模型作者条款。整理新增的文档、路径包装脚本与源文件的来源信息见 `docs/source_manifest.json`。
