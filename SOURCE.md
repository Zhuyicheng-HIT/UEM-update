# 代码来源与精简记录

- 上游仓库：https://github.com/sxh-kk/UEM-update
- 提取提交：`156ab79d5f692a6e8db3c3eb769abf1e4cb1fc08`
- 原始实验配置：`ablation/configs/e7_x0_global_w8_u84k.yaml`
- 原始方法：UniEgoMotion，ICCV 2025，Chaitanya Patel 等。
- 官方完整身体网络参考：`chaitanya100100/UniEgoMotion`，提交 `580c92c6d70a91672c4106bab30ca82cdd80f379`。

网络精简为完整身体 Transformer decoder 与 E7 连续时间嵌入。为兼容上游 E7/EMA 权重，保留参数名称和顺序、未使用的 text 兼容参数及其冻结设置。工作树不再包含 MoE、稀疏关节、TaskFiLM、双输出头和 Diffusion 模型。

保留 E7 的加权 x0 目标、随机条件屏蔽、Euler 采样、Beta 时间分布、训练配置和 EMA。新增推理时可选历史约束与显式初始噪声参数，开关默认关闭。约束的具体含义见主 README；本次未证明它能改善真实运动恢复精度。

数据转换、几何、指标及可视化工具继承原作者代码。`module/ema.py` 等文件的第三方许可头保留；本次整理未为上游代码、SMPL-X 或数据资产授予新的许可。

Git 历史保留来源。工作树已移除混合实验报告、其他实验脚本/配置和模型变体。仓库外的审计副本不属于上传分支。
