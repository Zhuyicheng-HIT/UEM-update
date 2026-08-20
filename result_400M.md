# K12 400M Motion Expert 测评结果

## 1. 结论

400M K12 MoE Motion Expert 已完成 300 epochs 训练和八卡 256 样本测评。与原始约
88M K12 模型相比，400M 模型在 Recon、Gen、Fore 三个任务的 K12 位置、旋转和
速度误差均有改善；其中 Gen 位置误差下降 4.10%，Fore 位置误差下降 6.44%。

但 400M 模型没有全面超过完整 22 关节 E7 在相同 K12 子集上的结果：它在 Gen
位置误差和 Fore 速度误差上略优，在其他多项指标上略差。因此目前证据支持
“扩大 K12 Motion Expert 容量优于原始 K12 模型”，不支持“400M 在所有指标上
超过 E7”。

## 2. 公平对比口径

本报告严格分开两种评测，二者数值不能直接横向比较：

1. **完整 22 关节评测**：原论文 Diffusion 和 E7 都输出完整 243D `v4_beta`，经
   SMPL-X 解码后在全部 SMPL22 身体关节上计算论文指标。
2. **K12 关节评测**：从原论文 Diffusion 和 E7 的 22 关节输出中抽取与 K12 模型
   完全相同的 12 个 canonical 关节块，再与原始 K12 和 400M K12 的直接输出比较。

固定 K12 索引为：

```text
[0, 4, 5, 10, 11, 13, 14, 15, 18, 19, 20, 21]
```

对应 pelvis、双膝、双脚端点、双肩、head、双肘和双腕。所有结果使用 EE4D val、
256 个相同 strided windows、8 个推理进程，rank 随机种子为 `62 + rank`。任务窗口为：

- Recon：完整有效 clip；
- Gen：`[0, 20)`，即前 2 秒；
- Fore：`[20, 40)`，即已知 2 秒之后的 2 秒。

## 3. K12 稀疏关节结果

下表所有指标均为越低越好。Position 和 Velocity 来自 `v4_beta` canonical joint
blocks，不是 SMPL-X 解码后的完整 22 关节 MPJPE。

| 任务 | 模型 | Position (mm) ↓ | Rotation (°) ↓ | Velocity (mm/s) ↓ |
| --- | --- | ---: | ---: | ---: |
| Recon | 原论文 Diffusion：22→K12 | **98.97** | **23.62** | **163.05** |
| Recon | E7：22→K12 | 100.36 | 24.07 | 173.10 |
| Recon | 原始 K12 约88M：直接K12 | 103.55 | 25.06 | 176.57 |
| Recon | **400M K12 MoE：直接K12** | 102.58 | 24.92 | 174.24 |
| Gen | 原论文 Diffusion：22→K12 | 161.36 | 30.47 | 232.63 |
| Gen | E7：22→K12 | 149.91 | **28.74** | **226.40** |
| Gen | 原始 K12 约88M：直接K12 | 155.74 | 29.71 | 234.38 |
| Gen | **400M K12 MoE：直接K12** | **149.35** | 29.69 | 232.00 |
| Fore | 原论文 Diffusion：22→K12 | 154.54 | 30.84 | 242.70 |
| Fore | E7：22→K12 | **139.83** | **29.75** | 231.86 |
| Fore | 原始 K12 约88M：直接K12 | 149.71 | 30.60 | 229.53 |
| Fore | **400M K12 MoE：直接K12** | 140.08 | 30.23 | **226.86** |

### 400M 相对原始 K12 的变化

| 任务 | Position | Rotation | Velocity |
| --- | ---: | ---: | ---: |
| Recon | -0.97 mm（-0.93%） | -0.14°（-0.57%） | -2.33 mm/s（-1.32%） |
| Gen | -6.39 mm（-4.10%） | -0.02°（-0.06%） | -2.38 mm/s（-1.02%） |
| Fore | -9.64 mm（-6.44%） | -0.37°（-1.21%） | -2.67 mm/s（-1.16%） |

这说明容量扩充最明显地改善了缺少完整动作条件时的 Gen/Fore 位置预测，而旋转和
短时速度改善较小。

## 4. 完整 22 关节论文指标

只有原论文 Diffusion 和 E7 能直接进入本表。K12 模型不输出剩余 10 个身体关节，
且本轮明确不使用恢复器，因此不能为原始 K12 或 400M K12 填写完整 22 关节结果。

单位均为 mm，越低越好：

| 任务 | 模型 | Body MPJPE ↓ | Body PA-MPJPE ↓ | Hand MPJPE ↓ | Hand PA-MPJPE ↓ | Root ↓ | Head ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Recon | 原论文 Diffusion 22 | 97.51 | **52.61** | **193.03** | 35.75 | 79.90 | 53.54 |
| Recon | **E7 22** | **90.73** | 54.20 | 199.24 | **35.65** | **71.58** | **41.70** |
| Gen | 原论文 Diffusion 22 | 210.57 | 73.89 | 326.63 | 42.02 | 191.35 | 193.94 |
| Gen | **E7 22** | **197.98** | **67.43** | **295.73** | **38.18** | **181.78** | **182.41** |
| Fore | 原论文 Diffusion 22 | 198.72 | 72.55 | 321.42 | 42.75 | 177.73 | 184.70 |
| Fore | **E7 22** | **183.44** | **67.17** | **294.59** | **40.49** | **162.01** | **169.88** |

完整 22 关节结果表明，E7 在 Gen/Fore 的所有论文指标上均优于原论文 Diffusion；
Recon 中 E7 的 Body MPJPE、Root 和 Head 更好，但 Diffusion 的 PA-Body 与原始
Hand MPJPE 略好。

## 5. 训练收敛状态

- 实验目录：`exp/smpl_ablation/k12_e7_x0_global_w8_u84k_400m_shared_chunk4`
- 最终 checkpoint：`last.ckpt`，epoch 299，global step 84300；
- 模型规模：约 400.24M 参数；
- 最终 `train/loss_epoch`：0.09699；
- 最终 `val/loss`：0.14457；
- 最终 `val/local_mse`：0.20078；
- 最终 `val/global_mse`：0.001005；
- 最终 `val/moe_aux_loss`：1.03795。

训练正常结束，日志中未发现 OOM、NCCL、NaN 或异常终止。

## 6. 原始结果文件

### K12 canonical 评测

- 400M K12：`exp/smpl_ablation/k12_e7_x0_global_w8_u84k_400m_shared_chunk4/metrics_sparse_256.json`
- 原始 K12：`exp/sparse_joint_eval_k12_comparison/k12.json`
- E7 抽取 K12：`exp/sparse_joint_eval_k12_comparison/e7.json`
- 原论文 Diffusion 抽取 K12：`exp/sparse_joint_eval_k12_comparison/diffusion.json`

### 完整 22 关节评测

- E7：`exp/ablation/e7_x0_global_w8_u84k/metrics_paper_protocol_dense22_n256_8gpu.pkl`
- 原论文 Diffusion：`exp/uem_v4b_dinov2/metrics_paper_protocol_dense22_n256_8gpu.pkl`

完整 22 关节的三任务预测文件保存在各自实验目录下，文件名为
`preds_ee4d_{recon,gen,fore}_dense22_n256_8gpu.pkl`。

## 7. 结果边界

- K12 的优势只能解释为指定 12 个关节及其辅助表示的预测能力，不能解释为完整
  22 关节动作恢复能力。
- 400M 使用 MoE，总参数量约 400M，但每个 token 仅执行 Shared Expert 和 Top-2
  Routed Experts；参数量不等同于每步激活计算量。
- 若后续要比较完整全身恢复，必须给所有 K12 模型使用同一个冻结恢复器，并将恢复器
  参数量和误差单独报告；本报告没有引入恢复器。
