# 400M Motion Expert：前三项实验结果

更新时间：2026-08-20

模型：`K12 400M Shared Global Expert + 11 Routed Experts`（400,242,729 参数）

Checkpoint：`exp/smpl_ablation/k12_e7_x0_global_w8_u84k_400m_shared_chunk4/last.ckpt`（epoch 299，global step 84300，使用 checkpoint 内 EMA 权重）
K12：`[0, 4, 5, 10, 11, 13, 14, 15, 18, 19, 20, 21]`

## 结论摘要

1. **Routed Experts 确实有用**：只保留 Shared Expert 时，三项任务的位置误差上升约 **10.66%～11.98%**。因此当前 400M 模型不是仅靠 Shared Expert 工作。
2. **但是当前 Router 的选择作用偏弱**：Top-1 和训练时的 Top-2 几乎具有相同的位置误差；Random Top-2 也只比正常 Top-2 差约 0.38%～0.58%。这说明 routed expert 的整体容量有贡献，但“输入应该交给哪两个专家”的专门化还不强。
3. **One-step hidden 可作为动作先验，但还不能视为 final hidden 的等价替代**：同一线性 probe 下，step-10 相比 step-1 的位置、旋转和速度误差分别降低 **6.55%、2.86%、14.89%**。不过 step-1 已经能恢复大部分未来 K12 运动信息，适合作为后续 VLM 融合的低延迟 motion prior。
4. **负载均衡实现得很好，但路由很分散**：11 个专家的总体 Top-1 使用率为 8.19%～9.93%，CV 仅 **4.78%**，没有死亡专家；Router 熵达到理论最大值的 92%～93%，任务偏好接近均匀，说明专家尚未形成明显的 recon/fore/gen 分工。
5. **Chunk 路由切换频繁**：相邻 4 帧 chunk 的 Top-1 专家切换率为 **81.55%～85.25%**。本模型按既定设计没有时间平滑损失，因此这是实际学到的行为；它略低于均匀随机路由的理论切换率 90.91%，但时间连续性仍然较弱。

---

## 实验一：Shared-only / Top-1 / Top-2 / Random Top-2

### 协议

- 官方 validation split，stride=10 后取 256 个窗口。
- 每个窗口 80 帧，10-step Euler Flow Matching。
- 四种路由使用相同样本顺序，并针对每项任务重置为相同的 Flow 初始噪声。
- `Random Top-2` 使用独立的确定性随机路由种子 62，不消耗或改变 Flow 噪声随机数。
- 指标只计算模型实际预测的 K12，未使用恢复器。
- 速度使用单张 NVIDIA H20、batch=16、CUDA Event；每个模式/任务先运行一个不计时 warm-up batch。Router 统计使用单独的非计时遍历，不污染 Top-2 延迟。
- 当前专家分发是普通 PyTorch indexing，并非 fused MoE kernel，因此速度代表当前代码实现。

### 完整结果

| 路由模式 | 任务 | 位置误差 ↓ (mm) | 旋转误差 ↓ (°) | 位置速度误差 ↓ (mm/s) | 延迟 (ms/window) ↓ | 吞吐 (window/s) ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Top-2 | Recon | 101.999 | 24.733 | 173.729 | 23.417 | 42.70 |
| Top-2 | Fore | 139.628 | 29.738 | 229.063 | 23.410 | 42.72 |
| Top-2 | Gen | 147.850 | 29.893 | 231.720 | 23.332 | 42.86 |
| Top-1 | Recon | 102.457 | 24.799 | 177.793 | 22.160 | 45.13 |
| Top-1 | Fore | 139.316 | 29.782 | 232.059 | 21.957 | 45.54 |
| Top-1 | Gen | 147.678 | 30.052 | 236.456 | 21.938 | 45.58 |
| Shared-only | Recon | 114.216 | 26.281 | 179.000 | 7.055 | 141.73 |
| Shared-only | Fore | 155.610 | 30.881 | 232.138 | 7.052 | 141.80 |
| Shared-only | Gen | 163.604 | 30.977 | 231.049 | 7.043 | 141.98 |
| Random Top-2 | Recon | 102.387 | 24.755 | 179.372 | 22.659 | 44.13 |
| Random Top-2 | Fore | 140.438 | 29.738 | 231.953 | 22.768 | 43.92 |
| Random Top-2 | Gen | 148.536 | 29.624 | 231.928 | 23.027 | 43.43 |

### 相对正常 Top-2 的变化

| 模式 | Recon 位置 | Fore 位置 | Gen 位置 | 平均吞吐 | 解释 |
|---|---:|---:|---:|---:|---|
| Top-1 | +0.45% | -0.22% | -0.12% | **1.06×** | 位置几乎不变，但速度误差增加 1.31%～2.34% |
| Shared-only | +11.98% | +11.45% | +10.66% | **3.32×** | Routed branch 对准确率有明确贡献 |
| Random Top-2 | +0.38% | +0.58% | +0.46% | **1.02×** | 专家选择仅产生很小影响，路由专门化不足 |

### 如何理解

Top-1 的平均位置误差为 129.817 mm，Top-2 为 129.826 mm，二者在本次 256 条、单随机种子测试中可视为持平；Top-1 的平均旋转和速度误差略差。Random Top-2 也非常接近正常 Top-2。因此现阶段能得到的稳妥结论是：

- 多个 Routed Experts 形成的附加变换能力有价值，因为 Shared-only 明显退化。
- 尚无充分证据证明 Router 学到了强专家选择策略；它更像是在均衡调用一组相近的专家。
- 如果优先考虑当前实现的推理速度，Top-1 是较合理的部署消融；如果保持训练设定和完整能力，仍使用 Top-2。

这不是统计显著性结论。要在论文中主张 Top-1 与 Top-2 等价，还需要多个采样种子和 take-level bootstrap 置信区间。

---

## 实验二：One-step / Step-5 / Final-step Hidden Probe

### Probe 定义

- 400M backbone、Router 和 Experts 全部冻结，仅训练独立 probe。
- 模式固定为训练时的 Top-2。
- 任务固定为 forecasting：输入可见帧 0～19，probe 预测未来帧 20～39 的 K12 参数。
- 每个 token 的目标为 12 个关节的 `6D rotation + 3D position`，共 108 维。
- 三个 step 使用完全相同的轻量结构：`LayerNorm(768) + Linear(768,108)`，每个 probe **84,588 参数**。
- Probe 使用官方 train split 的 1,024 条训练样本和 128 条 dev 样本；最终结果在官方 val split 的 256 条 held-out 样本上计算。
- `step-1 / step-5 / step-10` 分别是 Flow 时间 `t=1.0 / 0.6 / 0.1` 的 Transformer hidden。这里的 step-10 是第十次模型调用的 hidden；最终 Euler 更新后没有额外进行一次 `t=0` re-encode。

### 结果

| Hidden | Flow t | 最佳 dev epoch | Val normalized MSE ↓ | 位置误差 ↓ (mm) | 旋转误差 ↓ (°) | 位置速度误差 ↓ (mm/s) | 跨噪声 cosine ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Step-1 | 1.0 | 17 | 0.8319 | 149.168 | 27.785 | 289.163 | **0.9825** |
| Step-5 | 0.6 | 22 | 0.7925 | 142.184 | 27.086 | 323.439 | 0.7699 |
| Step-10 | 0.1 | 42 | **0.7837** | **139.394** | **26.990** | **246.098** | 0.8686 |

跨噪声稳定性使用相同的 64 条 validation 样本，比较种子 62 与种子 63、64 的对应帧 hidden cosine。

### 如何理解

- Step-10 的未来轨迹线性可解码性最好。相对 Step-1，其 normalized MSE、位置、旋转、速度误差分别下降 5.79%、6.55%、2.86%、14.89%。
- Step-1 并不是无意义的噪声 token：84K 参数的线性 probe 已能达到 149.17 mm，说明第一步已经由 egovideo、pose/traj 条件和 timestep 注入了明显的动作意图。
- Step-5 的位置和旋转比 Step-1 更好，但速度误差反而上升 11.85%，说明去噪过程中的时序平滑性并非单调改善。
- Step-1 的 cosine 最高不应直接解释成“语义最好”。它更可能表示第一步 hidden 被跨噪声共享的条件特征和残差结构主导；Step-10 虽然 cosine 较低，但对未来运动的可解码性最好。

因此，若目标是像 LaMP 一样提供一个低延迟 motion prior，**one-step hidden 是可行的**；若目标是获得尽可能准确、平滑的运动表征，当前证据仍支持使用最后一步 hidden。后续融合实验应同时报告 VLM 下游收益和额外延迟，而不能只看 hidden cosine。

---

## 实验三：Router 行为分析

统计范围为正常 Top-2 推理的 256 个 validation 窗口、三项任务、全部 10 次 Flow 模型调用和 12 层 Transformer。只统计有效的 4 帧 temporal chunk，不将 timestep prefix 当作动作 chunk。

### 熵与时间切换

| 任务 | 归一化 Router 熵 ↑ | 相邻 chunk Top-1 切换率 |
|---|---:|---:|
| Recon | 0.9211 | 81.55% |
| Fore | 0.9301 | 84.49% |
| Gen | 0.9317 | 85.25% |

熵除以 `log(11)`，1.0 表示完全均匀。三项任务均超过 0.92，说明 Router 概率分布较平坦；生成与预测的熵和切换率高于重建。

### 11 个 Routed Experts 的总体 Top-1 负载

| Expert | E1 | E2 | E3 | E4 | E5 | E6 | E7 | E8 | E9 | E10 | E11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 使用率 | 8.19% | 8.85% | 9.23% | 9.35% | 9.30% | 8.91% | 9.10% | 8.78% | 8.85% | 9.51% | 9.93% |

- 理想均匀使用率为 9.09%。
- 专家负载 CV 为 **4.78%**，最小 8.19%，最大 9.93%，没有死亡或被垄断的专家。
- 每个专家收到的 recon/fore/gen 比例大致都在 31.66%～35.07% 之间，接近三等分；尚未观察到明显的任务专家。

### Router 结论

负载均衡损失完成了“避免专家塌缩”的目标，但当前结果更接近“均衡、相似的专家集合”，而不是“不同输入激活不同能力专家”。正常 Top-2 与 Random Top-2 的小差距和高路由熵互相印证。下一轮若希望强化专家专门化，优先验证降低 load-balance 权重、增加 router logit temperature/稀疏度约束或增加 expert diversity loss；这些都应作为新实验，不能从当前结果直接假定有效。

### 可视化

- [各任务、各层专家负载热图](exp/smpl_ablation/k12_e7_x0_global_w8_u84k_400m_shared_chunk4/moe_analysis/router_figures/router_load_by_task_layer.png)
- [Router 熵与 chunk 切换率](exp/smpl_ablation/k12_e7_x0_global_w8_u84k_400m_shared_chunk4/moe_analysis/router_figures/router_entropy_and_switching.png)
- [各专家任务构成](exp/smpl_ablation/k12_e7_x0_global_w8_u84k_400m_shared_chunk4/moe_analysis/router_figures/router_task_preference.png)
- [三个去噪步骤的 hidden probe](exp/smpl_ablation/k12_e7_x0_global_w8_u84k_400m_shared_chunk4/moe_analysis/probe_figures/denoising_hidden_probe.png)

---

## 原始结果与复现

原始数据：

- [路由消融与 Router 统计 JSON](exp/smpl_ablation/k12_e7_x0_global_w8_u84k_400m_shared_chunk4/moe_analysis/routing_ablation_256.json)
- [Hidden probe JSON](exp/smpl_ablation/k12_e7_x0_global_w8_u84k_400m_shared_chunk4/moe_analysis/denoising_hidden_probe.json)
- Probe checkpoint：`exp/smpl_ablation/k12_e7_x0_global_w8_u84k_400m_shared_chunk4/moe_analysis/probe_checkpoints/`

路由消融：

```bash
/root/miniconda3/envs/uem/bin/python ablation/scripts/evaluate_400m_moe_routing.py \
  --config smpl_ablation/configs/k12_e7_x0_global_w8_u84k_400m.yaml \
  --checkpoint exp/smpl_ablation/k12_e7_x0_global_w8_u84k_400m_shared_chunk4/last.ckpt \
  --data-dir /gaozt-test1/guanzerong/datasets/ee4d_motion_uniegomotion \
  --output exp/smpl_ablation/k12_e7_x0_global_w8_u84k_400m_shared_chunk4/moe_analysis/routing_ablation_256.json \
  --num-samples 256 --batch-size 16 --device cuda:0
```

Hidden probe：

```bash
/root/miniconda3/envs/uem/bin/python ablation/scripts/probe_400m_denoising_hidden.py \
  --config smpl_ablation/configs/k12_e7_x0_global_w8_u84k_400m.yaml \
  --checkpoint exp/smpl_ablation/k12_e7_x0_global_w8_u84k_400m_shared_chunk4/last.ckpt \
  --data-dir /gaozt-test1/guanzerong/datasets/ee4d_motion_uniegomotion \
  --output exp/smpl_ablation/k12_e7_x0_global_w8_u84k_400m_shared_chunk4/moe_analysis/denoising_hidden_probe.json \
  --device cuda:1 --probe-device cuda:1 \
  --train-samples 1024 --dev-samples 128 --val-samples 256 \
  --extract-batch-size 16 --probe-epochs 60 \
  --stability-samples 64 --stability-seeds 3
```

## 代码与兼容性说明

- `model/core.py` 新增四种**仅评测时**生效的路由模式；模型处于 training mode 时仍强制使用配置中的 Top-2，不改变已有训练逻辑。
- `model/uniegomotion.py` 新增将评测路由模式同步到 12 层的接口。
- 新增两个独立分析脚本，不修改 checkpoint，不训练或覆盖 400M 主模型。
- 稀疏配置/模型边界测试已通过。环境没有安装 `pytest`，因此依赖 `pytest` 的第二个已有测试文件未执行；本次新增脚本已完成 2 条样本端到端冒烟测试和完整 256 条运行。
