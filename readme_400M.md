# K12 400M Motion Expert

本文档记录 K12 稀疏关节实验的 400M 参数 Motion Expert 设计。它是一个独立的运动生成专家，不是视觉语言模型，也不是全身恢复器：模型仍然只预测 K12 稀疏表示，恢复器和全身 243D 解码不属于该模型的训练目标。

## 1. 目标与边界

该实验只回答一个问题：在输入和输出都限定为 K12 稀疏运动表示时，扩大运动主干容量是否能改善 K12 的生成、重建和预测误差。

以下内容保持不变：

- 数据划分、窗口长度和评测协议；
- `v4_beta` 稀疏表示及 12 个关节集合；
- Aria trajectory、DINOv2 特征和有效帧 mask 的条件接口；
- Flow Matching、`PREDICTION_TYPE: x0`、Euler 采样和全局特征权重；
- 输出头的 153 维形状。

以下内容被替换：

- 原来每个 Transformer block 中的单个 FFN；
- 替换为每层一个独立 Shared Global Expert、11 个 Routed Experts 和一个
  条件 Top-2 router。

因此，该实验不能被解释为“400M 模型恢复了 22 个关节”。它仍然只输出 K12 的 153 维表示；全身恢复器应作为独立的 evaluation-only 模块进行测评。

## 2. 输入和输出

### 输入

对长度为 `T=80` 的窗口，模型输入为：

| 名称 | 形状 | 说明 |
| --- | --- | --- |
| `x_t` | `[B, 80, 153]` | 加噪/插值后的 K12 `v4_beta` 运动状态 |
| `timesteps` | `[B]` | Flow 时间 `t`，范围约为 `[0, 1]` |
| `y["traj"]` | `[B, 80, 18]` | Aria trajectory 条件 |
| `y["img_embs"]` | `[B, 80, 1024]` | 已提取的 DINOv2 特征 |
| `y["valid_frames"]` | `[B, 80]` | 有效帧 mask |

`DINOv2` 编码器不在本模型中重新训练；训练数据加载器提供的 1024 维特征直接经过条件投影层。模型也不读取全身 243D 标签作为输入。

### 输出

模型输出仍为：

```text
[B, 80, 153]
```

其中：

```text
12 个选定关节 × 9D + v4_beta 辅助 45D = 153D
```

这是当前 `FLOW.PREDICTION_TYPE: x0` 配置下的 clean `x0` 预测；如果将来切换为 velocity，张量形状不变，只改变输出的物理含义。不会输出剩余 10 个身体关节，也不会输出 SMPL-X 的完整 145 个节点。

## 3. 主干结构

### 3.1 共享部分

- hidden dimension：`768`；
- temporal blocks：`12` 层；
- attention heads：`12`；
- 每层保留原有 self-attention、condition cross-attention、残差和 LayerNorm；
- 输入投影：`153 -> 768`；
- 输出投影：`768 -> 153`。

### 3.2 MoE FFN

每个 block 的 FFN 改为：

```text
shared expert: always active, one independent copy per layer
router:       3072 -> 11 logits
expert_i:     LayerNorm(768)
              Linear(768, 1536)
              GELU
              Dropout
              Linear(1536, 768)
              Dropout
```

Router 同时使用 motion hidden、pose hidden、egovideo hidden 和 Flow timestep。
80 帧按 4 帧划分为 20 个 temporal chunks；一个 chunk 内的帧共享同一组
Top-2 Routed Experts。Shared Expert 始终激活，最终输出为 Shared 分支与
Top-2 Routed 分支的缩放加权和。实验不加入 temporal smoothness loss，只使用
Routed Experts 的负载均衡项。

### 3.3 参数量

在 `12 layers × 12 experts × hidden 768 × FFN 1536` 的设置下：

- 12 层各自拥有 `1 Shared + 11 Routed`，共 144 个 FFN expert；
- 12 层 MoE 主干加输入/条件/输出投影约 **400.24M 参数**；
- 训练时 Top-2 只执行被选中的两个 expert 的 token 子集，参数量和激活计算量是两个不同概念。

参数量是按 `model.parameters()` 统计的，包含 router、所有 expert、attention、条件投影和输出层；预计算的 DINOv2 特征提取器、SMPL-X、数据集和 evaluation-only 恢复器不计入 Motion Expert 参数量。

## 4. Router 负载均衡

每层计算 Switch-style auxiliary loss：

```text
L_balance = E × sum_e(mean_router_probability[e] × top1_load[e])
```

总训练损失为：

```text
L_total = L_flow + 0.01 × mean(L_balance over blocks)
```

权重由 `MODEL.MOTION_EXPERT.LOAD_BALANCE_WEIGHT` 控制。该项只约束 expert 使用均衡，不改变 153D 运动监督。日志中会记录 `train/moe_aux_loss` 或 `val/moe_aux_loss`。

由于 Top-2 动态路由可能在一个 batch 中没有 token 进入某些 expert，多卡训练自动使用 `find_unused_parameters=True`，并关闭 DDP static graph；这是必要的运行设置，不是模型结构变化。

训练初始化采用已有 K12 dense checkpoint：每层 dense FFN 复制到对应的
Shared Expert，Routed Experts 从 Shared Expert 复制并加入小扰动。前 30 个
epoch 先训练 Router/Routed Experts，之后解冻 Shared Expert 和主干。

## 5. One-step hidden

Flow 采样从 `t=1` 的纯噪声开始。第一次模型前向时，在最后一层输出投影之前
导出 motion hidden：

```text
x_1 --first reverse denoising call--> h_one_step [B, 80, 768]
     --continue Euler rollout-------------> final K12 motion [B, 80, 153]
```

该 hidden 是 LaMP 风格的部分去噪 motion prior，不替代完整十步 Flow 输出。
通过 `FlowMatching.sample_loop(..., return_one_step_hidden=True)` 可以获得
`(final_sample, one_step_hidden)`。当前模型的数值时间方向与其他工作可能相反，
不要直接复制外部论文的时间数值。

## 6. 配置和启动

新增配置文件：

```text
smpl_ablation/configs/k12_e7_x0_global_w8_u84k_400m.yaml
```

启动命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python run/train_uem.py CONFIG \
  smpl_ablation/configs/k12_e7_x0_global_w8_u84k_400m.yaml
```

配置默认使用：

- 8 卡 DDP；
- 每卡 batch size 16；
- 梯度累积 4，因此有效 global batch size 为 `16 × 8 × 4 = 512`；
- `bf16-mixed`；
- 300 epochs；
- 新实验目录 `exp/smpl_ablation/k12_e7_x0_global_w8_u84k_400m_shared_chunk4`。

如果显存不足，优先降低每卡 `DATA.BATCH_SIZE` 并同比增大 `TRAIN.ACCUMULATE_GRAD_BATCHES`，使有效 global batch size 保持不变。不要把该配置直接用于已有 88M K12 checkpoint 的 `last_ckpt` 恢复。

## 7. Checkpoint 迁移

88M K12 模型的 attention、输入投影、条件投影和输出投影可以作为初始化参考；
原来的 `ff.net.*` 会复制到对应层的 `ff.shared.*`，再初始化 11 个 routed 分支：

- 默认配置会自动读取 `INIT_DENSE_CKPT_PATH` 并完成上述转换；
- Router 从随机小权重开始，Routed 分支使用小 gate；
- 旧的 all-routed 400M checkpoint 不作为该架构的严格初始化来源。

## 8. 与 LaMP、HEX、Psi0 的借鉴关系

这里借鉴的是设计思想，不是直接复用它们的 checkpoint 或输入协议：

- LaMP 的启发是把运动序列视为独立的时序建模对象，并可在后续增加 mask-span motion pretraining；本实现首轮仍只使用现有 Flow Matching 目标，不混入语言损失。
- HEX 的启发是使用专门的 proprioceptive/motion expert 和门控专家分工；本实现对应为 K12 motion-only 的 Top-2 MoE FFN。
- Psi0 的启发是将运动/action expert 作为独立模块与视觉条件解耦；本实现保留 DINOv2 作为条件，但不引入语言模型、VLM 或机器人 action head。

因此，论文表述建议使用：

> a 400M K12 sparse-motion expert with top-2 mixture-of-experts temporal blocks

不要写成 LaMP/HEX/Psi0 的复现，也不要声称它本身完成稀疏到全身的恢复。

## 9. 公平对比建议

至少保留以下三个模型：

1. 原始 K12 约 88M backbone；
2. K12 400M MoE Motion Expert；
3. 同训练数据、同输入条件和同采样步数的 dense 400M 对照（如果后续实现）。

三者都只测 K12 的 observed joint MPJPE/MPJRE/MPJVE、全局轨迹误差和生成质量。若加入 evaluation-only 全身恢复器，应把恢复结果另列，不能将恢复器的能力归因给 Motion Expert。

## 10. 实现文件

- `model/core.py`：Shared + Routed `MoEFeedForward`、chunk 条件 Router；
- `model/uniegomotion.py`：K12 Motion Expert 开关、MoE block 构造和 auxiliary loss 汇总；
- `module/uem_module.py`：将 router 负载均衡项加入 Flow/ Diffusion 训练损失；
- `run/train_uem.py`：MoE 动态路由下的 DDP 设置；
- `config/defaults.py`：`MODEL.MOTION_EXPERT` 默认配置；
- `smpl_ablation/configs/k12_e7_x0_global_w8_u84k_400m.yaml`：可直接启动的 400M K12 实验配置。

## 11. 参考工作

- [LaMP: Learning Vision-Language-Action Policy with 3D Scene Flow as Latent Motion Prior](https://arxiv.org/abs/2603.25399)
- [HEX: Open-X-Humanoid](https://github.com/Open-X-Humanoid/HEX)
- [Psi0](https://github.com/physical-superintelligence-lab/Psi0)
