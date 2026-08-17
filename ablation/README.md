# UniEgoMotion Flow Matching 消融实验

B0 与 E1–E12 的相应训练/测评均已完成；E1 仅复用 B0 checkpoint 重新测评，不需要训练。256 样本结果、配对统计和速度实验见 [`result.md`](../result.md)。以下命令均假定已进入仓库根目录，Python 默认从 `PATH` 中解析，也可用 `PYTHON_BIN=/path/to/python` 覆盖。

## 官方 Diffusion 与 E7 推理速度实验

本实验只进行 checkpoint 推理和计时，不包含训练。比较对象固定为：

| 方法 | 配置 | Checkpoint | 采样器 | NFE |
| --- | --- | --- | --- | ---: |
| 原论文 Diffusion | `config/uem.yaml` | `exp/uem_v4b_dinov2/last.ckpt` | ancestral DDPM | 1000 |
| E7 | `ablation/configs/e7_x0_global_w8_u84k.yaml` | `exp/ablation/e7_x0_global_w8_u84k/last.ckpt` | Euler | 10 |

默认协议为单张相同 GPU、验证集固定前 256 个 stride-10 样本、batch size 64、seed 62、窗口 80 帧（8 秒）、Reconstruction/Generation/Forecasting 三任务分别测量、每组重复 3 次。运行入口会先检查 GPU 是否空闲；GPU 上存在其他计算进程时默认拒绝启动，避免得到受资源竞争污染的结果。

主指标：

- CUDA event 统计的纯 `model.sample()` GPU 时间；
- 每个任务 256 样本总时间、单样本毫秒数和 samples/s；
- 8 秒动作窗口对应的实时倍数；
- batch latency 的 P50/P90/P95；
- 峰值 allocated/reserved GPU 显存；
- NFE 降幅、E7 实测加速比及跨重复标准差；
- host-to-device、checkpoint 加载、预热和数据准备时间单独记录；
- 默认额外测一次 SMPL-X 后处理，给出 prepared-pipeline 时间，避免把公共后处理误认为采样器耗时。

正式运行：

```bash
cd /path/to/UniEgoMotion
SPEED_GPU=0 bash ablation/scripts/run_diffusion_vs_e7_speed.sh
```

常用覆盖参数：

```bash
SPEED_GPU=0 \
SPEED_NUM_SAMPLES=256 \
SPEED_BATCH_SIZE=64 \
SPEED_REPEATS=3 \
SPEED_INCLUDE_POSTPROCESS=1 \
SPEED_METHOD_ORDER=diffusion_first \
SPEED_OUTPUT_DIR=exp/speed_diffusion_vs_e7_n256_b64 \
bash ablation/scripts/run_diffusion_vs_e7_speed.sh
```

如需排除方法执行先后带来的温度/频率偏差，可使用新的输出目录再运行一次 `SPEED_METHOD_ORDER=e7_first`；两次都必须在 GPU 独占且系统状态稳定时执行。

输出包括两个方法的原始 JSON、逐方法 CSV、加速比 CSV/JSON、Markdown 主表、标准输出/错误日志、软件版本以及测试前后的 `nvidia-smi -q` 快照。汇总器会强制检查 GPU、样本键哈希、样本数、batch size、任务、重复次数和软件版本完全一致；任一协议不一致时拒绝生成加速比。

相关脚本：

```text
ablation/scripts/benchmark_inference_speed.py
ablation/scripts/summarize_speed_comparison.py
ablation/scripts/run_diffusion_vs_e7_speed.sh
```

## 第三轮：Global/Local 双分支融合实验（E9–E12）

本轮以 E7（x0、Global Weight 8、Euler10）为共同基线，固定数据、随机种子、损失、global batch 512、300 epochs 和约84k updates，只比较输出分支及融合拓扑。

### 实验矩阵

| 编号 | GPU | 双分支模式 | Local→Global | Global→Local | 融合源梯度 | 训练预算 |
| --- | --- | --- | --- | --- | --- | ---: |
| E7（已有基线） | 已完成 | 原始单头 | — | — | — | 约84k updates |
| E9 | 0–1 | 无融合 | 关闭 | 关闭 | — | 约84k updates |
| E10 | 2–3 | 单向融合 | 开启 | 关闭 | stop-gradient | 约84k updates |
| E11 | 4–5 | 单向融合 | 关闭 | 开启 | stop-gradient | 约84k updates |
| E12 | 6–7 | 双向门控 | 开启 | 开启 | 双向 stop-gradient | 约84k updates |

Global 为 v4_beta 的 `[198,207)` 九维 SE(3) 增量；Local 234维由 `[0,198)` 与 `[207,243)` 拼接。两个分支各有独立的 `LayerNorm–Linear–SiLU–Dropout–LayerNorm` 特征适配器，最后按原243维顺序重组，因此现有 Flow loss、采样器和评测代码无需改变。

融合使用逐通道门控残差，门控 logit 初始化为 `-4`（sigmoid约0.018）。stop-gradient 只阻止接收分支的 loss 沿融合边污染源分支；两类 loss 仍会通过各自适配器共同训练共享 Transformer。四组均构造并执行两套门控，通过固定 `(0,0)/(1,0)/(0,1)/(1,1)` 掩码控制拓扑，因此四组总参数均为 `89,230,323`，DDP计算图一致。

训练目标仍严格使用 E7 的加权 MSE：Global 每维权重8，Local 每维权重1，按总权重归一化。新增的 `local_mse`、`global_mse` 和两个有效融合门控仅用于 TensorBoard 诊断，不参与反向目标。

### 配置与启动

配置文件：

```text
ablation/configs/e9_dual_no_fusion_w8_u84k.yaml
ablation/configs/e10_dual_l2g_stopgrad_w8_u84k.yaml
ablation/configs/e11_dual_g2l_stopgrad_w8_u84k.yaml
ablation/configs/e12_dual_bidir_gated_w8_u84k.yaml
```

四组并行启动：

```bash
cd /path/to/UniEgoMotion
bash ablation/scripts/train_global_local_fusion.sh
```

启动器会原子检查 GPU、端口、重复进程和实验目录，然后用独立端口29609–29612启动四组两卡 Lightning DDP。当前 H20/driver-535 主机需要 CUDA forward compatibility；各训练脚本默认组合 forward-compatible `libcuda` 与宿主版本 NVML，并把 NCCL JIT 缓存上限设为4 GiB。其他驱动/GPU 环境可使用 `USE_H20_CUDA_COMPAT=0` 跳过这一主机专用步骤。

日志位置：

```text
exp/ablation/e9_dual_no_fusion_w8_u84k/train.log
exp/ablation/e10_dual_l2g_stopgrad_w8_u84k/train.log
exp/ablation/e11_dual_g2l_stopgrad_w8_u84k/train.log
exp/ablation/e12_dual_bidir_gated_w8_u84k/train.log
```

正式训练前的两卡静态图检查：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH="$PWD" \
bash -c 'source ablation/scripts/setup_h20_cuda_compat.sh && \
  python -m torch.distributed.run \
  --standalone --nproc_per_node=2 tests/smoke_global_local_ddp.py'
```

### 选择规则

- 首先要求 Reconstruction Root、Last Segment 和 Head Translation 不劣于 E7；
- Generation PA-MPJPE 至少低于 E6 的93.9 mm，理想目标接近 E4 的91.1 mm；
- Reconstruction/Forecasting PA-MPJPE不得出现显著退化；
- 同时检查 Generation Root、FID、Foot Sliding和多样性，避免通过压低动作幅度取巧；
- 若 E12 对 E10 没有显著收益，优先选择结构更简单、因果方向更明确的 E10。

---

## 第二轮：x0 全局增量权重实验（E5–E8）

状态：E5–E8 训练与 256 样本测评已完成。E4（x0、旋转权重1、平移权重1）作为本轮共同参照，未重复训练；完整结果见 [`result.md`](../result.md)。

### 实验矩阵

| 编号 | GPU | 预测目标 | 旋转 `[198, 204)` | 平移 `[204, 207)` | 全局加权总量 | 训练预算 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| E4（已有） | 已完成 | x0 | 1 | 1 | 9 | 约84k updates |
| E5 | 0–1 | x0 | 2 | 2 | 18 | 约84k updates |
| E6 | 2–3 | x0 | 4 | 4 | 36 | 约84k updates |
| E7 | 4–5 | x0 | 8 | 8 | 72 | 约84k updates |
| E8 | 6–7 | x0 | 6 | 12 | 72 | 约84k updates |

全局加权总量定义为 `6 × rotation_weight + 3 × translation_weight`。E7 与 E8 的总量及损失分母完全一致；E7 的旋转/平移模块总贡献为48/24，E8为36/36，因此二者只比较全局权重在旋转和平移之间的分配。

所有新实验固定：2卡、每卡batch 256、global batch 512、300 epochs、约84k optimizer updates、`bf16-mixed`、LR `8.5e-5`、EMA `0.992028`、Beta(1.5,1)时间采样以及Euler10。除损失权重外均与E4一致。

### 配置文件

```text
ablation/configs/e5_x0_global_w2_u84k.yaml
ablation/configs/e6_x0_global_w4_u84k.yaml
ablation/configs/e7_x0_global_w8_u84k.yaml
ablation/configs/e8_x0_rot6_trans12_u84k.yaml
```

E5–E7沿用 `FLOW.GLOBAL_WEIGHT`。E8设置：

```yaml
FLOW:
  GLOBAL_WEIGHT: 1.0
  GLOBAL_ROT_WEIGHT: 6.0
  GLOBAL_TRANS_WEIGHT: 12.0
  GLOBAL_FEATURE_START: 198
  GLOBAL_FEATURE_END: 207
```

当两个分权项都为 `None` 时，代码严格沿用原 `GLOBAL_WEIGHT`；两个分权项必须同时设置，而且当前仅接受9维SE(3)切片。

### 启动训练

一键后台启动四组：

```bash
cd /path/to/UniEgoMotion
bash ablation/scripts/train_x0_global_sweep.sh
```

启动器会先检查重复进程和已有非空实验目录；任一预检查失败时不会启动任何实验。也可以分别以前台方式启动：

```bash
bash ablation/scripts/train_e5_x0_global_w2.sh
bash ablation/scripts/train_e6_x0_global_w4.sh
bash ablation/scripts/train_e7_x0_global_w8.sh
bash ablation/scripts/train_e8_x0_rot6_trans12.sh
```

日志分别位于：

```text
exp/ablation/e5_x0_global_w2_u84k/train.log
exp/ablation/e6_x0_global_w4_u84k/train.log
exp/ablation/e7_x0_global_w8_u84k/train.log
exp/ablation/e8_x0_rot6_trans12_u84k/train.log
```

### 监控与测评

```bash
tail -n 5 exp/ablation/e5_x0_global_w2_u84k/train.log
tail -n 5 exp/ablation/e6_x0_global_w4_u84k/train.log
tail -n 5 exp/ablation/e7_x0_global_w8_u84k/train.log
tail -n 5 exp/ablation/e8_x0_rot6_trans12_u84k/train.log
```

训练完成后使用统一单卡、256样本、Euler10协议。例如E5：

```bash
CUDA_VISIBLE_DEVICES=0 EVAL_NUM_GPUS=1 bash ablation/scripts/eval_checkpoint.sh \
  ablation/configs/e5_x0_global_w2_u84k.yaml \
  last_ckpt 10 _ablation_e5_x0_global_w2_euler10
```

其余实验使用各自YAML和其中的 `TRAIN.EVAL_SUFFIX`。主指标为三任务的关键关节MPJPE、PA-MPJPE、root translation error和每20帧root漂移；E7与E8还必须比较head rotation/translation，避免用旋转退化换取平移改善。

### 选择规则

- root误差相对E4至少下降5%才视为实质收益；
- PA-MPJPE相对E4恶化不得超过2%；
- 手部MPJPE与head rotation恶化不得超过5%；
- 若E8优于E7，说明瓶颈主要位于平移增量；若E7优于E8，说明统一提高SE(3)权重更合适；
- 只对最终胜出配置运行完整约4400样本的Foot、Semantic Similarity和FID测评。

---

## 第一轮：velocity、更新次数、求解器与x0目标（已完成）

本目录固定当前已经训练完成的 velocity Flow Matching 模型作为基线，并定义四个单变量消融实验。当前机器使用 8 张 H20，并按互不重叠的 GPU 集合运行实验。所有脚本都会自动进入项目根目录并设置 `PYTHONPATH`，避免出现 `No module named config`。

## 实验矩阵

| 编号 | GPU | 预测目标 | 更新次数 | 全局特征权重 | Euler 步数 | 是否训练 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| B0 | 已完成 | velocity | 约 84k | 1 | 10 | 否，现有基线 |
| E1 | 0–1 | velocity | 约 84k | 1 | 50 | 否，只重新测评 |
| E2 | 2–3 | velocity | 约 84k | 8 | 10 | 是 |
| E3 | 0、1、4、5 | velocity | 约 168k | 1 | 10 | 是 |
| E4 | 6–7 | x0 | 约 84k | 1 | 10 | 是 |

基线 checkpoint：

```text
exp/uem_flow_8gpu_e300/last.ckpt
```

E2–E4 的输出分别保存到 `exp/ablation/` 下的独立目录，不会覆盖基线或其他实验。

## 多卡训练配置

所有训练组都保持有效全局 batch 512：

```text
E2、E4：每卡 batch 256 × 2 张 GPU × 梯度累计 1 = 512
E3：每卡 batch 128 × 4 张 GPU × 梯度累计 1 = 512
```

因此继续保持原实验的学习率、EMA 和更新次数口径：

- E2、E4：300 epochs，每个 epoch 约 280 次更新，总计约 84k updates；
- E3：600 epochs，总计约 168k updates；
- 学习率：`8.5e-5`；
- EMA decay：`0.992028`；
- 精度：`bf16-mixed`；
- 其他模型结构、数据、条件掩码、Beta(1.5, 1) 时间采样均保持不变。

注意，“全局特征权重”指动作表示中全局 SE(3) 增量的 loss 权重，不是 global batch size。

## 各实验改变的变量

### B0：现有基线

直接使用已经训练完成的 velocity checkpoint，Euler 10 步。它是所有实验的共同对照组。

### E1：Euler 50

完全复用 B0 checkpoint，只把推理积分步数从 10 改为 50。该实验不需要训练，用于判断全局偏移是否主要来自 ODE 离散积分误差。

### E2：提高全局位姿权重

将 v4_beta 表示中的全局 SE(3) delta 特征 `[198, 207)` 权重设为 8。加权损失按权重总和归一化：

```text
loss = sum(weight × squared_error) / sum(weight)
```

`GLOBAL_WEIGHT: 1.0` 与原始所有维度等权 MSE 完全一致。

### E3：增加更新次数

保持 velocity、全局权重 1 和 Euler 10 不变，只把训练从约 84k updates 增加到约 168k updates，用于判断当前模型是否欠训练。

实际执行时，E3 曾从双卡阶段生成的 epoch 4、global step 1400 checkpoint 恢复，并切换到 GPU 0、1、4、5 四卡训练；有效全局 batch 和 optimizer/scheduler 状态保持不变。这只是运行中断后的恢复记录。仓库中的 E3 配置默认 `MODEL.CKPT_PATH: None`，新环境可直接从零训练；只有继续同一实验目录时才显式覆盖为 `MODEL.CKPT_PATH last_ckpt`。

E3 在 epoch 300 附近产生了一个约 84k updates 的中间 checkpoint，用于和最终 168k 结果做同一训练轨迹内的严格比较。

### E4：直接预测 x0

使用与 velocity 实验完全相同的插值路径：

```text
x_t = (1 - t) x0 + t noise
```

训练目标改为干净动作 `x0`。推理时再转换成 ODE velocity：

```text
v = (x_t - x0_pred) / t
```

用于验证 FlowMotion 风格的 target prediction 是否能降低关键关节误差、抖动和全局漂移。

## 启动命令

四个脚本已经固定到互不重叠的 GPU 集合：

```bash
./ablation/scripts/eval_e1_euler50.sh
./ablation/scripts/train_e2_global_w8.sh
./ablation/scripts/train_e3_updates168k.sh
./ablation/scripts/train_e4_x0.sh
```

后台运行时的日志位置：

```text
exp/uem_flow_8gpu_e300/eval_e1_euler50.log
exp/ablation/e2_global_w8_u84k/train.log
exp/ablation/e3_velocity_u168k/train.log
exp/ablation/e4_x0_u84k/train.log
```

## checkpoint 测评

任意训练组完成后，可以指定 Euler 步数和唯一输出后缀：

```bash
./ablation/scripts/eval_checkpoint.sh \
  ablation/configs/e2_velocity_global_w8_u84k.yaml \
  last_ckpt \
  10 \
  _ablation_e2_global_w8_euler10
```

对 x0 模型建议分别测 Euler 10 和 Euler 50：

```bash
./ablation/scripts/eval_checkpoint.sh \
  ablation/configs/e4_x0_u84k.yaml \
  last_ckpt \
  10 \
  _ablation_e4_x0_euler10

./ablation/scripts/eval_checkpoint.sh \
  ablation/configs/e4_x0_u84k.yaml \
  last_ckpt \
  50 \
  _ablation_e4_x0_euler50
```

E3 最终 168k 结果使用 `last.ckpt`。约 84k 的中间结果应使用 epoch 300 附近的 checkpoint：

```bash
ls exp/ablation/e3_velocity_u168k/epoch=299-*.ckpt
```

若 E3 训练被中断，可从同一目录安全恢复：

```bash
python run/train_uem.py \
  CONFIG ablation/configs/e3_velocity_u168k.yaml \
  MODEL.CKPT_PATH last_ckpt
```

## 统一测评口径

快速筛选统一使用 256 个样本、固定随机种子和相同的 12 个关键关节。至少比较：

- 关键关节全局 MPJPE；
- root translation error；
- 最后一帧 root error；
- 首帧对齐后的 trajectory drift；
- 关键关节 PA-MPJPE；
- 头部平移和旋转误差；
- foot sliding；
- NFE、推理耗时和峰值显存。

### 论文口径与配对统计

已有预测可按论文的 22 关节和任务时间窗重新统计：

```bash
python ablation/scripts/compute_paper_joint_metrics.py \
  --exp-path exp/ablation/e7_x0_global_w8_u84k \
  --eval-suffix _ablation_e7_x0_global_w8_euler10 \
  --data-dir "${UEM_DATA_DIR:-data/ee4d_motion_uniegomotion}"
```

`result.md` 中的配对置信区间由 `paired_bootstrap_metrics.py` 复算。脚本会先校验预测字典的样本键和顺序，再以候选减基线为差值，默认使用固定 seed 62 做 10,000 次 percentile bootstrap：

```bash
python ablation/scripts/paired_bootstrap_metrics.py \
  --experiment E7 exp/ablation/e7_x0_global_w8_u84k _ablation_e7_x0_global_w8_euler10 \
  --experiment E9 exp/ablation/e9_dual_no_fusion_w8_u84k _ablation_e9_dual_no_fusion_w8_euler10 \
  --compare E7 E9 \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 62 \
  --output exp/ablation/e7_vs_e9_bootstrap.json
```

可以重复 `--experiment` 和 `--compare`，在一次运行中汇总多组比较；生成的 JSON 属于实验产物，不提交到 Git。

本轮遵循了先完成四个单变量实验、再组合改动的顺序；后续组合实验不替代单变量消融结论。
