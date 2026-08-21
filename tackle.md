# NymeriaPlus → EE4D-Motion / UniEgoMotion 处理方案

> 版本：v1（2026-08-20）  
> 原始数据目录：`dataset/NymeriaPlus_100h`  
> 目标：把约 100 小时 NymeriaPlus 转为与当前 EE4D-Motion / UniEgoMotion 训练、验证和推理管线兼容的数据，同时保留 NymeriaPlus 的头部轨迹、第一视角图像特征和多层级文本标注。

## 1. 总体结论

推荐采用“**离线标准化 + 统一适配层 + mask-aware 联合训练**”方案：

```text
NymeriaPlus 原始文件
    ├── standard SMPL（约 240 FPS，TIME_CODE）
    ├── Aria VRS + MPS 轨迹
    └── narration CSV（DEVICE_TIME）
               │
               ▼
完整性清单与 participant-disjoint 划分
               │
               ▼
统一 TIME_CODE 时间轴
    ├── 动作/头轨迹：10 FPS
    ├── RGB/DINOv2：5 FPS
    └── 文本：区间标注映射到 10 FPS 帧和 8 秒窗口
               │
               ▼
身体与坐标系标准化
    ├── SMPL24 → UniEgoMotion SMPL22
    ├── standard SMPL → SMPL-X 兼容桥接
    ├── 地面高度 + 足部接触
    └── T_world_device → aria_traj
               │
               ▼
243D v4_beta + 18D aria_traj_repre + DINOv2 + text + validity masks
               │
               ▼
EE4D / NymeriaPlus source-aware 联合 Dataset、损失、归一化和评价
```

不能直接把 NymeriaPlus 的标准 SMPL 参数当作 SMPL-X 参数使用。两者前 22 个关节的语义和姿态顺序可映射，但 shape space、模型拓扑及手部参数并不等价。因此主体姿态可直接映射，shape 和最终解码需要显式桥接或拟合。

另外，当前工程虽然存在部分 text condition 参数，但实际训练路径没有使用文本。第一阶段应完成可靠的文本时间对齐和数据接口；如果目标是文本条件动作生成，还需要单独增加文本编码器、条件融合和 classifier-free guidance，不能只靠写入文本字段实现。

## 2. 当前格式与关键约束

### 2.1 EE4D / UniEgoMotion 的目标格式

当前 `v4_beta` 动作表示共 243 维：

| 区间 | 维度 | 含义 | NymeriaPlus 有效性 |
|---|---:|---|---|
| `[0:198)` | 198 | 22 个关节，每个关节 6D 全局旋转 + 3D 平移 | 有效 |
| `[198:207)` | 9 | canonical/global residual transform | 有效 |
| `[207:219)` | 12 | 左手 PCA | 无效，置零并 mask |
| `[219:231)` | 12 | 右手 PCA | 无效，置零并 mask |
| `[231:233)` | 2 | 左/右足接触 | 有效 |
| `[233:243)` | 10 | SMPL-X betas | 桥接成功后有效，否则必须 mask |

必须特别区分“手指”和“手腕”：

- 左手腕是 joint 20，对应主体特征 `[180:189)`；
- 右手腕是 joint 21，对应主体特征 `[189:198)`；
- 两个手腕都参与训练、归一化和评价；
- 只屏蔽 `[207:231)` 的 24 维手部 PCA，不屏蔽手腕。

处理后的基础序列继续保持当前工程的公共字段：

```python
{
    "aria_traj": Tensor[T, 9],           # rotation-6D + translation
    "smpl_params": {
        "global_orient": Tensor[T, 6],
        "body_pose": Tensor[T, 21, 6],
        "left_hand_pose": Tensor[T, 12],
        "right_hand_pose": Tensor[T, 12],
        "betas": Tensor[1, 10],
        "transl": Tensor[T, 3],
    },
    "kp3d": Tensor[T, 76, 3],
    "body_root_offset": Tensor[3],
    "floor_height": float,
    "num_frames": int,
}
```

在此基础上新增时间、来源和有效性字段，见第 10 节。

### 2.2 NymeriaPlus 原始身体数据

每段数据的主体文件是：

```text
<sequence>/body/xdata_smpl_neutral.npz
```

关键字段为：

- `timestamps`: 微秒时间戳，属于 Aria `TIME_CODE` 域；
- `global_orient`: `N × 3`，root axis-angle；
- `body_pose`: `N × 69`，23 个 SMPL 关节的 axis-angle；
- `transl`: `N × 3`，米；
- `betas`: `N × 10`，标准 SMPL neutral shape；
- 坐标已位于 Aria world frame，世界坐标 Z 轴朝上。

源动作约 240 FPS，不能先按数组下标粗暴抽帧；必须按真实时间戳插值到目标时间轴。

### 2.3 需要提前解决的资源依赖

当前仓库中的 `body_models` 只有 SMPL-X 模型，NymeriaPlus 原始姿态求关节位置还需要合法获取的 **standard SMPL neutral 模型**。处理程序启动时必须检查模型文件、许可证和加载结果，缺失时直接失败，不能用 SMPL-X 模型强行读取 69 维 SMPL pose。

## 3. 数据清单、完整性检查和划分

### 3.1 先建立 manifest

第一步只扫描，不做昂贵计算。为每个 sequence 记录：

- sequence ID、参与者匿名 ID、关联参与者/配对关系、场景和 activity；
- SMPL、head VRS、MPS trajectory、calibration、narration 三类 CSV 是否存在；
- 每种数据的起止时间、帧数、时长、文件大小；
- timestamp 是否单调、是否存在重复或大间断；
- VRS RGB stream ID、标称 FPS、设备序列号；
- `quality_score` 或官方质量字段；
- 处理状态、失败原因和处理版本。

建议输出：

```text
processed/nymeriaplus_v1/manifests/raw_inventory.jsonl
processed/nymeriaplus_v1/manifests/failures.jsonl
```

只有同时具备 SMPL、可查询的 head pose 和 RGB 的序列才能进入完整多模态处理。缺少 narration 的序列仍可用于无文本动作训练，但必须设置 `text_valid=False`。

### 3.2 数据集划分

NymeriaPlus 必须按参与者划分，禁止随机切 8 秒窗口，否则同一个人的外观、动作和环境会泄漏到训练集和验证集。

推荐：

1. 用 participant ID 建图；如果 `pt2` 或配对字段连接两名参与者，则将其视为一个 connected component；
2. 以 connected component 为最小划分单位；
3. 在不泄漏的前提下，按地点、activity、scripted/free-form 和性别等字段分层；
4. 初始采用约 85% train / 15% val，最终按实际有效小时数平衡；
5. 保持 EE4D 官方 train/val 原样，不重新划分 EE4D；
6. 固化 sequence 列表和随机种子，不以处理顺序决定划分。

## 4. 统一时间轴和多模态同步

### 4.1 主时间域

统一使用 Aria `TIME_CODE` 纳秒作为 master time domain：

- NymeriaPlus SMPL timestamp 原始为 `TIME_CODE` 微秒，先乘 `1000` 转为纳秒；
- RGB capture 和 MPS trajectory 通过 VRS 的时间转换接口映射到 `TIME_CODE`；
- narration CSV 的秒数属于 `DEVICE_TIME`，不能直接与 SMPL 时间相减；先经对应 head VRS 将 `DEVICE_TIME` 转换到 `TIME_CODE`，或把 master sample 反向转换到 `DEVICE_TIME` 后做区间查询；
- 所有输出同时保留整数纳秒，避免用 float seconds 作为永久主键。

有效区间定义为：

```text
SMPL 可用区间 ∩ head pose 可用区间 ∩ RGB 可查询区间
```

建议在首尾各裁 1 秒，避开 MPS/插值边界。narration 不限制动作有效区间，只产生单独的 `text_coverage_mask`。

### 4.2 10 FPS 动作时间轴

设有效区间起点为 `t0_ns`，以整数算术生成：

```python
t_motion[k] = t0_ns + k * 100_000_000   # 10 FPS
```

SMPL 重采样规则：

- `global_orient` 和每个 `body_pose` joint：axis-angle → quaternion，使用 SLERP；
- `transl`：线性插值；
- `betas`：通常每序列取稳健中位数/首个一致值，并检查时间方差；
- 不跨 timestamp 大间断插值；间断处分段；
- 保存左右源帧时间、插值权重和最大时间残差用于 QC。

由于源数据约 240 FPS，正常情况下目标时间会被非常近的源帧包围。若采用最近邻作为快速基线，仍须记录误差并设置不超过 8 ms 的硬上限；正式版本优先 SLERP/线性插值。

### 4.3 10 FPS 头部轨迹

在相同 `t_motion` 上查询 MPS closed-loop trajectory：

- translation 使用线性插值；
- rotation 使用 quaternion SLERP；
- 严禁对 4×4 matrix 逐元素线性插值；
- 插值前后检查旋转正交性和 determinant；
- MPS 时间查询残差建议不超过 3 ms，超出则该帧或该连续段无效。

### 4.4 5 FPS RGB / DINOv2 时间轴

从同一个 `t0_ns` 生成：

```python
t_dino[j] = t0_ns + j * 200_000_000     # 5 FPS
```

对每个目标时间按 timestamp 查询最近的 RGB capture，而不是假设原视频恒定 30 FPS 后每 6 帧抽一帧。保存：

- `target_timestamp_ns`；
- `capture_timestamp_ns`；
- `capture_time_residual_ns`；
- `dino_valid_mask`。

RGB 残差初始阈值可设为 20–25 ms；最终根据 pilot 的分布收紧。10 FPS 动作帧到 5 FPS DINO 特征的映射要保存为 `motion_to_dino_idx`。为了完全复现当前 EE4D 行为，可在 Dataset 层将每个 5 FPS 特征重复两次，但底层文件仍保留原始 timestamp，避免相位偏移被隐藏。

### 4.5 文本对齐

保留三类 annotation，不要提前混成一种文本：

- `motion_narration`：姿态、手臂、腿部、注意力等字段；
- `atomic_action`：短时动作；
- `activity_summarization`：长时活动总结。

每条文本统一为：

```python
{
    "text_id": str,
    "text_type": "motion_narration" | "atomic_action" | "activity_summary",
    "raw_fields": dict,
    "normalized_text": str,
    "start_device_ns": int,
    "end_device_ns": int,
    "start_timecode_ns": int,
    "end_timecode_ns": int,
}
```

区间采用半开区间 `[start, end)`。每个 10 FPS frame 保存重叠文本 ID 列表；每个训练窗口选择与窗口有交集的所有文本，并同时返回 overlap ratio。存在多条重叠标注时全部保留，不静默覆盖。

文本存储和文本条件建模分两阶段：

- 阶段 A：完成 timestamp 对齐、窗口查询、文本有效性 mask；
- 阶段 B：若需要文本条件生成，再增加 tokenizer/text encoder、条件融合、text dropout 和无条件分支，并做单独消融实验。

## 5. 身体动作转换：standard SMPL → SMPL22 → SMPL-X 兼容格式

### 5.1 22 关节映射

NymeriaPlus standard SMPL 的前 22 个关节与工程中的 SMPL22 顺序一致：

| ID | 关节 | ID | 关节 |
|---:|---|---:|---|
| 0 | pelvis/hips | 11 | right toe |
| 1 | left hip | 12 | neck |
| 2 | right hip | 13 | left collar |
| 3 | spine1 | 14 | right collar |
| 4 | left knee | 15 | head |
| 5 | right knee | 16 | left shoulder |
| 6 | spine2 | 17 | right shoulder |
| 7 | left ankle | 18 | left elbow |
| 8 | right ankle | 19 | right elbow |
| 9 | spine3 | 20 | left wrist |
| 10 | left toe | 21 | right wrist |

转换时保留 root + `body_pose` 前 21 个 joint，丢弃 standard SMPL 的末尾 left/right hand 关节 22、23。这里丢弃的是 SMPL 末端手部关节，不是 joint 20/21 的手腕。

### 5.2 推荐的严格兼容桥接

为了让现有 SMPL-X decoder、可视化和 EE4D 评价代码继续工作，推荐离线生成 SMPL-X 兼容参数：

1. 用 standard SMPL neutral 模型在 10 FPS 上前向，得到原始 SMPL22 的 global rotations 和 joints；
2. 将 root + 21 个 body local rotations 映射到 SMPL-X 的对应 body joints；
3. jaw、eyes 和所有 finger rotations 设为 identity；
4. 对每个 sequence 优化一组常量 SMPL-X neutral betas，使采样帧上的 SMPL-X 22 joints 拟合 standard SMPL 22 joints，同时使用 L2 shape regularization；
5. 每帧调整 SMPL-X translation，使 pelvis 与原 standard SMPL pelvis 对齐；
6. 用拟合后的 SMPL-X 模型重新计算 `kp3d`、`body_root_offset` 和可供现有 decoder 使用的 `smpl_params`；
7. 保存拟合误差和优化状态，超过阈值的 sequence 不进入正式训练。

初始 QC 阈值可采用：22-joint 拟合误差 median < 20 mm、p95 < 40 mm。该阈值需要在 3–5 个 pilot sequence 上结合可视化调整。

这样得到的 `[233:243)` 是真实拟合后的 SMPL-X betas，可参与联合训练。原始 standard SMPL betas 只作为 provenance 保存，不能直接填入 SMPL-X beta 位置。

### 5.3 备用的直接表示方案

如果 shape bridge 的拟合质量暂时不够，可直接从 standard SMPL22 global transforms 构造前 198 维表示，但需要：

- 设置 `body_model="smpl"`；
- `[233:243)` betas 置零并 mask；
- 使用 standard SMPL 专用重建/评价路径；
- 禁止把该参数送入现有 SMPL-X decoder。

此方案适合快速验证时间轴和训练可行性，但不属于完全 EE4D 解码兼容，正式联合版本优先严格桥接方案。

## 6. 地面高度和脚接触

在 10 FPS 的、位于统一 Z-up world frame 的 SMPL22 joints 上计算：

1. 用静止 toe 高度做 DBSCAN 聚类，估计最低可信地面；
2. 使用当前 `determine_floor_height_and_contacts(..., fps=10)` 的阈值作为基线；
3. 长 sequence 若出现楼梯、台阶或明显地面变化，按 timestamp discontinuity、活动边界或 floor change point 切成连续 segment 后分别估计；
4. 原始层保留 4 维接触：left heel/toe、right heel/toe；
5. `v4_beta` 的 2 维接触由同侧 heel/toe 做 OR/max 得到；
6. 保存 `floor_height`、`floor_confidence`、`contacts_4d`、`contacts_2d` 和 `terrain_flag`；
7. 表示转换函数优先使用已经计算的 contacts，避免运行时又用另一套 toe-only 规则重算。

应抽查奔跑、蹲下、坐下、上台阶等情况。对 terrain sequence 不建议一律删除，先标记并在训练配置中决定是否纳入。

## 7. Aria 头部轨迹转换

MPS 提供的通常是 `T_world_device`。UniEgoMotion 需要与当前 canonicalization 一致的“头部局部坐标”：固定轴定义必须为右手系，并明确哪个轴是 forward/up。

转换形式为：

```text
T_world_uem_head = T_world_device @ T_device_sensor @ T_sensor_uem_head
```

其中固定外参必须来自每个 VRS 的 calibration，或来自经过验证的 sensor-to-device 物理外参。不要未经验证直接复用 EE4D 中的固定 `bla` 旋转，因为 EE4D 输入外参和 NymeriaPlus 的 `T_world_device` 语义可能不同。

实施步骤：

1. 通过 VRS calibration 获取 device 与 RGB/CPF 的变换及其明确方向；
2. 定义 `T_sensor_uem_head`，使最终 UEM head frame 与现有 canonicalization 的 forward/up 约定一致；
3. 以 4×4 matrix 做一次方向验证，确认是 `T_A_B` 还是其 inverse；
4. 将最终 rotation 转为连续 6D，拼接 translation，保存 `aria_traj: T × 9`；
5. 再用现有 representation 逻辑得到 18D `aria_traj_repre`（local 9D + temporal delta 9D）；
6. 每条 sequence 保存所用 calibration ID、固定转换矩阵和代码版本。

必须做三类验证：

- 几何：`RᵀR≈I`、`det(R)≈1`，无突变；
- 身体一致性：head joint 到 device trajectory 的相对距离和方向在合理范围，p95 初始阈值 0.30 m；
- 图像一致性：将身体/头部朝向投影或叠加到 RGB，抽样人工确认 forward、left/right 和 up 没有翻转。

## 8. DINOv2 特征处理

为了与现有 checkpoint 和 EE4D 特征一致，默认使用当前工程配置：

- 模型：`dinov2_vitl14_reg`；
- 输入尺寸：短边 resize 到 336；
- normalization：ImageNet mean/std，与现有脚本一致；
- 输出：CLS + 4 register tokens，共 `5 × 1024`；
- 保存 dtype：FP16；
- 时间频率：5 FPS；
- `dinov2` 配置只消费 CLS token，`dinov2_reg` 才消费全部 5 个 token。

需要先对少量 VRS 图像确认旋转方向。当前 EE4D 脚本有 `rot90(-1)`，NymeriaPlus 是否需要同样旋转应由 calibration 和实际画面决定，不能盲目照搬。

工程实现上应：

- 按 sequence 流式解码 VRS，避免先导出全部 JPEG；
- 批量 GPU 推理；
- 每个 sequence 独立 shard，支持断点续跑和原子完成标记；
- 同时保存 timestamps、frame validity 和 capture residual；
- 数据加载时通过 `motion_to_dino_idx` 对齐，不从文件名猜时间；
- 对缺失图像使用显式 `image_valid_mask`，不能用全零特征隐式判断。

约 100 小时数据在 5 FPS 下约有 180 万个目标图像。若保存 5 个 token 的 FP16 特征，仅特征主体约 18 GB；若只保存 CLS，约 3.7 GB。建议保留 5-token 原始特征，Dataset 层按配置选择 token。

## 9. 手部 24 维 mask、归一化和损失

### 9.1 数据约定

NymeriaPlus 没有可用的 SMPL-X finger articulation：

```python
motion[..., 207:231] = 0
feature_valid_mask[..., 207:231] = False
feature_valid_mask[..., 0:207] = True
feature_valid_mask[..., 231:243] = True   # 仅严格 SMPL-X bridge 成功时
```

EE4D 的真实手部 PCA 对应 mask 为 `True`。绝对不能用 `motion == 0` 推断无效性，因为零旋转、零平移和归一化后的零值都可能是合法数据。

建议区分：

- `frame_valid_mask: T`：padding、时间缺口等帧级有效性；
- `feature_valid_mask: T × 243`：逐帧逐维有效性；
- 静态 hand mask 可压缩存储，但 Dataset 返回 batch 时必须展开到 `B × T × 243`。

### 9.2 mask-aware 归一化

统计量只从 train split 的有效数值计算：

```python
count[d] = sum(mask[..., d])
mean[d] = sum(mask[..., d] * x[..., d]) / count[d]
std[d]  = masked_std(x[..., d])
x_norm  = where(mask, (x - mean) / clamp(std, eps), 0)
```

可用 masked Welford 避免大文件一次加载和数值不稳定。必须保存每一维的 `count`，检查手部维在联合数据中只统计 EE4D 样本。

提供两种明确模式：

- `joint_stats`：从 EE4D + NymeriaPlus train 的有效数据重算，推荐用于从头联合训练；
- `ee4d_compat_stats`：沿用原 EE4D stats，推荐用于加载既有 checkpoint 后微调；NymeriaPlus 无效手部仍归一化后强制为零。

不要在加载旧 checkpoint 的同时悄悄切换统计量。

### 9.3 mask-aware loss

当前 diffusion 和 flow-matching 路径只有 frame mask，需要统一修改为：

```python
valid = frame_valid_mask[..., None] & feature_valid_mask
loss = (error.square() * valid).sum() / valid.sum().clamp_min(1)
```

同时处理 diffusion state：

- 对无效维，`x_start` 和 noise 都强制为 0；
- model output 在无效维强制为 0；
- sampling 时接收 sample-specific feature mask，并持续 clamp 无效维；
- velocity、position、contact 或 FK 几何损失只消费对应有效主体关节；
- NymeriaPlus 不计算 finger/hand reconstruction loss。

需要改造的现有路径至少包括：

- `mydiffusion/gaussian_diffusion.py`；
- `mydiffusion/flow_matching.py`；
- 训练 batch 组装和 sampler；
- normalization/stats 计算；
- evaluation 与导出。

模型输入输出仍保持 243 维，因此线性层 shape 和已有 checkpoint 结构不变；变化的是数据有效性和损失计算语义。

## 10. 推荐的中间格式和目录结构

不要一开始就把所有 sequence 合成一个巨大 `.pt`。先按 sequence/shard 输出，验证完成后再导出兼容包：

```text
processed/nymeriaplus_v1/
├── manifests/
│   ├── raw_inventory.jsonl
│   ├── processing_qc.jsonl
│   └── failures.jsonl
├── splits/
│   ├── train.txt
│   └── val.txt
├── sequences/
│   └── <sequence_id>.pt
├── dinov2/
│   └── <sequence_id>.pt
├── text/
│   └── <sequence_id>.json
├── stats/
│   └── v4_beta_joint_train_stats.pt
└── uniegomotion_export/
    ├── nymeria_train.pt
    ├── nymeria_val.pt
    ├── egoview_dinov2_nymeria_train.pt
    └── egoview_dinov2_nymeria_val.pt
```

每个标准序列建议保存：

```python
{
    # identity/provenance
    "source": "nymeriaplus",
    "sequence_id": str,
    "participant_group": str,
    "processing_version": str,
    "body_model": "smplx_bridge",

    # explicit time contract
    "fps": 10,
    "timestamps_timecode_ns": LongTensor[T],
    "timestamps_device_ns": LongTensor[T],
    "t0_timecode_ns": int,
    "frame_valid_mask": BoolTensor[T],

    # current EE4D-compatible fields
    "aria_traj": FloatTensor[T, 9],
    "smpl_params": dict,
    "kp3d": FloatTensor[T, 76, 3],
    "body_root_offset": FloatTensor[3],
    "floor_height": float,
    "num_frames": int,

    # new validity/auxiliary fields
    "feature_valid_mask": BoolTensor[T, 243],
    "contacts_4d": BoolTensor[T, 4],
    "contacts_2d": BoolTensor[T, 2],
    "motion_to_dino_idx": LongTensor[T],
    "image_valid_mask": BoolTensor[T],
    "text_ids_per_frame": list[list[str]],
    "text_coverage_mask": BoolTensor[T],

    # quality and compatibility
    "smpl_source_betas": FloatTensor[10],
    "smplx_fitted_betas": FloatTensor[10],
    "smpl_to_smplx_fit_metrics": dict,
    "coordinate_transform_metadata": dict,
}
```

现有 `ImageFeats` 从 sequence key 中解析 30 FPS frame index。正式改造应让 loader 读取显式 timestamp/offset；为临时兼容旧逻辑，exporter 可额外生成 nominal 30 FPS key：

```text
<sequence_id>___<3 * start_10fps>___<3 * end_10fps>
```

但 nominal index 只能作为兼容别名，真实同步必须依赖 timestamps。

## 11. 与 EE4D-Motion 联合加载和训练

推荐增加 source-aware 的统一 Dataset，而不是覆盖原始 `ee_train.pt`：

```python
sample = {
    "motion": FloatTensor[80, 243],
    "aria_traj": FloatTensor[80, 18],
    "image_feats": FloatTensor[80, 1024],
    "frame_valid_mask": BoolTensor[80],
    "feature_valid_mask": BoolTensor[80, 243],
    "image_valid_mask": BoolTensor[80],
    "text": ...,
    "text_valid_mask": ...,
    "source": "ee4d" | "nymeriaplus",
}
```

窗口规则保持当前配置：

- motion 10 FPS；
- window 80 frames，即 8 秒；
- stride 20 frames，即 2 秒；
- 连续有效段不足 80 帧时，训练集优先丢弃；验证/推理如需保留则 padding，并依靠 frame mask；
- 不允许窗口跨越 timestamp gap、floor segment 或 sequence 边界。

联合训练建议：

- 使用 source-balanced sampler，避免某个数据源因窗口数量更多而支配训练；
- NymeriaPlus 内部再按 participant/activity 平衡；
- 保留 `source` 字段，支持 domain embedding 或 source-specific ablation；
- 第一阶段先只训练现有 image/head → motion 任务，确认与 EE4D baseline 等价；
- 第二阶段再启用 aligned text，避免同时引入时间轴、body bridge、mask 和文本模型四类变量。

## 12. 评价协议

评价必须按数据源分别报告：

1. 原 EE4D val：保持现有 benchmark，不改变样本；
2. NymeriaPlus participant-disjoint val；
3. combined 汇总值，仅作为附加项。

NymeriaPlus 评价规则：

- 主体动作指标只计算 22 个 body joints，包含 left/right wrist；
- 不计算 finger/hand PCA 指标，显式排除 `[207:231)`；
- 若 SMPL-X bridge 失败，beta 指标也不计算；
- foot contact 计算 precision/recall/F1；
- 头部轨迹计算 translation、rotation 和速度误差；
- DINO/text 有效性只在对应 mask 为真时计算；
- FID/TMR/semantic 指标需确认 encoder 是否受手部零值影响，优先提供 body-only 和 source-separated 版本。

需要防止一种假象：只在 loss 中屏蔽手部，但 sampling 后保留随机手部输出。这会污染全维 FID 和可视化，所以 sampling/export 也必须按 sample mask 将 NymeriaPlus 手部维归零。

## 13. 质量控制和验收

### 13.1 单元测试

- axis-angle ↔ quaternion/rotation matrix ↔ 6D round-trip；
- SLERP 端点、180°附近和 batch joint 插值；
- standard SMPL24 → SMPL22 关节顺序；
- wrist 20/21 保留、手部 `[207:231)` 精确屏蔽；
- masked mean/std、masked loss、全无效维 denominator；
- DEVICE_TIME ↔ TIME_CODE 转换；
- `[start, end)` 文本边界和多文本重叠；
- 5 FPS → 10 FPS 映射相位；
- `aria_traj` 旋转方向和 inverse；
- 表示转换/逆转换的一致性。

### 13.2 每序列自动 QC

`processing_qc.jsonl` 至少记录：

- 原始/有效/输出时长及各模态 coverage；
- timestamp 单调性、gap 数和各模态时间残差 p50/p95/p99/max；
- NaN/Inf、rotation orthogonality 和 determinant；
- SMPL→SMPL-X 22-joint fit median/p95/max；
- pelvis/head 速度和加速度极值；
- head-device 与 body-head 相对位置统计；
- floor height、接触比例、terrain flag；
- DINO 有效率、文本覆盖率；
- hand mask 和 beta mask 是否符合 body model mode。

初始告警阈值：

| 项目 | 初始阈值 |
|---|---:|
| SMPL 最近源样本残差 | ≤ 8 ms |
| MPS pose 残差 | ≤ 3 ms |
| RGB capture 残差 | ≤ 20–25 ms |
| `|det(R)-1|` | < 1e-4 |
| `||RᵀR-I||` | < 1e-4 |
| SMPL→SMPL-X joint fit median / p95 | < 20 / 40 mm |
| head-body translation residual p95 | < 0.30 m |
| NaN / Inf | 0 |

这些是 pilot 阶段的初始阈值，不应在观察真实分布前当作不可修改的真值。

### 13.3 人工可视化

先选择 3–5 条覆盖室内、室外、快速运动、坐下/起身和双人活动的 sequence，输出：

- 10 FPS skeleton + head trajectory + floor plane；
- RGB 上的身体/头朝向投影；
- 同步显示 DINO frame timestamp 和 narration；
- standard SMPL 与拟合 SMPL-X skeleton 叠加。

只有 pilot 全部通过后才批量处理 100 小时数据。

## 14. 建议的代码改造边界

建议新增：

```text
run/build_nymeriaplus_manifest.py
run/process_nymeriaplus_for_uniegomotion.py
run/extract_nymeriaplus_dinov2.py
dataset/nymeriaplus_time_alignment.py
dataset/nymeriaplus_motion_dataset.py
dataset/combined_motion_dataset.py
tests/test_nymeriaplus_alignment.py
tests/test_feature_masks.py
```

建议修改：

- `dataset/representation_utils.py`
  - 允许显式传入 hand PCA、contacts 和 validity mask；
  - 不再要求每条数据都有真实 hand pose；
  - 避免重复估计 foot contact。
- `dataset/feats.py`
  - 支持 timestamp/index map；
  - 返回 image validity mask；
  - 保留旧 EE4D key 兼容路径。
- `dataset/ee4d_motion_dataset.py`
  - 抽出公共 window/canonicalization 逻辑；
  - batch 返回 source 和 feature masks。
- `mydiffusion/gaussian_diffusion.py`
  - 使用 frame × feature mask；
  - mask noise、target 和 sampling state。
- `mydiffusion/flow_matching.py`
  - `_valid_frame_mse` 扩展为逐特征 validity；
  - 与现有 feature weights 正确相乘。
- stats/evaluation 相关代码
  - masked stats；
  - body-only、source-separated metrics。

处理脚本需要具备：`--sequence-list`、`--resume`、`--overwrite-failed`、`--workers`、`--device`、`--dry-run`、`--processing-version` 和确定性 seed。成功输出先写临时文件，再原子 rename，避免中断留下看似完整的 `.pt`。

## 15. 分阶段实施顺序

### Phase 0：清单和契约

- 建 manifest、完整性检查、participant-disjoint split；
- 固定坐标系、时间域、joint order 和输出 schema；
- 获取 standard SMPL neutral 模型。

验收：能准确列出所有可处理 sequence、各模态时段和失败原因。

### Phase 1：单序列时间轴与身体 pilot

- 完成 10 FPS SMPL 插值；
- SMPL22 映射；
- floor/contact；
- standard SMPL → SMPL-X bridge；
- 生成 243D v4_beta 和手部 mask。

验收：3–5 条序列通过 round-trip、拟合误差和 skeleton 可视化。

### Phase 2：Aria 与 DINOv2

- 完成 MPS 时间插值、坐标转换和 `aria_traj`；
- 5 FPS VRS 解码及 DINOv2；
- 验证 10/5 FPS 相位和 RGB 方向。

验收：head/body/RGB 对齐可视化通过，时间残差满足阈值。

### Phase 3：narration

- DEVICE_TIME ↔ TIME_CODE 转换；
- 三层文本区间化、窗口查询和 coverage mask。

验收：随机抽样窗口的画面、动作和文本人工一致。

### Phase 4：联合 Dataset 与 mask-aware 训练

- source-aware Dataset/sampler；
- masked stats、loss、sampling 和 evaluation；
- 先跑小规模 overfit，再跑 EE4D baseline regression。

验收：关闭 NymeriaPlus 时 EE4D 指标与原实现一致；开启后无手部统计污染、无 NaN、loss denominator 正确。

### Phase 5：批量 100 小时和正式实验

- sequence shards 断点处理；
- 汇总 QC；
- 导出 train/val compatibility pack；
- 分别报告 EE4D、NymeriaPlus 和 combined 指标。

## 16. 最终推荐的默认决策

| 问题 | 默认决策 |
|---|---|
| master time domain | `TIME_CODE` int64 ns |
| motion / head FPS | 10 FPS |
| DINOv2 FPS | 5 FPS |
| training window | 80 帧 / 8 秒，stride 20 帧 |
| body joint set | SMPL22，保留双手腕 |
| fingers | 24D 置零 + 显式 feature mask |
| body compatibility | 离线拟合 SMPL-X shape 的严格 bridge |
| foot contact | 原始 4D，模型表示 2D |
| head coordinate | 基于 VRS calibration 显式转换并验证 |
| DINO variant | `dinov2_vitl14_reg`，5×1024 FP16 |
| text | 先对齐存储，条件生成作为第二阶段 |
| split | participant/pair connected-component disjoint |
| combined storage | source-specific shards + unified loader，不覆盖 EE4D 原文件 |
| stats | 从头训练用 joint masked stats；旧 checkpoint 微调用 EE4D-compatible stats |
| metrics | EE4D、NymeriaPlus、combined 分开，NymeriaPlus 不评手指 |

## 17. 参考实现与资料

仓库内重点参考：

- `run/process_for_uniegomotion.py`：EE4D 基础序列处理；
- `dataset/representation_utils.py`：243D `v4_beta` 和 18D `aria_traj_repre`；
- `dataset/egoego_utils.py`：SMPL22、地面和接触；
- `run/extract_img_feats.py`：DINOv2 配置；
- `dataset/feats.py`：5 FPS 特征到 10 FPS 动作的现有对齐；
- `mydiffusion/gaussian_diffusion.py`、`mydiffusion/flow_matching.py`：需要增加逐特征 mask 的损失路径。

外部资料：

- NymeriaPlus 官方工具：<https://github.com/facebookresearch/nymeria_dataset>
- Project Aria 3D 坐标约定：<https://facebookresearch.github.io/projectaria_tools/docs/data_formats/coordinate_convention/3d_coordinate_frame_convention>
- Project Aria MPS trajectory：<https://facebookresearch.github.io/projectaria_tools/gen2/technical-specs/mps/data_formats/slam/mps_trajectory>

