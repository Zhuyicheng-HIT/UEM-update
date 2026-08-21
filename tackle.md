# NymeriaPlus → EE4D-Motion 补充数据处理方案

> 版本：v2（2026-08-21）
> 目标：将 NymeriaPlus 处理为 EE4D-Motion 的补充数据集，使现有 UniEgoMotion 训练、验证和推理管线可以直接读取。
> 原则：首版追求字段、维度、坐标系、采样率和文件组织兼容，不追求严格的 SMPL-X 网格级拟合。

## 1. 首版目标和边界

NymeriaPlus 在首版作为额外的 body/head/RGB 数据源加入 EE4D-Motion。EE4D-Motion 原始数据和官方验证集保持不变，NymeriaPlus 主要扩充训练数据，同时保留其文本标注供后续文本条件实验使用。

首版必须完成：

1. Nymeria SMPL 动作转换为 EE4D-Motion 的 22 关节和 243 维 v4_beta 格式；
2. 头部轨迹转换为现有 aria_traj / aria_traj_repre 格式；
3. RGB 按 EE4D 现有配置提取 DINOv2 特征；
4. 动作、头部和 RGB 使用统一时间戳，并对齐到 10 FPS / 5 FPS；
5. 生成与 EE4D-Motion 相同字段名、shape、dtype 和窗口组织的数据；
6. 保留 motion_narration、atomic_action 和 activity_summarization，并对齐到时间区间和 8 秒训练窗口。

首版暂不做：

- SMPL→SMPL-X 的 beta 拟合和网格级误差优化；
- 逐特征、逐关节的复杂 mask-aware loss、sampling 和 stats 系统；
- 复杂地形识别、接触置信度和接触专用 QC；
- source-aware sampler、source-aware loss 和联合评价协议；
- 文本编码器、文本条件融合和 classifier-free guidance。

这些功能在基础数据能够被 EE4D-Motion 直接读取后再逐项加入。

## 2. 首版数据流

NymeriaPlus 原始序列
→ 统一时间戳和有效区间
→ SMPL 映射到 SMPL22 并重采样到 10 FPS
→ head trajectory 重采样到 10 FPS
→ RGB 提取 DINOv2 并保存为 5 FPS
→ narration 转为文本区间和 8 秒窗口索引
→ EE4D-Motion 同格式序列和索引
→ 追加 Nymeria train 到 EE4D train

## 3. EE4D-Motion 兼容格式

每个 Nymeria 序列最终保存与 EE4D-Motion 相同的公共字段：

- aria_traj：T×9
- smpl_params.global_orient：T×6
- smpl_params.body_pose：T×21×6
- smpl_params.left_hand_pose：T×12
- smpl_params.right_hand_pose：T×12
- smpl_params.betas：1×10
- smpl_params.transl：T×3
- kp3d：T×76×3
- body_root_offset：3
- floor_height：float
- num_frames：int

同时保存 sequence_id、source、timestamps_ns、dino_timestamps_ns、motion_to_dino_idx、text_annotations 和 text_window_ids 等辅助字段。

### 3.1 v4_beta 维度

保持当前工程的 243 维布局，不新建 Nymeria 专用维度：

| 区间 | 含义 | 首版处理 |
|---|---|---|
| [0:198) | 22 个关节的 6D 旋转和 3D 平移 | 使用 Nymeria 身体动作 |
| [198:207) | canonical/global residual | 按 EE4D 表示逻辑计算 |
| [207:219) | 左手 PCA | 置零占位 |
| [219:231) | 右手 PCA | 置零占位 |
| [231:233) | 左右脚接触 | 复用 EE4D 基础计算方法 |
| [233:243) | SMPL-X beta | 首版使用 0 或固定中性值 |

左右手腕仍然属于 22 个身体关节：左手腕为 joint 20，右手腕为 joint 21。只将手指 PCA 作为缺失占位，不能删除腕部关节。

## 4. Nymeria SMPL 转换

### 4.1 输入字段

NymeriaPlus 的主体文件为：

    <sequence>/body/xdata_smpl_neutral.npz

首版读取：

- timestamps：微秒时间戳；
- global_orient：N×3，root axis-angle；
- body_pose：N×69，23 个 SMPL 关节 axis-angle；
- transl：N×3，米；
- betas：N×10，standard SMPL neutral shape。

官方 loader 确认了这些字段和单位，但 narration 的具体时间域仍需查看真实 CSV 后确认，参见 NymeriaPlus SMPL loader：
https://github.com/facebookresearch/nymeria_dataset/blob/main/nymeriaplus/loaders/smpl.py

### 4.2 22 关节映射

将 standard SMPL 的前 22 个语义关节映射为 EE4D-Motion 的 SMPL22 顺序：

pelvis、left/right hip、spine1、left/right knee、spine2、
left/right ankle、spine3、left/right toe、neck、
left/right collar、head、left/right shoulder、
left/right elbow、left/right wrist。

需要用一条真实序列做左右侧和关节顺序可视化检查。确认映射正确后，直接复用现有 representation 转换，不做 SMPL-X shape 拟合。

### 4.3 SMPL-X 参数占位

为了让现有 SMPL-X 解码和可视化接口保持不变，构造参数时使用：

- body pose：Nymeria SMPL body pose；
- global orient：Nymeria root orientation；
- transl：Nymeria translation；
- left/right hand PCA：0；
- jaw/eyes：identity；
- betas：0 或固定中性值。

这只保证接口和身体运动兼容，不宣称 Nymeria 已经拥有真实的手指或 SMPL-X shape 监督。Nymeria 的手部和 beta 不参与首版指标结论。

## 5. 时间轴和采样

所有模态使用整数纳秒时间戳保存。动作和头部轨迹以 10 FPS 作为目标时间轴，DINO 以 5 FPS 作为目标时间轴：

- t_motion[k] = t0 + k × 100 ms；
- t_dino[j] = t0 + j × 200 ms。

不能按照数组下标直接抽帧，应根据原始 timestamp 取最近帧或进行插值。动作旋转使用 quaternion SLERP，平移使用线性插值。遇到明显时间间断时切分 sequence，不跨间断插值。

首版只要求动作、head pose 和 RGB 同时存在的区间进入完整多模态数据。没有 narration 的序列仍可进入无文本动作训练，文本只标记为缺失，不删除整个序列。首尾可以裁掉约 1 秒以避开轨迹查询边界。

## 6. 头部轨迹

将 MPS 的设备轨迹转换为当前 UniEgoMotion 使用的头部轨迹表示，最终得到：

- aria_traj：T×9；
- aria_traj_repre：T×18。

必须确认矩阵方向、坐标轴和左右手系。不要直接假设 Nymeria 和 EE4D 使用相同的固定旋转矩阵。只做一次几何和可视化检查，不引入复杂轨迹 QC。

首版验收标准：

- 旋转矩阵正交且 determinant 接近 1；
- 头部轨迹没有明显跳变；
- 可视化中 forward、up、left-right 方向正确；
- 头部轨迹和 body root 的相对位置合理。

## 7. RGB 和 DINOv2

必须复用 EE4D-Motion 的现有配置：

- 模型：dinov2_vitl14_reg；
- 输入：resize 到 336；
- normalization：ImageNet mean/std；
- 频率：5 FPS；
- dtype：FP16。

DINO 输出的 token 数量必须与现有 EE4D 特征文件一致。如果当前 checkpoint 使用 CLS token，就保存 CLS；如果当前配置使用 CLS+register tokens，就保存完整的 5×1024 特征，不能让两个数据源的特征 shape 不同。

Nymeria VRS 是否需要旋转 90°，先抽取少量图像人工确认，不能无条件照搬 EE4D 的旋转操作。

## 8. 文本保留和时间对齐

NymeriaPlus 的文本作为额外元数据保存，不改变 EE4D-Motion 原有动作字段：

- narration/motion_narration.csv；
- narration/atomic_action.csv；
- narration/activity_summarization.csv。

保留三种粒度，不提前合并：

- motion_narration：细粒度动作、身体部位和注意力描述；
- atomic_action：短时原子动作；
- activity_summarization：较长时间范围的活动总结。

每条文本至少保存：

- text_id；
- text_type；
- text；
- start；
- end；
- sequence_id。

处理时把文本区间映射到 10 FPS 帧和 8 秒训练窗口。窗口中可以保留多个文本 ID，同时保存原始开始时间和结束时间。文本和动作的时间域必须先通过真实 CSV 样例确认，不能仅依据文件名或假设进行换算。

首版不增加 tokenizer、text encoder 或文本条件 loss；这些文本先作为 aligned sidecar 保存，后续文本条件实验直接读取。

## 9. 首版不引入复杂 mask

首版为了保持现有 EE4D-Motion 代码最小改动：

- Nymeria hand PCA = 0；
- Nymeria beta = 0 或固定中性值。

不新增逐维 mask loss、mask noise、mask sampling 和 masked Welford 统计。

但必须在 manifest 中记录：

- has_hand_pose = false；
- has_shape_gt = false。

并遵守两个限制：

1. Nymeria 样本不单独报告手指或 beta 指标；
2. 如果重新计算联合统计量，不能把 Nymeria 的手部占位值当作真实手部监督。

首版身体训练验证通过后，再增加显式 feature_valid_mask 和 mask-aware loss。

## 10. 脚部接触和地面

首版复用 EE4D-Motion 当前的 floor_height 和 2D foot-contact 计算方法，保证输出字段一致：

- floor_height：float；
- contacts：T×2。

不新增 DBSCAN 地形分段、接触置信度和复杂 terrain QC。只过滤明显异常情况，例如 NaN、轨迹断裂和极端速度；奔跑、蹲下、坐下、上台阶等动作先保留。

后续如发现 Nymeria 接触质量影响训练，再单独增加 contact confidence 和 terrain flag。

## 11. 数据划分和加入 EE4D

EE4D 官方 train/val 不改变。NymeriaPlus 建议按 participant 划分，避免同一个人的窗口同时出现在 train 和 val：

- Nymeria train：约 85%；
- Nymeria val：约 15%。

首版训练时：

- training set = official EE4D train + Nymeria train；
- validation = official EE4D val。

Nymeria val 只用于观察跨数据源泛化，不并入官方 EE4D 指标。

首版不使用 source-balanced sampler，也不修改模型结构。可以先按照两个数据源窗口列表直接拼接；如果 Nymeria 窗口数量过多，再通过简单采样比例控制其占比。

## 12. 推荐的最小实现顺序

### Phase 0：单序列检查

- 检查 SMPL、VRS、SLAM、标定和 narration 文件；
- 确认 SMPL22 关节顺序；
- 确认时间戳和坐标轴；
- 可视化 body、head trajectory 和 RGB。

### Phase 1：身体补充数据

- SMPL 重采样到 10 FPS；
- 生成 243D v4_beta；
- 手部和 beta 使用中性占位；
- 生成 kp3d、body_root_offset、floor_height；
- 通过现有 Dataset 读取 80 帧窗口。

### Phase 2：头部和 DINO

- 生成 10 FPS aria_traj_repre；
- RGB 按 5 FPS 提取 DINO；
- 检查 DINO 特征 shape 与 EE4D 完全一致；
- 完成 motion-to-DINO 索引。

### Phase 3：文本 sidecar

- 读取三类 narration CSV；
- 确认时间域；
- 保存文本区间；
- 将文本 ID 对齐到 10 FPS 帧和 8 秒窗口。

### Phase 4：联合训练验证

- EE4D train 与 Nymeria train 拼接；
- 保持官方 EE4D val 不变；
- 先跑单批 collate + forward + loss；
- 再跑少量 reconstruction/generation smoke test。

### Phase 5：批量处理

- 先处理 1 小时 Nymeria 数据；
- 检查失败率、时长、DINO 特征和窗口数量；
- 确认无误后扩展到完整数据集。

## 13. 后置工作

首版数据和训练链路稳定后，再按以下顺序增加：

1. SMPL-X beta 拟合和网格级 QC；
2. 手部、beta、图像和轨迹的显式 validity mask；
3. mask-aware normalization、loss 和 sampling；
4. 接触置信度、terrain flag 和复杂 foot-contact QC；
5. source-aware sampler、loss 和跨数据源评价；
6. 文本 encoder、文本条件融合和 classifier-free guidance。

## 14. 首版验收标准

NymeriaPlus 补充数据首版完成的判断标准：

1. 数据字段名与 EE4D-Motion 一致；
2. 所有 tensor 的 shape、dtype 和帧率符合现有 Dataset 要求；
3. Nymeria 样本可以被现有 Dataset 读取；
4. 可以完成 batch collate、模型 forward 和 loss；
5. 可以完成一次推理和 SMPL-X 可视化；
6. DINO 特征可以和 EE4D 特征放入同一 batch；
7. 文本区间能够根据窗口时间正确查询；
8. 官方 EE4D 验证集指标不因数据接入而改变。

达到这些标准后，NymeriaPlus 就已经完成了 EE4D-Motion 补充数据集的目标，不需要等待严格 SMPL-X 拟合或复杂联合训练模块完成。

