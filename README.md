# UniEgoMotion
This repository provides the dataset and code for the following paper: 
```
UniEgoMotion: A Unified Model for Egocentric Motion Reconstruction, Forecasting, and Generation
Chaitanya Patel, Hiroki Nakamura, Yuta Kyuragi, Kazuki Kozuka, Juan Carlos Niebles, Ehsan Adeli
ICCV 2025  
```
[[Project Website](https://chaitanya100100.github.io/UniEgoMotion)]
[[ArXiv](https://arxiv.org/abs/2508.01126)]

![Teaser](assets/teaser.png)

## Citation
Cite us if you use our model, code or data:
```
@inproceedings{patel2025uniegomotion,
        title={UniEgoMotion: A Unified Model for Egocentric Motion Reconstruction, Forecasting, and Generation},
        author={Patel, Chaitanya and Nakamura, Hiroki and Kyuragi, Yuta and Kozuka, Kazuki and Niebles, Juan Carlos and Adeli, Ehsan},
        booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
        pages={10318--10329},
        year={2025}
      }
```

## Installation
The code is tested on Python 3.10, PyTorch 2.5.1, and CUDA 11.8 on a single L40S GPU. However, it should work on most modern GPUs with any recent PyTorch and CUDA version.
```
conda create --name uem python=3.10
conda activate uem

# Install Pytorch with your own CUDA version
pip3 install torch --index-url https://download.pytorch.org/whl/cu118

pip3 install pytorch_lightning==2.4.0
pip3 install -r requirements.txt
```
Download **SMPL-X v1.1** from [here](https://smpl-x.is.tue.mpg.de/download.php) and extract
`SMPLX_NEUTRAL.npz` to `body_models/smplx/`. Alternatively, set `SMPLX_MODEL_PATH` to the directory
containing `SMPLX_NEUTRAL.npz`.

## EE4D-Motion Dataset
See [DATASET.md](DATASET.md) to download and setup the dataset. This is required to run the model.
Set `UEM_DATA_DIR` to the extracted `ee4d_motion_uniegomotion` directory, or place it at
`data/ee4d_motion_uniegomotion`.

## Flow Matching Experiments (E1–E12)

The Flow Matching implementation, B0/E1–E12 configurations, launch/evaluation scripts, and
reproducibility notes are documented in [ablation/README.md](ablation/README.md). The 256-sample
ablation results and inference benchmark are reported in [result.md](result.md). Generated
checkpoints, predictions, dataset files, and SMPL-X model files are intentionally not versioned.

## K12 稀疏关节测评

K12 模型不再预测完整的 22 个 SMPL22 身体关节块，而是只预测固定的 12 个稀疏关节：

```text
[0, 4, 5, 10, 11, 13, 14, 15, 18, 19, 20, 21]
```

它们分别对应 pelvis、双膝、双脚端点、双肩、head、双肘和双腕。测评将 K12 与稠密 E7
模型、原论文 Diffusion 模型在相同的 12 个关节上进行比较，使用验证集 256 个窗口，包含
recon、forecast 和 generation 三个任务。本对比不使用全身恢复器。

误差在 canonical `v4_beta` 关节块表示空间中计算：位置误差单位为 mm，旋转误差单位为
degree，位置速度误差单位为 mm/s。

| 模型 | 任务 | 位置误差 (mm) | 旋转误差 (deg) | 位置速度误差 (mm/s) |
|---|---|---:|---:|---:|
| K12 | Recon | 103.55 | 25.06 | 176.57 |
| E7 | Recon | 100.36 | 24.07 | 173.10 |
| 原论文 Diffusion | Recon | **98.97** | **23.62** | **163.05** |
| K12 | Generation | 155.74 | 29.71 | **234.38** |
| E7 | Generation | **149.91** | **28.74** | 226.40 |
| 原论文 Diffusion | Generation | 161.36 | 30.47 | 232.63 |
| K12 | Forecasting | 149.71 | 30.60 | **229.53** |
| E7 | Forecasting | **139.83** | **29.75** | 231.86 |
| 原论文 Diffusion | Forecasting | 154.54 | 30.84 | 242.70 |

原始 K12 指标保存在本地实验文件
[`exp/sparse_joint_eval_k12_comparison/k12.json`](exp/sparse_joint_eval_k12_comparison/k12.json)
中（如果保留实验产物）。详细测评报告见 [`result.md`](result.md)。这些结果只衡量选定的
12 个预测关节，不能代表未预测的 10 个关节或完整身体 SMPL-X 的恢复质量。

## Pretrained UniEgoMotion Model
Download the pretrained model from [here](https://downloads.cs.stanford.edu/simurgh/chpatel/uem_v4b_dinov2.zip) or [here](https://huggingface.co/datasets/chaitanya100100/uniegomotion/tree/main) and place it in the `exp/` directory.

## Visualizing UniEgoMotion Outputs
Run the following command to visualize UniEgoMotion outputs on some validation samples for all three tasks.
```
python run/vis_uem.py \
CONFIG ./config/uem.yaml \
TRAIN.EXP_PATH ./exp/uem_v4b_dinov2/ \
MODEL.CKPT_PATH last_ckpt
```

## Training and Evaluating UniEgoMotion from Scratch
Run the following to train UniEgoMotion model. Use an appropriate experiment path to save training logs and model checkpoints.
```
python run/train_uem.py \
CONFIG ./config/uem.yaml \
TRAIN.EXP_PATH <exp_path>
```

Run the following command to evaluate UniEgoMotion model on the validation set. It will save the predictions for all three tasks in the experiment directory, and compute 3D and semantic metrics. You will need to setup TMR model through [this repo](https://github.com/nv-tlabs/stmc) and set the path in `model/tmr_eval_model.py`.
```
python eval/eval_exp.py \
CONFIG ./config/uem.yaml \
TRAIN.EXP_PATH <exp_path> \
MODEL.CKPT_PATH last_ckpt
```

Run the following command to visualize some saved predictions for a particular task.
```
python eval/vis_uem_preds.py \
--EXP_PATH <exp_path> \
--EVAL_TASK <task>
```

### Baselines and Ablations
Commands to train UniEgoMotion model and some ablation experiments:
- `python run/train_uem.py CONFIG ./config/uem.yaml TRAIN.EXP_PATH <exp_path>` for original UniEgoMotion model.
- Use `MODEL.ENCODER_TSFM add` to train transformer encoder baseline.
- Use `DATA.REPRE_TYPE v1_beta` to train with global motion representation. See `dataset.representation_utils.py` for the explanation of different motion representations. You will need to compute the statistics for this representation first. See [DATASET.md](DATASET.md) for more details.
- Use `DATA.COND_IMG_FEAT False` to train without egocentric video conditioning i.e. only conditioned on egocentric trajectory.
- Use `DATA.IMG_FEAT_TYPE clip_all` to train with CLIP features instead of DINOv2. Note that you will need to extra CLIP features for all takes and collect them into one file per split. See [DATASET.md](DATASET.md) for more details.

Commands to train baselines:
- See [this](https://github.com/chaitanya100100/UniEgoMotion/issues/2) issue for the updated head tracking metrics of egoallo.
- `python run/train_uem.py CONFIG ./config/lstm.yaml TRAIN.EXP_PATH <exp_path> MODEL.LSTM_TYPE gen` to train LSTM baseline for egocentric motion generation. Use `MODEL.LSTM_TYPE fore` for forecasting baseline.
- `python run/train_uem.py CONFIG ./config/uem_traj.yaml TRAIN.EXP_PATH <exp_path>` to train UniEgoMotion trajectory prediction model (instead of whole body pose). The trajectory conditioned motion prediction model will be the original UniEgoMotion model. Use `CONFIG ./config/uem_twostage.yaml` with appropriate arguments for evaluation and visualization.


If you have any questions or concerns about the code or dataset, please open an issue.
