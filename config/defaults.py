import os

from yacs.config import CfgNode as CN

_C = CN()

_C.DATA = CN()
_C.DATA.DATA_DIR = os.environ.get("UEM_DATA_DIR", "data/ee4d_motion_uniegomotion")
_C.DATA.DATASET_NAME = None
_C.DATA.BATCH_SIZE = 32
_C.DATA.NUM_WORKERS = 4
_C.DATA.PIN_MEMORY = False
_C.DATA.PERSISTENT_WORKERS = False
_C.DATA.PREFETCH_FACTOR = 2
_C.DATA.DROP_LAST = False
_C.DATA.WINDOW = 80  # Length of the motion sequence. At 10 fps, window of 80 is 8 seconds.
_C.DATA.REPRE_TYPE = "v4_beta"  # motion representation type
_C.DATA.COND_IMG_FEAT = False  # whether to condition on image features
_C.DATA.COND_TRAJ = True  # whether to condition on aria trajectory
# whether to condition on betas. We do NOT condition on betas in the paper but rather predict them.
_C.DATA.COND_BETAS = False
_C.DATA.IMG_FEAT_TYPE = "dinov2"  # image feature type if conditioning on image features

_C.MODEL = CN()
_C.MODEL.CKPT_PATH = None
_C.MODEL.PREDICT_XSTART = True
_C.MODEL.DIFFUSION_STEPS = 1000
_C.MODEL.NOISE_SCHEDULE = "cosine"
_C.MODEL.GENERATIVE_TYPE = "diffusion"  # "diffusion" or continuous-time "flow"
_C.MODEL.MODEL_NAME = "uem"  # uem, lstm, unet
_C.MODEL.LEARN_TRAJ = False  # trajectory model for two-stage model
_C.MODEL.TRAJ_CKPT_PATH = None  # for two-stage model
_C.MODEL.MOTION_CKPT_PATH = None  # for two-stage model
_C.MODEL.ENCODER_TSFM = None  # use "add" for ablation with tsfm encoder
_C.MODEL.LSTM_TYPE = "gen"  # ["gen", "fore"]. If model is lstm, this specifies the task lstm is trained for.
_C.MODEL.FINETUNE_TYPE = None
_C.MODEL.ZERO_MASK_TOKEN = False  # whether to use zero mask token instead of a learnable one.
# Output decoder used by the Global/Local topology ablations. "single" keeps
# the original checkpoint-compatible Linear(768, input_feats) head. Other
# choices: no_fusion, local_to_global, global_to_local, bidirectional.
_C.MODEL.OUTPUT_BRANCH_MODE = "single"
_C.MODEL.FUSION_GATE_INIT = -4.0
_C.MODEL.FUSION_STOP_GRAD = True
# Optional explicit task identity used by E14--E16.  The mapping is fixed to
# recon=0, fore=1, gen=2 so evaluation and training share checkpoint semantics.
_C.MODEL.COND_TASK = False
_C.MODEL.NUM_TASKS = 3
# E18: identity-initialized task modulation after each decoder pre-norm.
# The task embedding remains enabled, so this only adds task-specific feature
# scaling/shifting without changing the shared Transformer topology.
_C.MODEL.TASK_FILM = CN()
_C.MODEL.TASK_FILM.ENABLED = False

_C.FLOW = CN()
_C.FLOW.NUM_STEPS = 10
_C.FLOW.SOLVER = "euler"
_C.FLOW.BETA_ALPHA = 1.5
_C.FLOW.BETA_BETA = 1.0
_C.FLOW.T_MIN = 0.001
_C.FLOW.PREDICTION_TYPE = "velocity"  # "velocity" or target-predictive "x0"
# Optional per-feature weighting for the v4_beta global SE(3) delta at
# feature indices [198, 207). A weight of 1.0 exactly recovers the original
# all-feature mean squared error.
_C.FLOW.GLOBAL_WEIGHT = 1.0
_C.FLOW.GLOBAL_FEATURE_START = 198
_C.FLOW.GLOBAL_FEATURE_END = 207
# Optional split weighting within the 9D global SE(3) delta. When both are
# set, the first six dimensions use GLOBAL_ROT_WEIGHT and the final three use
# GLOBAL_TRANS_WEIGHT. Leaving both as None preserves GLOBAL_WEIGHT exactly.
_C.FLOW.GLOBAL_ROT_WEIGHT = None
_C.FLOW.GLOBAL_TRANS_WEIGHT = None
# E17: optional recon/fore/gen-specific weights for the same global slice.
# None preserves GLOBAL_WEIGHT exactly. It is mutually exclusive with the
# rotation/translation split above.
_C.FLOW.TASK_GLOBAL_WEIGHTS = None

_C.TRAIN = CN()
_C.TRAIN.LR = 3.0e-5
_C.TRAIN.WEIGHT_DECAY = 0.0
_C.TRAIN.USE_CKPT_LR = False  # whether to use lr from the checkpoint rather than the config lr.
_C.TRAIN.EXP_PATH = None  # experiment log path to save logs and checkpoints

_C.TRAIN.NUM_EPOCHS = 200
_C.TRAIN.MAX_STEPS = -1  # positive values stop at an exact optimizer-step budget.
_C.TRAIN.LOG_EVERY_N_STEPS = 50
# _C.TRAIN.VAL_CHECK_INTERVAL = 1.0
_C.TRAIN.CHECK_VAL_EVERY_N_EPOCHS = 1
# _C.TRAIN.SAVE_EVERY_N_STEPS = None
_C.TRAIN.SAVE_EVERY_N_EPOCHS = 10
_C.TRAIN.ONLY_VALIDATE = False
_C.TRAIN.NUM_GPUS = 1
_C.TRAIN.PRECISION = "32-true"
_C.TRAIN.NUM_SANITY_VAL_STEPS = 2
_C.TRAIN.ACCUMULATE_GRAD_BATCHES = 1
_C.TRAIN.GRADIENT_CLIP_VAL = 1.0
_C.TRAIN.DDP_STATIC_GRAPH = False
_C.TRAIN.CUDNN_BENCHMARK = False
_C.TRAIN.FUSED_ADAMW = False
_C.TRAIN.EMA_DECAY = 0.999
_C.TRAIN.SCHEDULER = "step"
_C.TRAIN.WARMUP_EPOCHS = 0
_C.TRAIN.SCHEDULER_TOTAL_EPOCHS = 0  # 0 follows NUM_EPOCHS; set explicitly for stable extension runs.
_C.TRAIN.MIN_LR_RATIO = 0.1
_C.TRAIN.EARLY_STOP_PATIENCE = 0
_C.TRAIN.EARLY_STOP_MIN_DELTA = 1.0e-4
_C.TRAIN.PROGRESS_REFRESH_RATE = 1

_C.TRAIN.EVAL_SUFFIX = ""  # suffix to append to the evaluation and visualization results file
_C.TRAIN.EVAL_TASK = None  # task to evaluate or visualize. Should be one of ["recon", "gen", "fore"]
_C.TRAIN.COND_SCALE = None  # classifier free guidance scale. We do not use this for UniEgoMotion evaluation.

# Explicit three-task training used by E13--E16.  When disabled, the original
# independent random condition masking remains unchanged.
_C.TRAIN.TASK_SAMPLER = CN()
_C.TRAIN.TASK_SAMPLER.ENABLED = False
_C.TRAIN.TASK_SAMPLER.MODE = "fixed"  # fixed, curriculum, adaptive
_C.TRAIN.TASK_SAMPLER.BATCH_MODE = "step"  # step or mixed (E20)
_C.TRAIN.TASK_SAMPLER.TOTAL_STEPS = 84000
_C.TRAIN.TASK_SAMPLER.SEED = 62
_C.TRAIN.TASK_SAMPLER.FORECAST_PREFIX = 20
_C.TRAIN.TASK_SAMPLER.FIXED_PROBS = [0.40, 0.30, 0.30]  # recon, fore, gen
_C.TRAIN.TASK_SAMPLER.CURRICULUM_FRACTIONS = [0.15, 0.20, 0.25, 0.40]
_C.TRAIN.TASK_SAMPLER.CURRICULUM_RECON_PROBS = [0.80, 0.55, 0.35, 0.20]
_C.TRAIN.TASK_SAMPLER.CURRICULUM_FORE_PROBS = [0.15, 0.35, 0.35, 0.30]
_C.TRAIN.TASK_SAMPLER.CURRICULUM_GEN_PROBS = [0.05, 0.10, 0.30, 0.50]
# E16 updates the final-phase replay mix from deterministic validation flow
# losses.  This is a stable in-training proxy; paper metrics remain the final
# selection criterion outside the optimization loop.
_C.TRAIN.TASK_SAMPLER.ADAPTIVE_START_FRACTION = 0.60
_C.TRAIN.TASK_SAMPLER.ADAPTIVE_MIN_PROB = 0.15
_C.TRAIN.TASK_SAMPLER.ADAPTIVE_MAX_PROB = 0.55
_C.TRAIN.TASK_SAMPLER.ADAPTIVE_SHIFT = 0.05
_C.TRAIN.TASK_SAMPLER.ADAPTIVE_THRESHOLD = 0.01
_C.TRAIN.TASK_SAMPLER.ADAPTIVE_UPDATE_INTERVAL = 5000
_C.TRAIN.TASK_SAMPLER.ADAPTIVE_VAL_MAX_BATCHES = 8
_C.TRAIN.TASK_SAMPLER.ADAPTIVE_VAL_T = 0.75
_C.TRAIN.TASK_SAMPLER.ADAPTIVE_VAL_SEED = 6200

# E19: training-only differentiable SMPL-X kinematic losses.  The loss uses
# the 55-joint SMPL-X kinematic tree and never constructs mesh vertices.
_C.TRAIN.GEOMETRY_LOSS = CN()
_C.TRAIN.GEOMETRY_LOSS.ENABLED = False
_C.TRAIN.GEOMETRY_LOSS.WEIGHT = 0.10
_C.TRAIN.GEOMETRY_LOSS.WARMUP_STEPS = 10000
_C.TRAIN.GEOMETRY_LOSS.TASK_WEIGHTS = [1.0, 1.0, 0.25]  # recon, fore, gen
_C.TRAIN.GEOMETRY_LOSS.JOINT_WEIGHT = 1.0
_C.TRAIN.GEOMETRY_LOSS.ROOT_WEIGHT = 2.0
_C.TRAIN.GEOMETRY_LOSS.RELATIVE_WEIGHT = 1.0
_C.TRAIN.GEOMETRY_LOSS.HAND_WEIGHT = 0.5
_C.TRAIN.GEOMETRY_LOSS.FOOT_VELOCITY_WEIGHT = 0.1
_C.TRAIN.GEOMETRY_LOSS.FOOT_HEIGHT_WEIGHT = 0.1
_C.TRAIN.GEOMETRY_LOSS.HUBER_DELTA = 0.01

_C.EVAL = CN()
_C.EVAL.KEY_JOINTS_ONLY = False
# SMPL22: pelvis, knees, ankles, feet, head, elbows, wrists.
_C.EVAL.KEY_JOINT_INDICES = [0, 4, 5, 7, 8, 10, 11, 15, 18, 19, 20, 21]
_C.EVAL.RUN_SEMANTIC = True
_C.EVAL.NUM_GPUS = 1
_C.EVAL.NUM_SAMPLES = 0  # 0 evaluates the full strided validation split.
_C.EVAL.BATCH_SIZE = 64  # Per-rank inference batch size.


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


def get_cfg():
    import sys
    import warnings

    # Example command: python train/train_uem.py CONFIG ./config/uem.yaml TRAIN.EXP_PATH ./exp/uem_v4b_dinov2 TRAIN.LR 1e-4

    warnings.filterwarnings(
        "ignore", message=".*You are using `torch.load` with `weights_only=False`.*", category=FutureWarning
    )

    cfg = get_cfg_defaults()
    argv = sys.argv.copy()

    if len(argv) > 1:
        if argv[1] == "CONFIG":
            cfg.merge_from_file(argv[2])
            argv = argv[3:]
        else:
            argv = argv[1:]
        cfg.merge_from_list(argv)
    cfg.freeze()
    print(cfg.dump())
    return cfg
