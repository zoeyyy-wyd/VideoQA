"""
Standalone evaluation script for Moment-DETR on QVHighlights.

Place this file at the project root (same level as moment_detr/, utils/,
standalone_eval/, data/). Run with `python evaluate.py` directly --
no PYTHONPATH setup, no command-line arguments needed.

Parameters are aligned with the official train.sh:
    v_feat_types = slowfast_clip  -> v_feat_dim = 2304 + 512 = 2816
    t_feat_type  = clip           -> t_feat_dim = 512
    ctx_mode     = video_tef

Outputs:
    results_eval/
      |- inference_hl_test_test_preds.jsonl          # predictions
      |- inference_hl_test_test_preds_metrics.json   # 14 metrics
"""
import os
import sys
import pprint
import logging
from easydict import EasyDict as edict

# ===== Add project root to sys.path automatically =====
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.backends.cudnn as cudnn

from moment_detr.inference import eval_epoch, setup_model
from moment_detr.start_end_dataset import StartEndDataset


# =====================================================================
#                  EDIT THIS SECTION FOR YOUR SETUP
# =====================================================================
CONFIG = dict(
    # ---------- Required: confirm these match your setup ----------
    resume          = "model_final.ckpt",                       # your checkpoint
    eval_split_name = "test",                                   # "val" or "test"
    eval_path       = "data/highlight_test_with_gt.jsonl",      # GT file
    results_dir     = "results_eval",                           # output dir

    # ---------- Feature paths and dims (matches train.sh: slowfast + clip) ----------
    v_feat_dirs = [
        "features/slowfast_features",
        "features/clip_features",
    ],
    t_feat_dir = "features/clip_text_features",
    v_feat_dim = 2818,   # SlowFast(2304) + CLIP(512), aligned with train.sh
    t_feat_dim = 512,

    # ---------- Data settings (must match training) ----------
    dset_name        = "hl",
    ctx_mode         = "video_tef",
    span_loss_type   = "l1",
    max_q_l          = 32,
    max_v_l          = 75,
    clip_length      = 2,
    max_windows      = 5,
    data_ratio       = 1.0,
    no_norm_vfeat    = False,
    no_norm_tfeat    = False,

    # ---------- Inference hyperparams ----------
    eval_bsz         = 100,
    num_workers      = 4,
    pin_memory       = True,
    nms_thd          = -1,           # -1 = no NMS; set to 0.7 to enable
    max_before_nms   = 10,
    max_after_nms    = 10,
    no_sort_results  = False,

    # ---------- Model architecture (defaults; train.sh did not override) ----------
    position_embedding = "sine",
    enc_layers         = 2,
    dec_layers         = 2,
    dim_feedforward    = 1024,
    hidden_dim         = 256,
    input_dropout      = 0.5,
    dropout            = 0.1,
    txt_drop_ratio     = 0,
    use_txt_pos        = False,
    n_input_proj       = 2,
    nheads             = 8,
    num_queries        = 10,
    pre_norm           = False,

    # ---------- Loss settings (criterion is built even at eval time) ----------
    aux_loss          = True,
    set_cost_span     = 10,
    set_cost_giou     = 1,
    set_cost_class    = 4,
    span_loss_coef    = 10,
    giou_loss_coef    = 1,
    label_loss_coef   = 4,
    eos_coef          = 0.1,
    temperature       = 0.07,
    saliency_margin   = 0.2,
    lw_saliency       = 1.0,
    contrastive_align_loss     = False,
    contrastive_align_loss_coef = 0.0,
    contrastive_hdim           = 64,
    no_aux_loss                = False,

    # ---------- Misc ----------
    debug         = False,
    seed          = 2018,
    eval_id       = "test",
    resume_all    = False,
    no_pin_memory = False,
    lr            = 1e-4,
    wd            = 1e-4,
    lr_drop       = 400,
)
# =====================================================================


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def build_opt():
    """Build an opt object compatible with what TestOptions().parse() returns."""
    opt = edict(CONFIG)

    # device: build_model uses torch.device(args.device), so we pass torch.device
    if torch.cuda.is_available():
        opt.device = torch.device("cuda:0")
    else:
        opt.device = torch.device("cpu")
        logger.warning("CUDA not available, falling back to CPU (will be slow)")

    # Create output directory
    os.makedirs(opt.results_dir, exist_ok=True)
    opt.model_dir = opt.results_dir

    # Validate critical paths
    if not os.path.isfile(opt.resume):
        raise FileNotFoundError(
            f"Checkpoint not found: {opt.resume} "
            f"(absolute path: {os.path.abspath(opt.resume)})"
        )
    if not os.path.isfile(opt.eval_path):
        raise FileNotFoundError(f"GT file not found: {opt.eval_path}")
    for d in opt.v_feat_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Video feature directory not found: {d}")
    if not os.path.isdir(opt.t_feat_dir):
        raise FileNotFoundError(f"Text feature directory not found: {opt.t_feat_dir}")

    return opt


def main():
    logger.info("=" * 60)
    logger.info("Moment-DETR Standalone Evaluation")
    logger.info("=" * 60)

    opt = build_opt()
    logger.info(f"checkpoint  : {opt.resume}")
    logger.info(f"eval split  : {opt.eval_split_name}")
    logger.info(f"GT file     : {opt.eval_path}")
    logger.info(f"v_feat_dim  : {opt.v_feat_dim} (slowfast 2304 + clip 512)")
    logger.info(f"t_feat_dim  : {opt.t_feat_dim}")
    logger.info(f"results dir : {opt.results_dir}")
    logger.info(f"device      : {opt.device}")

    cudnn.benchmark = True
    cudnn.deterministic = False

    # Build dataset
    eval_dataset = StartEndDataset(
        dset_name=opt.dset_name,
        data_path=opt.eval_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        load_labels=True,                    # must be True to compute metrics
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=0,
    )
    logger.info(f"dataset size: {len(eval_dataset)}")

    # Build model and load checkpoint
    model, criterion, _, _ = setup_model(opt)

    # Run evaluation
    save_submission_filename = f"inference_{opt.dset_name}_{opt.eval_split_name}_{opt.eval_id}_preds.jsonl"
    logger.info("Starting inference & evaluation...")
    with torch.no_grad():
        metrics_no_nms, metrics_nms, _, latest_file_paths = eval_epoch(
            model, eval_dataset, opt, save_submission_filename, criterion=criterion
        )

    # Print metrics
    print("\n" + "=" * 60)
    print("Evaluation Results (no NMS)")
    print("=" * 60)
    pprint.pprint(metrics_no_nms["brief"], indent=4)

    if metrics_nms is not None:
        print("\n" + "=" * 60)
        print(f"Evaluation Results (NMS thd = {opt.nms_thd})")
        print("=" * 60)
        pprint.pprint(metrics_nms["brief"], indent=4)

    print("\nFiles saved:")
    for p in latest_file_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()