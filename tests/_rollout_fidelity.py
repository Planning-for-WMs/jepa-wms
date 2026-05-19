"""
Rollout fidelity sanity check.

Loads the trained FlexTok WM, takes a real PushT trajectory, and measures how
well the predictor's rollouts match the encoder's features of the actual future
frames. If rollouts diverge fast (cosine < 0.5 within 2-3 steps), the WM is too
weak for CEM planning regardless of planner tuning.

Two regimes are tested:
  - Teacher-forcing (one-step): predictor sees GT context, predicts next step.
    Measures pure one-step prediction quality.
  - Autoregressive (multi-step): predictor consumes its own predictions.
    Measures rollout stability — the metric that matters for planning.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tensordict import TensorDict

REPO = Path("/home/aarav/wms/jepa-wms")
sys.path.insert(0, str(REPO))

from app.plan_common.datasets.preprocessor import Preprocessor
from app.plan_common.datasets.transforms import make_transforms
from app.plan_common.datasets.pusht_dset import PushTDataset
from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module

DEVICE = torch.device("cuda:0")
EVAL_CFG = REPO / "configs/evals/simu_env_planning/pt/flextok-wm/pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_flextok_k64_nodecode.yaml"
DSET_PATH = "/home/aarav/wms/jepa-wms-data/pusht_noise/train"
H = 6  # rollout horizon (matches planner)
CTXT = 2  # context window (matches eval config)
TRAJ_IDX = 7  # which trajectory to inspect


def build_model():
    cfg = yaml.safe_load(EVAL_CFG.read_text())
    mk = cfg["model_kwargs"]
    pretrain_kwargs = mk["pretrain_kwargs"]  # the *actual* model config

    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_scale=(1.777, 1.777),
        random_resize_aspect_ratio=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        img_size=mk["data"]["img_size"],
        normalize=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
    )
    tmp = PushTDataset(data_path=DSET_PATH, transform=None, normalize_action=True, with_velocity=True)
    preprocessor = Preprocessor(
        action_mean=tmp.action_mean, action_std=tmp.action_std,
        state_mean=tmp.state_mean, state_std=tmp.state_std,
        proprio_mean=tmp.proprio_mean, proprio_std=tmp.proprio_std,
        transform=transform, inverse_transform=None,
    )

    model = init_module(
        folder=Path(mk["checkpoint"]).parent,
        checkpoint=Path(mk["checkpoint"]).name,
        model_kwargs=pretrain_kwargs,
        device=DEVICE,
        action_dim=tmp.action_dim,
        proprio_dim=tmp.proprio_dim,
        preprocessor=preprocessor,
        cfgs_data=mk["data"],
        wrapper_kwargs=mk.get("wrapper_kwargs", {}),
    )
    return model, transform


def load_trajectory(transform):
    FSK = 5  # frameskip from training config
    dset = PushTDataset(
        data_path=DSET_PATH,
        transform=transform,
        normalize_action=True,
        with_velocity=True,
    )
    T_model = CTXT + H
    # Frames at model-step granularity: 0, 5, 10, ..., 5*(T_model-1)
    model_frame_idx = list(range(0, FSK * T_model, FSK))
    # Need raw actions [FSK*(CTXT-1) : FSK*(CTXT-1 + H)] grouped into H blocks of FSK
    # i.e. starting from action that takes us from frame CTXT-1 (model) to frame CTXT
    raw_act_start = FSK * (CTXT - 1)
    raw_act_end = raw_act_start + FSK * H
    raw_frame_idx = model_frame_idx + list(range(raw_act_start, raw_act_end))
    # Use the dset to load observations + normalized actions in one shot.
    # We only need obs at model-frame indices, but the action array we want
    # is the raw ones at indices [raw_act_start..raw_act_end).
    obs, _, state, _, _ = dset.get_frames(TRAJ_IDX, model_frame_idx)
    # Extract raw normalized actions directly from dataset tensors
    raw_act = dset.actions[TRAJ_IDX, raw_act_start:raw_act_end]  # (H*FSK, 2)
    block_act = raw_act.reshape(H, FSK * 2)  # (H, 10)
    return obs, block_act, state, dset


@torch.no_grad()
def run():
    model, transform = build_model()
    obs, block_act, state, dset = load_trajectory(transform)

    # obs["visual"]: (T_model, C, H, W) in [-1, 1]
    visual = obs["visual"].unsqueeze(0).to(DEVICE)  # (1, T_model, C, H, W)
    proprio = obs["proprio"].unsqueeze(0).to(DEVICE)  # (1, T_model, P)
    block_act = block_act.unsqueeze(0).to(DEVICE)  # (1, H, 10)

    T = visual.shape[1]
    assert T == CTXT + H
    print(f"Loaded trajectory {TRAJ_IDX}: T={T} model-frames, block_act shape={block_act.shape}")

    # Encode all frames
    z_dict = model.model.encode_obs({"visual": visual, "proprio": proprio})
    gt_vid = z_dict["visual"]  # (1, T, V=1, h=8, w=8, d=1152)
    gt_prop = z_dict["proprio"]  # (1, T, P, d)
    print(f"GT visual features: {gt_vid.shape}")
    print(f"GT proprio features: {gt_prop.shape}")

    # =================================================================
    # (A) Autoregressive rollout from frame (CTXT-1): the planner-style setup
    # z_init = single frame (the last context frame), then roll H steps with
    # actions block_act[0..H-1]. Compare predictions to gt_vid[CTXT..CTXT+H-1].
    # =================================================================
    # Use last GT context frame as z_init (T=1)
    z_init = TensorDict({"visual": gt_vid[:, CTXT - 1 : CTXT],
                          "proprio": gt_prop[:, CTXT - 1 : CTXT]})
    act_suffix = block_act.permute(1, 0, 2)  # (H, 1, 10)
    print(f"\n--- Autoregressive rollout from frame {CTXT-1} (H={H}) ---")
    print(f"z_init visual: {z_init['visual'].shape}, act_suffix: {act_suffix.shape}")
    pred = model.unroll(z_init, act_suffix=act_suffix)
    pred_vid = pred["visual"]  # (T_out, B, ...) — T_out = 1 + H
    pred_vid_bt = pred_vid.permute(1, 0, *range(2, pred_vid.ndim))  # (B, T_out, ...)
    print(f"Predicted visual features: {pred_vid_bt.shape}")
    assert pred_vid_bt.shape[1] == 1 + H, f"got {pred_vid_bt.shape[1]} frames"

    print(f"\nStep    | cos_sim       | rel_L2_err   | gt-vs-gt cos")
    print(f"--------+---------------+--------------+--------------")
    for h in range(H):
        # pred_vid_bt[:, h+1] is prediction for absolute frame CTXT-1+h+1 = CTXT+h
        gt_idx = CTXT + h
        p = pred_vid_bt[:, 1 + h].flatten(1)
        g = gt_vid[:, gt_idx].flatten(1)
        g_prev = gt_vid[:, gt_idx - 1].flatten(1)
        cos_pred = F.cosine_similarity(p, g, dim=-1).item()
        l2_err = ((p - g).norm(dim=-1) / g.norm(dim=-1)).item()
        cos_gt = F.cosine_similarity(g_prev, g, dim=-1).item()
        print(f"  +{h+1:>2}   | {cos_pred:>+.4f}       | {l2_err:>.4f}       | {cos_gt:>+.4f}")

    # =================================================================
    # (B) Teacher-forcing: at each step, give predictor the actual GT context
    # frame and the actual action. Measures pure one-step accuracy.
    # =================================================================
    print(f"\n--- Teacher-forced one-step predictions ---")
    print(f"Step    | cos_sim       | rel_L2_err")
    print(f"--------+---------------+-----------")
    for h in range(H):
        # Predict frame CTXT+h from context = last GT frame CTXT+h-1
        z_ctxt_tf = TensorDict({"visual": gt_vid[:, CTXT + h - 1 : CTXT + h],
                                "proprio": gt_prop[:, CTXT + h - 1 : CTXT + h]})
        act_tf = block_act[:, h : h + 1].permute(1, 0, 2)  # (1, 1, 10)
        pred_tf = model.unroll(z_ctxt_tf, act_suffix=act_tf)
        pv = pred_tf["visual"].permute(1, 0, *range(2, pred_tf["visual"].ndim))
        # pv shape: (1, 2, ...): [GT ctxt, prediction]
        p = pv[:, 1].flatten(1)
        g = gt_vid[:, CTXT + h].flatten(1)
        cos = F.cosine_similarity(p, g, dim=-1).item()
        l2_err = ((p - g).norm(dim=-1) / g.norm(dim=-1)).item()
        print(f"  +{h+1:>2}   | {cos:>+.4f}       | {l2_err:>.4f}")

    # =================================================================
    # (C) Identity baseline: predicting "no change" (return last GT)
    # Tells us if our predictions beat doing nothing.
    # =================================================================
    print(f"\n--- Identity baseline (predict = last GT, no model) ---")
    print(f"Step    | cos_sim       | rel_L2_err")
    print(f"--------+---------------+-----------")
    for h in range(H):
        # Predict frame CTXT+h as if it equals frame CTXT-1 (no model at all)
        p = gt_vid[:, CTXT - 1].flatten(1)
        g = gt_vid[:, CTXT + h].flatten(1)
        cos = F.cosine_similarity(p, g, dim=-1).item()
        l2_err = ((p - g).norm(dim=-1) / g.norm(dim=-1)).item()
        print(f"  +{h+1:>2}   | {cos:>+.4f}       | {l2_err:>.4f}")


if __name__ == "__main__":
    run()
