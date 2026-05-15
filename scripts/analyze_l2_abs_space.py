"""Diagnostics for the L2 BSM abstract latent space.

This uses the same HDF5 feature cache and waypoint sampler as L2 training.
It can be run with or without an L2 checkpoint:

    python scripts/analyze_l2_abs_space.py \
        --config configs/vjepa_wm/l2/pt_l2_abs.yaml \
        --checkpoint saved_checkpoints/l2/pt_l2_abs/l2-latest.pth.tar \
        --batches 8 --batch-size 32

The goal is not to prove the representation is "good", but to separate
representation collapse from planner/model wiring issues.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.plan_common.datasets.l2_hdf5_dset import (
    L2FeatureHDF5,
    L2WaypointDataset,
    collate_l2,
)
from app.plan_common.models.AdaLN_vit import vit_predictor_AdaLN
from app.plan_common.models.latent_action_encoder import LatentActionEncoder
from app.vjepa_wm.bsm_encoder import encode_segment
from app.vjepa_wm.l2_world_model import L2WorldModel
from src.utils.yaml_utils import expand_env_vars


def _load_yaml(path):
    with open(path, "r") as f:
        return expand_env_vars(yaml.safe_load(f))


def _flatten_wp(wp_features):
    # (B, N, 1, H, W, D) -> (B, N, H*W, D)
    B, N, V, H, W, D = wp_features.shape
    return wp_features.reshape(B, N, V * H * W, D)


def _state_l1(a, b):
    return (a.float() - b.float()).abs().mean(dim=tuple(range(1, a.ndim)))


def _pairwise_state_dist(x, max_states=192):
    # x: (S, tokens, D)
    if x.shape[0] > max_states:
        x = x[:max_states]
    flat = x.float().flatten(1)
    return torch.cdist(flat, flat, p=2)


def _corrcoef(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    mask = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[mask], b[mask]
    if a.numel() < 3:
        return torch.tensor(float("nan"))
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return (a @ b) / denom.clamp(min=1e-12)


def _spearman_temporal(step_dists):
    # step_dists: (B, N-1), distances from waypoint 0 to waypoint 1..N-1
    ranks = torch.arange(1, step_dists.shape[1] + 1, device=step_dists.device).float()
    ranks = ranks.unsqueeze(0).expand_as(step_dists)
    vals = []
    for row, rank in zip(step_dists, ranks):
        vals.append(_corrcoef(row.argsort().argsort().float(), rank))
    return torch.stack(vals).nanmean()


def _participation_ratio(states):
    # states: (S, tokens, D). PR near 1 means collapse; larger means spread.
    x = states.float().flatten(1)
    x = x - x.mean(dim=0, keepdim=True)
    gram = x @ x.T / max(1, x.shape[1] - 1)
    eig = torch.linalg.eigvalsh(gram).clamp(min=0)
    return (eig.sum() ** 2 / eig.square().sum().clamp(min=1e-12)).item()


def _build_l2_model(cfg, h5_dset, device):
    cfg_l2 = cfg["l2"]
    cfg_model = cfg["model"]
    cfg_pred = cfg_l2["predictor"]
    cfg_bsm = cfg_l2.get("bsm", {})
    cfg_ae = cfg_l2["action_encoder"]

    grid_size = cfg_model.get("grid_size", h5_dset.grid_size)
    embed_dim = cfg_model.get("visual_encoder", {}).get("embed_dim", h5_dset.embed_dim)
    bsm_r = cfg_bsm.get("r", grid_size * grid_size // 2)
    n_abs = grid_size * grid_size - bsm_r
    patch_size = 14

    predictor = vit_predictor_AdaLN(
        img_size=grid_size * patch_size,
        patch_size=patch_size,
        num_frames=cfg_l2["num_waypoints"] - 1,
        tubelet_size=1,
        embed_dim=embed_dim,
        predictor_embed_dim=cfg_pred.get("pred_embed_dim", 768),
        depth=cfg_pred.get("pred_depth", 10),
        num_heads=cfg_pred.get("pred_num_heads", 12),
        use_rope=cfg_pred.get("use_rope", True),
        action_dim=cfg_l2["latent_action_dim"],
        action_encoder_inpred=True,
        proprio_dim=h5_dset.proprio_dim,
        use_proprio=cfg_pred.get("use_proprio", True),
        proprio_encoding=cfg_pred.get("proprio_encoding", "token"),
        proprio_tokens=cfg_pred.get("proprio_tokens", 1),
        proprio_emb_dim=cfg_pred.get("proprio_emb_dim", 0),
        proprio_encoder_inpred=True,
        local_window=tuple(cfg_pred.get("local_window", [3, -1, -1])),
        external_merge=True,
        external_n_abs=n_abs,
    )
    action_encoder = LatentActionEncoder(
        action_dim=h5_dset.action_dim,
        latent_action_dim=cfg_l2["latent_action_dim"],
        max_chunk_size=cfg_l2["max_chunk_size"],
        encoder_depth=cfg_ae.get("encoder_depth", 2),
        encoder_heads=cfg_ae.get("encoder_heads", 4),
        encoder_dim=cfg_ae.get("encoder_dim", 128),
    )
    return L2WorldModel(
        l2_predictor=predictor,
        action_encoder=action_encoder,
        embed_dim=embed_dim,
        grid_size=grid_size,
        normalize_reps=cfg_model.get("wm_encoding", {}).get("normalize_reps", False),
        ctxt_window=cfg_l2.get("ctxt_window", 2),
        r=bsm_r,
        bsm_metric_dim=cfg_bsm.get("metric_dim", 64),
    ).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vjepa_wm/l2/pt_l2_abs.yaml")
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-states-for-pairwise", type=int, default=192)
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    cache_path = args.cache_path or cfg["precompute"]["cache_path"]
    cache_path = os.path.expandvars(cache_path)
    device = torch.device(args.device)

    h5_dset = L2FeatureHDF5(cache_path, keys_to_cache=["actions", "proprios"])
    cfg_l2 = cfg["l2"]
    dataset = L2WaypointDataset(
        h5_dset=h5_dset,
        num_waypoints=cfg_l2["num_waypoints"],
        segment_range=tuple(cfg_l2.get("segment_length_range", [25, 70])),
        max_chunk_size=cfg_l2["max_chunk_size"],
        grid_size=cfg["model"].get("grid_size", h5_dset.grid_size),
        embed_dim=cfg["model"].get("visual_encoder", {}).get("embed_dim", h5_dset.embed_dim),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_l2,
    )

    l2_model = None
    if args.checkpoint:
        l2_model = _build_l2_model(cfg, h5_dset, device)
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        l2_model.load_state_dict(ckpt["l2_model"])
        l2_model.eval()
        print(f"loaded_checkpoint: {args.checkpoint} epoch={ckpt.get('epoch', '?')}")

    raw_step_dists = []
    abs_step_dists = []
    raw_consecutive = []
    abs_consecutive = []
    pair_corrs = []
    raw_prs = []
    abs_prs = []
    pred_losses = []
    copy_prev_losses = []
    copy_first_losses = []
    latent_norms = []

    autocast_enabled = device.type == "cuda"
    for bi, batch in enumerate(loader):
        if bi >= args.batches:
            break
        wp_features, action_chunks, chunk_lengths, wp_proprios = batch
        wp_features = wp_features.to(device)
        action_chunks = action_chunks.to(device)
        chunk_lengths = chunk_lengths.to(device)
        if wp_proprios is not None:
            wp_proprios = wp_proprios.to(device)

        raw = _flatten_wp(wp_features)
        enc = encode_segment(
            wp_features,
            reference_idx=0,
            r=cfg_l2.get("bsm", {}).get("r", 128),
            metric_dim=cfg_l2.get("bsm", {}).get("metric_dim", 64),
        )
        abs_feats = enc["abstract"]

        raw0 = raw[:, 0]
        abs0 = abs_feats[:, 0]
        raw_step = torch.stack([_state_l1(raw0, raw[:, i]) for i in range(1, raw.shape[1])], dim=1)
        abs_step = torch.stack([_state_l1(abs0, abs_feats[:, i]) for i in range(1, abs_feats.shape[1])], dim=1)
        raw_step_dists.append(raw_step.detach().cpu())
        abs_step_dists.append(abs_step.detach().cpu())

        raw_consecutive.append(
            torch.stack([_state_l1(raw[:, i - 1], raw[:, i]) for i in range(1, raw.shape[1])], dim=1).detach().cpu()
        )
        abs_consecutive.append(
            torch.stack([_state_l1(abs_feats[:, i - 1], abs_feats[:, i]) for i in range(1, abs_feats.shape[1])], dim=1).detach().cpu()
        )

        raw_states = raw.reshape(-1, raw.shape[-2], raw.shape[-1])
        abs_states = abs_feats.reshape(-1, abs_feats.shape[-2], abs_feats.shape[-1])
        rd = _pairwise_state_dist(raw_states, args.max_states_for_pairwise)
        ad = _pairwise_state_dist(abs_states, args.max_states_for_pairwise)
        mask = torch.triu(torch.ones_like(rd, dtype=torch.bool), diagonal=1)
        pair_corrs.append(_corrcoef(rd[mask].cpu(), ad[mask].cpu()))
        raw_prs.append(_participation_ratio(raw_states[: args.max_states_for_pairwise].cpu()))
        abs_prs.append(_participation_ratio(abs_states[: args.max_states_for_pairwise].cpu()))

        if l2_model is not None:
            with torch.no_grad(), torch.amp.autocast(device.type, enabled=autocast_enabled, dtype=torch.bfloat16):
                pred, _pred_prop, latent, target = l2_model.forward_teacher_forcing(
                    wp_features, action_chunks, chunk_lengths, wp_proprios
                )
                pred_loss = (pred.float() - target.float()).abs().mean()
                copy_prev_loss = (abs_feats[:, :-1].float() - target.float()).abs().mean()
                copy_first_loss = (abs_feats[:, :1].expand_as(target).float() - target.float()).abs().mean()
            pred_losses.append(pred_loss.detach().cpu())
            copy_prev_losses.append(copy_prev_loss.detach().cpu())
            copy_first_losses.append(copy_first_loss.detach().cpu())
            latent_norms.append(latent.float().norm(dim=-1).detach().cpu())

    raw_step_dists = torch.cat(raw_step_dists)
    abs_step_dists = torch.cat(abs_step_dists)
    raw_consecutive = torch.cat(raw_consecutive)
    abs_consecutive = torch.cat(abs_consecutive)

    print("\n== Geometry Preservation ==")
    print(f"raw_abs_pairwise_distance_corr: {torch.stack(pair_corrs).mean().item():.4f}")
    print(f"raw_participation_ratio:        {sum(raw_prs) / len(raw_prs):.2f}")
    print(f"abs_participation_ratio:        {sum(abs_prs) / len(abs_prs):.2f}")

    print("\n== Temporal Signal ==")
    print(f"raw_dist_from_wp0_by_index:     {[round(x, 5) for x in raw_step_dists.mean(0).tolist()]}")
    print(f"abs_dist_from_wp0_by_index:     {[round(x, 5) for x in abs_step_dists.mean(0).tolist()]}")
    print(f"raw_temporal_spearman:          {_spearman_temporal(raw_step_dists).item():.4f}")
    print(f"abs_temporal_spearman:          {_spearman_temporal(abs_step_dists).item():.4f}")
    print(f"raw_consecutive_dist_by_index:  {[round(x, 5) for x in raw_consecutive.mean(0).tolist()]}")
    print(f"abs_consecutive_dist_by_index:  {[round(x, 5) for x in abs_consecutive.mean(0).tolist()]}")

    if pred_losses:
        pred_loss = torch.stack(pred_losses).mean().item()
        copy_prev = torch.stack(copy_prev_losses).mean().item()
        copy_first = torch.stack(copy_first_losses).mean().item()
        norms = torch.cat(latent_norms).flatten()
        print("\n== Trained L2 In Abstract Space ==")
        print(f"l2_teacher_forced_l1:           {pred_loss:.5f}")
        print(f"copy_previous_l1:               {copy_prev:.5f}")
        print(f"copy_first_l1:                  {copy_first:.5f}")
        print(f"l2_vs_copy_previous_ratio:      {pred_loss / max(copy_prev, 1e-12):.4f}")
        print(f"latent_action_norm_mean/std:    {norms.mean().item():.4f} / {norms.std().item():.4f}")
        print(f"latent_action_norm_p95:         {norms.quantile(0.95).item():.4f}")


if __name__ == "__main__":
    main()
