"""Standalone L2 world model training script.

Precomputes DINO features once and streams them to HDF5. Subsequent runs
load directly from the cache — no encoder or video decoding needed.

Usage:
    python -m app.vjepa_wm.train_l2 --fname configs/vjepa_wm/l2/pt_l2.yaml [--devices cuda:0] [--debug]
"""

import argparse
import os
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from app.plan_common.datasets.l2_hdf5_dset import (
    L2FeatureHDF5,
    L2WaypointDataset,
    collate_l2,
)
from app.plan_common.models.AdaLN_vit import vit_predictor_AdaLN
from app.plan_common.models.latent_action_encoder import LatentActionEncoder
from app.vjepa_wm.l2_world_model import L2WorldModel
from app.vjepa_wm.utils import init_video_model
from src.utils.logging import get_logger
from src.utils.yaml_utils import expand_env_vars

logger = get_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, required=True, help="Path to L2 config YAML")
parser.add_argument("--devices", type=str, nargs="+", default=["cuda:0"])
parser.add_argument("--debug", action="store_true")

# ── Precomputation helpers ──────────────────────────────────────────────


def _make_transforms(img_size, normalize):
    import torchvision.transforms as T

    return T.Compose([
        T.Resize((img_size, img_size)),
        T.Normalize(mean=normalize[0], std=normalize[1]),
    ])


class _TrajIndexDataset(torch.utils.data.Dataset):
    """Thin wrapper so DataLoader workers decode videos in parallel."""

    def __init__(self, raw_dataset, n_trajs):
        self.ds = raw_dataset
        self.n = n_trajs

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        T = self.ds.get_seq_length(idx)
        obs, act, state, _, _ = self.ds[idx]
        vis = obs["visual"][:T]
        prop = obs.get("proprio")
        prop = prop[:T] if prop is not None else None
        return vis, act[:T], prop, T


@torch.no_grad()
def precompute_and_save(
    raw_dataset,
    encoder,
    grid_size,
    embed_dim,
    normalize_reps,
    h5_path,
    device,
    encode_batch_size=2048,
    quick_debug=False,
):
    """Encode all trajectory frames through frozen DINO and stream to HDF5.

    Writes features directly to the HDF5 file during encoding, avoiding the
    need to hold all features in RAM simultaneously.
    """
    n_trajs = len(raw_dataset)
    if quick_debug:
        n_trajs = min(n_trajs, 20)

    ep_lengths = [raw_dataset.get_seq_length(i) for i in range(n_trajs)]
    total_frames = sum(ep_lengths)
    ep_len = np.array(ep_lengths, dtype=np.int32)
    ep_offset = np.zeros(n_trajs, dtype=np.int64)
    if n_trajs > 1:
        ep_offset[1:] = np.cumsum(ep_len[:-1])

    n_patches = grid_size * grid_size
    action_dim = raw_dataset.action_dim
    proprio_dim = raw_dataset.proprio_dim

    num_workers = min(32, os.cpu_count() or 8)
    if quick_debug:
        num_workers = 0

    num_gpus = torch.cuda.device_count()
    enc = nn.DataParallel(encoder) if num_gpus > 1 else encoder

    loader = torch.utils.data.DataLoader(
        _TrajIndexDataset(raw_dataset, n_trajs),
        batch_size=1,
        num_workers=num_workers,
        collate_fn=lambda b: b[0],
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    parent = os.path.dirname(h5_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    ct = 256
    logger.info(f"Streaming {n_trajs} trajectories ({total_frames} frames) → {h5_path}")

    with h5py.File(h5_path, "w") as hf:
        fd = hf.create_dataset(
            "features",
            shape=(total_frames, n_patches, embed_dim),
            dtype=np.float16,
            chunks=(min(ct, total_frames), n_patches, embed_dim),
        )
        ad = hf.create_dataset(
            "actions",
            shape=(total_frames, action_dim),
            dtype=np.float16,
            chunks=(min(ct * 4, total_frames), action_dim),
        )
        pd = hf.create_dataset(
            "proprios",
            shape=(total_frames, proprio_dim),
            dtype=np.float16,
            chunks=(min(ct * 4, total_frames), proprio_dim),
        )
        hf.create_dataset("ep_len", data=ep_len)
        hf.create_dataset("ep_offset", data=ep_offset)
        for k, v in [
            ("embed_dim", embed_dim),
            ("grid_size", grid_size),
            ("action_dim", action_dim),
            ("proprio_dim", proprio_dim),
            ("n_trajs", n_trajs),
            ("total_frames", total_frames),
        ]:
            hf.attrs[k] = v

        frame_buf = []
        meta_buf = []
        buf_n = 0
        t0 = time.time()
        done = 0

        def _flush():
            nonlocal buf_n
            if not frame_buf:
                return
            frames = torch.cat(frame_buf, dim=0).to(device)
            fc = []
            for s in range(0, len(frames), encode_batch_size):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    fc.append(enc(frames[s : s + encode_batch_size]).cpu())
            del frames
            feats = torch.cat(fc, dim=0)

            off = 0
            for ti, T_i, a, p in meta_buf:
                f = feats[off : off + T_i]
                if normalize_reps:
                    f = F.layer_norm(f.float(), (f.size(-1),))
                o = int(ep_offset[ti])
                fd[o : o + T_i] = f.half().numpy()
                ad[o : o + T_i] = a[:T_i].float().half().numpy()
                if p is not None:
                    pd[o : o + T_i] = p[:T_i].float().half().numpy()
                off += T_i

            frame_buf.clear()
            meta_buf.clear()
            buf_n = 0

        for idx, (vis, act, prop, T_i) in enumerate(loader):
            if isinstance(T_i, torch.Tensor):
                T_i = T_i.item()
            frame_buf.append(vis)
            meta_buf.append((idx, T_i, act, prop))
            buf_n += T_i
            done += 1

            if buf_n >= encode_batch_size:
                _flush()

            if done % 500 == 0:
                elapsed = time.time() - t0
                total_done = sum(ep_lengths[:done])
                logger.info(
                    f"  {done}/{n_trajs} trajectories ({total_done / elapsed:.0f} fps)"
                )

        _flush()

    elapsed = time.time() - t0
    sz = os.path.getsize(h5_path) / 1e9
    logger.info(
        f"Saved {h5_path}: {n_trajs} trajs, {total_frames} frames, "
        f"{sz:.1f} GB, {elapsed:.1f}s"
    )


# ── Main ────────────────────────────────────────────────────────────────


def main(args):
    args = expand_env_vars(args)

    folder = args["folder"]
    checkpoint_folder = args.get("checkpoint_folder", folder)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(checkpoint_folder, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = args.get("seed", 234)
    torch.manual_seed(seed)

    quick_debug = args.get("quick_debug", False)
    dtype = torch.bfloat16

    cfgs_model = args["model"]
    cfgs_data = args["data"]
    cfgs_data_aug = args.get("data_aug", {})
    img_size = cfgs_data.get("img_size", 224)
    grid_size = cfgs_model.get("grid_size", 16)
    embed_dim = cfgs_model["visual_encoder"].get("embed_dim", 384)
    normalize_reps = cfgs_model.get("wm_encoding", {}).get("normalize_reps", False)

    # ── HDF5 feature cache ──────────────────────────────────────────────
    cfgs_pre = args.get("precompute", {})
    default_cache = os.path.join(checkpoint_folder, "dino_features.h5")
    cache_path = cfgs_pre.get("cache_path", default_cache)
    keys_to_cache = cfgs_pre.get("keys_to_cache", ["features", "actions", "proprios"])

    if quick_debug:
        cache_path = cache_path.replace(".h5", "_debug.h5")

    if os.path.exists(cache_path):
        logger.info(f"Loading cached features from {cache_path}")
    else:
        logger.info("No HDF5 cache found — precomputing DINO features...")
        normalize = cfgs_data_aug.get(
            "normalize", [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
        )
        transform = _make_transforms(img_size=img_size, normalize=normalize)

        from app.plan_common.datasets.pusht_dset import PushTDataset
        from src.datasets.utils.utils import get_dataset_paths

        dataset_paths = get_dataset_paths(cfgs_data.get("datasets", []))
        cfgs_custom = cfgs_data.get("custom", {})
        raw_dataset = PushTDataset(
            data_path=dataset_paths[0] + "/train",
            transform=transform,
            normalize_action=cfgs_custom.get("normalize_action", True),
            with_velocity=True,
        )
        logger.info(
            f"Raw dataset: {len(raw_dataset)} trajs, "
            f"action_dim={raw_dataset.action_dim}, "
            f"proprio_dim={raw_dataset.proprio_dim}"
        )

        enc_kwargs = {}
        if "visual_encoder" in cfgs_model:
            enc_kwargs.update(cfgs_model["visual_encoder"])
        enc_kwargs.update({
            "device": device,
            "img_size": img_size,
            "pred_type": "none",
            "use_action": False,
            "use_proprio": False,
        })
        _, encoder, _, _ = init_video_model(**enc_kwargs)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False

        precompute_and_save(
            raw_dataset,
            encoder,
            grid_size,
            embed_dim,
            normalize_reps,
            cache_path,
            device,
            encode_batch_size=cfgs_pre.get("encode_batch_size", 2048),
            quick_debug=quick_debug,
        )

        del encoder, raw_dataset
        torch.cuda.empty_cache()
        logger.info("Encoder freed from GPU")

    h5_dset = L2FeatureHDF5(cache_path, keys_to_cache=keys_to_cache)

    # ── L2 model ────────────────────────────────────────────────────────
    cfgs_l2 = args["l2"]
    latent_action_dim = cfgs_l2["latent_action_dim"]
    max_chunk_size = cfgs_l2["max_chunk_size"]
    num_waypoints = cfgs_l2["num_waypoints"]
    segment_range = tuple(cfgs_l2.get("segment_length_range", [25, 70]))
    ctxt_window = cfgs_l2.get("ctxt_window", 2)

    cfgs_l2_pred = cfgs_l2["predictor"]
    cfgs_bsm = cfgs_l2.get("bsm", {})
    bsm_r = cfgs_bsm.get("r", grid_size * grid_size // 2)
    bsm_metric_dim = cfgs_bsm.get("metric_dim", 64)
    n_abs = grid_size * grid_size - bsm_r
    local_window = tuple(cfgs_l2_pred.get("local_window", [3, -1, -1]))
    patch_size = 14  # DINO ViT-S/14
    l2_predictor = vit_predictor_AdaLN(
        img_size=grid_size * patch_size,
        patch_size=patch_size,
        num_frames=num_waypoints - 1,
        tubelet_size=1,
        embed_dim=embed_dim,
        predictor_embed_dim=cfgs_l2_pred.get("pred_embed_dim", 768),
        depth=cfgs_l2_pred.get("pred_depth", 10),
        num_heads=cfgs_l2_pred.get("pred_num_heads", 12),
        use_rope=cfgs_l2_pred.get("use_rope", True),
        action_dim=latent_action_dim,
        action_encoder_inpred=True,
        proprio_dim=h5_dset.proprio_dim,
        use_proprio=cfgs_l2_pred.get("use_proprio", True),
        proprio_encoding=cfgs_l2_pred.get("proprio_encoding", "token"),
        proprio_tokens=cfgs_l2_pred.get("proprio_tokens", 1),
        proprio_emb_dim=cfgs_l2_pred.get("proprio_emb_dim", 0),
        proprio_encoder_inpred=True,
        local_window=local_window,
        external_merge=True,
        external_n_abs=n_abs,
    ).to(device)

    cfgs_ae = cfgs_l2["action_encoder"]
    action_encoder = LatentActionEncoder(
        action_dim=h5_dset.action_dim,
        latent_action_dim=latent_action_dim,
        max_chunk_size=max_chunk_size,
        encoder_depth=cfgs_ae.get("encoder_depth", 2),
        encoder_heads=cfgs_ae.get("encoder_heads", 4),
        encoder_dim=cfgs_ae.get("encoder_dim", 128),
    ).to(device)

    l2_model = L2WorldModel(
        l2_predictor=l2_predictor,
        action_encoder=action_encoder,
        embed_dim=embed_dim,
        grid_size=grid_size,
        normalize_reps=normalize_reps,
        ctxt_window=ctxt_window,
        r=bsm_r,
        bsm_metric_dim=bsm_metric_dim,
    ).to(device)

    pred_params = sum(p.numel() for p in l2_predictor.parameters())
    ae_params = sum(p.numel() for p in action_encoder.parameters())
    logger.info(f"L2: {pred_params + ae_params:,} trainable params")

    # ── Resume checkpoint (before compile) ──────────────────────────────
    start_epoch = 0
    latest_path = os.path.join(checkpoint_folder, "l2-latest.pth.tar")
    ckpt = None
    if os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location="cpu")
        l2_model.load_state_dict(ckpt["l2_model"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed model from epoch {start_epoch}")

    l2_model = torch.compile(l2_model)

    # ── Optimizer ───────────────────────────────────────────────────────
    cfgs_opt = args["optimization"]
    lr = cfgs_opt.get("lr", 3e-4)
    num_epochs = cfgs_opt.get("num_epochs", 500)
    batch_size = cfgs_opt.get("batch_size", 64)
    grad_accum = cfgs_opt.get("gradient_accumulation_steps", 1)
    clip_grad = cfgs_opt.get("clip_grad", 1.0)

    optimizer = torch.optim.AdamW(
        l2_model.parameters(),
        lr=lr,
        weight_decay=cfgs_opt.get("weight_decay", 1e-5),
        betas=tuple(cfgs_opt.get("betas", [0.9, 0.999])),
    )
    scaler = torch.amp.GradScaler("cuda")
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        del ckpt
        logger.info("Resumed optimizer state")

    # ── Wandb ───────────────────────────────────────────────────────────
    cfgs_logging = args.get("logging", {})
    cfgs_wandb = cfgs_logging.get("wandb", {})
    use_wandb = cfgs_wandb.get("use_wandb", False)
    if use_wandb:
        import wandb

        project = cfgs_wandb.get("project", "vjepa_wm_l2")
        if quick_debug:
            project += "_debug"
        try:
            wandb.init(
                project=project,
                dir=folder,
                config=args,
                settings=wandb.Settings(init_timeout=300),
            )
            wandb.run.name = os.path.basename(folder)
        except Exception as e:
            logger.warning(f"wandb init failed: {e}. Continuing without wandb.")
            use_wandb = False

    # ── DataLoader ──────────────────────────────────────────────────────
    train_dset = L2WaypointDataset(
        h5_dset,
        num_waypoints,
        segment_range,
        max_chunk_size,
        grid_size,
        embed_dim,
    )
    loader_workers = cfgs_pre.get("loader_workers", 8)
    if quick_debug:
        loader_workers = 0
    effective_bs = min(batch_size, len(train_dset))
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=effective_bs,
        shuffle=True,
        num_workers=loader_workers,
        pin_memory=True,
        drop_last=len(train_dset) > effective_bs,
        persistent_workers=loader_workers > 0,
        collate_fn=collate_l2,
    )

    # ── Training loop ───────────────────────────────────────────────────
    save_every = cfgs_opt.get("save_every", 50)
    log_freq = cfgs_opt.get("log_freq", 10)
    logger.info(
        f"Training: {len(train_dset)} trajs | micro-batch {effective_bs} | "
        f"grad_accum {grad_accum} (eff. batch {effective_bs * grad_accum}) | "
        f"{len(train_loader)} iters/epoch | {loader_workers} workers"
    )

    for epoch in range(start_epoch, num_epochs):
        t_epoch = time.time()
        l2_model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad()

        for itr, (wp_features, action_chunks, chunk_lengths, wp_proprios) in enumerate(
            train_loader
        ):
            if quick_debug and itr > 5:
                break

            wp_features = wp_features.to(device, non_blocking=True)
            action_chunks = action_chunks.to(device, non_blocking=True)
            chunk_lengths = chunk_lengths.to(device, non_blocking=True)
            if wp_proprios is not None:
                wp_proprios = wp_proprios.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=dtype):
                pred_features, pred_proprios, latent_actions, target_features = (
                    l2_model.forward_teacher_forcing(
                        wp_features, action_chunks, chunk_lengths, wp_proprios,
                    )
                )
                target_proprios = (
                    wp_proprios[:, 1:] if wp_proprios is not None else None
                )
                loss, loss_dict = l2_model.compute_loss(
                    pred_features, target_features, pred_proprios, target_proprios,
                )
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (itr + 1) % grad_accum == 0 or (itr + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(l2_model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * grad_accum
            epoch_steps += 1

            if itr % log_freq == 0:
                logger.info(
                    f"  [{epoch + 1}, {itr:4d}/{len(train_loader)}] "
                    f"loss={loss.item() * grad_accum:.4f}"
                )
                if use_wandb:
                    log_d = {
                        "epoch": epoch + 1,
                        "itr": epoch * len(train_loader) + itr,
                        **loss_dict,
                    }
                    log_d["latent_action_norm"] = (
                        latent_actions.detach().norm(dim=-1).mean().item()
                    )
                    wandb.log(log_d)

        avg_loss = epoch_loss / max(epoch_steps, 1)
        dt = time.time() - t_epoch
        logger.info(f"Epoch {epoch + 1} avg_loss={avg_loss:.4f} ({dt:.1f}s)")

        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            model_sd = (
                l2_model._orig_mod.state_dict()
                if hasattr(l2_model, "_orig_mod")
                else l2_model.state_dict()
            )
            ckpt_path = os.path.join(checkpoint_folder, f"l2-ep{epoch + 1}.pth.tar")
            save_dict = {
                "l2_model": model_sd,
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": args,
            }
            torch.save(save_dict, ckpt_path)
            torch.save(save_dict, latest_path)
            logger.info(f"Saved: {ckpt_path}")

    h5_dset.close()
    logger.info("L2 training complete.")


if __name__ == "__main__":
    cli_args = parser.parse_args()
    with open(cli_args.fname, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if cli_args.debug:
        config["quick_debug"] = True
    main(config)
