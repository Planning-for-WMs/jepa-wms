"""Bipartite Soft Matching (BSM) encoder for the L2 abstract state space.

Wraps the ``bipartite_soft_matching`` primitive in
``app.plan_common.models.AdaLN_vit`` so it can be used as a deterministic,
external encoder between frozen DINO features and the L2 predictor.

The encoder is "deterministic" in the sense that the partition is fully
determined by a reference state (``reference_idx`` for training segments,
``z_init`` for planning) — every other timestep is merged with the same
partition so abstract features stay temporally aligned.
"""

import torch

from app.plan_common.models.AdaLN_vit import bipartite_soft_matching


def _flatten_visual(features: torch.Tensor) -> torch.Tensor:
    """(B, T, V, H, W, D) → (B, T, V*H*W, D)."""
    B, T, V, H, W, D = features.shape
    return features.reshape(B, T, V * H * W, D)


def _apply_merge(
    feat_t: torch.Tensor,   # (B, n_visual, D)
    s_init: torch.Tensor,   # (B, n_visual, 1) — all ones for first pass
    merge_fn,
):
    """Mass-weighted merge of one timestep — returns (merged_feat, merged_s)."""
    weighted = feat_t * s_init
    merged = merge_fn(weighted)
    s_merged = merge_fn(s_init)
    return merged / s_merged.clamp(min=1.0), s_merged


def _build_outputs(
    flat_features: torch.Tensor,  # (B, T, n_visual, D)
    merge_fn,
    merge_pos,
    n_visual: int,
    device: torch.device,
    dtype: torch.dtype,
):
    B, T, _, _ = flat_features.shape
    s_init = torch.ones(B, n_visual, 1, device=device, dtype=dtype)

    abs_per_t = []
    s_final = None
    for t in range(T):
        merged, s_merged = _apply_merge(flat_features[:, t], s_init, merge_fn)
        abs_per_t.append(merged)
        s_final = s_merged
    abs_features = torch.stack(abs_per_t, dim=1)  # (B, T, N_abs, D)

    pos_ids = merge_pos(
        torch.arange(n_visual, device=device).unsqueeze(0).expand(B, -1).contiguous()
    )  # (B, N_abs)

    sizes = s_final.squeeze(-1)  # (B, N_abs)

    return abs_features, pos_ids, sizes


def encode_segment(
    features: torch.Tensor,
    reference_idx: int = 0,
    r: int = 128,
    metric_dim: int = 64,
) -> dict:
    """Encode a (B, T, V, H, W, D) trajectory segment into abstract space.

    The BSM partition is locked to ``features[:, reference_idx]`` and applied
    identically to every timestep along T.

    Returns a dict with:
        ``abstract``: (B, T, N_abs, D)  merged features (N_abs = V*H*W - r)
        ``pos_ids``:  (B, N_abs)        original 2D-flat indices of dst tokens
        ``sizes``:    (B, N_abs)        per-token merge mass (1 or 2 for r=N/2)
        ``merge_fn``: callable          per-batch merge function
        ``merge_pos``: callable         position-merging function
    """
    B, T, V, H, W, D = features.shape
    n_visual = V * H * W
    flat = _flatten_visual(features)

    ref = flat[:, reference_idx]  # (B, n_visual, D)
    merge_fn, merge_pos, _ = bipartite_soft_matching(ref, r=r, metric_dim=metric_dim)

    abs_features, pos_ids, sizes = _build_outputs(
        flat, merge_fn, merge_pos, n_visual, features.device, features.dtype
    )

    return {
        "abstract": abs_features,
        "pos_ids": pos_ids,
        "sizes": sizes,
        "merge_fn": merge_fn,
        "merge_pos": merge_pos,
    }


def encode_with_reference(
    features: torch.Tensor,
    reference: torch.Tensor,
    r: int = 128,
    metric_dim: int = 64,
) -> dict:
    """Encode features using a partition built from a separately-passed reference.

    Used at planning time, where ``reference`` is ``z_init`` (the agent's
    current state) and ``features`` is whatever needs to be projected into
    the same abstract space (e.g. an L1 unroll, the goal).

    Args:
        features: (B, T, V, H, W, D) — features to merge.
        reference: (B, V, H, W, D) or (B, 1, V, H, W, D) — state used to build
            the BSM partition.

    Returns same dict as ``encode_segment``.
    """
    if reference.dim() == 6:
        # accept a (B, 1, V, H, W, D) reference (e.g. ctxt with one timestep)
        reference = reference[:, 0]
    B, V, H, W, D = reference.shape
    n_visual = V * H * W
    ref_flat = reference.reshape(B, n_visual, D)
    merge_fn, merge_pos, _ = bipartite_soft_matching(ref_flat, r=r, metric_dim=metric_dim)

    flat = _flatten_visual(features)
    abs_features, pos_ids, sizes = _build_outputs(
        flat, merge_fn, merge_pos, n_visual, features.device, features.dtype
    )

    return {
        "abstract": abs_features,
        "pos_ids": pos_ids,
        "sizes": sizes,
        "merge_fn": merge_fn,
        "merge_pos": merge_pos,
    }


def apply_merge_to_features(
    features: torch.Tensor,
    merge_fn,
) -> torch.Tensor:
    """Apply a precomputed ``merge_fn`` to a feature tensor.

    Accepts (B, T, V, H, W, D) or (B, V, H, W, D) — returns the same layout
    minus the spatial dims, replaced by N_abs.

    Output: (B, T, N_abs, D) or (B, N_abs, D) accordingly.
    """
    if features.dim() == 6:
        B, T, V, H, W, D = features.shape
        flat = features.reshape(B, T, V * H * W, D)
        s_init = torch.ones(B, V * H * W, 1, device=features.device, dtype=features.dtype)
        outs = []
        for t in range(T):
            merged, _ = _apply_merge(flat[:, t], s_init, merge_fn)
            outs.append(merged)
        return torch.stack(outs, dim=1)
    elif features.dim() == 5:
        B, V, H, W, D = features.shape
        flat = features.reshape(B, V * H * W, D)
        s_init = torch.ones(B, V * H * W, 1, device=features.device, dtype=features.dtype)
        merged, _ = _apply_merge(flat, s_init, merge_fn)
        return merged
    else:
        raise ValueError(f"apply_merge_to_features: unexpected ndim={features.dim()}")
