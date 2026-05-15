import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tensordict.tensordict import TensorDict

from app.plan_common.models.latent_action_encoder import LatentActionEncoder
from app.vjepa_wm.bsm_encoder import (
    apply_merge_to_features,
    encode_segment,
    encode_with_reference,
)


class L2WorldModel(nn.Module):
    """Level-2 hierarchical world model operating on abstract waypoint states.

    A frozen DINO encoder produces (B, T, V, H, W, D) features; the L2 model
    further compresses each frame into N_abs<H*W tokens via Bipartite Soft
    Matching (BSM) — a deterministic encoder locked to a single reference
    state per segment (or to z_init at planning time). The predictor then
    operates entirely in this abstract space.

    Reference: "Hierarchical World Models" (arxiv 2604.03208), with the L2
    abstract-encoder modification documented in L2_ABSTRACT_ENCODER_PLAN.md.
    """

    def __init__(
        self,
        l2_predictor,
        action_encoder,
        embed_dim=384,
        grid_size=16,
        normalize_reps=False,
        ctxt_window=2,
        r=128,
        bsm_metric_dim=64,
    ):
        super().__init__()
        self.predictor = l2_predictor
        self.action_encoder = action_encoder
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.normalize_reps = normalize_reps
        self.ctxt_window = ctxt_window
        self.latent_action_dim = action_encoder.latent_action_dim
        self.r = r
        self.bsm_metric_dim = bsm_metric_dim
        self.n_abs = grid_size * grid_size - r

    @property
    def z_dim(self):
        return self.latent_action_dim

    def encode_action_chunks(self, action_chunks, chunk_lengths=None):
        """Encode variable-length action chunks into latent macro-actions."""
        B, N, T_max, A = action_chunks.shape
        chunks_flat = action_chunks.reshape(B * N, T_max, A)
        lengths_flat = chunk_lengths.reshape(B * N) if chunk_lengths is not None else None
        latent_flat = self.action_encoder(chunks_flat, chunk_lengths=lengths_flat)
        return latent_flat.reshape(B, N, -1)

    def forward_teacher_forcing(
        self,
        waypoint_features,
        action_chunks,
        chunk_lengths=None,
        waypoint_proprios=None,
    ):
        """Training forward pass with teacher-forcing on waypoint sequences.

        Args:
            waypoint_features: (B, N, V, H, W, D) — N waypoint states from DINO
            action_chunks: (B, N-1, max_chunk_size, action_dim)
            chunk_lengths: (B, N-1)
            waypoint_proprios: (B, N, P) or None

        Returns:
            pred_abs:        (B, N-1, N_abs, D) — predicted next-waypoint features in abstract space
            pred_proprios:   (B, N-1, 1, P) or None
            latent_actions:  (B, N-1, latent_action_dim)
            target_abs:      (B, N-1, N_abs, D) — BSM-encoded ground-truth next waypoints
        """
        # Lock the BSM partition to the first waypoint of the segment, then
        # apply the same merge_fn to every waypoint.
        enc = encode_segment(
            waypoint_features, reference_idx=0, r=self.r, metric_dim=self.bsm_metric_dim
        )
        abs_feats = enc["abstract"]   # (B, N, N_abs, D)
        pos_ids = enc["pos_ids"]      # (B, N_abs)
        sizes = enc["sizes"]          # (B, N_abs)

        latent_actions = self.encode_action_chunks(action_chunks, chunk_lengths)

        inp = abs_feats[:, :-1]                                  # (B, N-1, N_abs, D)
        T_in = inp.size(1)
        pos_t = pos_ids.unsqueeze(1).expand(-1, T_in, -1)        # (B, T_in, N_abs)
        sizes_t = sizes.unsqueeze(1).expand(-1, T_in, -1)        # (B, T_in, N_abs)

        pred_abs, _, pred_proprio = self.predictor(
            inp,
            latent_actions,
            waypoint_proprios[:, :-1] if waypoint_proprios is not None else None,
            external_pos_ids=pos_t,
            external_sizes=sizes_t,
        )

        if self.normalize_reps:
            pred_abs = F.layer_norm(pred_abs, (pred_abs.size(-1),))

        return pred_abs, pred_proprio, latent_actions, abs_feats[:, 1:]

    def compute_loss(self, pred_features, target_features, pred_proprios=None, target_proprios=None):
        """L1 loss between predicted and actual abstract waypoint features.

        Both inputs are (B, T, N_abs, D) — no spatial grid axes.
        """
        visual_loss = F.l1_loss(pred_features, target_features)

        loss = visual_loss
        loss_dict = {"l2_visual_loss": visual_loss.item()}

        if pred_proprios is not None and target_proprios is not None:
            if pred_proprios.shape == target_proprios.shape:
                proprio_loss = F.l1_loss(pred_proprios, target_proprios)
                loss = loss + proprio_loss
                loss_dict["l2_proprio_loss"] = proprio_loss.item()

        loss_dict["l2_total_loss"] = loss.item()
        return loss, loss_dict

    def unroll(self, z_ctxt, act_suffix=None, debug=False, reference=None):
        """Autoregressive prediction for planning.

        Builds a BSM partition from ``reference`` (or, if not provided, from
        the first context frame) and runs the L2 predictor entirely in the
        abstract space.

        Args:
            z_ctxt: (B, tau, V, H, W, D) raw DINO features, or TensorDict with
                a ``visual`` key of that shape.
            act_suffix: (T, B, latent_action_dim) — latent macro-actions.
            reference: (B, V, H, W, D) state used to lock the BSM partition.
                Defaults to ``z_ctxt[:, 0]`` (or its visual component).

        Returns:
            (T+tau, B, N_abs, D) abstract trajectory, or TensorDict with that
            visual tensor plus passthrough proprio.
        """
        T, B_act, A = act_suffix.shape
        has_proprio = False

        raw_prop = None
        out_prop = None

        if isinstance(z_ctxt, (TensorDict, dict)):
            vid_feats = z_ctxt["visual"].expand(B_act, *z_ctxt["visual"].shape[1:])
            if "raw_proprio" in z_ctxt.keys():
                raw_prop = z_ctxt["raw_proprio"].expand(B_act, *z_ctxt["raw_proprio"].shape[1:])
                out_prop = z_ctxt["proprio"].expand(B_act, *z_ctxt["proprio"].shape[1:])
            else:
                raw_prop = z_ctxt["proprio"].expand(B_act, *z_ctxt["proprio"].shape[1:])
                if raw_prop.ndim == 4 and raw_prop.shape[2] == 1:
                    raw_prop = raw_prop.squeeze(2)
                out_prop = raw_prop
            has_proprio = True
        else:
            vid_feats = z_ctxt.expand(B_act, *z_ctxt.shape[1:])
            if hasattr(self.predictor, 'use_proprio') and self.predictor.use_proprio:
                pdim = getattr(self.predictor, 'proprio_dim', 1)
                raw_prop = torch.zeros(
                    B_act, vid_feats.shape[1], pdim,
                    device=vid_feats.device, dtype=vid_feats.dtype,
                )
                out_prop = raw_prop

        # Build the BSM partition. By default the reference is the latest ctxt
        # frame (the agent's "current state"), matching what the hierarchical
        # planner uses to lock its goal/L1 partition.
        if reference is None:
            reference_visual = vid_feats[:, -1]
        else:
            reference_visual = reference[:, 0] if reference.dim() == 6 else reference

        enc = encode_with_reference(
            vid_feats, reference_visual, r=self.r, metric_dim=self.bsm_metric_dim
        )
        abs_feats = enc["abstract"]   # (B_act, tau, N_abs, D)
        pos_ids = enc["pos_ids"]      # (B_act, N_abs)
        sizes = enc["sizes"]          # (B_act, N_abs)
        merge_fn = enc["merge_fn"]

        act_suffix = rearrange(act_suffix, "t b a -> b t a")

        act_feats = None
        for h in range(T):
            new_act = act_suffix[:, h: h + 1]
            act_feats = new_act if act_feats is None else torch.cat([act_feats, new_act], dim=1)

            ctx_abs = abs_feats[:, -self.ctxt_window:]                # (B_act, T_ctx, N_abs, D)
            ctx_prop = raw_prop[:, -self.ctxt_window:] if raw_prop is not None else None
            ctx_act = act_feats[:, -self.ctxt_window:]
            T_ctx = ctx_abs.size(1)

            pos_t = pos_ids.unsqueeze(1).expand(-1, T_ctx, -1)
            sizes_t = sizes.unsqueeze(1).expand(-1, T_ctx, -1)

            pred_abs, _, _ = self.predictor(
                ctx_abs,
                ctx_act,
                ctx_prop,
                external_pos_ids=pos_t,
                external_sizes=sizes_t,
            )

            if self.normalize_reps:
                pred_abs = F.layer_norm(pred_abs, (pred_abs.size(-1),))

            next_abs = pred_abs[:, -1:]                               # (B_act, 1, N_abs, D)
            abs_feats = torch.cat([abs_feats, next_abs], dim=1)

            if raw_prop is not None:
                raw_prop = torch.cat([raw_prop, raw_prop[:, -1:]], dim=1)
            if out_prop is not None:
                out_prop = torch.cat([out_prop, out_prop[:, -1:]], dim=1)

        if has_proprio:
            visual_out = rearrange(abs_feats, "b t n d -> t b n d")
            prop_out = out_prop if out_prop.ndim >= 4 else out_prop.unsqueeze(2)
            prop_out = rearrange(prop_out, "b t ... -> t b ...")
            return TensorDict({"visual": visual_out, "proprio": prop_out})
        else:
            return rearrange(abs_feats, "b t n d -> t b n d")

    def encode(self, features, reference=None):
        """Project (B, T, V, H, W, D) features into abstract space.

        Used by the planner to wrap the goal and L1 unrolls into the same
        partition that L2 just produced.
        """
        if reference is None:
            reference = features[:, 0] if features.dim() == 6 else features
        enc = encode_with_reference(
            features if features.dim() == 6 else features.unsqueeze(1),
            reference,
            r=self.r,
            metric_dim=self.bsm_metric_dim,
        )
        return enc

    def apply_merge(self, features, merge_fn):
        """Apply a precomputed merge_fn to a feature tensor."""
        return apply_merge_to_features(features, merge_fn)


def sample_waypoints(seq_len, num_waypoints, segment_range=(25, 70)):
    """Sample random waypoint indices from a trajectory segment."""
    min_len, max_len = segment_range
    max_len = min(max_len, seq_len)
    min_len = min(min_len, max_len)

    seg_len = torch.randint(min_len, max_len + 1, (1,)).item()
    seg_start = torch.randint(0, seq_len - seg_len + 1, (1,)).item()
    seg_end = seg_start + seg_len

    indices = torch.sort(
        torch.randperm(seg_len)[:num_waypoints]
    ).values + seg_start

    return indices.tolist(), seg_start, seg_end
