import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tensordict.tensordict import TensorDict

from app.plan_common.models.latent_action_encoder import LatentActionEncoder


class L2WorldModel(nn.Module):
    """Level-2 hierarchical world model operating on waypoint states with latent macro-actions.

    Wraps a scaled-up AdaLN ViT predictor and a transformer-based latent action encoder.
    Both levels share the same frozen DINO encoder, so L2 operates in the same spatial
    feature space as L1.

    Reference: "Hierarchical World Models" (arxiv 2604.03208).
    """

    def __init__(
        self,
        l2_predictor,
        action_encoder,
        embed_dim=384,
        grid_size=16,
        normalize_reps=False,
        ctxt_window=2,
    ):
        super().__init__()
        self.predictor = l2_predictor
        self.action_encoder = action_encoder
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.normalize_reps = normalize_reps
        self.ctxt_window = ctxt_window
        self.latent_action_dim = action_encoder.latent_action_dim

    @property
    def z_dim(self):
        return self.latent_action_dim

    def encode_action_chunks(self, action_chunks, chunk_lengths=None):
        """Encode variable-length action chunks into latent macro-actions.

        Args:
            action_chunks: (B, N_transitions, max_chunk_size, action_dim)
            chunk_lengths: (B, N_transitions) or None

        Returns:
            latent_actions: (B, N_transitions, latent_action_dim)
        """
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
            waypoint_features: (B, N, V, H, W, D) — N waypoint states from frozen DINO encoder
            action_chunks: (B, N-1, max_chunk_size, action_dim) — raw actions between waypoints
            chunk_lengths: (B, N-1) — actual length of each chunk
            waypoint_proprios: (B, N, P) or None — proprio at waypoints

        Returns:
            pred_features: (B, N-1, V, H, W, D) — predicted next waypoint features
            pred_proprios: (B, N-1, 1, P) or None
            latent_actions: (B, N-1, latent_action_dim) — encoded latent actions
        """
        B, N, V, H, W, D = waypoint_features.shape

        latent_actions = self.encode_action_chunks(action_chunks, chunk_lengths)

        input_features = waypoint_features[:, :-1]
        latent_acts_for_pred = latent_actions

        pred_video, _, pred_proprio = self.predictor(
            input_features,
            latent_acts_for_pred,
            waypoint_proprios[:, :-1] if waypoint_proprios is not None else None,
        )

        pred_video = rearrange(
            pred_video, "b t (v h w) d -> b t v h w d",
            h=self.grid_size, w=self.grid_size, v=1,
        )

        if self.normalize_reps:
            pred_video = F.layer_norm(pred_video, (pred_video.size(-1),))

        return pred_video, pred_proprio, latent_actions

    def compute_loss(self, pred_features, target_features, pred_proprios=None, target_proprios=None):
        """L1 loss between predicted and actual waypoint features (paper Equation 1).

        Args:
            pred_features: (B, N-1, V, H, W, D)
            target_features: (B, N-1, V, H, W, D) — actual next waypoint features
            pred_proprios: (B, N-1, ...) or None
            target_proprios: (B, N-1, ...) or None

        Returns:
            loss: scalar
            loss_dict: dict with detailed losses for logging
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

    def unroll(self, z_ctxt, act_suffix=None, debug=False):
        """Autoregressive prediction for planning — matches EncPredWM.unroll interface.

        Args:
            z_ctxt: (B, tau, V, H, W, D) or TensorDict — initial waypoint state(s)
            act_suffix: (T, B, latent_action_dim) — latent macro-actions from planner

        Returns:
            (T+tau, B, V, H, W, D) or TensorDict — predicted waypoint trajectory
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

        act_suffix = rearrange(act_suffix, "t b a -> b t a")

        act_feats = None
        for h in range(T):
            new_act = act_suffix[:, h: h + 1]
            act_feats = new_act if act_feats is None else torch.cat([act_feats, new_act], dim=1)

            ctx_vid = vid_feats[:, -self.ctxt_window:]
            ctx_prop = raw_prop[:, -self.ctxt_window:] if raw_prop is not None else None
            ctx_act = act_feats[:, -self.ctxt_window:]

            pred_video, _, _ = self.predictor(
                ctx_vid, ctx_act, ctx_prop,
            )

            pred_video = rearrange(
                pred_video, "b t (v h w) d -> b t v h w d",
                h=self.grid_size, w=self.grid_size, v=1,
            )

            if self.normalize_reps:
                pred_video = F.layer_norm(pred_video, (pred_video.size(-1),))

            next_vid = pred_video[:, -1:]
            vid_feats = torch.cat([vid_feats, next_vid], dim=1)

            # L2 does not predict proprio — repeat the last value forward
            if raw_prop is not None:
                raw_prop = torch.cat([raw_prop, raw_prop[:, -1:]], dim=1)
            if out_prop is not None:
                out_prop = torch.cat([out_prop, out_prop[:, -1:]], dim=1)

        if has_proprio:
            vid_feats = rearrange(vid_feats, "b t ... -> t b ...")
            prop_out = out_prop if out_prop.ndim >= 4 else out_prop.unsqueeze(2)
            prop_out = rearrange(prop_out, "b t ... -> t b ...")
            return TensorDict({"visual": vid_feats, "proprio": prop_out})
        else:
            return rearrange(vid_feats, "b t ... -> t b ...")


def sample_waypoints(seq_len, num_waypoints, segment_range=(25, 70)):
    """Sample random waypoint indices from a trajectory segment.

    Args:
        seq_len: total trajectory length
        num_waypoints: N — number of waypoints to sample
        segment_range: (min_len, max_len) — range for segment length

    Returns:
        waypoint_indices: sorted list of N indices
        segment_start: start index of the segment
        segment_end: end index of the segment
    """
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
