# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Union

import torch
from tensordict import TensorDict

# TODO: Add the classical estimated discounted reward objective.

# #######################
# OBJECTIVES TO MINIMIZE
# #######################


def cos(a, b):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return (a * b).sum(-1)


class BaseMPCObjective:
    """Base class for MPC objective.
    This is a callable that takes encodings and returns a tensor -
    objective to be optimized.
    """

    def __call__(self, encodings: torch.Tensor, actions: torch.Tensor, keepdims: bool = False) -> torch.Tensor:
        pass


class ReprTargetCosMPCObjective(BaseMPCObjective):
    """Objective to minimize minus the cosine similarity to the target representation."""

    def __init__(
        self,
        cfg: dict,
        target_enc: torch.Tensor,
        sum_all_diffs: bool = False,
        alpha: float = 1.0,  # weight for proprioceptive loss
        **kwargs,
    ):
        self.cfg = cfg
        self.target_enc = target_enc
        self.sum_all_diffs = sum_all_diffs
        self.alpha = alpha

    def __call__(
        self, encodings: Union[torch.Tensor, TensorDict], actions: torch.Tensor, keepdims: bool = False
    ) -> torch.Tensor:
        """
        Args:
            encodings: tensor or TensorDict,
                if tensor: (T x B x ... x D) for visual, (T x B x ... x P) for proprio
                if TensorDict: {'visual': (T x B x ... x D), 'proprio': (T x B x ... x P)}
                in general: D = P and ... = N or ... = V, H, W
            target_enc: tensor or TensorDict,
                if tensor: (1 x ... x D) for visual or (1 x ... x P) for proprio
                if TensorDict: {'visual': (1 x ... x D), 'proprio': (1 x ... x P)}
                in general: D = P and ... = N or ... = V, H, W
            actions: tensor, (T x B x A)
        Returns:
            loss: tensor, (T x B) or (B) if not keepdims
        """
        if isinstance(encodings, TensorDict) and isinstance(self.target_enc, TensorDict):
            sims_visual = cos(
                self.target_enc["visual"].reshape(1, -1),
                encodings["visual"].reshape(encodings["visual"].shape[0], encodings["visual"].shape[1], -1),
            )
            sims_proprio = cos(
                self.target_enc["proprio"].reshape(1, -1),
                encodings["proprio"].reshape(encodings["proprio"].shape[0], encodings["proprio"].shape[1], -1),
            )
            sims = sims_visual + self.alpha * sims_proprio
        elif isinstance(encodings, torch.Tensor) and isinstance(self.target_enc, torch.Tensor):
            sims = cos(
                self.target_enc.reshape(1, -1),
                encodings.reshape(encodings.shape[0], encodings.shape[1], -1),
            )
        else:
            raise ValueError("Input type mismatch")
        if not keepdims:
            if self.sum_all_diffs:
                sims = sims.sum(0)
            else:
                sims = sims[-1]
        elif self.sum_all_diffs:
            sims = sims.cumsum(0).flip(0)
        return -1 * sims


class ReprTargetDistMPCObjective(BaseMPCObjective):
    """Objective to minimize distance to the target representation."""

    def __init__(
        self,
        cfg: dict,
        target_enc: Union[torch.Tensor, TensorDict],
        sum_all_diffs: bool = False,
        alpha: float = 1.0,  # weight for proprioceptive loss
        **kwargs,
    ):
        self.cfg = cfg
        self.target_enc = target_enc
        self.sum_all_diffs = sum_all_diffs
        self.alpha = alpha

    def __call__(
        self, encodings: Union[torch.Tensor, TensorDict], actions: torch.Tensor, keepdims: bool = False
    ) -> torch.Tensor:
        """
        Args:
            encodings: tensor or TensorDict,
                if tensor: (T x B x ... x D) for visual, (T x B x ... x P) for proprio
                if TensorDict: {'visual': (T x B x ... x D), 'proprio': (T x B x ... x P)}
                in general: D = P and ... = N or ... = V, H, W
            target_enc: tensor or TensorDict,
                if tensor: (1 x ... x D) for visual or (1 x ... x P) for proprio
                if TensorDict: {'visual': (1 x ... x D), 'proprio': (1 x ... x P)}
                in general: D = P and ... = N or ... = V, H, W
            actions: tensor, (T x B x A)
        Returns:
            loss: tensor, (T x B) or (B) if not keepdims
        """
        if isinstance(encodings, TensorDict) and isinstance(self.target_enc, TensorDict):
            diff_visual = (
                (self.target_enc["visual"] - encodings["visual"])
                .pow(2)
                .mean(dim=tuple(range(2, encodings["visual"].ndim)))
            )
            diff_proprio = (
                (self.target_enc["proprio"] - encodings["proprio"])
                .pow(2)
                .mean(dim=tuple(range(2, encodings["proprio"].ndim)))
            )
            diff = diff_visual + self.alpha * diff_proprio
        elif isinstance(encodings, torch.Tensor) and isinstance(self.target_enc, torch.Tensor):
            diff = (self.target_enc - encodings).pow(2).mean(dim=tuple(range(2, encodings.ndim)))
        else:
            raise ValueError("Input type mismatch")
        if not keepdims:
            if self.sum_all_diffs:
                diff = diff.sum(0)
            else:
                diff = diff[-1]
        elif self.sum_all_diffs:
            diff = diff.cumsum(0).flip(0)
        return diff


class AbstractL1MPCObjective(BaseMPCObjective):
    """L1 distance between L1-predicted features and an abstract-space target.

    The L1 model produces 256-token features; the L2 abstract space holds
    N_abs tokens after Bipartite Soft Matching. This objective wraps the L1
    rollout with the same merge_fn that produced the abstract target before
    measuring distance.

    Args:
        target_abs: (1, N_abs, D) or TensorDict with that visual + a passthrough
            proprio target.
        merge_fn: per-batch BSM callable that maps (B, V*H*W, D) → (B, N_abs, D).
        sum_all_diffs: aggregate distances across the rollout.
        alpha: weight for the proprio loss term.
    """

    def __init__(
        self,
        cfg: dict,
        target_abs: Union[torch.Tensor, TensorDict],
        merge_fn,
        sum_all_diffs: bool = False,
        alpha: float = 1.0,
        **kwargs,
    ):
        self.cfg = cfg
        self.target_abs = target_abs
        self.merge_fn = merge_fn
        self.sum_all_diffs = sum_all_diffs
        self.alpha = alpha

    def _merge_visual(self, visual: torch.Tensor) -> torch.Tensor:
        """Apply merge_fn to a (T, B, V, H, W, D) L1 rollout.

        Returns (T, B, N_abs, D).
        """
        T, B = visual.shape[:2]
        V, H, W, D = visual.shape[2:]
        flat = visual.reshape(T * B, V * H * W, D)
        sizes = torch.ones(T * B, V * H * W, 1, device=visual.device, dtype=visual.dtype)
        merged_sizes = self.merge_fn(sizes)
        merged = self.merge_fn(flat * sizes) / merged_sizes.clamp(min=1.0)
        return merged.reshape(T, B, merged.size(1), D)

    def __call__(
        self,
        encodings: Union[torch.Tensor, TensorDict],
        actions: torch.Tensor,
        keepdims: bool = False,
    ) -> torch.Tensor:
        if isinstance(encodings, TensorDict) and isinstance(self.target_abs, TensorDict):
            visual_abs = self._merge_visual(encodings["visual"])
            tgt_v = self.target_abs["visual"]
            diff_visual = torch.abs(tgt_v - visual_abs).mean(
                dim=tuple(range(2, visual_abs.ndim))
            )
            diff_proprio = torch.abs(self.target_abs["proprio"] - encodings["proprio"]).mean(
                dim=tuple(range(2, encodings["proprio"].ndim))
            )
            diff = diff_visual + self.alpha * diff_proprio
        elif isinstance(encodings, torch.Tensor) and isinstance(self.target_abs, torch.Tensor):
            visual_abs = self._merge_visual(encodings)
            diff = torch.abs(self.target_abs - visual_abs).mean(
                dim=tuple(range(2, visual_abs.ndim))
            )
        else:
            raise ValueError("AbstractL1MPCObjective: input/target type mismatch")
        if not keepdims:
            if self.sum_all_diffs:
                diff = diff.sum(0)
            else:
                diff = diff[-1]
        elif self.sum_all_diffs:
            diff = diff.cumsum(0).flip(0)
        return diff


class ReprTargetDistL1MPCObjective(BaseMPCObjective):
    """Objective to minimize L1 distance to the target representation."""

    def __init__(
        self,
        cfg: dict,
        target_enc: Union[torch.Tensor, TensorDict],
        sum_all_diffs: bool = False,
        alpha: float = 1.0,  # weight for proprioceptive loss
        **kwargs,
    ):
        self.cfg = cfg
        self.target_enc = target_enc
        self.sum_all_diffs = sum_all_diffs
        self.alpha = alpha

    def __call__(
        self, encodings: Union[torch.Tensor, TensorDict], actions: torch.Tensor, keepdims: bool = False
    ) -> torch.Tensor:
        """
        Args:
            encodings: tensor or TensorDict,
                if tensor: (T x B x ... x D) for visual, (T x B x ... x P) for proprio
                if TensorDict: {'visual': (T x B x ... x D), 'proprio': (T x B x ... x P)}
                in general: D = P and ... = N or ... = V, H, W
            target_enc: tensor or TensorDict,
                if tensor: (1 x ... x D) for visual or (1 x ... x P) for proprio
                if TensorDict: {'visual': (1 x ... x D), 'proprio': (1 x ... x P)}
                in general: D = P and ... = N or ... = V, H, W
            actions: tensor, (T x B x A)
        Returns:
            loss: tensor, (T x B) or (B) if not keepdims
        """
        if isinstance(encodings, TensorDict) and isinstance(self.target_enc, TensorDict):
            diff_visual = torch.abs(self.target_enc["visual"] - encodings["visual"]).mean(
                dim=tuple(range(2, encodings["visual"].ndim))
            )
            diff_proprio = torch.abs(self.target_enc["proprio"] - encodings["proprio"]).mean(
                dim=tuple(range(2, encodings["proprio"].ndim))
            )
            diff = diff_visual + self.alpha * diff_proprio
        elif isinstance(encodings, torch.Tensor) and isinstance(self.target_enc, torch.Tensor):
            diff = torch.abs(self.target_enc - encodings).mean(dim=tuple(range(2, encodings.ndim)))
        else:
            raise ValueError("Input type mismatch")
        if not keepdims:
            if self.sum_all_diffs:
                diff = diff.sum(0)
            else:
                diff = diff[-1]
        elif self.sum_all_diffs:
            diff = diff.cumsum(0).flip(0)
        return diff
