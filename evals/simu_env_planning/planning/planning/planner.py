# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
from typing import Callable, List, NamedTuple

import nevergrad as ng
import numpy as np
import torch
import torch.distributed as dist

from evals.simu_env_planning.planning.planning import objectives
from src.utils.logging import get_logger

logger = get_logger(__name__)

########### PLANNERS IN LATENT SPACE ###############


class PlanningResult(NamedTuple):
    actions: torch.Tensor
    # locations that the model has planned to achieve
    losses: torch.Tensor = None
    prev_elite_losses_mean: torch.Tensor = None
    prev_elite_losses_std: torch.Tensor = None
    info: dict = None
    plan_metrics: dict = None
    pred_frames_over_iterations: List = None
    predicted_best_encs_over_iterations: List = None


class Planner(ABC):
    def __init__(self, unroll: Callable):
        self.objective = None
        self.unroll = unroll

    def set_objective(self, objective: objectives.BaseMPCObjective):
        self.objective = objective

    @abstractmethod
    def plan(self, obs: torch.Tensor, steps_left: int):
        pass

    def cost_function(self, actions: torch.Tensor, z_init: torch.Tensor) -> torch.Tensor:
        predicted_encs = self.unroll(z_init, actions)
        return self.objective(predicted_encs, actions)


class NevergradPlanner(Planner):
    def __init__(
        self,
        unroll: Callable,
        action_dim: int,
        iterations: int,
        var_scale: float = 1,
        max_norms: List[float] = None,
        max_norm_dims: List[List[int]] = [[0, 1, 2], [6]],
        num_samples: int = 1,
        horizon: int = None,
        num_act_stepped: int = None,
        decode_each_iteration: bool = False,
        decode_unroll: Callable = None,
        num_elites: int = 10,
        optimizer_name: str = "NgIohTuned",
        **kwargs,
    ):
        super().__init__(unroll)
        self.action_dim = action_dim
        self.iterations = iterations
        self.var_scale = var_scale
        self.max_norms = max_norms
        self.max_norm_dims = max_norm_dims
        self.num_samples = num_samples
        self.horizon = horizon
        self.num_act_stepped = num_act_stepped
        self.decode_each_iteration = decode_each_iteration
        self.decode_unroll = decode_unroll
        self.num_elites = num_elites  # just for logging
        self.optimizer_name = optimizer_name
        self.optimizer_map = {
            "NgIohTuned": ng.optimizers.NgIohTuned,
            "NGOpt": ng.optimizers.NGOpt,
            # CMA-ES variants - numerically stable, good for continuous optimization
            "CMA": ng.optimizers.CMA,
            "ParametrizedCMA": ng.optimizers.ParametrizedCMA,
            "DiagonalCMA": ng.optimizers.DiagonalCMA,
            # Other stable alternatives
            "PSO": ng.optimizers.PSO,
            "DE": ng.optimizers.DE,
            "OnePlusOne": ng.optimizers.OnePlusOne,
            "TwoPointsDE": ng.optimizers.TwoPointsDE,
        }

    def build_optimizer(self, optimizer_name, **kwargs):
        """Build an optimizer by name."""
        if optimizer_name in self.optimizer_map:
            return self.optimizer_map[optimizer_name](**kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _get_optimizer(self, plan_length: int):
        parametrization = ng.p.Array(shape=(self.horizon, self.action_dim))
        if self.max_norms is not None:
            lower_bounds = -np.ones((plan_length, self.action_dim))
            upper_bounds = np.ones((plan_length, self.action_dim))

            for max_norm_group, dims in zip(self.max_norms, self.max_norm_dims):
                for d in dims:
                    lower_bounds[:, d] = -max_norm_group
                    upper_bounds[:, d] = max_norm_group

            parametrization.set_bounds(lower=lower_bounds, upper=upper_bounds)
        optimizer = self.build_optimizer(
            self.optimizer_name,
            parametrization=parametrization,
            budget=self.iterations * self.num_samples,
            num_workers=self.num_samples,
        )
        logger.info(f"⚙️  Optimizer: {optimizer}")
        logger.info(f"   Optimizer info: {optimizer._info()}")

        # Check if NGOpt selected MetaModel - it causes numerical instability
        # due to polynomial regression overflow when loss variance is low.
        # In this case, replace with DiagonalCMA which is what NGOpt typically
        # selects in other configurations and is more numerically stable.
        if hasattr(optimizer, "optim") and optimizer.optim.name == "MetaModel":
            logger.warning(
                "NGOpt selected MetaModel optimizer which can cause numerical instability. "
                "Switching to DiagonalCMA for better numerical stability."
            )
            optimizer = self.build_optimizer(
                "DiagonalCMA",
                parametrization=parametrization,
                budget=self.iterations * self.num_samples,
                num_workers=self.num_samples,
            )
            logger.info(f"⚙️  Replacement optimizer: {optimizer}")

        if hasattr(optimizer, "optim"):
            if optimizer.optim.name in ["MetaModel", "CMApara"]:
                if hasattr(optimizer.optim, "_optim"):
                    if hasattr(optimizer.optim._optim, "_es") and optimizer.optim._optim._es is not None:
                        logger.info(f"{optimizer.optim._optim._es.inopts=}")
                    else:
                        logger.info("No _es in optimizer")
        return optimizer

    @torch.no_grad()
    def plan(
        self,
        z_init: torch.Tensor,
        steps_left: int = None,
    ) -> PlanningResult:
        if steps_left is not None:
            plan_length = min(self.horizon, steps_left)
        else:
            plan_length = self.horizon
        optimizer = self._get_optimizer(plan_length)
        costs = []
        prev_elite_losses_mean = []
        prev_elite_losses_std = []
        pred_frames_over_iterations = []
        predicted_best_encs_over_iterations = []

        for itr in range(self.iterations):
            candidates = [optimizer.ask() for _ in range(self.num_samples)]
            candidate_values = torch.from_numpy(np.array([c.value for c in candidates])).to(
                device=z_init.device, dtype=torch.float32
            )
            loss = self.cost_function(candidate_values.permute(1, 0, 2), z_init)

            # Check for NaN or Inf values in loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logger.warning(f"NaN or Inf detected in loss at iteration {itr}. Replacing with large values.")
                loss = torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=-1e6)

            # for logging
            elite_losses = torch.topk(loss, k=self.num_elites, largest=False).values
            prev_elite_losses_mean.append(elite_losses.mean().item())
            prev_elite_losses_std.append(elite_losses.std().item())

            for i, c in enumerate(candidates):
                optimizer.tell(c, loss[i].item())
            costs.append(loss.min().item())

            best_solution = optimizer.provide_recommendation().value
            actions = torch.tensor(best_solution, device=z_init.device, dtype=torch.float32).unsqueeze(1)
            predicted_best_encs = self.unroll(z_init, act_suffix=actions)
            predicted_best_encs_over_iterations.append(predicted_best_encs)
            if self.decode_each_iteration and self.decode_unroll is not None:
                pred_frames = self.decode_unroll(predicted_best_encs)
                pred_frames_over_iterations.append(pred_frames)

        best_solution = optimizer.provide_recommendation().value
        actions = torch.tensor(best_solution, device=z_init.device)
        result = PlanningResult(
            actions=actions[: self.num_act_stepped],
            losses=torch.tensor(costs).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(prev_elite_losses_mean).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(prev_elite_losses_std).unsqueeze(-1),
            pred_frames_over_iterations=pred_frames_over_iterations if self.decode_each_iteration else None,
            predicted_best_encs_over_iterations=predicted_best_encs_over_iterations,
        )
        return result


class CEMPlanner(Planner):
    def __init__(
        self,
        unroll: Callable,
        iterations: int = 6,
        num_samples: int = 512,
        horizon: int = 32,
        action_dim: int = 4,
        var_scale: float = 1,
        num_elites: int = 64,
        momentum_mean: float = 0.0,
        momentum_std: float = 0.0,
        max_norms: List[float] = None,
        max_norm_dims: List[List[int]] = [[0, 1, 2], [6]],
        distribute_planner: bool = False,
        local_generator: torch.Generator = None,
        num_act_stepped: int = None,
        decode_each_iteration: bool = False,
        decode_unroll: Callable = None,
        **kwargs,
    ):
        super().__init__(unroll)
        self.iterations = iterations
        self.num_samples = num_samples
        self.horizon = horizon
        self.action_dim = action_dim
        self.device = torch.device("cuda")
        self.var_scale = var_scale
        self.num_elites = num_elites
        self.momentum_mean = momentum_mean
        self.momentum_std = momentum_std
        self.max_norms = max_norms
        self.max_norm_dims = max_norm_dims
        self._prev_mean = None
        self.distribute_planner = distribute_planner
        self.local_generator = local_generator
        self.num_act_stepped = num_act_stepped
        self.decode_each_iteration = decode_each_iteration
        self.decode_unroll = decode_unroll

    @torch.no_grad()
    def plan(
        self,
        z_init,
        steps_left=None,
    ):
        """
        Same as MPPIPlanner but without a policy network.
        Plan a sequence of actions using the learned world model.
        This planner assumes independence between temporal dimensions: we sample actions according
        to a diagonal Gaussian

        Args:
                z_init (torch.Tensor): Latent state from which to plan.
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.
                task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        if steps_left is None:
            plan_length = self.horizon
        else:
            plan_length = min(self.horizon, steps_left)
        mean = torch.zeros(plan_length, self.action_dim, device=self.device)
        std = self.var_scale * torch.ones(plan_length, self.action_dim, device=self.device)
        actions = torch.empty(
            plan_length,
            self.num_samples,
            self.action_dim,
            device=self.device,
        )
        losses, elite_means, elite_stds = [], [], []
        predicted_best_encs_over_iterations = []
        if self.decode_each_iteration:
            pred_frames_over_iterations = []
        # Iterate CEM
        for itr in range(self.iterations):
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                plan_length, self.num_samples, self.action_dim, device=std.device, generator=self.local_generator
            )
            # Mean sample inclusion trick to never loose best previous action
            actions[:, 0, :] = mean
            # Apply clipping if max_norms is specified
            if self.max_norms is not None:
                for h in range(plan_length):
                    # Loop through each group of dimensions to clip
                    for i, (dims, maxnorm) in enumerate(zip(self.max_norm_dims, self.max_norms)):
                        # Clip the specified dimensions to [-maxnorm, maxnorm]
                        actions[h, :, dims] = torch.clip(actions[h, :, dims], min=-maxnorm, max=maxnorm)
            # Compute elite actions
            cost = self.cost_function(actions, z_init).unsqueeze(1)
            losses.append(cost.min().item())
            # Gather all values
            if self.distribute_planner:
                cost = torch.cat(FullGatherLayer.apply(cost), dim=0)
                all_actions = torch.cat(FullGatherLayer.apply(actions), dim=1)
            else:
                all_actions = actions
            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss, elite_actions = cost[elite_idxs], all_actions[:, elite_idxs]  # [EL,1] , [H,EL,A]
            # Log the mean and std of the elite values
            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())
            # Update parameters with momentum
            new_mean = torch.mean(elite_actions, dim=1)
            new_std = torch.std(elite_actions, dim=1)
            # Apply momentum to mean and std updates
            mean = new_mean * (1 - self.momentum_mean) + mean * self.momentum_mean
            std = new_std * (1 - self.momentum_std) + std * self.momentum_std
            # Decoding logic
            predicted_best_encs = self.unroll(z_init, act_suffix=mean.unsqueeze(1))
            predicted_best_encs_over_iterations.append(predicted_best_encs)
            if self.decode_each_iteration and self.decode_unroll is not None:
                pred_frames = self.decode_unroll(
                    predicted_best_encs,
                )
                pred_frames_over_iterations.append(pred_frames)
                # [T H W 3]: uint 8 in [0, 255]

        self._prev_mean = mean
        a = mean[: self.num_act_stepped]
        if self.distribute_planner:
            dist.broadcast(a, src=0)
        result = PlanningResult(
            actions=a,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
            pred_frames_over_iterations=pred_frames_over_iterations if self.decode_each_iteration else None,
            predicted_best_encs_over_iterations=predicted_best_encs_over_iterations,
        )
        return result

class GRASPlanner(Planner):
    """GRASP: Gradient RelAxed Stochastic Planner.

    Implements the planner from "Parallel Stochastic Gradient-Based Planning for
    World Models" (Psenka et al., 2025). Core components:

    1. **Parallelized planning with virtual states**: Introduces auxiliary "virtual
       states" z_1,...,z_T as optimization variables alongside actions a_0,...,a_{T-1}.
       All one-step world model evaluations F(z_t, a_t) are computed in parallel.

    2. **Langevin state noise**: Injects isotropic Gaussian noise into state iterates
       each optimization step, allowing escape from local minima (Eq. 5 in paper).

    3. **Grad-cut dynamics loss with dense goal shaping**: Detaches gradients through
       state inputs of the world model (stop-gradient on z_t when computing F(z_t, a_t))
       to avoid adversarial exploitation of brittle state Jacobians. Adds a dense goal
       loss on one-step predictions to provide task-aligned signal at every timestep
       (Eqs. 8-10 in paper).

    4. **Full-rollout synchronization**: Periodically runs standard gradient descent
       on the serial rollout objective for refinement (Eqs. 13-15 in paper).

    Unlike other planners that only need the `unroll` callable, GRASP requires direct
    access to the EncPredWM model for parallel one-step predictions via `forward_pred`
    and action encoding via `encode_act`.
    """

    def __init__(
        self,
        unroll: Callable,
        action_dim: int,
        horizon: int = 32,
        # GRASP optimization parameters
        steps: int = 100,
        action_lr: float = 0.01,
        state_lr: float = 0.01,
        state_noise_std: float = 0.01,
        gamma: float = 1.0,
        # Full-rollout sync parameters
        sync_every: int = 20,
        sync_steps: int = 5,
        sync_lr: float = 0.01,
        # Initialization
        var_scale: float = 1.0,
        action_init: str = "zero",
        state_init: str = "rollout",
        # Action clipping
        max_norms: List[float] = None,
        max_norm_dims: List[List[int]] = [[0, 1, 2], [6]],
        # Output
        num_act_stepped: int = None,
        decode_each_iteration: bool = False,
        decode_unroll: Callable = None,
        # World model access
        enc_pred_wm=None,
        **kwargs,
    ):
        """
        Args:
            unroll: Serial rollout function (from EncPredWM).
            action_dim: Dimension of the action space.
            horizon: Planning horizon T (number of action steps).
            steps: Total number of GRASP optimization iterations.
            action_lr: Learning rate for action gradient updates.
            state_lr: Learning rate for state gradient updates.
            state_noise_std: Std of Langevin noise injected into virtual states (σ).
            gamma: Weight for dense goal shaping loss relative to dynamics loss.
            sync_every: Run full-rollout sync every K_sync iterations.
            sync_steps: Number of GD steps per sync phase (J_sync).
            sync_lr: Learning rate for sync-phase gradient descent.
            var_scale: Scale for random initialization of actions.
            action_init: How to initialize actions ("zero" or "randn").
            state_init: How to initialize virtual states ("zero", "randn", or "rollout").
            max_norms: List of max norm values for action clipping per dim group.
            max_norm_dims: List of dimension groups to clip.
            num_act_stepped: Number of actions to return (execute).
            decode_each_iteration: Whether to decode predictions at each iteration.
            decode_unroll: Function to decode latent predictions to frames.
            enc_pred_wm: The full EncPredWM model for direct access to forward_pred.
        """
        super().__init__(unroll)
        self.action_dim = action_dim
        self.horizon = horizon
        self.device = torch.device("cuda")
        # GRASP-specific
        self.steps = steps
        self.action_lr = action_lr
        self.state_lr = state_lr
        self.state_noise_std = state_noise_std
        self.gamma = gamma
        # Sync
        self.sync_every = sync_every
        self.sync_steps = sync_steps
        self.sync_lr = sync_lr
        # Init
        self.var_scale = var_scale
        self.action_init = action_init
        self.state_init = state_init
        # Clipping
        self.max_norms = max_norms
        self.max_norm_dims = max_norm_dims
        # Output
        self.num_act_stepped = num_act_stepped
        self.decode_each_iteration = decode_each_iteration
        self.decode_unroll = decode_unroll
        # World model
        self.enc_pred_wm = enc_pred_wm
        # Goal encoding (set via set_goal_enc before planning)
        self.goal_enc = None

    def set_goal_enc(self, goal_enc):
        """Store the goal encoding for dense goal shaping loss.

        Args:
            goal_enc: Goal state encoding, same type/shape as z_ctxt from EncPredWM.encode().
                For visual-only: Tensor [1, tau, V, H, W, D] (tau=1 typically).
                For multimodal: TensorDict with "visual" and "proprio" keys.
        """
        self.goal_enc = goal_enc

    def _clip_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply per-dimension-group clipping to actions.

        Args:
            actions: (T, A) or (1, T, A) action tensor.

        Returns:
            Clipped actions of the same shape.
        """
        if self.max_norms is not None:
            for dims, maxnorm in zip(self.max_norm_dims, self.max_norms):
                actions[..., dims] = torch.clip(actions[..., dims], min=-maxnorm, max=maxnorm)
        return actions

    def _one_step_predict(self, state_t: torch.Tensor, action_raw_t: torch.Tensor):
        """Perform a single one-step prediction through the world model.

        Encodes the action, then calls forward_pred with ctxt_window=1 (single frame).

        Args:
            state_t: Virtual state at time t. Shape (B, 1, V, H, W, D).
            action_raw_t: Raw action at time t. Shape (B, 1, A).

        Returns:
            pred_vid: Predicted next visual state (B, 1, V, H, W, D).
            pred_prop: Predicted next proprio state or None.
        """
        wm = self.enc_pred_wm.model
        # Encode the raw action
        act_feats = wm.encode_act(action_raw_t)  # (B, 1, ...) encoded action features
        # forward_pred expects: video (B, tau, V, H, W, D), action (B, T, ...), proprio (B, T, ...)
        pred_vid, _, pred_prop = wm.forward_pred(
            state_t,
            act_feats,
            None,  # No proprio in virtual state optimization
        )
        return pred_vid, pred_prop

    def _compute_grasp_loss(
        self,
        virtual_states: torch.Tensor,
        actions: torch.Tensor,
        z_init_vid: torch.Tensor,
        goal_vid: torch.Tensor,
        plan_length: int,
    ):
        """Compute the GRASP loss: grad-cut dynamics consistency + dense goal shaping.

        Implements Eq. 10 from the paper:
            L(s, a) = Σ_t ‖z_{t+1} − F(z̄_t, a_t)‖² + γ · ‖F(z̄_t, a_t) − g‖²

        The stop-gradient on z_t is achieved by using z_t.detach() as input to F.

        Args:
            virtual_states: (plan_length, 1, V, H, W, D) — z_1 to z_T.
            actions: (1, plan_length, A) — a_0 to a_{T-1}.
            z_init_vid: (1, 1, V, H, W, D) — initial state z_0 (from encoder).
            goal_vid: (1, 1, V, H, W, D) — goal state g.
            plan_length: T, the planning horizon.

        Returns:
            total_loss: Scalar loss for gradient computation.
        """
        # Build the full state sequence: z_0 (fixed), z_1, ..., z_T (optimized)
        # z_0 is from the encoder, z_1..z_T are virtual states
        # For the dynamics loss: F(z̄_t, a_t) should match z_{t+1}
        #   t=0: F(z̄_0, a_0) → z_1
        #   t=1: F(z̄_1, a_1) → z_2
        #   ...
        #   t=T-1: F(z̄_{T-1}, a_{T-1}) → (checked against goal for last step)

        dynamics_loss = 0.0
        goal_loss = 0.0

        for t in range(plan_length):
            # Get z_t (detached for stop-gradient)
            if t == 0:
                z_t = z_init_vid.detach()  # z_0 is always fixed and detached
            else:
                z_t = virtual_states[t - 1: t].detach()  # z̄_t: stop gradient

            # Get target next state z_{t+1}
            if t < plan_length - 1:
                z_next = virtual_states[t: t + 1]  # z_{t+1} (gets gradients!)
            else:
                z_next = None  # Last step: no dynamics target (or use goal)

            # One-step prediction: F(z̄_t, a_t)
            # z_t shape: (1, 1, V, H, W, D), action: (1, 1, A)
            a_t = actions[:, t: t + 1, :]  # (1, 1, A)
            pred_vid, _ = self._one_step_predict(z_t, a_t)
            # pred_vid shape: (1, 1, V, H, W, D)

            # Dynamics consistency loss (Eq. 8): ‖z_{t+1} − F(z̄_t, a_t)‖²
            if z_next is not None:
                # z_next has shape (1, 1, V, H, W, D) — need to match pred_vid
                dyn_diff = (z_next - pred_vid).pow(2).mean()
                dynamics_loss = dynamics_loss + dyn_diff

            # Dense goal shaping (Eq. 9): ‖F(z̄_t, a_t) − g‖²
            goal_diff = (pred_vid - goal_vid).pow(2).mean()
            goal_loss = goal_loss + goal_diff

        total_loss = dynamics_loss + self.gamma * goal_loss
        return total_loss

    def _sync_full_rollout(
        self,
        actions: torch.Tensor,
        z_init,
        plan_length: int,
    ):
        """Full-rollout synchronization phase (Section 3.4).

        Runs J_sync steps of standard gradient descent on the serial rollout
        objective, updating actions via full backpropagation through the T-step
        rollout.

        Args:
            actions: (1, plan_length, A) current action sequence (modified in-place).
            z_init: Initial latent state for unroll.
            plan_length: Planning horizon T.

        Returns:
            actions: Updated actions tensor (still requires_grad).
        """
        for _ in range(self.sync_steps):
            if actions.grad is not None:
                actions.grad.zero_()

            # Serial rollout: unroll expects actions as (T, B, A)
            actions_for_unroll = actions.squeeze(0).unsqueeze(1)  # (T, 1, A)
            predicted_encs = self.unroll(z_init, act_suffix=actions_for_unroll)
            # Compute objective using the existing planning objective
            loss = self.objective(predicted_encs, actions_for_unroll)
            sync_loss = loss.mean()
            sync_loss.backward()

            with torch.no_grad():
                actions.data -= self.sync_lr * actions.grad
                actions.data = self._clip_actions(actions.data)
            actions.grad.zero_()

        return actions

    def plan(
        self,
        z_init,
        steps_left=None,
    ) -> PlanningResult:
        """Plan using the GRASP algorithm.

        Optimizes virtual states and actions jointly using:
        1. Parallel one-step predictions with grad-cut dynamics loss
        2. Dense goal shaping on every one-step prediction
        3. Langevin noise on state iterates for exploration
        4. Periodic full-rollout synchronization for refinement

        Args:
            z_init: Initial latent state from EncPredWM.encode().
                Tensor (1, tau, V, H, W, D) or TensorDict with "visual"/"proprio".
            steps_left: Optional number of steps left in episode.

        Returns:
            PlanningResult with optimized actions and planning metrics.
        """
        if steps_left is not None:
            plan_length = min(self.horizon, steps_left)
        else:
            plan_length = self.horizon

        # Extract visual features from z_init
        if isinstance(z_init, dict) or hasattr(z_init, 'keys'):
            z_init_vid = z_init["visual"]  # (1, tau, V, H, W, D)
        else:
            z_init_vid = z_init  # (1, tau, V, H, W, D)

        # Use only the last frame as the initial state for planning
        z_init_vid_last = z_init_vid[:, -1:, ...]  # (1, 1, V, H, W, D)

        # Extract goal visual encoding
        assert self.goal_enc is not None, "Goal encoding must be set via set_goal_enc() before planning."
        if isinstance(self.goal_enc, dict) or hasattr(self.goal_enc, 'keys'):
            goal_vid = self.goal_enc["visual"]  # (1, tau, V, H, W, D)
        else:
            goal_vid = self.goal_enc
        goal_vid = goal_vid[:, -1:, ...]  # (1, 1, V, H, W, D)

        # Get state shape from z_init
        _, _, V, H, W, D = z_init_vid_last.shape

        # --- Initialize actions ---
        if self.action_init == "zero":
            actions = torch.zeros(1, plan_length, self.action_dim, device=self.device)
        else:  # "randn"
            actions = torch.randn(1, plan_length, self.action_dim, device=self.device) * self.var_scale
        actions = self._clip_actions(actions)
        actions = actions.detach().requires_grad_(True)

        # --- Initialize virtual states z_1, ..., z_T ---
        if self.state_init == "rollout" and plan_length > 0:
            # Initialize by doing a serial rollout with initial actions
            with torch.no_grad():
                init_acts_for_unroll = actions.squeeze(0).unsqueeze(1)  # (T, 1, A)
                init_rollout = self.unroll(z_init, act_suffix=init_acts_for_unroll)
                if isinstance(init_rollout, dict) or hasattr(init_rollout, 'keys'):
                    init_vid = init_rollout["visual"]  # (T+tau, 1, V, H, W, D)
                else:
                    init_vid = init_rollout
                # Take the predicted states (skip context frames)
                tau = z_init_vid.shape[1]
                virtual_states = init_vid[tau:, ...].clone()  # (T, 1, V, H, W, D)
        elif self.state_init == "zero":
            virtual_states = torch.zeros(plan_length, 1, V, H, W, D, device=self.device)
        else:  # "randn"
            virtual_states = torch.randn(plan_length, 1, V, H, W, D, device=self.device) * self.var_scale
        virtual_states = virtual_states.detach().requires_grad_(True)

        # --- Tracking ---
        losses = []
        predicted_best_encs_over_iterations = []
        pred_frames_over_iterations = [] if self.decode_each_iteration else None

        # --- Main GRASP optimization loop ---
        self.enc_pred_wm.model.eval()

        for itr in range(self.steps):
            # Zero any existing gradients
            if actions.grad is not None:
                actions.grad.zero_()
            if virtual_states.grad is not None:
                virtual_states.grad.zero_()

            # Compute GRASP loss (grad-cut dynamics + dense goal shaping)
            total_loss = self._compute_grasp_loss(
                virtual_states, actions, z_init_vid_last, goal_vid, plan_length,
            )
            total_loss.backward()
            losses.append(total_loss.item())

            # --- Update actions via gradient descent ---
            with torch.no_grad():
                actions.data -= self.action_lr * actions.grad
                actions.data = self._clip_actions(actions.data)

            # --- Update virtual states via gradient descent + Langevin noise ---
            with torch.no_grad():
                if virtual_states.grad is not None:
                    virtual_states.data -= self.state_lr * virtual_states.grad
                # Langevin noise injection (Eq. 5)
                if self.state_noise_std > 0:
                    noise = torch.randn_like(virtual_states) * self.state_noise_std
                    virtual_states.data += noise

            # Zero gradients after update
            if actions.grad is not None:
                actions.grad.zero_()
            if virtual_states.grad is not None:
                virtual_states.grad.zero_()

            # --- Full-rollout synchronization (Section 3.4) ---
            if self.sync_every > 0 and (itr + 1) % self.sync_every == 0:
                actions = self._sync_full_rollout(actions, z_init, plan_length)

            # --- Record predictions for logging (periodic, not every iteration) ---
            if itr == self.steps - 1 or (self.decode_each_iteration and itr % max(1, self.steps // 10) == 0):
                with torch.no_grad():
                    acts_for_unroll = actions.squeeze(0).unsqueeze(1)  # (T, 1, A)
                    predicted_best_encs = self.unroll(z_init, act_suffix=acts_for_unroll)
                    predicted_best_encs_over_iterations.append(predicted_best_encs)
                    if self.decode_each_iteration and self.decode_unroll is not None:
                        pred_frames = self.decode_unroll(predicted_best_encs)
                        pred_frames_over_iterations.append(pred_frames)

        # --- Final actions ---
        final_actions = actions.squeeze(0).detach()  # (T, A)
        a = final_actions[: self.num_act_stepped] if self.num_act_stepped else final_actions
        losses_tensor = torch.tensor(losses).detach().unsqueeze(-1)

        result = PlanningResult(
            actions=a,
            losses=losses_tensor,
            prev_elite_losses_mean=losses_tensor,  # No elites in GRASP; use raw losses
            prev_elite_losses_std=torch.zeros_like(losses_tensor),
            pred_frames_over_iterations=pred_frames_over_iterations if self.decode_each_iteration else None,
            predicted_best_encs_over_iterations=predicted_best_encs_over_iterations,
        )
        return result


class MPPIPlanner(Planner):
    def __init__(
        self,
        unroll: Callable,
        iterations: int = 6,
        num_samples: int = 512,
        horizon: int = 32,
        action_dim: int = 4,
        max_std: float = 2,
        min_std: float = 0.05,
        num_elites: int = 64,
        temperature: float = 0.5,
        distribute_planner: bool = False,
        local_generator: torch.Generator = None,
        num_act_stepped: int = None,
        decode_each_iteration: bool = False,
        decode_unroll: Callable = None,
        **kwargs,
    ):
        super().__init__(unroll)
        self.iterations = iterations
        self.num_samples = num_samples
        self.horizon = horizon
        self.action_dim = action_dim
        self.device = torch.device("cuda")
        self.max_std = max_std
        self.min_std = min_std
        self.num_elites = num_elites
        self.temperature = temperature
        self._prev_mean = None
        self.distribute_planner = distribute_planner
        self.local_generator = local_generator
        self.num_act_stepped = num_act_stepped
        self.decode_each_iteration = decode_each_iteration
        self.decode_unroll = decode_unroll

    @torch.no_grad()
    def plan(self, z_init, eval_mode=False, task=None, steps_left=None):
        """
        MPPIPlanner without a policy network.
        Plan a sequence of actions using the learned world model.

        Args:
                z_init (torch.Tensor): Latent state from which to plan.
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.
                task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        if steps_left is None:
            plan_length = self.horizon
        else:
            plan_length = min(self.horizon, steps_left)

        # Initialize state and parameters
        mean = torch.zeros(plan_length, self.action_dim, device=self.device)
        std = self.max_std * torch.ones(plan_length, self.action_dim, device=self.device)
        actions = torch.empty(
            plan_length,
            self.num_samples,
            self.action_dim,
            device=self.device,
        )

        losses, elite_means, elite_stds = [], [], []
        predicted_best_encs_over_iterations = []
        if self.decode_each_iteration:
            pred_frames_over_iterations = []
        # Iterate MPPI
        for _ in range(self.iterations):
            # Sample actions
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                plan_length,
                self.num_samples,
                self.action_dim,
                device=std.device,
                generator=self.local_generator,
            )
            # Compute costs
            cost = self.cost_function(actions, z_init).unsqueeze(1)
            losses.append(cost.min().item())
            # Get elite actions
            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss, elite_actions = cost[elite_idxs], actions[:, elite_idxs]
            # Record statistics
            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())
            # Update parameters
            min_cost = cost.min(0)[0]
            score = torch.exp(self.temperature * (min_cost - elite_loss[:, 0]))  # increasing with elite_value
            score /= score.sum(0)
            mean = torch.sum(score.unsqueeze(0).unsqueeze(2) * elite_actions, dim=1) / (score.sum(0) + 1e-9)  # T B A
            std = torch.sqrt(
                torch.sum(
                    score.unsqueeze(0).unsqueeze(2) * (elite_actions - mean.unsqueeze(1)) ** 2,
                    dim=1,  # T B A
                )
                / (score.sum(0) + 1e-9)
            )
            # Decoding logic
            predicted_best_encs = self.unroll(z_init, act_suffix=mean.unsqueeze(1))
            predicted_best_encs_over_iterations.append(predicted_best_encs)
            if self.decode_each_iteration and self.decode_unroll is not None:
                pred_frames = self.decode_unroll(
                    predicted_best_encs,
                )
                pred_frames_over_iterations.append(pred_frames)
                # [T H W 3]: uint 8 in [0, 255]
        # Select action
        score = score.cpu().numpy()  # [EL,]
        # actions: [H, A]
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]  # [H,A]
        self._prev_mean = mean
        a, std = actions[: self.num_act_stepped], std[: self.num_act_stepped]  # [N, A], [N, A]
        if not eval_mode:
            a += std * torch.randn(self.action_dim, device=std.device, generator=self.local_generator)
        # to make sure each GPU outputs same action
        if self.distribute_planner:
            dist.broadcast(a, src=0)

        result = PlanningResult(
            actions=a,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
            pred_frames_over_iterations=pred_frames_over_iterations if self.decode_each_iteration else None,
            predicted_best_encs_over_iterations=predicted_best_encs_over_iterations,
        )
        return result


class GradientDescentPlanner(Planner):
    def __init__(
        self,
        unroll: Callable,
        action_dim: int,
        horizon: int,
        iterations: int = 500,
        lr: float = 1,
        action_noise: float = 0.003,
        sample_type: str = "randn",
        var_scale: float = 1,
        max_norms: List[float] = None,
        max_norm_dims: List[List[int]] = [[0, 1, 2], [6]],
        num_act_stepped: int = None,
        decode_each_iteration: bool = False,
        decode_unroll: Callable = None,
        optimizer_type: str = "sgd",
        adam_betas: tuple = (0.9, 0.995),
        adam_eps: float = 1e-8,
        **kwargs,
    ):
        """
        Gradient Descent Planner for action optimization in latent space.

        Args:
            unroll: Function to unroll the world model
            action_dim: Dimension of the action space
            horizon: Planning horizon (number of timesteps)
            iterations: Number of optimization iterations
            lr: Learning rate for gradient descent
            action_noise: Standard deviation of Gaussian noise to add after each gradient step
            sample_type: Type of action initialization ("randn" or "zero")
            max_norms: List of maximum norm values for each group of dimensions (None to disable clipping)
            max_norm_dims: List of dimension groups to clip (e.g., [[0, 1, 2], [6]])
            num_act_stepped: Number of actions to execute (default: all)
            decode_each_iteration: Whether to decode predictions at each iteration
            decode_unroll: Function to decode latent predictions to frames
            optimizer_type: Type of optimizer to use ("sgd" or "adam")
            adam_betas: Betas for Adam optimizer (default: (0.9, 0.995))
            adam_eps: Epsilon for Adam optimizer (default: 1e-8)
        """
        super().__init__(unroll)
        self.action_dim = action_dim
        self.horizon = horizon
        self.iterations = iterations
        self.lr = lr
        self.action_noise = action_noise
        self.var_scale = var_scale
        self.sample_type = sample_type
        self.max_norms = max_norms
        self.max_norm_dims = max_norm_dims
        self.num_act_stepped = num_act_stepped
        self.decode_each_iteration = decode_each_iteration
        self.decode_unroll = decode_unroll
        self.optimizer_type = optimizer_type.lower()
        self.adam_betas = adam_betas
        self.adam_eps = adam_eps
        self.device = torch.device("cuda")

    def init_actions(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize actions for planning.

        Args:
            device: Device to place actions on

        Returns:
            actions: (1, horizon, action_dim) initialized actions
        """
        if self.sample_type == "randn":
            actions = torch.randn(1, self.horizon, self.action_dim, device=device) * self.var_scale
        elif self.sample_type == "zero":
            actions = torch.zeros(1, self.horizon, self.action_dim, device=device)
        else:
            raise ValueError(f"Unknown sample_type: {self.sample_type}")
        return actions

    def plan(
        self,
        z_init: torch.Tensor,
        steps_left: int = None,
    ) -> PlanningResult:
        """
        Plan a sequence of actions using gradient descent optimization.

        Args:
            z_init: Initial latent state
            steps_left: Number of steps left in episode (optional)

        Returns:
            PlanningResult with optimized actions and planning metrics
        """
        if steps_left is not None:
            plan_length = min(self.horizon, steps_left)
        else:
            plan_length = self.horizon

        # Initialize actions: (batch_size, plan_length, action_dim)
        actions = self.init_actions(1, self.device)[:, :plan_length, :]
        actions.requires_grad = True

        # Setup optimizer based on optimizer_type
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam([actions], lr=self.lr, betas=self.adam_betas, eps=self.adam_eps)
        else:
            optimizer = torch.optim.SGD([actions], lr=self.lr)

        losses = []
        predicted_best_encs_over_iterations = []
        if self.decode_each_iteration:
            pred_frames_over_iterations = []

        # Optimization loop
        for itr in range(self.iterations):
            optimizer.zero_grad()

            # Unroll world model with current actions
            # actions shape: (1, plan_length, action_dim)
            # Need to transpose to (plan_length, 1, action_dim) for unroll
            actions_transposed = actions.transpose(0, 1)

            predicted_encs = self.unroll(z_init, act_suffix=actions_transposed)
            loss = self.objective(predicted_encs, actions_transposed)  # (1,)

            total_loss = loss.mean()
            total_loss.backward()

            # Manual gradient descent update with noise
            with torch.no_grad():
                actions_new = actions - self.lr * actions.grad

                # Add Gaussian noise if specified
                if self.action_noise > 0:
                    actions_new += torch.randn_like(actions_new) * self.action_noise

                # Apply clipping if max_norms is specified (similar to CEM)
                if self.max_norms is not None:
                    for dims, maxnorm in zip(self.max_norm_dims, self.max_norms):
                        actions_new[:, :, dims] = torch.clip(actions_new[:, :, dims], min=-maxnorm, max=maxnorm)

                actions.copy_(actions_new)

            # Reset gradients after manual update
            actions.grad.zero_()

            losses.append(total_loss.item())

            # Store predictions for this iteration
            with torch.no_grad():
                predicted_best_encs = self.unroll(z_init, act_suffix=actions.transpose(0, 1))
                predicted_best_encs_over_iterations.append(predicted_best_encs)

                if self.decode_each_iteration and self.decode_unroll is not None:
                    pred_frames = self.decode_unroll(predicted_best_encs)
                    pred_frames_over_iterations.append(pred_frames)

        # Return the optimized actions
        final_actions = actions.squeeze(0).detach()
        losses = torch.tensor(losses).detach().unsqueeze(-1)

        result = PlanningResult(
            actions=final_actions[: self.num_act_stepped] if self.num_act_stepped else final_actions,
            losses=losses,
            prev_elite_losses_mean=losses,
            prev_elite_losses_std=torch.zeros_like(losses),
            pred_frames_over_iterations=pred_frames_over_iterations if self.decode_each_iteration else None,
            predicted_best_encs_over_iterations=predicted_best_encs_over_iterations,
        )
        return result


class AdamPlanner(GradientDescentPlanner):
    """Adam optimizer-based planner for action optimization in latent space.

    This is a convenience wrapper around GradientDescentPlanner with optimizer_type="adam".
    """

    def __init__(
        self,
        unroll: Callable,
        action_dim: int,
        horizon: int,
        iterations: int = 500,
        lr: float = 1,
        action_noise: float = 0.003,
        sample_type: str = "randn",
        var_scale: float = 1,
        max_norms: List[float] = None,
        max_norm_dims: List[List[int]] = [[0, 1, 2], [6]],
        num_act_stepped: int = None,
        decode_each_iteration: bool = False,
        decode_unroll: Callable = None,
        adam_betas: tuple = (0.9, 0.995),
        adam_eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(
            unroll=unroll,
            action_dim=action_dim,
            horizon=horizon,
            iterations=iterations,
            lr=lr,
            action_noise=action_noise,
            sample_type=sample_type,
            var_scale=var_scale,
            max_norms=max_norms,
            max_norm_dims=max_norm_dims,
            num_act_stepped=num_act_stepped,
            decode_each_iteration=decode_each_iteration,
            decode_unroll=decode_unroll,
            optimizer_type="adam",
            adam_betas=adam_betas,
            adam_eps=adam_eps,
            **kwargs,
        )


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]
