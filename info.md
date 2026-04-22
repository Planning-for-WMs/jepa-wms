# JEPA-WMs Codebase Guide for Agents

This document gives an agent enough context to implement new experiments (new loss, new planner, new objective, new environment, new model component, etc.) without needing to read through the full codebase.

---

## Repository Structure (key directories only)

```
jepa-wms/
  app/
    main.py                     # Training entry point (local multi-GPU)
    scaffold.py                 # Routes app name -> app/{name}/train.py
    vjepa_wm/
      train.py                  # Training loop (~1700 lines, the big file)
      video_wm.py               # VideoWM class: encode, forward_pred, compute_loss, rollout
      utils.py                  # init_video_model(), init_opt(), checkpoint load/save
      modelcustom/
        simu_env_planning/
          vit_enc_preds.py      # EncPredWM wrapper (inference-time encode/unroll/decode)
    plan_common/
      models/
        AdaLN_vit.py            # AdaLN predictor (with Token Merging)
        vit.py                  # ViTPredictor (dino_wm style)
        prop_embedding.py       # ProprioceptiveEmbedding (action & proprio encoders)
        dino.py                 # DinoEncoder (frozen DINOv2)
        wm_heads.py             # Decoder heads: image, pose/state, reward
        decoder.py              # VisionTransformerDecoder (image head backbone)
        state_decoder.py        # StateReadoutViT (state head backbone)
      datasets/
        utils.py                # init_data() — dataset factory
        preprocessor.py         # Preprocessor (normalization stats, transforms)
        pusht_dset.py           # PushT dataset
        metaworld_hf_dset.py    # MetaWorld dataset
        wall_dset.py            # Wall dataset
        point_maze_dset.py      # PointMaze dataset
        droid_dset.py           # DROID dataset
        robocasa_dset.py        # RoboCasa dataset
        precomputed_dset.py     # Precomputed features dataset
  evals/
    main.py                     # Eval entry point (local multi-GPU)
    main_distributed.py         # Eval entry point (SLURM submitit)
    scaffold.py                 # Routes eval_name -> evals/{name}/eval.py
    utils.py                    # make_datasets() shared utility
    simu_env_planning/
      eval.py                   # Eval orchestration: init model, loop over tasks/episodes
      envs/
        init.py                 # make_env() factory — routes task prefix to environment
        metaworld.py            # MetaWorld env wrapper
        pusht_gym_wrap.py       # PushT env wrapper
        wall_gym_wrap.py        # Wall env wrapper
        pointmaze_gym_wrap.py   # PointMaze env wrapper
        robocasa.py             # RoboCasa env wrapper
        wrappers/               # PixelWrapper, TensorWrapper, MultitaskWrapper, TimeLimit
      planning/
        gc_agent.py             # GC_Agent: planner instantiation, set_goal, plan, act
        plan_evaluator.py       # PlanEvaluator: episode setup, agent rollout, metrics
        planning/
          planner.py            # All planner classes (CEM, GRASP, CEM-GD, MPPI, GD, Adam, Nevergrad)
          objectives.py         # Planning objectives (L2, L1, cosine similarity)
        common/
          gc_logger.py          # Eval Logger (CSV + wandb)
          parser.py             # parse_cfg()
          __init__.py           # TASK_SET dict (multi-task groups)
  configs/
    vjepa_wm/                   # Training configs
    evals/                      # Eval configs (for main_distributed.py)
    online_plan_evals/          # Planning eval configs (pt/, mw/, wall/, mz/, droid/, rcasa_custom/)
  src/
    utils/
      run_id.py                 # Run ID generation, config snapshots, metadata
      logging.py                # CSVLogger, git_information()
      yaml_utils.py             # expand_env_vars(), .env loading
      distributed.py            # init_distributed()
    models/
      ac_predictor.py           # VJEPA2-AC predictor (action-conditioned)
      vision_transformer.py     # V-JEPA v1 encoder
      vision_transformer_v2.py  # V-JEPA v2 encoder
  scripts/
    profile_planning.py         # Benchmark planning loop
    profile_predictor.py        # Benchmark predictor forward pass
```

---

## How to Run

### Environment setup
```bash
source env.sh              # Sets JEPAWM_LOGS, JEPAWM_DSET, JEPAWM_CKPT, JEPAWM_OSSCKPT
conda activate jepa-wms    # Or however the env is activated
```

### Training (local)
```bash
# Single GPU debug
python -m app.main --fname configs/vjepa_wm/<config>.yaml --debug

# Multi-GPU (spawns one process per device)
python -m app.main --fname configs/vjepa_wm/<config>.yaml --devices cuda:0 cuda:1
```

### Evaluation (local)
```bash
# Single GPU debug — quickest way to test changes
python -m evals.main --fname configs/evals/simu_env_planning/pt/jepa-wm/<config>.yaml --debug

# Multi-GPU
python -m evals.main --fname configs/evals/simu_env_planning/pt/jepa-wm/<config>.yaml
```

### Evaluation (SLURM)
```bash
python -m evals.main_distributed \
  --fname configs/evals/simu_env_planning/pt/jepa-wm/<config>.yaml \
  --account <slurm_account> --partition gpu --qos <qos> --time 120
```

### Quick debug mode
Set `meta.quick_debug: true` in the eval config. This forces `eval_episodes=1`, `iterations=2`, `num_samples=2`, `num_elites=2` for fast iteration.

---

## Architecture Overview

### Training data flow
```
Batch: obs["visual"] (B,T,C,H,W) + obs["proprio"] (B,T,P) + action (B,T,A)
  |
  v
VideoWM.encode()
  ├── encode_obs() -> Frozen encoder (DINO/VJEPA) -> video_features (B,T,1,H,W,D)
  ├── encode_act() -> Action encoder -> action_features (B,T,1,D) or (B,T,H*W,D)
  └── encode_proprio() -> Proprio encoder -> proprio_features (B,T,1,D) or (B,T,H*W,D)
  |
  v
VideoWM.forward_pred()
  └── Predictor (AdaLN/dino_wm/vjepa2_ac) -> pred_video (B,T,1,H,W,D), pred_proprio (B,T,1,D)
  |
  v
VideoWM.compute_loss()  ->  weighted sum of {L2, L1, cosine, smooth-L1} losses
  |
  v
Optional: VideoWM.rollout() -> multi-step prediction with sequential or parallel unrolling
Optional: Head decoding -> image_head, state_head, reward_head losses
```

### Planning data flow (eval)
```
Env observation
  |
  v
GC_Agent.act(obs)
  ├── EncPredWM.encode(obs)  ->  z_init (context latent state)
  └── Planner.plan(z_init)
        ├── Sample/optimize action candidates
        ├── For each candidate: EncPredWM.unroll(z_init, actions) -> predicted latents
        ├── Objective(predicted_latents, target_enc) -> cost
        └── Return best action sequence
  |
  v
Execute action in env, observe next state, repeat
```

---

## How to Add a New Training Loss

### Where losses are defined
**File:** `app/vjepa_wm/video_wm.py` — `VideoWM.compute_loss()` (line ~384)

The method receives predicted and ground-truth features, computes 4 loss types (L2, L1, cosine, smooth-L1), and returns a weighted sum. The weights come from config under `loss:`.

### Steps

1. **Add loss computation in `VideoWM.compute_loss()`** (`app/vjepa_wm/video_wm.py:384-465`):
   ```python
   # After the existing loss computations (~line 428):
   visual_my_loss = my_loss_fn(visual_features_, visual_targets_).mean(dim=-1)
   ```

2. **Add the weight to the loss combination** (~line 440):
   ```python
   loss += self.cfgs_loss["my_loss_weight"] * visual_my_loss
   ```

3. **Add the weight to config** — in your training YAML:
   ```yaml
   loss:
     l2_loss_weight: 1.0
     my_loss_weight: 0.5
   ```

4. **Initialize the weight with a default** — in `train.py` where `cfgs_loss` is read (~line 100), ensure a default exists. The `cfgs_loss` dict is passed directly from config, so adding the key to YAML is sufficient. If you want a default of 0 (backward compatible), check for it in `compute_loss()`:
   ```python
   my_weight = self.cfgs_loss.get("my_loss_weight", 0.0)
   ```

5. **Add to the returned dict** for logging (~line 449):
   ```python
   out["visual_my_loss"] = visual_my_loss.mean() if reduce_mean else visual_my_loss
   ```

### Head losses (image decoder, state decoder, reward)
Head losses are separate from `compute_loss()`. They live in `app/plan_common/models/wm_heads.py`:
- `WorldModelViTImageHead.compute_loss()` (line 67) — pixel + optional LPIPS
- `WorldModelPoseReadoutHead.compute_loss()` (line 153) — L1 on state dims
- `WorldModelRewardReadoutHead.compute_loss()` (line 187) — L1 on reward scalar

To add a new head loss, modify the relevant class's `compute_loss()` method. Head training is controlled by `optimization.train_heads: true` in config.

---

## How to Add a New Planner

### Where planners are defined
**File:** `evals/simu_env_planning/planning/planning/planner.py`

All planners inherit from `Planner` (line 37) and implement `plan(z_init, steps_left) -> PlanningResult`.

### Existing planners (for reference)
| Name | Config value | Class | Line | Description |
|------|-------------|-------|------|-------------|
| CEM | `cem` | `CEMPlanner` | 212 | Cross-Entropy Method, sampling-based |
| GRASP | `grasp` | `GRASPlanner` | 347 | Gradient relaxed, needs full model (`enc_pred_wm`) |
| CEM-GD | `cem_gd` | `CEMGDPlanner` | 563 | CEM + gradient descent refinement |
| MPPI | `mppi` | `MPPIPlanner` | 766 | Model Predictive Path Integral |
| GD | `gd` | `GradientDescentPlanner` | 899 | Pure gradient descent on actions |
| Adam | `adam` | `AdamPlanner` | 1070 | Adam optimizer on actions |
| Nevergrad | `nevergrad` | `NevergradPlanner` | 54 | Black-box optimization |

### Steps

1. **Add planner class in `planner.py`**:
   ```python
   class MyPlanner(Planner):
       def __init__(
           self,
           unroll,               # Callable: EncPredWM.unroll(z_ctxt, act_suffix)
           action_dim,
           horizon=6,
           iterations=10,
           my_param=0.5,
           num_act_stepped=None,  # How many actions to execute before re-planning
           decode_each_iteration=False,
           decode_unroll=None,    # Callable: EncPredWM.decode_unroll()
           local_generator=None,  # torch.Generator for reproducibility
           max_norms=None,        # Action clipping bounds
           max_norm_dims=None,
           **kwargs,              # IMPORTANT: always accept **kwargs (config passes all planner keys)
       ):
           super().__init__(unroll)
           # store params...

       @torch.no_grad()  # Remove if your planner needs gradients (like GD/GRASP)
       def plan(self, z_init, steps_left=None):
           plan_length = min(self.horizon, steps_left) if steps_left else self.horizon

           # Your optimization loop:
           # 1. Initialize actions: (plan_length, 1, action_dim) on cuda
           # 2. For each iteration:
           #    cost = self.cost_function(actions, z_init)
           #    # cost_function calls self.unroll(z_init, actions) then self.objective(pred, actions)
           #    Update actions based on cost
           # 3. Return PlanningResult

           return PlanningResult(
               actions=best_actions[:self.num_act_stepped],
               losses=torch.tensor(losses).unsqueeze(-1),
               prev_elite_losses_mean=...,
               prev_elite_losses_std=...,
           )
   ```

   The `cost_function` is inherited from `Planner` base class (line 42):
   ```python
   def cost_function(self, actions, z):
       pred = self.unroll(z, actions)
       return self.objective(pred, actions)
   ```

2. **Register in `gc_agent.py`** (`evals/simu_env_planning/planning/gc_agent.py:53-113`):
   ```python
   from evals.simu_env_planning.planning.planning.planner import MyPlanner

   # Inside GC_Agent.__init__, add an elif:
   elif self.cfg.planner.planner_name == "my_planner":
       self.planner = MyPlanner(
           unroll=self.model.unroll,
           action_dim=self.model.action_dim,
           local_generator=self.local_gpu_generator,
           decode_unroll=self.model.decode_unroll,
           **self.cfg.planner,
       )
   ```

3. **Add config**:
   ```yaml
   planner:
     planner_name: my_planner
     horizon: 6
     iterations: 10
     num_act_stepped: 3
     my_param: 0.5
     planning_objective:
       objective_type: L2
       alpha: 0
   ```

### Key interfaces your planner interacts with
- **`self.unroll(z_ctxt, act_suffix)`**: Takes context `z_ctxt` (TensorDict or Tensor with shape involving `(B, T, V, H, W, D)`) and actions `(T_act, B, A)`. Returns predicted latents `(T_total, B, V, H, W, D)`.
- **`self.objective(encodings, actions)`**: Takes predicted latents and actions, returns scalar cost per sample `(B,)`. Set by `Planner.set_objective()`.
- **`self.cost_function(actions, z_init)`**: Convenience wrapper that calls unroll then objective.
- **Actions are clipped** via `max_norms` / `max_norm_dims` (see CEMPlanner for clipping logic).
- **`PlanningResult`** fields: `actions`, `losses`, `prev_elite_losses_mean`, `prev_elite_losses_std`, `info`, `plan_metrics`, `pred_frames_over_iterations`, `predicted_best_encs_over_iterations`.

---

## How to Add a New Planning Objective

### Where objectives are defined
**File:** `evals/simu_env_planning/planning/planning/objectives.py`

All objectives inherit from `BaseMPCObjective` and implement `__call__(encodings, actions, keepdims) -> Tensor`.

### Existing objectives
| Config value | Class | Description |
|-------------|-------|-------------|
| `L2` | `ReprTargetDistMPCObjective` | MSE to target encoding |
| `L1` | `ReprTargetDistL1MPCObjective` | L1 distance to target |
| `repr_sim` | `ReprTargetCosMPCObjective` | Negative cosine similarity to target |

### Steps

1. **Add class in `objectives.py`**:
   ```python
   class MyObjective(BaseMPCObjective):
       def __init__(self, cfg, target_enc, sum_all_diffs=False, alpha=1.0, **kwargs):
           self.cfg = cfg
           self.target_enc = target_enc    # Encoded goal state (TensorDict or Tensor)
           self.sum_all_diffs = sum_all_diffs
           self.alpha = alpha              # Weight for proprio component

       def __call__(self, encodings, actions, keepdims=False):
           # encodings: (T, B, ..., D) — predicted latents from unroll
           # self.target_enc: (1, ..., D) — encoded goal
           # Must handle both TensorDict and Tensor inputs
           if isinstance(encodings, TensorDict):
               diff_visual = ...  # your metric on encodings["visual"] vs self.target_enc["visual"]
               diff_proprio = ... # your metric on encodings["proprio"] vs self.target_enc["proprio"]
               diff = diff_visual + self.alpha * diff_proprio
           else:
               diff = ...  # your metric on pure tensors

           if not keepdims:
               return diff[-1] if not self.sum_all_diffs else diff.sum(0)
           elif self.sum_all_diffs:
               return diff.cumsum(0).flip(0)
           return diff
   ```

2. **Register in `gc_agent.py`** (`evals/simu_env_planning/planning/gc_agent.py:116-141`):
   ```python
   from evals.simu_env_planning.planning.planning.objectives import MyObjective

   # Inside GC_Agent.set_goal(), add elif:
   elif self.cfg.planner.planning_objective.objective_type == "my_objective":
       self.objective = MyObjective(
           self.cfg, target_enc=self.goal_state_enc, **self.cfg.planner.planning_objective
       )
   ```

3. **Config**:
   ```yaml
   planner:
     planning_objective:
       objective_type: my_objective
       alpha: 0.1
       sum_all_diffs: false
   ```

---

## How to Add a New Environment

### Where environments are routed
**File:** `evals/simu_env_planning/envs/init.py` — `make_env(cfg)` (line 70)

Routing is by task name prefix: `pusht-` -> PushT, `mw-` -> MetaWorld, etc.

### Steps

1. **Create env wrapper** in `evals/simu_env_planning/envs/my_env.py`. Must provide:
   - `reset(seed, task_idx)` -> `(obs, info)`
   - `step(action)` -> `(obs, reward, done, info)` where `info` contains `"success"`
   - `obs_shape` dict, `action_dim`, `action_range`

2. **Add routing** in `evals/simu_env_planning/envs/init.py`:
   ```python
   elif cfg.task_specification.task.startswith("myenv-"):
       env = make_my_env(cfg)
   ```

3. **Wrap with standard wrappers** (TensorWrapper, PixelWrapper if needed).

4. **Add dataset** if needed (for goal sampling with `goal_source: dset`) in `app/plan_common/datasets/`.

---

## How to Add a New Predictor Type

### Where predictors are initialized
**File:** `app/vjepa_wm/utils.py` — `init_video_model()` (line ~726)

The predictor type is selected by `pretrain_kwargs.predictor.pred_type`. Currently: `dino_wm`, `vjepa2_ac`, `AdaLN`.

### Steps

1. **Define the predictor module** in `app/plan_common/models/my_predictor.py`.
   Must accept the same forward signature as other predictors. For `dino_wm` style:
   - Input: concatenated `(B, T, N_patches, D_concat)` where `D_concat = embed_dim + action_emb + proprio_emb`
   - Output: `(B, T, N_patches, D_concat)` (same shape)
   
   For `AdaLN` / `vjepa2_ac` style, inputs are separate tensors.

2. **Add initialization** in `init_video_model()` (`app/vjepa_wm/utils.py:726+`):
   ```python
   elif pred_type == "my_predictor":
       predictor = MyPredictor(...)
   ```

3. **Add forward_pred handling** in `VideoWM.forward_pred()` (`app/vjepa_wm/video_wm.py:301-371`) if the new predictor has a different calling convention.

4. **Config**:
   ```yaml
   model:
     predictor:
       pred_type: my_predictor
       # ... predictor-specific params
   ```

---

## How to Add a New Encoder

### Where encoders are initialized
**File:** `app/vjepa_wm/utils.py` — `init_video_model()` (line ~631)

Encoder type is selected by `pretrain_kwargs.visual_encoder.enc_type`. Currently: `dino`, `vjepa`, `precomputed`.

### Steps

1. **Define encoder** in `app/plan_common/models/my_encoder.py`:
   - Input: `(B*T, C, H, W)` images (when `batchify_video=True`)
   - Output: `(B*T, N_patches, D)` features

2. **Add initialization** in `init_video_model()`.

3. **Add encode_obs handling** in `VideoWM.encode_obs()` (`app/vjepa_wm/video_wm.py:146-209`).

---

## How to Add a New Decoder Head

### Where heads are defined
**File:** `app/plan_common/models/wm_heads.py`

All heads inherit from `TrainableModel` (in `app/plan_common/models/trainable_model.py`) which provides `backward()`, `optimization_step()`, and gradient scaling.

### Steps

1. **Add head class** in `wm_heads.py`:
   ```python
   class MyHead(TrainableModel):
       def __init__(self, head_config, train_config=None, device="cpu"):
           model = MyDecoder(**head_config)
           super().__init__(model, train_config=train_config, device=device)

       def compute_loss(self, predicted_features, targets):
           output = self.model(predicted_features)
           loss = ...
           return {"loss": loss}
   ```

2. **Register in `init_module()`** (`app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py:32-250`) and in `train.py` head initialization.

3. **Config**:
   ```yaml
   model:
     heads_cfg:
       architectures:
         my_head: /path/to/pretrained_head.pth.tar
   ```

---

## Key Classes Reference

### VideoWM (`app/vjepa_wm/video_wm.py:27`)
The core world model. Holds encoder, predictor, action/proprio encoders.

| Method | Line | Purpose |
|--------|------|---------|
| `encode_obs()` | 146 | Encode visual observations through frozen encoder |
| `encode_act()` | 211 | Encode actions (token or feature mode) |
| `encode_proprio()` | 234 | Encode proprioception (token or feature mode) |
| `encode()` | 250 | Calls all three encoders |
| `forward_pred()` | 301 | Run predictor on encoded features |
| `concat_obs_act()` | 373 | Concatenate visual + proprio + action (dino_wm only) |
| `compute_loss()` | 384 | Compute training loss (L2, L1, cosine, smooth-L1) |
| `rollout()` | 467 | Multi-step prediction (sequential or parallel) |

### EncPredWM (`app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py:253`)
Inference wrapper around VideoWM. Used by planners.

| Method | Line | Purpose |
|--------|------|---------|
| `encode(obs)` | 356 | Preprocess raw obs and encode through frozen encoder |
| `unroll(z_ctxt, act_suffix)` | 289 | Autoregressively predict forward from context + actions |
| `decode_unroll(pred_encs)` | 422 | Decode predicted latents to RGB via image head |

### GC_Agent (`evals/simu_env_planning/planning/gc_agent.py:34`)
Agent wrapper. Instantiates planner, handles goal encoding.

| Method | Line | Purpose |
|--------|------|---------|
| `__init__()` | 34 | Instantiate planner based on config |
| `set_goal()` | 116 | Encode goal state, instantiate objective |
| `plan()` | 143 | Call planner, store metrics |
| `act()` | 169 | Encode obs -> plan -> return action |

### Planner base (`evals/simu_env_planning/planning/planning/planner.py:37`)

| Method | Line | Purpose |
|--------|------|---------|
| `plan(z_init, steps_left)` | — | Abstract: optimize action sequence (each subclass implements) |
| `set_objective(objective)` | 47 | Set the cost function |
| `cost_function(actions, z)` | 42 | Unroll world model + evaluate objective |

---

## Tensor Shapes Cheat Sheet

| Tensor | Shape | Notes |
|--------|-------|-------|
| Raw visual obs | `(B, T, C, H, W)` | C=3, H=W=224 typically |
| Encoded visual features | `(B, T, 1, H_p, W_p, D)` | H_p=W_p=grid_size (e.g. 16), D=embed_dim (e.g. 384) |
| Encoded proprio features | `(B, T, 1, D)` | D=embed_dim |
| Action features (token) | `(B, T, 1, D)` | 1 token per timestep |
| Action features (feature) | `(B, T, H_p*W_p, D_act)` | Broadcast to spatial grid |
| Predicted features | Same as encoded features | Output of forward_pred() |
| Planning actions | `(T_plan, B, A)` | T_plan=horizon, B=num_samples, A=action_dim |
| Unroll output | `(T_total, B, V, H_p, W_p, D)` | T_total = context + predicted steps |
| Objective output | `(B,)` | Scalar cost per sample |

---

## Config Quick Reference

### Training config (`configs/vjepa_wm/*.yaml`)
```yaml
app: vjepa_wm
folder: ${JEPAWM_LOGS}/...
data:
  datasets: [PushT]
  img_size: 224
  custom: { frameskip: 5, action_skip: 1, num_hist: 9, num_pred: 1 }
loss:
  l2_loss_weight: 1.0
  cos_loss_weight: 0.0
  l1_loss_weight: 0.0
  smooth_l1_loss_weight: 0.0
model:
  predictor: { pred_type: AdaLN, pred_depth: 6, pred_embed_dim: 384, pred_num_heads: 16 }
  visual_encoder: { enc_type: dino, enc_version: dinov2_vits14, embed_dim: 384 }
  action_encoder: { action_tokens: 1, action_emb_dim: 0, action_encoder_inpred: true }
  proprio_encoder: { proprio_tokens: 0, proprio_emb_dim: 16 }
  rollout_cfg: { rollout_steps: 2, do_sequential_rollout: true }
  heads_cfg: { architectures: {} }  # empty = no decoder heads
optimization:
  transition_model: { num_epochs: 50, ref_lr: 5.e-4, clip_grad: 1 }
```

### Eval config (`configs/evals/simu_env_planning/.../*.yaml` or `configs/online_plan_evals/.../*.yaml`)
```yaml
eval_name: simu_env_planning
folder: ${JEPAWM_LOGS}/...
tag: gc_zeroshot/my_experiment
meta: { seed: 1, eval_episodes: 96, quick_debug: false }
distributed: { distribute_multitask_eval: true, local_rng_samplers: true, seed_shift: horizon_1000 }
logging: { exp_name: my_exp, save_csv: true, tqdm_silent: false, optional_plots: true }
model_kwargs:
  module_name: app.vjepa_wm.modelcustom.simu_env_planning
  checkpoint: /path/to/model.pth.tar
  pretrain_kwargs: { ... }   # Same model architecture params as training
  data: { ... }              # Same dataset params
  wrapper_kwargs: { ctxt_window: 2 }
task_specification:
  task: pusht-base           # Task name (prefix determines environment)
  obs: rgb_state             # rgb, rgb_state, or state
  goal_source: dset          # dset, expert, random_state, random_action
  succ_def: simu
  goal_H: 6
  img_size: 224
planner:
  planner_name: cem
  horizon: 6
  iterations: 30
  num_samples: 300
  num_elites: 10
  num_act_stepped: 3
  var_scale: 1.0
  planning_objective: { objective_type: L2, alpha: 0.1, sum_all_diffs: false }
```

---

## Config-Driven Feature Flags

| Feature | Config key | Default | Effect |
|---------|-----------|---------|--------|
| Token Merging (ToMe) | `predictor.tome_r` | `0` (off) | Set >0 to merge tokens in predictor |
| CLS vs patch tokens | `visual_encoder.feature_key` | `x_norm_patchtokens` | `x_norm_clstoken` for CLS |
| Precomputed features | `visual_encoder.enc_type` | `dino` | `precomputed` skips encoding |
| Planner selection | `planner.planner_name` | `cem` | Any of: `cem`, `grasp`, `cem_gd`, `gd`, `adam`, `mppi`, `nevergrad` |
| Action conditioning | `action_conditioning` | `token` | `token` or `feature` |
| Proprio encoding | `proprio_encoding` | `feature` | `token` or `feature` |
| Decoder heads | `heads_cfg.architectures` | `{}` (none) | Add `image_head`, `state_head`, `reward_head` |
| Rollout steps | `rollout_cfg.rollout_steps` | `1` | Multi-step training prediction |
| Wandb (training) | `logging.wandb.project` | `vjepa_wm` | Set `use_wandb: true` to enable |
| Wandb (eval) | `logging.wandb.project` | `jepa_wm_eval` | Set `use_wandb: true` to enable |

---

## Checkpoint Structure

Saved by `train.py:705-732`, loaded by `utils.py:358-450`.

```python
{
    "predictor": predictor.state_dict(),
    "action_encoder": action_encoder.state_dict(),  # if external
    "proprio_encoder": proprio_encoder.state_dict(),  # if external
    "opt": optimizer.state_dict(),
    "scaler": scaler.state_dict(),
    "epoch": int,
}
# Heads saved separately as: {checkpoint_path}_{head_name}.pth.tar
```

---

## Run Tracking

Every training and eval run generates a unique ID (`YYYYMMDD_HHMMSS_hexhash`) and creates:
```
{folder}/runs/{run_id}/
  config_resolved.yaml    # Full expanded config
  run_meta.json           # Git commit, hostname, timestamp
  eval.csv / log_r0.csv   # Metrics
{folder}/latest -> runs/{run_id}   # Symlink to most recent
```

---

## Environment Variables

Set via `source env.sh` or a `.env` file (auto-loaded from repo root):

| Variable | Purpose |
|----------|---------|
| `JEPAWM_LOGS` | Training and eval output root |
| `JEPAWM_DSET` | Dataset root |
| `JEPAWM_CKPT` | Checkpoint root |
| `JEPAWM_OSSCKPT` | Pretrained open-source encoder checkpoints |

---

## Common Gotchas

1. **`**kwargs` in planner constructors** — Config passes all planner keys to the constructor. Your planner MUST accept `**kwargs` or it will crash on unknown keys.

2. **Action shape convention differs between training and planning** — Training: `(B, T, A)`. Planning: `(T, B, A)` (time-first). The `EncPredWM.unroll()` does `rearrange(act_suffix, "t b ... -> b t ...")` internally.

3. **TensorDict vs Tensor** — When `obs: rgb_state`, features are TensorDict with `"visual"` and `"proprio"` keys. When `obs: rgb`, features are plain Tensors. Objectives and planners must handle both.

4. **Frozen encoder** — The visual encoder (DINO/VJEPA) is always frozen. Only the predictor, action encoder, and proprio encoder are trained.

5. **`forward_pred()` differs by predictor type** — `dino_wm` concatenates all features before calling the predictor. `AdaLN` and `vjepa2_ac` pass features separately. If adding a new predictor, add a branch in `VideoWM.forward_pred()`.

6. **`grid_size` vs `img_size`** — `grid_size = img_size / patch_size` (e.g., 224/14 = 16). Features are `(grid_size, grid_size, embed_dim)` per frame.

7. **`frameskip` vs `action_skip`** — `frameskip` = simulator steps per WM step. `action_skip` = how many low-level actions the model outputs per step. `action_ratio = frameskip / action_skip`.
