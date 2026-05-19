"""Quick check: did the saved predictor actually train (i.e. did weights change
from fresh init)? If saved weights == fresh init weights up to layernorm/bias,
the optimizer never updated them."""
import sys
from pathlib import Path

import torch

REPO = Path("/home/aarav/wms/jepa-wms")
sys.path.insert(0, str(REPO))

CKPT_PATH = "/home/aarav/wms/jepa-wms-logs/pt_sweep/pt_4f_fsk5_ask1_r224_flextok_d18d28dfn_k64_predAdaLN_depth6_2roll_save/jepa-latest.pth.tar"

ckpt = torch.load(CKPT_PATH, map_location="cpu")
print(f"Checkpoint keys: {list(ckpt.keys())}")
print(f"Epoch: {ckpt['epoch']}")

# Predictor weights
pred_sd = ckpt["predictor"]
print(f"\nPredictor weights:")
print(f"  total tensors: {len(pred_sd)}")

# Print stats for a few representative tensors
for k in list(pred_sd.keys())[:10]:
    v = pred_sd[k]
    print(f"  {k:60s}  shape={tuple(v.shape)}  mean={v.float().mean():+.5f}  std={v.float().std():.5f}  min={v.float().min():+.3f}  max={v.float().max():+.3f}")

# Now build a fresh predictor with the same config and compare
print("\n" + "="*70)
print("Building fresh predictor for comparison...")

from app.plan_common.models.AdaLN_vit import vit_predictor_AdaLN

# patch_size = img_size // grid_size = 224 // 8 = 28
fresh = vit_predictor_AdaLN(
    img_size=224,
    patch_size=28,
    embed_dim=1152,
    predictor_embed_dim=384,
    num_heads=16,
    depth=6,
    tubelet_size=1,
    proprio_dim=4,
    proprio_emb_dim=16,
    action_dim=10,
    use_rope=True,
    use_silu=False,
    action_encoder_inpred=True,
    proprio_encoder_inpred=False,
    use_proprio=True,
    use_activation_checkpointing=False,
    proprio_encoding="feature",
    num_frames=4,
    init_scale_factor_adaln=10,
)
print("Fresh predictor built.")

fresh_sd = fresh.state_dict()

# Strip DDP "module." prefix from saved checkpoint
pred_sd = {k.replace("module.", ""): v for k, v in pred_sd.items()}

# Compare
common_keys = set(pred_sd.keys()) & set(fresh_sd.keys())
saved_only = set(pred_sd.keys()) - set(fresh_sd.keys())
print(f"\nCommon keys: {len(common_keys)}, saved-only: {len(saved_only)}")

# Compute Frobenius distance for each tensor
import torch.nn.functional as F
total_change = 0.0
total_norm = 0.0
unchanged = 0
changed = 0
print(f"\nWeight comparison (saved vs fresh):")
print(f"  {'param':70s} {'fresh_norm':>11s} {'saved_norm':>11s} {'rel_diff':>11s}")
n_shown = 0
for k in sorted(common_keys):
    f = fresh_sd[k].float()
    s = pred_sd[k].float()
    if f.shape != s.shape:
        continue
    diff = (s - f).norm().item()
    fn = f.norm().item()
    total_change += diff**2
    total_norm += fn**2
    rel = diff / (fn + 1e-8)
    if rel < 1e-5:
        unchanged += 1
    else:
        changed += 1
    if n_shown < 25:
        print(f"  {k:70s} {fn:>11.4f} {s.norm().item():>11.4f} {rel:>11.4f}")
        n_shown += 1

print(f"\nSummary:")
print(f"  Params unchanged (rel_diff < 1e-5): {unchanged}")
print(f"  Params changed:                     {changed}")
print(f"  Global rel change: {total_change**0.5 / (total_norm**0.5 + 1e-8):.4f}")
