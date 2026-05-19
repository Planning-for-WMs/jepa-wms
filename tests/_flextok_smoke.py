"""Standalone smoke test for FlexTokEncoder. Not part of the test suite — delete after use."""

import sys
import traceback

import torch

print("python:", sys.executable, flush=True)
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), flush=True)

try:
    from app.plan_common.models.flextok_enc import FlexTokEncoder

    print("[1/4] import ok", flush=True)
    enc = FlexTokEncoder(
        model_id="EPFL-VILAB/flextok_d18_d28_dfn",
        img_size=224,
        eval_keep_k=64,
    )
    print(
        f"[2/4] built encoder: patch_size={enc.patch_size}, emb_dim={enc.emb_dim}, num_features={enc.num_features}",
        flush=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = enc.to(device).eval()
    print(f"[3/4] moved to {device}", flush=True)

    x = torch.rand(2, 3, 224, 224, device=device) * 2 - 1
    with torch.no_grad():
        out = enc(x)
    print(f"[4/4] forward ok: out.shape={tuple(out.shape)} dtype={out.dtype}", flush=True)
    assert out.shape == (2, 64, 1152), f"Expected (2, 64, 1152), got {tuple(out.shape)}"
    print("SHAPE_OK", flush=True)
except Exception as e:
    print("FAIL:", type(e).__name__, str(e), flush=True)
    traceback.print_exc()
