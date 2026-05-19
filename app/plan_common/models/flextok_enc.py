# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License

import math

import torch
import torch.nn as nn


class FlexTokEncoder(nn.Module):
    """
    Wraps the FlexTok encoder (EPFL-VILAB/flextok_*) as a frozen visual feature extractor.

    FlexTok's encoder pipeline is a SequentialModuleDictWrapper:
        vae_latents → channels_to_last → patch_embed → posemb → register_module
        → seq_packer → flex_transformer → unpacker → [enc_to_latents → FSQ]

    We run all stages up to (not including) the final LinearHead that projects
    from native_dim (1152 for d18) down to FSQ input dim (6).  After the unpacker,
    data_dict['enc_registers'] holds one [1, 256, native_dim] tensor per image.

    We then slice to eval_keep_k tokens. The Registers1D module is trained with
    Matryoshka / nested ordering, so the first K tokens carry the most information
    at any K ≤ 256.

    eval_keep_k must be a perfect square so that VideoWM can re-interpret the 1-D
    token sequence as a sqrt(K) × sqrt(K) spatial grid (grid_size = sqrt(K)).

    Args:
        model_id:      HuggingFace repo, e.g. 'EPFL-VILAB/flextok_d18_d28_dfn'.
        img_size:      Spatial resolution of input images (square assumed).
        eval_keep_k:   Number of tokens to keep after truncation (must be a perfect square).
        proj_dim:      If set, add a trainable linear projection from native_dim → proj_dim.
                       Leave None to let the predictor's built-in embed projection handle it.
    """

    def __init__(self, model_id: str, img_size: int, eval_keep_k: int, proj_dim: int = None):
        super().__init__()

        grid_size = int(math.isqrt(eval_keep_k))
        assert grid_size * grid_size == eval_keep_k, (
            f"eval_keep_k={eval_keep_k} must be a perfect square so tokens form a 2-D grid"
        )
        assert img_size % grid_size == 0, (
            f"img_size={img_size} must be divisible by grid_size={grid_size}"
        )

        from flextok.flextok_wrapper import FlexTokFromHub

        _full = FlexTokFromHub.from_pretrained(model_id).eval()

        # --- VAE (frozen, always called under torch.no_grad) ---
        self.vae = _full.vae
        # Key under which the VAE expects input images
        self._vae_images_key: str = _full.vae.images_read_key  # 'rgb'

        # --- Identify which modules to run and which to skip ---
        # The final LinearHead writes to FSQ.latents_read_key ('enc_registers').
        # We skip that module so enc_registers still holds 1152-dim features.
        fsq_latents_key: str = _full.regularizer.latents_read_key  # 'enc_registers'

        enc_modules_ordered = {}
        native_dim: int = None
        for name, module in _full.encoder.module_dict.items():
            # The LinearHead we want to skip: its write_key == fsq_latents_key
            if getattr(module, "write_key", None) == fsq_latents_key:
                # Grab native_dim from the projection it would perform (dim_in)
                native_dim = module.dim_in
                break
            enc_modules_ordered[name] = module

        assert native_dim is not None, (
            "Could not find the enc_to_latents LinearHead in encoder.module_dict. "
            "Check that the FlexTok model is EPFL-VILAB/flextok_d18_d28_dfn or compatible."
        )

        # Register as nn.ModuleDict to ensure parameters are tracked (for device moves etc.)
        self._enc_modules = nn.ModuleDict(enc_modules_ordered)
        # Ordered list of keys (Python 3.7+ dicts are insertion-ordered)
        self._enc_module_keys = list(enc_modules_ordered.keys())
        # Key where 1152-dim register features live after enc_unpacker
        self._feature_key: str = fsq_latents_key  # 'enc_registers'

        # Freeze everything — the entire FlexTok encoder is a frozen feature extractor
        for p in self.parameters():
            p.requires_grad = False

        # --- Optional trainable projection ---
        if proj_dim is not None:
            self.proj = nn.Linear(native_dim, proj_dim)
            self.emb_dim = proj_dim
        else:
            self.proj = None
            self.emb_dim = native_dim

        # --- Attributes expected by JEPA-WMS (mirrors DinoEncoder interface) ---
        self.eval_keep_k = eval_keep_k
        # Virtual patch size: the grid is grid_size × grid_size, so each "patch" spans
        # img_size // grid_size pixels. The JEPA assertion checks img_size % patch_size == 0.
        self.patch_size = img_size // grid_size
        self.num_features = self.emb_dim  # DinoEncoder compat

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) float in [-1, 1]  (VAE expects this range)
        Returns:
            (B, eval_keep_k, emb_dim) continuous features
        """
        # FlexTok uses a list-of-tensors API (each item: [1, N, D])
        data_dict = {self._vae_images_key: x.split(1)}

        # VAE encode under no_grad (SD VAE is always frozen)
        with torch.no_grad():
            data_dict = self.vae.encode(data_dict)

        # Run the encoding pipeline up to (not including) enc_to_latents
        for key in self._enc_module_keys:
            data_dict = self._enc_modules[key](data_dict)

        # data_dict[self._feature_key] is a list of [1, 256, native_dim] tensors
        feats_list = data_dict[self._feature_key]
        feats = torch.cat(feats_list, dim=0)  # (B, 256, native_dim)

        # Truncate to eval_keep_k — Matryoshka ordering makes first K most informative
        feats = feats[:, : self.eval_keep_k, :]  # (B, K, native_dim)

        if self.proj is not None:
            feats = self.proj(feats)

        return feats
