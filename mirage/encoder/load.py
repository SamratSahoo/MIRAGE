from __future__ import annotations

import torch

from .models import MaskedStateEncoder, StateEncoder


def build_encoder(enc_cfg: dict):
    mask_prob = float(enc_cfg.get("mask_prob", 0.0))
    if mask_prob > 0:
        return MaskedStateEncoder(
            latent_dim=enc_cfg["latent_dim"],
            hidden_dim=enc_cfg["hidden_dim"],
            n_hidden=enc_cfg["n_hidden"],
            l2_normalize=enc_cfg["l2_normalize"],
            cold_init_eps=enc_cfg["cold_init_eps"],
        )
    state_dim = {"xy": 2, "proprio": 27, "full": 29}[enc_cfg["input_mode"]]
    return StateEncoder(
        in_dim=state_dim,
        latent_dim=enc_cfg["latent_dim"],
        hidden_dim=enc_cfg["hidden_dim"],
        n_hidden=enc_cfg["n_hidden"],
        l2_normalize=enc_cfg["l2_normalize"],
        cold_init_eps=enc_cfg["cold_init_eps"],
    )


def load_encoder(ckpt_path: str, device="cpu", eval_mode: bool = True):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc_cfg = ck["config"]
    encoder = build_encoder(enc_cfg)
    encoder.load_state_dict(ck["encoder"])
    encoder = encoder.to(device)
    if eval_mode:
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
    return encoder, enc_cfg, ck


def is_masked(enc_cfg: dict) -> bool:
    return float(enc_cfg.get("mask_prob", 0.0)) > 0
