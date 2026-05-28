#!/usr/bin/env python3
"""
Regenerate the Particle-Transformer attention plots with BALANCED sampling.
===========================================================================
The original cls_attention / attention_maps plots sampled the first 10
(unshuffled) test jets, which were all QCD, leaving the Top-jet panel
empty. With the fixed Trainer.get_attention_maps (balanced 5-Top/5-QCD),
this regenerates both plots locally from the downloaded checkpoint —
no retraining.

Lean: loads only the particle features (no jet-substructure loops), uses a
train slice for the normalisation stats, and stops after the first test
batch that yields both classes.
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from config import PARTFORMER_CONFIG, MODEL_DIR
from src.data_utils import load_raw_data, compute_particle_features, ParticleDataset
from src.particle_transformer import ParticleTransformer
from src.trainer import Trainer
from src.evaluation import plot_attention_maps, plot_cls_attention

MAXP = 100  # matches the pipeline's --max-particles default for ParT eval


def main():
    ckpt = os.path.join(MODEL_DIR, "particle_transformer_best.pt")
    if not os.path.exists(ckpt):
        sys.exit(f"Checkpoint not found: {ckpt}")

    print("Loading train slice for normalisation stats...")
    tr_con, _ = load_raw_data("train")
    tr_feat, tr_mask = compute_particle_features(tr_con)
    valid = tr_mask.flatten().astype(bool)
    flat = tr_feat.reshape(-1, tr_feat.shape[-1])
    mean = flat[valid].mean(axis=0)
    std = flat[valid].std(axis=0) + 1e-8
    del tr_con, tr_feat, tr_mask, flat

    print("Loading test slice for attention...")
    te_con, te_lab = load_raw_data("test")
    te_feat, te_mask = compute_particle_features(te_con)
    te_norm = ((te_feat - mean) / std) * te_mask[:, :, None]

    ds = ParticleDataset(
        te_norm.astype(np.float32), te_mask, te_lab,
        max_particles=MAXP, compute_pairs=True, features_raw=te_feat,
    )
    loader = DataLoader(ds, batch_size=128, shuffle=False)

    print("Loading Particle Transformer checkpoint...")
    model = ParticleTransformer()
    tr = Trainer(model, "particle_transformer", PARTFORMER_CONFIG, model_type="transformer")
    tr.load_best_model()

    print("Extracting balanced attention (5 Top + 5 QCD)...")
    attn_maps, feats, labels = tr.get_attention_maps(loader, n_samples=10)
    print("  labels of sampled jets:", labels)

    plot_attention_maps(attn_maps, feats, labels)
    plot_cls_attention(attn_maps, feats, labels)
    print("Done — regenerated attention_maps.{pdf,png} and cls_attention.{pdf,png}")


if __name__ == "__main__":
    main()
