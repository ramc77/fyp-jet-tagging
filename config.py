"""
Configuration for FYP: Deep Learning for Jet Classification
============================================================
All hyperparameters, paths, and settings in one place.
Modify this file to change any aspect of the pipeline.
"""

import os
import torch

# ── Paths ──────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
MODEL_DIR = os.path.join(PROJECT_DIR, "results", "models")

# Create directories
for d in [DATA_DIR, RESULTS_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Device ─────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4 if torch.cuda.is_available() else 0

# ── Dataset ────────────────────────────────────────────────
# Top Quark Tagging Reference Dataset (Zenodo: 10.5281/zenodo.2603256)
# Full dataset: 1.2M train, 400k val, 400k test (~15 GB)
# Set USE_SUBSET=True for development/testing on limited hardware
USE_SUBSET = True           # Set False for full 1.2M-jet training
SUBSET_TRAIN = 50_000
SUBSET_VAL   = 50_000
SUBSET_TEST  = 50_000

MAX_CONSTITUENTS = 200       # Max particles per jet (zero-padded)
NUM_PARTICLE_FEATURES = 7    # (pT, eta, phi, E, delta_eta, delta_phi, log_pT)

# ── Jet Image Settings ─────────────────────────────────────
IMG_SIZE = 40                # 40x40 pixels in (delta_eta, delta_phi) plane
IMG_CHANNELS = 3             # pT-weighted, multiplicity, pT^2
IMG_RANGE = 1.0              # Range in delta_eta and delta_phi

# ── CNN Hyperparameters ────────────────────────────────────
# epochs are intentionally trimmed from the literature defaults so the
# full pipeline finishes in <1 h on a CPU laptop while still saturating
# learning for the 50k-jet subset (early-stopping kicks in around 12).
CNN_CONFIG = {
    "lr": 1e-3,
    "batch_size": 256,
    "epochs": 50,
    "weight_decay": 1e-4,
    "dropout": 0.2,
    "filters": [64, 128, 256],
    "fc_dim": 128,
    "scheduler_patience": 5,
    "early_stop_patience": 10,
}

# ── ParticleNet Hyperparameters ────────────────────────────
PARTICLENET_CONFIG = {
    "lr": 1e-3,
    "batch_size": 512,
    "epochs": 30,
    "weight_decay": 1e-4,
    "dropout": 0.1,
    "k_neighbors": 16,
    "edge_conv_dims": [
        [64, 64, 64],
        [128, 128, 128],
        [256, 256, 256],
    ],
    "fc_dims": [256],
    "input_features": 4,
    "coord_features": 2,
    "scheduler_patience": 3,
    "early_stop_patience": 7,
}

# ── Particle Transformer Hyperparameters ───────────────────
PARTFORMER_CONFIG = {
    "lr": 1e-4,
    "batch_size": 256,
    "epochs": 30,
    "weight_decay": 1e-4,
    "dropout": 0.1,
    "embed_dim": 128,
    "num_heads": 4,
    "num_layers": 3,
    "ff_dim": 256,
    "input_features": 7,
    "pair_features": 4,
    "scheduler_patience": 3,
    "early_stop_patience": 7,
}

# ── BDT Baseline ───────────────────────────────────────────
BDT_CONFIG = {
    "n_estimators": 500,
    "max_depth": 7,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
}

# ── Physics Constants ──────────────────────────────────────
TOP_MASS = 172.76   # GeV, PDG 2023
W_MASS = 80.377     # GeV, PDG 2023

# ── Plot Settings ──────────────────────────────────────────
PLOT_STYLE = {
    "figure.figsize": (8, 6),
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}
