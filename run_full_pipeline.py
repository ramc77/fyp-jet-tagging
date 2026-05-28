#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  FYP: Deep Learning for Jet Classification in pp Collisions     ║
║  BNBWU-CERN Collaboration                                       ║
║                                                                  ║
║  Master Pipeline: Downloads data, trains all models,             ║
║  evaluates, and generates all publication-quality plots.         ║
║                                                                  ║
║  Usage:                                                          ║
║    python run_full_pipeline.py              # Run everything     ║
║    python run_full_pipeline.py --model cnn  # Train only CNN     ║
║    python run_full_pipeline.py --eval-only  # Only evaluate      ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ─── macOS / OpenMP safety ─────────────────────────────────────────
# MUST run before numpy/torch/xgboost imports. The stack pulls in two
# OpenMP runtimes on macOS (libiomp5 via numpy/openblas, libomp via
# XGBoost), which corrupts thread state and segfaults during
# xgb.DMatrix construction on Intel Macs. KMP_DUPLICATE_LIB_OK tells
# the Intel runtime to tolerate a second OpenMP being loaded; it's the
# workaround recommended by Intel/XGBoost for exactly this crash.
# See: https://github.com/dmlc/xgboost/issues/1715
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Keep OMP thread count modest — large counts amplify OMP-runtime races.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import sys
import time
import json
import numpy as np
import torch

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from config import *
from src.data_utils import prepare_all_data, make_dataloaders
from src.jet_cnn import JetCNN
from src.particle_net import ParticleNet
from src.particle_transformer import ParticleTransformer
from src.trainer import Trainer, train_bdt_baseline
from src.evaluation import run_full_evaluation


# ─── Resumability helpers ──────────────────────────────────────────
PRED_DIR = os.path.join(RESULTS_DIR, "predictions")
os.makedirs(PRED_DIR, exist_ok=True)


def _pred_path(model_name):
    return os.path.join(PRED_DIR, f"{model_name}_test.npz")


def save_predictions(model_name, test_probs, test_labels, n_params, extra=None):
    """Persist per-model test predictions so the next run can skip
    retraining + re-prediction and go straight to evaluation."""
    payload = {
        "test_probs": np.asarray(test_probs, dtype=np.float32),
        "test_labels": np.asarray(test_labels, dtype=np.float32),
        "n_params": str(n_params),
    }
    if extra is not None:
        for k, v in extra.items():
            payload[k] = np.asarray(v)
    np.savez_compressed(_pred_path(model_name), **payload)


def load_predictions(model_name):
    p = _pred_path(model_name)
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    n_params = d["n_params"].item() if d["n_params"].ndim == 0 else str(d["n_params"])
    return {
        "labels": d["test_labels"],
        "probs": d["test_probs"],
        "n_params": n_params,
    }


def _trainer_done(model_name):
    """True if a previous run finished this model (early-stop or max-epochs)."""
    return os.path.exists(os.path.join(MODEL_DIR, f"{model_name}_done.json"))


def parse_args():
    parser = argparse.ArgumentParser(description="Jet Classification Pipeline")
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["all", "bdt", "cnn", "particlenet", "transformer"],
        help="Which model to train (default: all)"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training, only run evaluation on saved models"
    )
    parser.add_argument(
        "--max-particles", type=int, default=100,
        help="Max constituents for ParticleNet/Transformer (default: 100)"
    )
    parser.add_argument(
        "--no-bdt", action="store_true",
        help="Skip BDT baseline training"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-train models even if a previous run marked them done. "
             "Without this flag, models with a saved *_done.json are skipped "
             "and their cached predictions are reused."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    total_start = time.time()

    print("╔" + "═" * 62 + "╗")
    print("║  FYP: Deep Learning for Jet Classification                   ║")
    print("║  Top Quark Tagging in 14 TeV pp Collisions                   ║")
    print("║  BNBWU-CERN Collaboration                                    ║")
    print("╚" + "═" * 62 + "╝")
    print(f"\nDevice: {DEVICE}")
    print(f"Subset mode: {USE_SUBSET} (train={SUBSET_TRAIN if USE_SUBSET else '1.2M'})")
    print(f"Max particles: {args.max_particles}")

    # ─── Step 1: Prepare Data ──────────────────────────────────
    print("\n" + "█" * 60)
    print("  STEP 1: DATA PREPARATION")
    print("█" * 60)

    # Determine which models will be trained, so we only prepare needed data
    if args.model == "all":
        models_needed = ["bdt", "cnn", "particlenet", "transformer"]
        if args.no_bdt:
            models_needed.remove("bdt")
    else:
        models_needed = [args.model]
        if not args.no_bdt and args.model == "all":
            models_needed.append("bdt")

    data = prepare_all_data(max_particles=args.max_particles, models_needed=models_needed)

    # Store all results for final comparison
    results_dict = {}
    histories = {}
    attn_data = None

    # ─── Step 2: BDT Baseline ─────────────────────────────────
    if not args.no_bdt and args.model in ("all", "bdt"):
        print("\n" + "█" * 60)
        print("  STEP 2: BDT BASELINE (XGBoost)")
        print("█" * 60)

        cached = load_predictions("bdt_baseline") if not args.force else None
        if cached is not None or args.eval_only:
            if cached is None:
                print("  [eval-only] no cached BDT predictions; skipping.")
            else:
                print(f"  [resume] using cached BDT predictions "
                      f"({_pred_path('bdt_baseline')})")
                results_dict["BDT (XGBoost)"] = cached
        else:
            bdt_results = train_bdt_baseline(
                data["train"]["jet_features_norm"],
                data["train"]["labels"],
                data["val"]["jet_features_norm"],
                data["val"]["labels"],
                data["test"]["jet_features_norm"],
                data["test"]["labels"],
                data["feature_names"],
            )

            n_params = f"{bdt_results['model'].num_boosted_rounds()} trees"
            results_dict["BDT (XGBoost)"] = {
                "labels": bdt_results["test_labels"],
                "probs": bdt_results["test_probs"],
                "n_params": n_params,
            }
            save_predictions("bdt_baseline",
                             bdt_results["test_probs"],
                             bdt_results["test_labels"],
                             n_params)

            # Plot feature importance
            from src.evaluation import plot_feature_importance
            plot_feature_importance(
                bdt_results["importance"],
                bdt_results["feature_names"]
            )

    # ─── Step 3: CNN on Jet Images ────────────────────────────
    if args.model in ("all", "cnn"):
        print("\n" + "█" * 60)
        print("  STEP 3: CNN ON JET IMAGES")
        print("█" * 60)

        cached = load_predictions("jet_cnn") if not args.force else None
        if cached is not None and (_trainer_done("jet_cnn") or args.eval_only):
            print(f"  [resume] using cached CNN predictions "
                  f"({_pred_path('jet_cnn')})")
            results_dict["CNN (Jet Images)"] = cached
        else:
            model_cnn = JetCNN()
            trainer_cnn = Trainer(model_cnn, "jet_cnn", CNN_CONFIG, model_type="cnn")

            if not args.eval_only:
                train_loader, val_loader, test_loader = make_dataloaders(data, "cnn")
                history = trainer_cnn.train(train_loader, val_loader)
                histories["CNN (Jet Images)"] = history

            trainer_cnn.load_best_model()
            _, _, test_loader = make_dataloaders(data, "cnn")
            test_probs, test_labels = trainer_cnn.predict(test_loader)

            n_params = model_cnn.count_parameters()
            results_dict["CNN (Jet Images)"] = {
                "labels": test_labels,
                "probs": test_probs,
                "n_params": n_params,
            }
            save_predictions("jet_cnn", test_probs, test_labels, n_params)

    # ─── Step 4: ParticleNet ──────────────────────────────────
    if args.model in ("all", "particlenet"):
        print("\n" + "█" * 60)
        print("  STEP 4: PARTICLENET (Graph Neural Network)")
        print("█" * 60)

        cached = load_predictions("particlenet") if not args.force else None
        if cached is not None and (_trainer_done("particlenet") or args.eval_only):
            print(f"  [resume] using cached ParticleNet predictions "
                  f"({_pred_path('particlenet')})")
            results_dict["ParticleNet"] = cached
        else:
            # Adjust input features for ParticleNet
            pnet_config = PARTICLENET_CONFIG.copy()
            pnet_config["input_features"] = NUM_PARTICLE_FEATURES  # 7 features

            model_pnet = ParticleNet(config=pnet_config)
            trainer_pnet = Trainer(model_pnet, "particlenet", PARTICLENET_CONFIG,
                                   model_type="particlenet")

            if not args.eval_only:
                train_loader, val_loader, test_loader = make_dataloaders(
                    data, "particlenet", args.max_particles)
                history = trainer_pnet.train(train_loader, val_loader)
                histories["ParticleNet"] = history

            trainer_pnet.load_best_model()
            _, _, test_loader = make_dataloaders(data, "particlenet", args.max_particles)
            test_probs, test_labels = trainer_pnet.predict(test_loader)

            n_params = model_pnet.count_parameters()
            results_dict["ParticleNet"] = {
                "labels": test_labels,
                "probs": test_probs,
                "n_params": n_params,
            }
            save_predictions("particlenet", test_probs, test_labels, n_params)

    # ─── Step 5: Particle Transformer ─────────────────────────
    if args.model in ("all", "transformer"):
        print("\n" + "█" * 60)
        print("  STEP 5: PARTICLE TRANSFORMER")
        print("█" * 60)

        cached = load_predictions("particle_transformer") if not args.force else None
        if cached is not None and (_trainer_done("particle_transformer") or args.eval_only):
            print(f"  [resume] using cached ParT predictions "
                  f"({_pred_path('particle_transformer')})")
            results_dict["Particle Transformer"] = cached

            # We still want attention maps for the plots; rebuild only if a
            # checkpoint exists.
            if os.path.exists(os.path.join(MODEL_DIR, "particle_transformer_best.pt")):
                model_pt = ParticleTransformer()
                trainer_pt = Trainer(model_pt, "particle_transformer",
                                     PARTFORMER_CONFIG, model_type="transformer")
                trainer_pt.load_best_model()
                _, _, test_loader = make_dataloaders(data, "transformer", args.max_particles)
                attn_data = trainer_pt.get_attention_maps(test_loader, n_samples=10)
        else:
            model_pt = ParticleTransformer()
            trainer_pt = Trainer(model_pt, "particle_transformer",
                                 PARTFORMER_CONFIG, model_type="transformer")

            if not args.eval_only:
                train_loader, val_loader, test_loader = make_dataloaders(
                    data, "transformer", args.max_particles)
                history = trainer_pt.train(train_loader, val_loader)
                histories["Particle Transformer"] = history

            trainer_pt.load_best_model()
            _, _, test_loader = make_dataloaders(data, "transformer", args.max_particles)
            test_probs, test_labels = trainer_pt.predict(test_loader)

            n_params = model_pt.count_parameters()
            results_dict["Particle Transformer"] = {
                "labels": test_labels,
                "probs": test_probs,
                "n_params": n_params,
            }
            save_predictions("particle_transformer", test_probs, test_labels, n_params)

            # Extract attention maps for interpretability
            print("\nExtracting attention maps for interpretability analysis...")
            attn_data = trainer_pt.get_attention_maps(test_loader, n_samples=10)

    # ─── Step 6: Full Evaluation ──────────────────────────────
    print("\n" + "█" * 60)
    print("  STEP 6: EVALUATION & PUBLICATION PLOTS")
    print("█" * 60)

    attn = attn_data

    run_full_evaluation(
        results_dict=results_dict,
        data=data,
        histories=histories if histories else None,
        attn_data=attn,
    )

    # ─── Done ─────────────────────────────────────────────────
    total_time = time.time() - total_start
    print("\n" + "╔" + "═" * 62 + "╗")
    print(f"║  PIPELINE COMPLETE                                           ║")
    print(f"║  Total time: {total_time/60:.1f} minutes                                    ║")
    print(f"║  Results: {RESULTS_DIR:<50} ║")
    print(f"║  Plots:   {os.path.join(RESULTS_DIR, 'plots'):<50} ║")
    print("╚" + "═" * 62 + "╝")

    print("\n  Generated outputs:")
    print("  ├── results/model_comparison.json    (metrics)")
    print("  ├── results/comparison_table.tex     (LaTeX table)")
    print("  ├── results/models/                  (saved checkpoints)")
    print("  └── results/plots/")
    print("      ├── roc_comparison.pdf           (ROC curves)")
    print("      ├── score_distributions.pdf      (classifier output)")
    print("      ├── confusion_matrices.pdf       (confusion matrices)")
    print("      ├── training_history.pdf         (loss & AUC curves)")
    print("      ├── jet_substructure.pdf         (physics variables)")
    print("      ├── jet_images.pdf               (example jet images)")
    print("      ├── average_jet_images.pdf       (averaged images)")
    print("      ├── attention_maps.pdf           (transformer attention)")
    print("      ├── cls_attention.pdf            (CLS attention vs ΔR)")
    print("      └── bdt_importance.pdf           (feature importance)")


if __name__ == "__main__":
    main()
