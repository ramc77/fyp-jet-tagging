#!/usr/bin/env python3
"""
Data-efficiency and calibration study for top-tagging models.
=============================================================
Trains each architecture at multiple training-set sizes, computes
test-set AUC (with bootstrap uncertainty) and 1/eps_B at fixed signal
efficiency, fits a temperature scaling on the validation set, then
plots learning curves + reliability diagrams.

This is the research-angle complement to run_full_pipeline.py.

Usage:
  python run_study.py                     # default sweep
  python run_study.py --sizes 5000 10000 25000
  python run_study.py --models bdt cnn    # subset of models
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import copy
import json
import time
import numpy as np
import torch

import sys
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from config import (
    DEVICE, MODEL_DIR, RESULTS_DIR,
    CNN_CONFIG, PARTICLENET_CONFIG, PARTFORMER_CONFIG, BDT_CONFIG,
    NUM_PARTICLE_FEATURES,
)
from src.data_utils import prepare_all_data, make_dataloaders, JetImageDataset, ParticleDataset
from src.jet_cnn import JetCNN
from src.particle_net import ParticleNet
from src.particle_transformer import ParticleTransformer
from src.trainer import Trainer, train_bdt_baseline
from src.study_utils import (
    auc_with_bootstrap, bg_rejection_at,
    expected_calibration_error, reliability_curve,
    fit_temperature, logits_from_probs,
    plot_learning_curves, plot_calibration, save_study_json,
)
from torch.utils.data import DataLoader

ALL_MODELS = ["bdt", "cnn", "particlenet", "transformer"]

STUDY_PRED_DIR = os.path.join(RESULTS_DIR, "study_predictions")
os.makedirs(STUDY_PRED_DIR, exist_ok=True)


def _study_pred_path(model, n):
    return os.path.join(STUDY_PRED_DIR, f"{model}_n{n}.npz")


def _load_study_run(model, n):
    """Return a previously-saved run for (model, n_train), or None."""
    p = _study_pred_path(model, n)
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    return {
        "val_probs": d["val_probs"], "val_labels": d["val_labels"],
        "test_probs": d["test_probs"], "test_labels": d["test_labels"],
        "n_params": str(d["n_params"]),
    }


def _save_study_run(model, n, r):
    np.savez_compressed(
        _study_pred_path(model, n),
        val_probs=np.asarray(r["val_probs"], dtype=np.float32),
        val_labels=np.asarray(r["val_labels"], dtype=np.float32),
        test_probs=np.asarray(r["test_probs"], dtype=np.float32),
        test_labels=np.asarray(r["test_labels"], dtype=np.float32),
        n_params=str(r["n_params"]),
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int,
                   default=[5_000, 10_000, 25_000, 50_000],
                   help="Training-set sizes to sweep over.")
    p.add_argument("--models", nargs="+", default=ALL_MODELS,
                   choices=ALL_MODELS)
    p.add_argument("--epochs-cap", type=int, default=15,
                   help="Cap per-size epochs so the sweep finishes in reasonable time.")
    p.add_argument("--max-particles", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true",
                   help="Re-train every (model, N) cell even if a cached "
                        "result exists. Without --force, cached cells are "
                        "reused so the study can be run in parts.")
    return p.parse_args()


# -------------------------------------------------------------- subset helpers
def subset_indices(n_total, n_take, rng):
    """Stratification by index order is fine here — labels are already shuffled
    in the upstream HDF5. We just take the first n_take indices."""
    return np.arange(min(n_take, n_total))


def loaders_for_size(data, model_type, n_take, max_particles=100):
    """Build train/val/test DataLoaders restricted to the first n_take training jets."""
    # validation/test are kept at full SUBSET size for a stable estimate
    if model_type == "cnn":
        from config import CNN_CONFIG as CFG
        idx = np.arange(min(n_take, len(data["train"]["labels"])))
        train_ds = JetImageDataset(
            data["train"]["images"][idx], data["train"]["labels"][idx])
        val_ds = JetImageDataset(data["val"]["images"], data["val"]["labels"])
        test_ds = JetImageDataset(data["test"]["images"], data["test"]["labels"])
        bs = CFG["batch_size"]
    elif model_type == "particlenet":
        from config import PARTICLENET_CONFIG as CFG
        idx = np.arange(min(n_take, len(data["train"]["labels"])))
        train_ds = ParticleDataset(
            data["train"]["features_norm"][idx], data["train"]["mask"][idx],
            data["train"]["labels"][idx], max_particles=max_particles)
        val_ds = ParticleDataset(
            data["val"]["features_norm"], data["val"]["mask"],
            data["val"]["labels"], max_particles=max_particles)
        test_ds = ParticleDataset(
            data["test"]["features_norm"], data["test"]["mask"],
            data["test"]["labels"], max_particles=max_particles)
        bs = CFG["batch_size"]
    elif model_type == "transformer":
        from config import PARTFORMER_CONFIG as CFG
        idx = np.arange(min(n_take, len(data["train"]["labels"])))
        train_ds = ParticleDataset(
            data["train"]["features_norm"][idx], data["train"]["mask"][idx],
            data["train"]["labels"][idx], max_particles=max_particles,
            compute_pairs=True, features_raw=data["train"]["features"][idx])
        val_ds = ParticleDataset(
            data["val"]["features_norm"], data["val"]["mask"],
            data["val"]["labels"], max_particles=max_particles,
            compute_pairs=True, features_raw=data["val"]["features"])
        test_ds = ParticleDataset(
            data["test"]["features_norm"], data["test"]["mask"],
            data["test"]["labels"], max_particles=max_particles,
            compute_pairs=True, features_raw=data["test"]["features"])
        bs = CFG["batch_size"]
    else:
        raise ValueError(model_type)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
    return train_loader, val_loader, test_loader


# -------------------------------------------------------------- per-model run
def train_neural(model_type, n_train, data, max_particles, epochs_cap):
    """Train one of CNN / ParticleNet / Particle Transformer at given N_train."""
    if model_type == "cnn":
        cfg = copy.deepcopy(CNN_CONFIG); model = JetCNN(cfg)
    elif model_type == "particlenet":
        cfg = copy.deepcopy(PARTICLENET_CONFIG)
        cfg["input_features"] = NUM_PARTICLE_FEATURES
        model = ParticleNet(config=cfg)
    elif model_type == "transformer":
        cfg = copy.deepcopy(PARTFORMER_CONFIG); model = ParticleTransformer(config=cfg)
    else:
        raise ValueError(model_type)
    cfg["epochs"] = min(cfg["epochs"], epochs_cap)
    name = f"{model_type}_n{n_train}"
    tr = Trainer(model, name, cfg, model_type=model_type)
    tl, vl, te = loaders_for_size(data, model_type, n_train, max_particles)
    history = tr.train(tl, vl)
    tr.load_best_model()
    val_probs, val_labels = tr.predict(vl)
    test_probs, test_labels = tr.predict(te)
    return {
        "val_probs": val_probs, "val_labels": val_labels,
        "test_probs": test_probs, "test_labels": test_labels,
        "history": history, "n_params": model.count_parameters(),
    }


def train_bdt_size(n_train, data):
    """Train BDT on n_train jets."""
    idx = np.arange(min(n_train, len(data["train"]["labels"])))
    res = train_bdt_baseline(
        data["train"]["jet_features_norm"][idx],
        data["train"]["labels"][idx],
        data["val"]["jet_features_norm"], data["val"]["labels"],
        data["test"]["jet_features_norm"], data["test"]["labels"],
        data["feature_names"],
    )
    return {
        "val_probs": res["val_probs"], "val_labels": res["val_labels"],
        "test_probs": res["test_probs"], "test_labels": res["test_labels"],
        "n_params": f"{res['model'].num_boosted_rounds()} trees",
    }


# -------------------------------------------------------------- main
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"Device: {DEVICE}")
    print(f"Sizes : {args.sizes}")
    print(f"Models: {args.models}")

    models_needed = []
    if "bdt" in args.models: models_needed.append("bdt")
    if "cnn" in args.models: models_needed.append("cnn")
    if "particlenet" in args.models: models_needed.append("particlenet")
    if "transformer" in args.models: models_needed.append("transformer")

    data = prepare_all_data(max_particles=args.max_particles,
                            models_needed=models_needed)

    pretty = {
        "bdt": "BDT (XGBoost)", "cnn": "CNN (Jet Images)",
        "particlenet": "ParticleNet", "transformer": "Particle Transformer",
    }

    study = {pretty[m]: {"n_train": [], "auc": [], "auc_err": [],
                         "bg_rej_50": [], "bg_rej_30": []}
             for m in args.models}
    # Calibration uses the largest-size run only
    calibration = {}

    t0 = time.time()
    for m in args.models:
        print("\n" + "#" * 70)
        print(f"# Model: {pretty[m]}")
        print("#" * 70)
        last_run = None
        for n in args.sizes:
            print(f"\n--- {pretty[m]}: N_train = {n} ---")
            cached = None if args.force else _load_study_run(m, n)
            if cached is not None:
                print(f"    [resume] cached run found ({_study_pred_path(m, n)})")
                r = cached
            elif m == "bdt":
                r = train_bdt_size(n, data)
                _save_study_run(m, n, r)
            else:
                r = train_neural(m, n, data, args.max_particles, args.epochs_cap)
                _save_study_run(m, n, r)
            auc, auc_err = auc_with_bootstrap(
                r["test_labels"], r["test_probs"], rng=rng)
            rej50 = bg_rejection_at(r["test_labels"], r["test_probs"], 0.5)
            rej30 = bg_rejection_at(r["test_labels"], r["test_probs"], 0.3)
            print(f"    AUC = {auc:.4f} +- {auc_err:.4f} "
                  f"| 1/eps_B@0.5 = {rej50:.1f} | @0.3 = {rej30:.1f}")
            study[pretty[m]]["n_train"].append(n)
            study[pretty[m]]["auc"].append(auc)
            study[pretty[m]]["auc_err"].append(auc_err)
            study[pretty[m]]["bg_rej_50"].append(rej50)
            study[pretty[m]]["bg_rej_30"].append(rej30)
            last_run = r

        # Calibration analysis on the largest-size run
        if last_run is not None:
            val_logits = logits_from_probs(last_run["val_probs"])
            T = fit_temperature(last_run["val_labels"], val_logits)
            probs = last_run["test_probs"]
            labels = last_run["test_labels"]
            ece = expected_calibration_error(labels, probs)
            confs, accs, counts = reliability_curve(labels, probs)
            # Temperature-scaled probs
            test_logits = logits_from_probs(probs)
            probs_T = 1.0 / (1.0 + np.exp(-test_logits / max(T, 1e-3)))
            ece_T = expected_calibration_error(labels, probs_T)
            confs_T, accs_T, counts_T = reliability_curve(labels, probs_T)
            calibration[pretty[m]] = {
                "probs": probs, "labels": labels,
                "confs": confs, "accs": accs, "counts": counts,
                "confs_tscale": confs_T, "accs_tscale": accs_T,
                "counts_tscale": counts_T,
                "ece": ece, "T": T, "ece_tscale": ece_T,
            }

    print(f"\nStudy wall-time: {(time.time()-t0)/60:.1f} min")

    plot_learning_curves(study)
    plot_calibration(calibration)
    save_study_json(study, calibration)

    # also dump a flat CSV for the LaTeX article
    csv_path = os.path.join(RESULTS_DIR, "study_summary.csv")
    with open(csv_path, "w") as f:
        f.write("model,n_train,auc,auc_err,bg_rej_50,bg_rej_30\n")
        for name, s in study.items():
            for i in range(len(s["n_train"])):
                f.write(f"{name},{s['n_train'][i]},{s['auc'][i]:.5f},"
                        f"{s['auc_err'][i]:.5f},{s['bg_rej_50'][i]:.1f},"
                        f"{s['bg_rej_30'][i]:.1f}\n")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
