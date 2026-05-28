"""
Research-angle utilities: data-efficiency, calibration, and uncertainty.
========================================================================
Beyond the bare reproduction of model rankings, this module provides the
metrics and plots that turn the pipeline into a small research study:

  1. Data-efficiency curves : AUC and background-rejection as a function
     of the training-set size (proxy for how cheaply each architecture
     can be trained on a budget — important when MC simulation is
     expensive).

  2. Probability calibration : reliability diagrams + Expected
     Calibration Error (ECE). A high-AUC classifier is not automatically
     well-calibrated; if downstream physics uses the score as a
     probability (e.g., for likelihood fits), the calibration matters
     more than the ranking.

  3. Temperature scaling : a single-parameter post-hoc recalibration
     baseline (Guo et al., 2017) shown alongside the raw scores.

  4. Per-class bootstrap uncertainty on the AUC (so the comparison
     between models comes with error bars).
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR

PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ---------------------------------------------------------------- helpers
def bg_rejection_at(labels, probs, target_eff):
    """Background rejection 1/eps_B at fixed signal efficiency eps_S."""
    fpr, tpr, _ = roc_curve(labels, probs)
    idx = np.argmin(np.abs(tpr - target_eff))
    return float("inf") if fpr[idx] == 0 else 1.0 / fpr[idx]


def auc_with_bootstrap(labels, probs, n_boot=200, rng=None):
    """Bootstrap mean and 1-sigma uncertainty on the AUC."""
    rng = rng or np.random.default_rng(0)
    n = len(labels)
    aucs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        # Skip degenerate single-class samples
        if labels[idx].sum() in (0, n):
            aucs[i] = np.nan
            continue
        aucs[i] = roc_auc_score(labels[idx], probs[idx])
    aucs = aucs[~np.isnan(aucs)]
    return float(np.mean(aucs)), float(np.std(aucs))


def expected_calibration_error(labels, probs, n_bins=15):
    """
    Expected Calibration Error (Guo et al. 2017, eq.3).
    ECE = sum_b |B_b|/N * |acc(B_b) - conf(B_b)|
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs >= lo) & (probs < hi if hi < 1.0 else probs <= hi)
        if not m.any():
            continue
        acc = labels[m].mean()
        conf = probs[m].mean()
        ece += m.sum() / n * abs(acc - conf)
    return float(ece)


def reliability_curve(labels, probs, n_bins=15):
    """Per-bin mean-confidence and mean-accuracy for a reliability diagram."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    confs, accs, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs >= lo) & (probs < hi if hi < 1.0 else probs <= hi)
        if not m.any():
            continue
        confs.append(float(probs[m].mean()))
        accs.append(float(labels[m].mean()))
        counts.append(int(m.sum()))
    return np.array(confs), np.array(accs), np.array(counts)


# ---------------------------------------------------------------- temperature scaling
def fit_temperature(val_labels, val_logits, max_iter=200, lr=0.05):
    """
    Find a single scalar T > 0 that minimises NLL of
    sigma(logits / T) on validation data.
    Returns T as a float.
    """
    import torch
    t = torch.tensor([1.0], requires_grad=True)
    lg = torch.from_numpy(np.asarray(val_logits, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(val_labels, dtype=np.float32))
    opt = torch.optim.LBFGS([t], lr=lr, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lg / t.clamp(min=1e-3), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.detach().clamp(min=1e-3).item())


def logits_from_probs(probs, eps=1e-7):
    """Recover logits from probabilities (inverse sigmoid)."""
    p = np.clip(probs, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


# ---------------------------------------------------------------- plots
def plot_learning_curves(study, save_name="learning_curves"):
    """
    Plot AUC and 1/eps_B@50% versus N_train for each model.

    study : dict
        { model_name: {
            "n_train": [...],
            "auc": [...],
            "auc_err": [...],
            "bg_rej_50": [...],
            "bg_rej_30": [...],
        }, ... }
    """
    from src.evaluation import MODEL_COLORS

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for name, s in study.items():
        c = MODEL_COLORS.get(name, None)
        axes[0].errorbar(
            s["n_train"], s["auc"], yerr=s.get("auc_err"),
            marker="o", color=c, label=name, capsize=3, linewidth=1.6
        )
        axes[1].plot(
            s["n_train"], s["bg_rej_50"], marker="s",
            color=c, label=name, linewidth=1.6
        )

    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"Training jets $N_{\mathrm{train}}$")
    axes[0].set_ylabel("Test AUC")
    axes[0].set_title("Data-efficiency: AUC vs training-set size")
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].legend(loc="lower right", fontsize=10)

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"Training jets $N_{\mathrm{train}}$")
    axes[1].set_ylabel(r"Background rejection $1/\varepsilon_B$ at $\varepsilon_S=0.5$")
    axes[1].set_title("Data-efficiency: rejection at 50% signal eff.")
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"{save_name}.pdf")
    plt.savefig(path)
    plt.savefig(path.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {path}")


def plot_calibration(calibration, save_name="calibration"):
    """
    Reliability diagrams (top row) + score histograms (bottom row)
    for each model. `calibration` is a dict
        { model_name: {
            "confs": np.ndarray, "accs": np.ndarray, "counts": np.ndarray,
            "ece": float, "T": float, "ece_tscale": float,
            "probs": np.ndarray, "labels": np.ndarray
        }, ... }
    """
    from src.evaluation import MODEL_COLORS
    names = list(calibration.keys())
    n = len(names)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 8.5), sharey="row")
    if n == 1:
        axes = axes[:, None]

    for j, name in enumerate(names):
        c = MODEL_COLORS.get(name, "#444")
        cal = calibration[name]

        ax = axes[0, j]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
        ax.plot(cal["confs"], cal["accs"], marker="o", color=c,
                linewidth=1.8, label=f"ECE = {cal['ece']:.3f}")
        if "confs_tscale" in cal:
            ax.plot(cal["confs_tscale"], cal["accs_tscale"], marker="s",
                    color=c, linestyle=":", alpha=0.7,
                    label=f"T-scaled (T={cal['T']:.2f}, ECE={cal['ece_tscale']:.3f})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        if j == 0:
            ax.set_ylabel("Empirical fraction of top jets")
        ax.set_title(name, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper left")

        ax2 = axes[1, j]
        bins = np.linspace(0, 1, 41)
        labels = cal["labels"]
        probs = cal["probs"]
        ax2.hist(probs[labels == 0], bins=bins, alpha=0.55,
                 color="#1f77b4", label="QCD", density=True,
                 histtype="stepfilled")
        ax2.hist(probs[labels == 1], bins=bins, alpha=0.55,
                 color="#d62728", label="Top", density=True,
                 histtype="stepfilled")
        ax2.set_xlabel("Classifier score")
        if j == 0:
            ax2.set_ylabel("Normalised counts")
        ax2.set_yscale("log")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

    plt.suptitle("Probability calibration", y=1.02, fontsize=15)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"{save_name}.pdf")
    plt.savefig(path)
    plt.savefig(path.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {path}")


def save_study_json(study, calibration, path=None):
    """Persist the study + calibration summary as JSON for the article."""
    path = path or os.path.join(RESULTS_DIR, "study_summary.json")
    payload = {
        "data_efficiency": {
            name: {k: (list(v) if hasattr(v, "__iter__") and not isinstance(v, str) else v)
                   for k, v in s.items()}
            for name, s in study.items()
        },
        "calibration": {
            name: {
                "ece": cal["ece"],
                "T": cal["T"],
                "ece_tscale": cal["ece_tscale"],
                "n_samples": int(len(cal["labels"])),
            }
            for name, cal in calibration.items()
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {path}")
    return path
