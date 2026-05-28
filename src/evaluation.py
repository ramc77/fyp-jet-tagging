"""
Evaluation and Publication-Quality Plotting
=============================================
Produces all plots and metrics needed for thesis and publications:
  1. ROC curves with AUC comparison
  2. Background rejection vs signal efficiency
  3. Score distributions for signal and background
  4. Training history curves
  5. Jet substructure variable distributions
  6. Attention map visualizations
  7. Feature importance (BDT)
  8. Confusion matrices
  9. Model comparison summary table
  10. Jet image examples

All plots follow CMS/ATLAS publication standards using mplhep.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, PLOT_STYLE, TOP_MASS, W_MASS

# Apply plot style
plt.rcParams.update(PLOT_STYLE)

# Try to use CMS style from mplhep
try:
    import mplhep as hep
    plt.style.use(hep.style.CMS)
    HAS_MPLHEP = True
except ImportError:
    HAS_MPLHEP = False
    print("mplhep not available, using default matplotlib style")

# Color scheme for models
MODEL_COLORS = {
    "BDT (XGBoost)": "#1f77b4",
    "CNN (Jet Images)": "#ff7f0e",
    "ParticleNet": "#2ca02c",
    "Particle Transformer": "#d62728",
}

PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 1. CORE METRICS
# ═══════════════════════════════════════════════════════════════

def compute_metrics(labels, probs, threshold=0.5):
    """
    Compute comprehensive classification metrics.

    Physics-relevant metrics:
      - AUC-ROC: Overall discrimination power
      - 1/εB at εS=50%: Background rejection at 50% signal efficiency
        (how many background jets are rejected while keeping 50% of signal)
      - 1/εB at εS=30%: Same at 30% signal efficiency (high-purity regime)
    """
    preds = (probs > threshold).astype(int)

    # Standard metrics
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    # ROC
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    # Background rejection at fixed signal efficiency
    def bg_rejection_at_sig_eff(target_eff):
        idx = np.argmin(np.abs(tpr - target_eff))
        if fpr[idx] > 0:
            return 1.0 / fpr[idx]
        return float("inf")

    rej_50 = bg_rejection_at_sig_eff(0.5)
    rej_30 = bg_rejection_at_sig_eff(0.3)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "bg_rej_50": rej_50,
        "bg_rej_30": rej_30,
    }


# ═══════════════════════════════════════════════════════════════
# 2. ROC CURVES
# ═══════════════════════════════════════════════════════════════

def plot_roc_curves(results_dict, save_name="roc_comparison"):
    """
    Publication-quality ROC curve comparison.

    Args:
        results_dict: {model_name: {"labels": array, "probs": array}}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left: Standard ROC ---
    for name, res in results_dict.items():
        metrics = compute_metrics(res["labels"], res["probs"])
        color = MODEL_COLORS.get(name, None)
        ax1.plot(
            metrics["fpr"], metrics["tpr"],
            label=f'{name} (AUC = {metrics["auc"]:.4f})',
            color=color, linewidth=2,
        )

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax1.set_xlabel("False Positive Rate (Background Efficiency)")
    ax1.set_ylabel("True Positive Rate (Signal Efficiency)")
    ax1.set_title("ROC Curves: Top Quark Tagging")
    ax1.legend(loc="lower right", fontsize=11)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.02])
    ax1.grid(True, alpha=0.3)

    # --- Right: Background Rejection (1/εB) vs Signal Efficiency (εS) ---
    for name, res in results_dict.items():
        metrics = compute_metrics(res["labels"], res["probs"])
        fpr = metrics["fpr"]
        tpr = metrics["tpr"]

        # Avoid division by zero
        valid = fpr > 0
        color = MODEL_COLORS.get(name, None)
        ax2.semilogy(
            tpr[valid], 1.0 / fpr[valid],
            label=f'{name} (1/εB@50% = {metrics["bg_rej_50"]:.0f})',
            color=color, linewidth=2,
        )

    ax2.set_xlabel("Signal Efficiency (εS)")
    ax2.set_ylabel("Background Rejection (1/εB)")
    ax2.set_title("Background Rejection vs Signal Efficiency")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.set_xlim([0.2, 1.0])
    ax2.set_ylim([1, 1e5])
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, f"{save_name}.pdf")
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════
# 3. SCORE DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════

def plot_score_distributions(results_dict, save_name="score_distributions"):
    """
    Classifier output score distributions for signal and background.
    Shows the separation between top jets and QCD jets.
    """
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results_dict.items()):
        labels = res["labels"]
        probs = res["probs"]

        sig_mask = labels == 1
        bkg_mask = labels == 0

        bins = np.linspace(0, 1, 51)
        ax.hist(probs[sig_mask], bins=bins, alpha=0.6, label="Top (signal)",
                color="#d62728", density=True, histtype="stepfilled")
        ax.hist(probs[bkg_mask], bins=bins, alpha=0.6, label="QCD (background)",
                color="#1f77b4", density=True, histtype="stepfilled")

        ax.set_xlabel("Classifier Score")
        ax.set_ylabel("Normalized Counts")
        ax.set_title(name)
        ax.legend()
        ax.set_yscale("log")

    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, f"{save_name}.pdf")
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════
# 4. TRAINING HISTORY
# ═══════════════════════════════════════════════════════════════

def plot_training_history(histories, save_name="training_history"):
    """
    Plot training and validation loss/AUC curves.
    Useful for diagnosing overfitting, learning rate issues, etc.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for name, hist in histories.items():
        color = MODEL_COLORS.get(name, None)
        epochs = range(1, len(hist["train_loss"]) + 1)

        # Loss
        axes[0].plot(epochs, hist["train_loss"], "--", color=color, alpha=0.5)
        axes[0].plot(epochs, hist["val_loss"], "-", color=color, label=name, linewidth=2)

        # AUC
        axes[1].plot(epochs, hist["val_auc"], "-", color=color, label=name, linewidth=2)

        # Learning rate
        axes[2].plot(epochs, hist["lr"], "-", color=color, label=name, linewidth=2)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (BCE)")
    axes[0].set_title("Training (dashed) & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_title("Validation AUC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].legend()
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, f"{save_name}.pdf")
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════
# 5. JET SUBSTRUCTURE DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════

def plot_jet_substructure(jet_features, labels, feature_names,
                          save_name="jet_substructure"):
    """
    Plot distributions of jet substructure variables for signal vs background.

    These are the classic physics plots that show WHY the classifiers work:
    - Top jets have higher mass (near 173 GeV)
    - Top jets have lower τ₂₁ (3-prong vs 1-prong structure)
    - Top jets have more constituents (more decay products)
    """
    sig_mask = labels == 1
    bkg_mask = labels == 0

    n_feats = len(feature_names)
    n_cols = 4
    n_rows = (n_feats + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    physics_labels = {
        "jet_mass": r"Jet Mass [GeV]",
        "jet_pT": r"Jet $p_T$ [GeV]",
        "jet_eta": r"Jet $\eta$",
        "n_constituents": r"$N_{\mathrm{constituents}}$",
        "jet_width": r"Jet Width",
        "lead_pT_frac": r"Leading $p_T$ Fraction",
        "sublead_pT_frac": r"Sub-leading $p_T$ Fraction",
        "tau1": r"$\tau_1$",
        "tau2": r"$\tau_2$",
        "tau3": r"$\tau_3$",
        "tau21": r"$\tau_{21} = \tau_2/\tau_1$",
        "tau32": r"$\tau_{32} = \tau_3/\tau_2$",
        "C2": r"$C_2$ (Energy Correlation)",
    }

    for i, fname in enumerate(feature_names):
        ax = axes[i]
        sig_vals = jet_features[sig_mask, i]
        bkg_vals = jet_features[bkg_mask, i]

        # Remove outliers for better visualization
        vmin = np.percentile(np.concatenate([sig_vals, bkg_vals]), 1)
        vmax = np.percentile(np.concatenate([sig_vals, bkg_vals]), 99)
        bins = np.linspace(vmin, vmax, 50)

        ax.hist(sig_vals, bins=bins, alpha=0.6, label="Top (signal)",
                color="#d62728", density=True, histtype="stepfilled")
        ax.hist(bkg_vals, bins=bins, alpha=0.6, label="QCD (background)",
                color="#1f77b4", density=True, histtype="stepfilled")

        xlabel = physics_labels.get(fname, fname)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized")
        ax.legend(fontsize=8)

        # Mark known masses
        if fname == "jet_mass":
            ax.axvline(TOP_MASS, color="red", linestyle=":", alpha=0.7, label=f"$m_t$={TOP_MASS}")
            ax.axvline(W_MASS, color="blue", linestyle=":", alpha=0.7, label=f"$m_W$={W_MASS}")
            ax.legend(fontsize=8)

    # Hide unused axes
    for i in range(n_feats, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Jet Substructure Variables: Top vs QCD", fontsize=16, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, f"{save_name}.pdf")
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════
# 6. JET IMAGE VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def plot_jet_images(images, labels, save_name="jet_images"):
    """
    Visualize example jet images for top quarks and QCD jets.
    Shows the characteristic 3-prong (top) vs 1-prong (QCD) structure.
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    channel_names = [r"$p_T$ weighted", "Multiplicity", r"$p_T^2$ weighted"]

    # Top 5 signal examples
    sig_idx = np.where(labels == 1)[0][:5]
    for j, idx in enumerate(sig_idx):
        # Show pT channel (most informative)
        im = axes[0, j].imshow(
            images[idx, 0], cmap="hot", origin="lower",
            extent=[-1, 1, -1, 1]
        )
        axes[0, j].set_title(f"Top Jet #{j+1}", fontsize=12)
        if j == 0:
            axes[0, j].set_ylabel(r"$\Delta\phi$", fontsize=14)
        axes[0, j].set_xlabel(r"$\Delta\eta$", fontsize=12)

    # Bottom 5 QCD examples
    bkg_idx = np.where(labels == 0)[0][:5]
    for j, idx in enumerate(bkg_idx):
        im = axes[1, j].imshow(
            images[idx, 0], cmap="hot", origin="lower",
            extent=[-1, 1, -1, 1]
        )
        axes[1, j].set_title(f"QCD Jet #{j+1}", fontsize=12)
        if j == 0:
            axes[1, j].set_ylabel(r"$\Delta\phi$", fontsize=14)
        axes[1, j].set_xlabel(r"$\Delta\eta$", fontsize=12)

    plt.suptitle(
        r"Jet Images ($p_T$-weighted): Top Quarks (top) vs QCD (bottom)",
        fontsize=16, y=1.02
    )
    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, f"{save_name}.pdf")
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {save_path}")


def plot_average_jet_images(images, labels, save_name="average_jet_images"):
    """
    Average jet images for signal and background.
    These are extremely useful for understanding what the CNN sees:
    - Top quarks show clear 3-prong structure in the average
    - QCD shows a single central blob
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sig_mask = labels == 1
    bkg_mask = labels == 0

    avg_sig = images[sig_mask, 0].mean(axis=0)
    avg_bkg = images[bkg_mask, 0].mean(axis=0)
    avg_diff = avg_sig - avg_bkg

    titles = ["Average Top Jet", "Average QCD Jet", "Difference (Top − QCD)"]
    data = [avg_sig, avg_bkg, avg_diff]
    cmaps = ["hot", "hot", "RdBu_r"]

    for ax, d, title, cmap in zip(axes, data, titles, cmaps):
        if cmap == "RdBu_r":
            vmax = np.abs(d).max()
            im = ax.imshow(d, cmap=cmap, origin="lower", extent=[-1, 1, -1, 1],
                          vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(d, cmap=cmap, origin="lower", extent=[-1, 1, -1, 1])
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(r"$\Delta\eta$")
        ax.set_ylabel(r"$\Delta\phi$")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle(r"Average Jet Images ($p_T$-weighted channel)", fontsize=16, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, f"{save_name}.pdf")
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════
# 7. ATTENTION MAP VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def plot_attention_maps(attn_maps, features, labels, save_name="attention_maps"):
    """
    Visualize attention patterns from the Particle Transformer.

    The attention weights show which particle pairs the model considers
    most important. For top quark jets, we expect attention to focus
    on the three subjets from the t→Wb→qqb decay.
    """
    n_samples = min(4, len(attn_maps))
    n_layers = len(attn_maps[0])

    fig, axes = plt.subplots(n_samples, n_layers, figsize=(5 * n_layers, 5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_samples):
        label_str = "Top" if labels[i] == 1 else "QCD"
        for j in range(n_layers):
            # Average over attention heads
            attn = attn_maps[i][j].mean(axis=0)  # (N+1, N+1)
            # Show particle-to-particle attention (skip CLS token row/col)
            attn_pp = attn[1:, 1:]

            # Only show top 50 particles for clarity
            n_show = min(50, attn_pp.shape[0])
            im = axes[i, j].imshow(
                attn_pp[:n_show, :n_show], cmap="viridis",
                aspect="auto"
            )
            axes[i, j].set_title(f"{label_str} Jet | Layer {j+1}", fontsize=11)
            if j == 0:
                axes[i, j].set_ylabel(f"Sample {i+1}\nParticle index")
            axes[i, j].set_xlabel("Particle index")

    plt.suptitle("Particle Transformer Attention Maps (head-averaged)", fontsize=16, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, f"{save_name}.pdf")
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {save_path}")


def plot_cls_attention(attn_maps, features, labels, save_name="cls_attention"):
    """
    Visualize [CLS] token attention weights as a function of particle ΔR from jet axis.

    This shows which particles the classifier "looks at" most strongly.
    For top quarks, we expect attention peaks at the three subjets.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, label_val, label_str in zip(axes, [1, 0], ["Top Jets", "QCD Jets"]):
        for i in range(len(attn_maps)):
            if labels[i] != label_val:
                continue

            # CLS attention from last layer, averaged over heads
            # CLS is at position 0, attending to positions 1..N
            last_layer_attn = attn_maps[i][-1].mean(axis=0)  # (N+1, N+1)
            cls_attn = last_layer_attn[0, 1:]  # Attention from CLS to particles

            # Compute ΔR for each particle
            feats = features[i]
            delta_eta = feats[:len(cls_attn), 4] if feats.shape[1] > 4 else feats[:len(cls_attn), 0]
            delta_phi = feats[:len(cls_attn), 5] if feats.shape[1] > 5 else feats[:len(cls_attn), 1]
            dR = np.sqrt(delta_eta**2 + delta_phi**2)

            # Only non-padded particles
            valid = dR > 0
            ax.scatter(dR[valid], cls_attn[valid], alpha=0.3, s=10)

        ax.set_xlabel(r"$\Delta R$ from jet axis")
        ax.set_ylabel("[CLS] Attention Weight")
        ax.set_title(f"{label_str}: [CLS] Attention vs Particle $\\Delta R$")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, f"{save_name}.pdf")
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════
# 8. BDT FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════

def plot_feature_importance(importance, feature_names, save_name="bdt_importance"):
    """
    Plot XGBoost feature importance (gain-based).
    Shows which jet substructure variables are most discriminating.
    """
    # Sort by importance
    gains = [importance.get(f, 0) for f in feature_names]
    sorted_idx = np.argsort(gains)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = range(len(feature_names))

    physics_labels = {
        "jet_mass": r"Jet Mass",
        "jet_pT": r"Jet $p_T$",
        "jet_eta": r"Jet $\eta$",
        "n_constituents": r"$N_{\mathrm{const}}$",
        "jet_width": "Jet Width",
        "lead_pT_frac": r"Lead $p_T$ frac",
        "sublead_pT_frac": r"Sub-lead $p_T$ frac",
        "tau1": r"$\tau_1$",
        "tau2": r"$\tau_2$",
        "tau3": r"$\tau_3$",
        "tau21": r"$\tau_{21}$",
        "tau32": r"$\tau_{32}$",
        "C2": r"$C_2$",
    }

    sorted_names = [physics_labels.get(feature_names[i], feature_names[i]) for i in sorted_idx]
    sorted_gains = [gains[i] for i in sorted_idx]

    ax.barh(y_pos, sorted_gains, color="#2ca02c", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("BDT Feature Importance: Jet Substructure Variables")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, f"{save_name}.pdf")
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════
# 9. CONFUSION MATRICES
# ═══════════════════════════════════════════════════════════════

def plot_confusion_matrices(results_dict, save_name="confusion_matrices"):
    """Plot confusion matrices for all models side by side."""
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results_dict.items()):
        labels = res["labels"]
        preds = (res["probs"] > 0.5).astype(int)
        cm = confusion_matrix(labels, preds)

        # Normalize
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["QCD", "Top"])
        ax.set_yticklabels(["QCD", "Top"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(name)

        # Annotate
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm_norm[i,j]:.3f}\n({cm[i,j]})",
                       ha="center", va="center",
                       color="white" if cm_norm[i,j] > 0.5 else "black",
                       fontsize=11)

    plt.suptitle("Confusion Matrices (threshold = 0.5)", fontsize=16, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, f"{save_name}.pdf")
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════
# 10. SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════

def print_summary_table(results_dict):
    """Print and save a LaTeX-ready comparison table."""
    print("\n" + "=" * 90)
    print(f"{'Model':<25} {'AUC':>8} {'Acc':>8} {'1/εB@50%':>10} {'1/εB@30%':>10} {'Params':>12}")
    print("=" * 90)

    rows = []
    for name, res in results_dict.items():
        metrics = compute_metrics(res["labels"], res["probs"])
        n_params = res.get("n_params", "N/A")
        print(
            f"{name:<25} "
            f"{metrics['auc']:>8.5f} "
            f"{metrics['accuracy']:>8.4f} "
            f"{metrics['bg_rej_50']:>10.0f} "
            f"{metrics['bg_rej_30']:>10.0f} "
            f"{str(n_params):>12}"
        )
        rows.append({
            "model": name,
            **metrics,
            "n_params": n_params,
        })

    print("=" * 90)

    # Save as JSON
    save_data = []
    for r in rows:
        save_data.append({
            "model": r["model"],
            "auc": float(r["auc"]),
            "accuracy": float(r["accuracy"]),
            "precision": float(r["precision"]),
            "recall": float(r["recall"]),
            "f1": float(r["f1"]),
            "bg_rej_50": float(r["bg_rej_50"]),
            "bg_rej_30": float(r["bg_rej_30"]),
            "n_params": r["n_params"],
        })

    save_path = os.path.join(RESULTS_DIR, "model_comparison.json")
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved comparison to: {save_path}")

    # Generate LaTeX table
    latex = generate_latex_table(save_data)
    latex_path = os.path.join(RESULTS_DIR, "comparison_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"Saved LaTeX table to: {latex_path}")

    return rows


def generate_latex_table(data):
    """Generate a compact LaTeX table (fits a narrow thesis text width)."""
    def _fmt_params(p):
        """Compact parameter count: ints -> '1.26M'/'366k', else as-is."""
        if isinstance(p, int):
            if p >= 1_000_000:
                return f"{p/1e6:.2f}M"
            if p >= 1_000:
                return f"{p/1e3:.0f}k"
            return str(p)
        return str(p)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Comparison of jet-classification models on the Top-Quark "
        r"Tagging benchmark. $R_{50}$ and $R_{30}$ are the background "
        r"rejections $1/\varepsilon_B$ at signal efficiencies "
        r"$\varepsilon_S=0.5$ and $0.3$.}",
        r"\label{tab:model_comparison}",
        r"\begin{tabular}{lccccc}",
        r"\hline\hline",
        r"Model & AUC & Acc. & $R_{50}$ & $R_{30}$ & Params \\",
        r"\hline",
    ]

    for row in data:
        params_str = _fmt_params(row["n_params"])
        lines.append(
            f"{row['model']} & {row['auc']:.4f} & {row['accuracy']:.4f} "
            f"& {row['bg_rej_50']:.0f} & {row['bg_rej_30']:.0f} & {params_str} \\\\"
        )

    lines.extend([
        r"\hline\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 11. FULL EVALUATION PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_full_evaluation(results_dict, data=None, histories=None,
                        attn_data=None):
    """
    Run the complete evaluation and generate all plots.

    Args:
        results_dict: {model_name: {"labels": array, "probs": array, "n_params": int}}
        data: preprocessed data dict (for substructure plots)
        histories: {model_name: training_history} (for loss curves)
        attn_data: (attn_maps, features, labels) from transformer
    """
    print("\n" + "=" * 60)
    print("GENERATING ALL EVALUATION PLOTS")
    print("=" * 60)

    # 1. ROC curves
    plot_roc_curves(results_dict)

    # 2. Score distributions
    plot_score_distributions(results_dict)

    # 3. Confusion matrices
    plot_confusion_matrices(results_dict)

    # 4. Training history
    if histories:
        plot_training_history(histories)

    # 5. Jet substructure
    if data is not None:
        if "jet_features" in data["test"] and "feature_names" in data:
            plot_jet_substructure(
                data["test"]["jet_features"],
                data["test"]["labels"],
                data["feature_names"]
            )
        # Image plots only when CNN data was prepared (skipped when running
        # --model bdt / particlenet / transformer without CNN).
        if "images" in data["test"]:
            plot_jet_images(data["test"]["images"], data["test"]["labels"])
            plot_average_jet_images(data["test"]["images"], data["test"]["labels"])

    # 6. Attention maps (Transformer)
    if attn_data is not None:
        attn_maps, features, labels = attn_data
        plot_attention_maps(attn_maps, features, labels)
        plot_cls_attention(attn_maps, features, labels)

    # 7. Summary table
    summary = print_summary_table(results_dict)

    print(f"\nAll plots saved to: {PLOT_DIR}/")
    return summary
