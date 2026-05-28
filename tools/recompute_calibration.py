#!/usr/bin/env python3
"""
Recompute the calibration analysis from cached study predictions.
=================================================================
The data-efficiency study caches each (model, N_train) run's validation
and test predictions in results/study_predictions/<model>_n<N>.npz. This
script re-derives the calibration metrics + plot WITHOUT retraining, using
the largest-N run for each model. It exists because the original run shipped
a temperature-fit that returned NaN for the well-separated CNN/ParticleNet
scores; with the fixed fit_temperature this regenerates correct outputs.

Outputs (into results/):
  plots/calibration.pdf, plots/calibration.png
  study_summary.json   (calibration section refreshed; data_efficiency kept)
  calibration_table.tex
"""
import os, sys, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

from config import RESULTS_DIR
from src.study_utils import (
    expected_calibration_error, reliability_curve,
    fit_temperature, logits_from_probs, plot_calibration,
)

PRED_DIR = os.path.join(RESULTS_DIR, "study_predictions")
PRETTY = {
    "bdt": "BDT (XGBoost)", "cnn": "CNN (Jet Images)",
    "particlenet": "ParticleNet", "transformer": "Particle Transformer",
}
ORDER = ["bdt", "cnn", "particlenet", "transformer"]


def largest_n_file(model):
    files = glob.glob(os.path.join(PRED_DIR, f"{model}_n*.npz"))
    if not files:
        return None
    # pick the largest N by parsing the trailing integer
    def n_of(p):
        base = os.path.basename(p)
        return int(base.split("_n")[1].split(".npz")[0])
    return max(files, key=n_of)


def main():
    calibration = {}
    for m in ORDER:
        f = largest_n_file(m)
        if f is None:
            print(f"[skip] no cached predictions for {m}")
            continue
        d = np.load(f, allow_pickle=True)
        val_probs, val_labels = d["val_probs"], d["val_labels"]
        test_probs, test_labels = d["test_probs"], d["test_labels"]

        # temperature on validation, evaluate on test
        T = fit_temperature(val_labels, logits_from_probs(val_probs))
        ece = expected_calibration_error(test_labels, test_probs)
        confs, accs, counts = reliability_curve(test_labels, test_probs)

        test_logits = logits_from_probs(test_probs)
        probs_T = 1.0 / (1.0 + np.exp(-test_logits / max(T, 1e-3)))
        ece_T = expected_calibration_error(test_labels, probs_T)
        confs_T, accs_T, counts_T = reliability_curve(test_labels, probs_T)

        calibration[PRETTY[m]] = {
            "probs": test_probs, "labels": test_labels,
            "confs": confs, "accs": accs, "counts": counts,
            "confs_tscale": confs_T, "accs_tscale": accs_T, "counts_tscale": counts_T,
            "ece": float(ece), "T": float(T), "ece_tscale": float(ece_T),
        }
        print(f"{PRETTY[m]:24s} ECE={ece:.4f}  T={T:.3f}  ECE_Tscaled={ece_T:.4f}")

    if not calibration:
        print("No predictions found — nothing to do.")
        return

    plot_calibration(calibration)

    # Refresh the calibration section of study_summary.json (keep the rest)
    summ_path = os.path.join(RESULTS_DIR, "study_summary.json")
    summ = {}
    if os.path.exists(summ_path):
        summ = json.load(open(summ_path))
    summ["calibration"] = {
        name: {"ece": c["ece"], "T": c["T"], "ece_tscale": c["ece_tscale"],
               "n_samples": int(len(c["labels"]))}
        for name, c in calibration.items()
    }
    json.dump(summ, open(summ_path, "w"), indent=2)
    print(f"Updated {summ_path}")

    # Regenerate the calibration LaTeX table
    de = summ.get("data_efficiency", {})
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Model & Test AUC & ECE & $T$ & ECE$_{T}$ \\", r"\midrule"]
    for name, c in calibration.items():
        aucs = de.get(name, {}).get("auc", [])
        auc_s = f"{aucs[-1]:.4f}" if aucs else "--"
        lines.append(f"{name} & {auc_s} & {c['ece']:.3f} & {c['T']:.2f} & {c['ece_tscale']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path = os.path.join(RESULTS_DIR, "calibration_table.tex")
    open(tex_path, "w").write("\n".join(lines))
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
