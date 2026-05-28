#!/usr/bin/env python3
"""
After run_study.py has produced results/study_summary.json, this
script writes results/calibration_table.tex — a LaTeX longtable
fragment with one row per model showing AUC, ECE, fitted T, and
ECE after temperature rescaling.
The thesis chapter 5 includes this fragment via \\input{}.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
SUMMARY = os.path.join(PROJECT, "results", "study_summary.json")
OUT = os.path.join(PROJECT, "results", "calibration_table.tex")

if not os.path.exists(SUMMARY):
    print(f"No {SUMMARY} found — run run_study.py first.", file=sys.stderr)
    sys.exit(1)

with open(SUMMARY) as f:
    s = json.load(f)

de = s["data_efficiency"]
cal = s["calibration"]

lines = [
    r"\begin{tabular}{lcccc}",
    r"\toprule",
    r"Model & Test AUC & ECE & $T$ & ECE$_{T}$ \\",
    r"\midrule",
]
for name, c in cal.items():
    # Largest-N AUC
    aucs = de.get(name, {}).get("auc", [None])
    auc = aucs[-1] if aucs else None
    auc_s = f"{auc:.4f}" if auc is not None else "--"
    lines.append(
        f"{name} & {auc_s} & {c['ece']:.3f} & {c['T']:.2f} & {c['ece_tscale']:.3f} \\\\"
    )
lines += [r"\bottomrule", r"\end{tabular}"]

with open(OUT, "w") as f:
    f.write("\n".join(lines))
print(f"Wrote {OUT}")
