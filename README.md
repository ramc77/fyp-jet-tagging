# FYP: Deep Learning for Jet Classification

Top-quark tagging on 14 TeV pp collisions (BNBWU–CERN collaboration).
Compares a BDT baseline (XGBoost on jet substructure features) with three
deep-learning architectures — **JetCNN** (jet images), **ParticleNet**
(graph neural network), and **Particle Transformer** — on the Top Quark
Tagging Reference Dataset (Zenodo 10.5281/zenodo.2603256).

Runs on **macOS** (Intel & Apple Silicon) and **Ubuntu / WSL**.

---

## 1. Requirements

| | |
|---|---|
| Python | 3.10 – 3.12 |
| RAM    | 8 GB minimum, 16 GB recommended (subset mode uses ~6 GB) |
| Disk   | ~2 GB for the dataset (subset) / ~17 GB (full) |
| GPU    | Optional. CUDA works on Linux/WSL; Apple MPS is not yet wired up |

---

## 2. Installation

### macOS

```bash
# 1. Clone / cd into the project
cd complete-project

# 2. Create a virtual environment (any name; we use "particle" below)
python3 -m venv particle
source particle/bin/activate

# 3. Upgrade pip and install deps
pip install --upgrade pip
pip install -r requirements.txt
```

No Homebrew packages are strictly required — the project sets
`KMP_DUPLICATE_LIB_OK=TRUE` internally to avoid the macOS OpenMP-runtime
collision between `libiomp5` (numpy/openblas) and `libomp` (XGBoost).

### Ubuntu / WSL

```bash
# 1. One-time system packages (skip if already installed)
sudo apt update
sudo apt install -y python3-venv python3-pip build-essential \
                    libhdf5-dev pkg-config

# 2. Clone / cd into the project
cd complete-project

# 3. Create a virtual environment
python3 -m venv particle
source particle/bin/activate

# 4. Upgrade pip and install deps
pip install --upgrade pip
pip install -r requirements.txt
```

**WSL tips**

- Put the project on the Linux filesystem (e.g. `~/FYP/...`), **not**
  `/mnt/c/...` — reading the 1 GB `train.h5` from the Windows drive is
  10–50× slower.
- Give WSL enough RAM. Create `%UserProfile%\.wslconfig` on Windows:

  ```
  [wsl2]
  memory=12GB
  ```

  Then `wsl --shutdown` in PowerShell and reopen.

### GPU support (optional, Linux/WSL only)

The default `torch` wheel is CPU. For CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

`config.py` auto-detects `torch.cuda.is_available()` and switches device.

---

## 3. Download the dataset

```bash
source particle/bin/activate
python3 download_data.py
```

Prompts for confirmation, then downloads `train.h5`, `val.h5`, `test.h5`
(~1.6 GB total) into `data/` from Zenodo. Files are resumable — rerun
if interrupted.

If the automatic download fails, grab the three `.h5` files manually
from <https://zenodo.org/records/2603256> and drop them in `data/`.

---

## 4. Run the pipeline

```bash
source particle/bin/activate
python3 run_full_pipeline.py
```

Default settings (in `config.py`) train on a **50 000-jet subset** of
each split — full pipeline finishes in roughly:

| Hardware        | Wall time |
|-----------------|-----------|
| M1/M2 / modern Intel Mac (CPU) | 20–40 min |
| Ubuntu + RTX 3070 / better     | 5–10 min  |
| WSL CPU-only                   | 25–50 min |

For the full 1.2 M-jet dataset, set `USE_SUBSET = False` in `config.py`.

### Useful flags

```bash
python3 run_full_pipeline.py --model bdt          # only BDT baseline
python3 run_full_pipeline.py --model cnn          # only CNN on jet images
python3 run_full_pipeline.py --model particlenet  # only ParticleNet
python3 run_full_pipeline.py --model transformer  # only Particle Transformer
python3 run_full_pipeline.py --eval-only          # regenerate plots from saved ckpts
python3 run_full_pipeline.py --no-bdt             # skip BDT (rest of models only)
python3 run_full_pipeline.py --max-particles 50   # fewer particles per jet (faster)
python3 run_full_pipeline.py --force              # re-train models even if a
                                                  # previous run completed them
```

### Stop / resume (give the laptop a break)

Everything in the pipeline is checkpointed. You can interrupt at any point
(Ctrl-C, close the lid, kill the process) and re-run the same command to
pick up where it stopped — no work is lost.

```bash
python3 run_full_pipeline.py                      # train + eval everything
# … later, after Ctrl-C in the middle of (say) the CNN …
python3 run_full_pipeline.py                      # resumes the CNN from
                                                  # the last completed epoch,
                                                  # then continues with
                                                  # ParticleNet and ParT
```

What gets saved as you go:

- `results/models/<model>_last.pt` — full training state (model, optimizer,
  scheduler, AMP scaler, epoch counter, history) snapshot **after every
  epoch**, so a re-run continues at the next epoch.
- `results/models/<model>_best.pt` — best-val-AUC checkpoint (used for
  final test-set prediction and plots).
- `results/models/<model>_done.json` — marker file written when training
  finishes for that model (either max epochs or early-stop). On the next
  run, models whose marker exists are skipped, and their cached test
  predictions in `results/predictions/<model>_test.npz` are loaded
  directly.
- `results/<model>_history.json` — loss/AUC history, updated **every
  epoch** (so the training-history plot reflects partial progress too).

To force a full re-train, pass `--force` (or delete `results/models/`).

The data-efficiency study (`run_study.py`) uses the same checkpoint
scheme per (model, N_train) cell. Cells whose `*_done.json` exists are
skipped, and cells in mid-training resume at the last epoch.

### Environment overrides

| Var | Default | Effect |
|---|---|---|
| `XGBOOST_NUM_THREADS` | `1` | Threads used by XGBoost. Safe to raise to 4 on Linux. |
| `OMP_NUM_THREADS`     | `4` (set by pipeline) | OpenMP threads for numpy/torch ops. |
| `KMP_DUPLICATE_LIB_OK`| `TRUE` (set by pipeline) | Allow duplicate OpenMP runtimes (macOS fix). |

Example: let XGBoost use more cores on Ubuntu:

```bash
XGBOOST_NUM_THREADS=8 python3 run_full_pipeline.py
```

---

## 5. Outputs

Written to `results/`:

```
results/
├── model_comparison.json       metrics: AUC, accuracy, BG rejection @ sig_eff
├── comparison_table.tex        LaTeX-ready comparison table
├── models/                     saved checkpoints (.pt / .json)
└── plots/
    ├── roc_comparison.pdf      ROC curves for all models
    ├── score_distributions.pdf classifier output histograms
    ├── confusion_matrices.pdf
    ├── training_history.pdf    loss & AUC vs epoch
    ├── jet_substructure.pdf    τ21, τ32, C2, mass, pT
    ├── jet_images.pdf          example event displays (CNN input)
    ├── average_jet_images.pdf  signal vs background averaged
    ├── attention_maps.pdf      Transformer per-head attention
    ├── cls_attention.pdf       CLS-token attention vs ΔR
    └── bdt_importance.pdf      XGBoost feature importance
```

---

## 6. Project layout

```
complete-project/
├── config.py                 all hyperparameters & paths
├── download_data.py          Zenodo fetcher
├── run_full_pipeline.py      top-level orchestrator
├── requirements.txt
├── src/
│   ├── data_utils.py         HDF5 loading, feature engineering, datasets
│   ├── jet_cnn.py            CNN on 40×40×3 jet images
│   ├── particle_net.py       EdgeConv-based graph network
│   ├── particle_transformer.py  self-attention on particle sequences
│   ├── trainer.py            unified train / val / predict + BDT
│   └── evaluation.py         metrics + all publication plots
├── notebooks/                exploratory analysis
├── data/                     (created by download_data.py)
└── results/                  (created at runtime)
```

---

## 7. Troubleshooting

### macOS — `zsh: segmentation fault python3 run_full_pipeline.py`

Almost always the duplicate-OpenMP collision (numpy's `libiomp5.dylib`
vs XGBoost's `libomp.dylib`). The pipeline sets
`KMP_DUPLICATE_LIB_OK=TRUE` and pins XGBoost to `nthread=1` by default,
which resolves it.

If you still see it, verify the fix is active:

```bash
python3 -c "import os; print(os.environ.get('KMP_DUPLICATE_LIB_OK'))"
# expected: TRUE   (after running run_full_pipeline.py at least once,
#                   or when the var is exported in your shell)
```

Force it at the shell level as a last resort:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
python3 run_full_pipeline.py
```

### macOS — `NumPy 2.x cannot be run… _ARRAY_API not found`

`requirements.txt` pins `numpy<2` to match the ABI of the stock
torch/xgboost wheels. Reinstall inside the venv:

```bash
pip install -r requirements.txt --upgrade
```

### Ubuntu / WSL — `TypeError: a bytes-like object is required, not 'str'` from `pd.read_hdf`

Already patched — `src/data_utils.py` no longer goes through
`pd.read_hdf`; it reads the HDF5 files via PyTables directly, which
sidesteps the pandas-vs-pytables attribute-decoding bug.

### Any OS — `FileNotFoundError: data/train.h5`

Run the downloader:

```bash
python3 download_data.py
```

### `ModuleNotFoundError` after `pip install`

You installed into the system Python instead of the venv. Make sure
the venv is active (you should see `(particle)` in your prompt):

```bash
which python3          # should point inside particle/bin
source particle/bin/activate
pip install -r requirements.txt
```

### Out-of-memory / process killed

Lower the subset sizes in `config.py`:

```python
SUBSET_TRAIN = 20_000
SUBSET_VAL   = 10_000
SUBSET_TEST  = 10_000
```

Or reduce particles:

```bash
python3 run_full_pipeline.py --max-particles 50
```

---

## 8. Citation

If you use this code, please cite the Top Quark Tagging Reference
Dataset:

> Kasieczka, G., Plehn, T., Thompson, J., & Russel, M. (2019).
> *Top Quark Tagging Reference Dataset.* Zenodo.
> <https://doi.org/10.5281/zenodo.2603256>

---

## 9. License / authorship

Project code: Dr. Ram Chand, The Begum Nusrat Bhutto Women University,
Sukkur (BNBWU), in collaboration with CERN.
