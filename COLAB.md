# Running the pipeline on Google Colab (free GPU)

Training the deep models on a laptop CPU is slow (~15 min/epoch). A free
Colab **T4 GPU** does the same in seconds per epoch. This guide takes you
from this repo to trained results downloaded back onto your Mac.

The workflow is: **push code to GitHub → open the Colab notebook → run on
GPU → download `results/` back to the Mac → compile the PDFs locally.**
The 1.6 GB dataset is *not* stored in GitHub; Colab fetches it from Zenodo
(fast there) and caches it on your Google Drive.

---

## 1. One-time: push this project to GitHub

From the project root on your Mac:

```bash
cd /Users/ramchand/Desktop/Climate/May2026/FYP_particle_physics

# initialise (skips the big data/venv dirs via .gitignore)
git init
git add .
git commit -m "FYP jet-tagging pipeline + docs"

# create the GitHub repo and push (you're logged in as ramc77)
gh repo create fyp-jet-tagging --public --source=. --remote=origin --push
```

This pushes only the code, docs and lightweight result summaries —
**not** the dataset or the virtual environments (they're git-ignored).

> If you'd rather keep it private, use `--private` instead of `--public`.

To push later updates:

```bash
git add -A && git commit -m "update" && git push
```

---

## 2. Open the notebook in Colab

Two ways:

- **Direct link** (after pushing): replace the user/repo and open
  ```
  https://colab.research.google.com/github/ramc77/fyp-jet-tagging/blob/main/notebooks/FYP_Colab.ipynb
  ```
- **Or** in Colab: `File → Open notebook → GitHub tab → paste`
  `ramc77/fyp-jet-tagging` → pick `notebooks/FYP_Colab.ipynb`.

Then set the GPU: **`Runtime → Change runtime type → T4 GPU`**.

---

## 3. Run the notebook top to bottom

The cells:

| Cell | What it does |
|------|--------------|
| 0 | Set `GITHUB_USER`, `REPO` (defaults already `ramc77` / `fyp-jet-tagging`). |
| 1 | `nvidia-smi` + confirm `torch.cuda.is_available()`. |
| 2 | Mount Google Drive (for persistence). |
| 3 | Clone/pull the repo. |
| 4 | `pip install -r requirements-colab.txt` (extras only — keeps Colab's GPU torch). |
| 5 | Symlink `data/` and `results/` onto Drive so checkpoints survive disconnects. |
| 6 | Download the dataset to Drive with `wget -c` (only once, ever). |
| 7a | *(optional)* flip `USE_SUBSET = False` to train on the full 1.2 M jets. |
| 7b | Run `run_full_pipeline.py` (BDT → CNN → ParticleNet → ParT → plots). |
| 8 | Run `run_study.py` + build the calibration table. |
| 9 | Print the metrics JSON and list the plots. |
| 10 | Zip `results/` and download it to your Mac. |

**If Colab disconnects** (free tier limits sessions to ~12 h and idles
out): just re-open and re-run cells 2→3→5→7b. Because `results/` lives on
Drive, the pipeline **resumes from the last completed epoch** and skips
already-finished models.

---

## 4. Back on the Mac

```bash
cd /Users/ramchand/Desktop/Climate/May2026/FYP_particle_physics
unzip -o ~/Downloads/fyp_results.zip -d results/

# rebuild the PDFs with the real figures and tables
cd docs/article  && pdflatex article.tex  && pdflatex article.tex
cd ../tutorial   && pdflatex tutorial.tex && pdflatex tutorial.tex
cd ../thesis     && pdflatex thesis.tex   && pdflatex thesis.tex
```

You now have the article, tutorial and thesis PDFs with GPU-trained
results, without ever maxing out the laptop.

---

## Notes / gotchas

- **Don't** `pip install -r requirements.txt` on Colab — that downgrades
  to CPU torch and breaks the GPU. Use `requirements-colab.txt`.
- The dataset on Drive is reused across sessions; you only pay the 1.6 GB
  download once.
- Free Colab has no guaranteed GPU; if you get "no GPU available", try
  again later or use CPU (slower, but the resume system still applies).
- All checkpoints, predictions, plots and JSON summaries land under
  `MyDrive/fyp_jet_tagging/results/` as well as in the downloaded zip.
