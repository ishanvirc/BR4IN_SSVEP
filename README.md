# BR41N.IO SSVEP — 33-hour hackathon repo

Decoding visual evoked potentials at multiple stimulus frequencies from g.tec
EEG `.mat` recordings. Pre-built CCA baseline, ITR + LOBO-CV scaffolding, and
stubs for FBCCA / TRCA. A synthetic data generator lets you exercise the full
pipeline before real data arrives.

## Install

**Windows:**

```powershell
.\scripts\setup.ps1
```

**Mac/Linux:**

```bash
conda env create -f environment.yml && conda activate br41n-ssvep && pip install -e .
```

Pure pip (no conda):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     |     Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` is what makes `from ssvep import ...` work everywhere
(notebooks, scripts, tests). Don't skip it.

## Run the baseline today (no data needed)

```bash
python scripts/run_baseline.py --synthetic
pytest tests/ -q
```

`--synthetic` generates fake SSVEP-shaped trials and runs the full
load → preprocess → CCA → ITR → LOBO-CV chain. If this prints metrics and
tests pass, the repo is wired correctly.

## Run the baseline against real data

1. Drop the `.mat` files into `data/raw/` (gitignored).
2. ```bash
   python scripts/run_baseline.py --mat data/raw/<file>.mat
   ```
3. If `load_mat` raises `KeyError`, the message lists the actual top-level
   keys in your file. Add them to the candidate lists at the top of
   [src/ssvep/io.py](src/ssvep/io.py) and re-run.

CLI options: `--bandpass LOW HIGH`, `--notch FREQ`, `--harmonics N`,
`--window-s SECONDS`.

## Pipeline stages → notebooks

| Stage | Notebook | Purpose |
|---|---|---|
| 01 | `01_data_exploration.ipynb` | shapes, fs, events, sanity plots |
| 02 | `02_preprocessing.ipynb` | notch + bandpass + epoch tuning |
| 03 | `03_cca_baseline.ipynb` | run pre-built CCA, accuracy + ITR |
| 04 | `04_fbcca.ipynb` | implement filter-bank CCA (Chen 2015) |
| 05 | `05_trca.ipynb` | implement TRCA (Nakanishi 2017) |
| 06 | `06_comparison_and_itr.ipynb` | classifier comparison, ITR sweep |
| 07 | `07_demo.ipynb` | clean demo for presentation |

## Team conventions

- **`src/ssvep/` is the canonical pipeline.** Notebooks are exploratory and
  disposable; production code lives in `src/`.
- **One owner per stage notebook.** Put your name in the first markdown cell.
  If two people need the same stage, fork as
  `03_cca_baseline_<initials>.ipynb`. Never co-edit a single `.ipynb` —
  Jupyter merge conflicts are unfixable under hackathon time pressure.
- **Clear outputs before saving** for any notebook you commit:
  `Kernel → Restart & Clear Output`. Keeps diffs reviewable.
- **Always import as `from ssvep import ...`**. Never `sys.path.insert`.
  This requires `pip install -e .` was run during setup.
- **Use `RANDOM_SEED = 42`** from `ssvep` for any stochastic step so
  teammates see the same numbers.
- **If you write a function in a notebook and use it twice, promote it to
  `src/ssvep/`.**
- Drop data only into `data/raw/`. Everything under `data/` except
  `data/README.md` is gitignored.

## Repo layout

```
br4in_ssvep/
├── src/ssvep/             # canonical pipeline package
│   ├── io.py              # load_mat (data-agnostic, key-list adjustable)
│   ├── preprocessing.py   # notch, bandpass, epoch, baseline (MNE-backed)
│   ├── features.py        # cca_reference_signals, psd, snr_at_freq
│   ├── synthetic.py       # make_synthetic_dataset (matches load_mat shape)
│   ├── classifiers/
│   │   ├── cca.py         # ✅ implemented
│   │   ├── fbcca.py       # 🚧 stub — Chen 2015
│   │   └── trca.py        # 🚧 stub — Nakanishi 2017
│   ├── evaluation.py      # accuracy, confusion, ITR, LOBO-CV
│   └── viz.py             # plot_psd, plot_topomap_at_freq, plot_confusion
├── notebooks/             # 01..07, one owner each
├── scripts/run_baseline.py
├── tests/test_smoke.py
├── data/raw/              # gitignored
├── results/{figures,tables}/
└── presentation/
```
