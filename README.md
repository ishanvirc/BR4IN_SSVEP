# BR41N.IO SSVEP — 33-hour hackathon repo

BR41N.IO 2026 Spring School Hackathon — SSVEP Data Analysis track. The
dataset (four g.tec `.mat` recordings) is in `data/raw/`, characterized in
[notebooks/01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb).
CH11 carries g.tec's live LDA classifier output, which we use as the SOTA
reference. Our job: implement CCA, FBCCA, and TRCA in `src/ssvep/`, then
compare ITR (bits/min) against CH11 across varying window lengths.

## Dataset

2 subjects × 2 sessions = 4 `.mat` files in `data/raw/` (gitignored).
Each file is continuous EEG, shape `(11, n_samples)` at 256 Hz:

- **CH1**: sample time
- **CH2-9**: EEG (PO7, PO3, POz, PO4, PO8, O1, Oz, O2 — all occipital)
- **CH10**: trigger (0 when stim off, otherwise stim freq in Hz)
- **CH11**: g.tec's live LDA classifier output (the SOTA reference)

Stimulation frequencies: 9, 10, 12, 15 Hz. 20 trials per file, 5 per
class, 7.36 s trials with 3.14 s gaps.

See [data/README.md](data/README.md) for full provenance and the CH11
mapping (descending: class 1 = 15 Hz, 2 = 12, 3 = 10, 4 = 9).

## Strategy

The track asks us to compare with state-of-the-art. We do that
properly: three classifiers along an axis of complexity, plus
g.tec's own live classifier as the SOTA truth-line.

| Method                | Sub-bands? | Subject-specific filters? | Training? |
|-----------------------|------------|---------------------------|-----------|
| CCA (Lin 2007)        | No         | No                        | No        |
| FBCCA (Chen 2015)     | Yes        | No                        | No        |
| TRCA (Nakanishi 2017) | Yes        | Yes                       | Yes       |

Each row adds one capability; this is our ablation. We compare all
three against **CH11**, g.tec's live LDA classifier output (Friman
2007 minimum-energy + LDA, trained on per-subject calibration). CH11
is the SOTA reference — see Guger et al. 2012 for the published
method on identical hardware.

Headline metric: **ITR (bits/min) at varying window lengths**, not
raw accuracy. The win condition is matching CH11's accuracy at a
shorter window — which boosts ITR — plus achieving 100% trial
coverage where CH11 fires on only ~49% of trials.

We are NOT inventing a novel method. Execution quality on a
well-scoped comparison beats novelty on a half-finished one.

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

## Sanity check (synthetic)

If you've just cloned the repo and want to verify it works before touching
real data, run the synthetic pipeline:

```bash
python scripts/run_baseline.py --synthetic
pytest tests/ -q
```

`--synthetic` generates fake SSVEP-shaped trials and runs the full
load → preprocess → CCA → ITR → LOBO-CV chain. If this prints metrics and
tests pass, the repo is wired correctly.

## Run on the hackathon data

The four `.mat` files should already be in `data/raw/` from the hackathon
Discord drop. If they're missing, download them and drop them in
(`data/raw/` is gitignored).

```bash
python scripts/run_baseline.py --mat data/raw/<file>.mat
```

**Caveat — open `io.py` issue:** the current
[src/ssvep/io.py](src/ssvep/io.py) loader expects an epoched-with-labels
schema (separate `X`, `y`, `stim_freqs` keys), but the hackathon files are
continuous `(11, n_samples)` arrays with labels embedded in CH10 and no
separate frequency key. Until the schema-aware loader update lands, the
loader will raise `KeyError` listing the actual top-level keys (`fs`, `y`)
— that's the signal, not a bug to work around. For now, see
[notebooks/01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb)
cell 2 for the loader pattern that works for this dataset.

CLI options: `--bandpass LOW HIGH`, `--notch FREQ`, `--harmonics N`,
`--window-s SECONDS`.

## Pipeline stages → notebooks

| Stage | Notebook | Purpose |
|---|---|---|
| 01 | `01_data_exploration.ipynb` | executed — full dataset characterization, trial structure, CH11 mapping recovery, PSD plots. Read this first. |
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

## References

- **Guger et al. 2012** — *How many people could use an SSVEP BCI?*
  Frontiers in Neuroscience. Bundled in `data/raw/`. Describes the
  g.tec hardware/pipeline that produced CH11.
- **Chen et al. 2015** — *Filter bank canonical correlation analysis
  for implementing a high-speed SSVEP-based BCI.* J. Neural Eng.
  The FBCCA paper.
- **Nakanishi et al. 2017** — *Enhancing detection of SSVEPs for a
  high-speed brain speller using task-related component analysis.*
  IEEE Trans. Biomed. Eng. The TRCA paper.
