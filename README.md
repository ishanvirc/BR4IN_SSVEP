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
| TRCA (Nakanishi 2018) | Yes        | Yes                       | Yes       |

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

Single-file CCA baseline:

```bash
python scripts/run_baseline.py --mat data/raw/<file>.mat
```

CLI options: `--bandpass LOW HIGH`, `--notch FREQ`, `--harmonics N`,
`--window-s SECONDS`.

Headline classifier comparison (CCA-8ch, CCA-top4-nested, FBCCA, TRCA, CH11):

```bash
# Cross-subject 4-block LOBO (default)
python scripts/compare_classifiers.py --window 3.0 --harmonics 2

# Within-subject LOBO (Nakanishi 2017 protocol; 4 folds, 2 per subject)
python scripts/compare_classifiers.py --protocol within_subject --window 3.0 --harmonics 2
```

CCA window-length sweep (8-ch vs per-subject top-4 vs per-fold nested):

```bash
python scripts/sweep_cca_windows.py
```

CH11 sanity (verifies g.tec's live LDA mapping via brute-force permutation
sweep):

```bash
python scripts/verify_ch11.py
```

## Results

**Headline (within-subject LOBO, 3.0s window, H=2):**

| Classifier | Accuracy | ITR (bpm) | Notes |
|---|---|---|---|
| CCA-8ch | 0.838 ± 0.163 | 22.0 | sin/cos refs, 8 occipital channels |
| CCA-top4 (per-fold nested) | 0.850 ± 0.150 | 23.0 | SNR-ranked top 4, leakage-free |
| **FBCCA** | **0.950 ± 0.061** | **32.7** | 5 sub-bands, Chen 2015 weighting |
| TRCA | 0.475 ± 0.202 | 3.4 | Nakanishi 2018; underperforms — see below |
| CH11 (g.tec live LDA) | 0.872 on fired | n/a | fires on only 39/80 trials (49% coverage) |

FBCCA is the recommended classifier on this dataset: highest mean
accuracy and lowest variance, with full 100% trial coverage versus
CH11's 49%. The 3.7pp gain over CCA concentrates on subject 2
(+10pp), where harmonic recovery rescues the weaker SSVEP response.

**Methodology and verification:**

- [results/within_subject_evaluation.md](results/within_subject_evaluation.md) — protocol justification (within-subject LOBO per Nakanishi 2017 / Chen 2015) and per-subject decomposition.
- [results/trca_methodology_note.md](results/trca_methodology_note.md) — TRCA negative result analysis. Briefly: 5 trials/class is the floor of Nakanishi's reported regime (saturation at ~11), and the dataset's wide frequency spacing (1–3 Hz) is exactly where CCA's matched-filter optimality dominates.
- [results/channel_selection.md](results/channel_selection.md) — Yuki's channel selection analysis: per-subject SNR-ranked top-4 matches 8-channel performance and improves robustness on subject 2.
- [results/audit_report.md](results/audit_report.md) — end-to-end provenance trace verifying every load-bearing number from raw `.mat` through CSV.
- [results/tables/](results/tables/) and [results/figures/](results/figures/) — full CSV/PNG artifacts at 1/2/3/5s windows for both protocols.

## Pipeline stages → notebooks

| Stage | Notebook | Purpose |
|---|---|---|
| 01 | `01_data_exploration.ipynb` | full dataset characterization, trial structure, CH11 mapping recovery, PSD plots. Read this first. |
| 02 | `02_preprocessing.ipynb` | notch + bandpass + epoch parameter scan; established the canonical 3-45 Hz bandpass + 50 Hz notch on continuous-before-epoching defaults. |
| 03 | `03_cca_baseline.ipynb` | run CCA, accuracy + ITR |
| 04 | `04_fbcca.ipynb` | filter-bank CCA (Chen 2015) |
| 05 | `05_trca.ipynb` | TRCA (Nakanishi 2018) |
| 06 | `06_comparison_and_itr.ipynb` | classifier comparison, ITR sweep |
| 07 | `07_demo.ipynb` | clean demo for presentation |
| 08 | `08_channel_selection.ipynb` | Yuki — per-subject SNR-ranked channel subsets vs 8-channel baseline. |
| 09 | `09_topomap_vt.ipynb` | VT — topomap visualizations of stimulus-locked response across electrodes. |

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
│   ├── io.py              # load_continuous, load_mat, load_all (continuous-schema, trigger-driven epoching)
│   ├── preprocessing.py   # filter_continuous (notch + bandpass before epoching, MNE-backed)
│   ├── features.py        # cca_reference_signals, psd, snr_at_freq
│   ├── channel_selection.py  # SNR ranking + per-subject top-N CCA sweep
│   ├── synthetic.py       # make_synthetic_dataset
│   ├── classifiers/
│   │   ├── cca.py         # ✅ Lin 2007
│   │   ├── fbcca.py       # ✅ Chen 2015 (5 sub-bands, Cheb I filter bank)
│   │   └── trca.py        # ✅ Nakanishi 2018 (ensemble TRCA, ported from meegkit BSD-3)
│   ├── evaluation.py      # accuracy, confusion, ITR (Wolpaw), LOBO-CV
│   └── viz.py             # plot_psd, plot_topomap_at_freq, plot_confusion
├── notebooks/             # 01..09, one owner each
├── scripts/
│   ├── run_baseline.py           # single-file CCA runner
│   ├── verify_ch11.py            # CH11 mapping permutation sweep
│   ├── sweep_cca_windows.py      # CCA window-length sweep (8-ch / per-subject top-4 / nested)
│   └── compare_classifiers.py    # headline comparison harness (--protocol cross_subject|within_subject)
├── tests/test_smoke.py
├── data/raw/              # gitignored
├── results/
│   ├── tables/            # comparison_*.csv, cca_window_sweep*.csv
│   ├── figures/           # comparison_*.png, channel_*, topomaps
│   ├── audit_report.md            # end-to-end provenance audit
│   ├── within_subject_evaluation.md  # protocol justification + headline numbers
│   ├── trca_methodology_note.md   # TRCA negative result analysis
│   └── channel_selection.md       # Yuki's channel-selection writeup
├── LICENSES/              # third-party licenses (meegkit-BSD-3 for TRCA port)
└── presentation/
```

## References

- **Guger et al. 2012** — *How many people could use an SSVEP BCI?*
  Frontiers in Neuroscience. Bundled in `data/raw/`. Describes the
  g.tec hardware/pipeline that produced CH11.
- **Chen et al. 2015** — *Filter bank canonical correlation analysis
  for implementing a high-speed SSVEP-based BCI.* J. Neural Eng.
  The FBCCA paper.
- **Nakanishi et al. 2018** — *Enhancing Detection of SSVEPs for a
  High-Speed Brain Speller Using Task-Related Component Analysis.*
  IEEE Trans. Biomed. Eng. 65(1):104-112. The TRCA paper.
- **Lin et al. 2007** — *Frequency Recognition Based on Canonical
  Correlation Analysis for SSVEP-Based BCIs.* IEEE Trans. Biomed. Eng.
  54(6):1172-1176. The CCA paper.
