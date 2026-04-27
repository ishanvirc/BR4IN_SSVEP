# BR41N.IO SSVEP — 33-hour hackathon repo

BR41N.IO 2026 Spring School Hackathon — SSVEP Data Analysis track. Four
g.tec `.mat` recordings (2 subjects × 2 sessions, ~225 s of continuous
EEG each) live in `data/raw/`. CH11 carries g.tec's live LDA classifier
output and serves as the SOTA reference. Our job: implement CCA, FBCCA,
and TRCA in `src/ssvep/`, then compare accuracy + ITR (bits/min) against
CH11 across varying analysis windows and two evaluation protocols.

The headline result is in [results/audit_report.md](results/audit_report.md);
see the [Results](#results) section below for the slide-ready summary.

## Dataset

2 subjects × 2 sessions = 4 `.mat` files in `data/raw/` (gitignored).
Each file is **continuous** EEG, shape `(11, n_samples)` at 256 Hz:

- **CH1** — sample time (seconds since recording start)
- **CH2–9** — EEG (PO7, PO3, POz, PO4, PO8, O1, Oz, O2 — all occipital)
- **CH10** — trigger (0 when stim off, otherwise stim freq in Hz)
- **CH11** — g.tec's live LDA classifier output (the SOTA reference)

Stimulation frequencies: 9, 10, 12, 15 Hz. 20 trials per file (5 per
class), 7.36 s stim + 3.14 s gap. 80 trials total across the 4 files.

CH11 mapping is **descending**: class 1 = 15 Hz, 2 = 12, 3 = 10, 4 = 9.
Empirically validated against CH10 ground truth — see
[data/README.md](data/README.md) for full provenance and the per-file
LDA fire-rate breakdown.

## Strategy

The track asks us to compare with state-of-the-art. We do that
properly: three classifiers along an axis of complexity, plus g.tec's
own live classifier as the SOTA truth-line.

| Method                | Sub-bands? | Subject-specific filters? | Training? |
|-----------------------|------------|---------------------------|-----------|
| CCA (Lin 2007)        | No         | No                        | No        |
| FBCCA (Chen 2015)     | Yes        | No                        | No        |
| TRCA (Nakanishi 2018) | Yes        | Yes                       | Yes       |

Each row adds one capability; this is our ablation. We compare all
three against **CH11**, g.tec's live LDA classifier output (Friman
2007 minimum-energy + LDA, trained on per-subject calibration). CH11
is the SOTA reference — see Guger et al. 2012 for the published method
on identical hardware.

Headline metric: **ITR (bits/min) at varying window lengths**, not raw
accuracy. The win condition is matching CH11's accuracy at a shorter
window — which boosts ITR — plus achieving 100% trial coverage where
CH11 fires on only ~49% of trials.

We are NOT inventing a novel method. Execution quality on a
well-scoped comparison beats novelty on a half-finished one.

## Install

**Windows:**

```powershell
.\scripts\setup.ps1
```

The script creates the `br41n-ssvep` conda env from `environment.yml`,
activates it, and runs `pip install -e .` so `from ssvep import ...`
works from anywhere (notebooks, scripts, tests).

**Mac/Linux:**

```bash
conda env create -f environment.yml && conda activate br41n-ssvep && pip install -e .
```

**Pure pip (no conda):**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     |     Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` is what makes `from ssvep import ...` work
everywhere. Don't skip it — `requires-python >=3.11` per
[pyproject.toml](pyproject.toml).

## Sanity check (synthetic)

If you've just cloned the repo and want to verify it works before
touching real data:

```bash
python scripts/run_baseline.py --synthetic
pytest tests/ -q
```

`--synthetic` generates fake SSVEP-shaped trials and runs the full
load → preprocess → CCA → ITR → LOBO-CV chain. The smoke test suite
covers imports, ITR formula edge cases, CCA on synthetic, LOBO-CV
mechanics, real-data loaders, continuous-first preprocessing, FBCCA
on synthetic + real, and TRCA on synthetic + real. Real-data tests
auto-skip if `data/raw/` is empty.

## Run on the hackathon data

The four `.mat` files should already be in `data/raw/` from the
hackathon Discord drop. If they're missing, download them and drop
them in (`data/raw/` is gitignored).

### Single-file CCA baseline

```bash
python scripts/run_baseline.py --mat data/raw/<file>.mat
python scripts/run_baseline.py --mat data/raw/                # whole directory → load_all + LOBO
```

CLI: `--bandpass LOW HIGH`, `--notch FREQ`, `--no-filter`,
`--harmonics N`, `--window-s SECONDS`. Defaults are the canonical
3–45 Hz bandpass + 50 Hz notch on continuous-before-epoching, 2
harmonics, full trial length.

### Headline classifier comparison

```bash
# Cross-subject 4-block LOBO (default)
python scripts/compare_classifiers.py --window 3.0 --harmonics 2

# Within-subject LOBO (Nakanishi 2018 protocol; 4 folds, 2 per subject)
python scripts/compare_classifiers.py --protocol within_subject --window 3.0 --harmonics 2
```

Writes `results/tables/comparison{,_within_subject}.csv` and matching
PNGs. Five rows per run: CCA-8ch, CCA-top4-nested (or CCA-top4 in the
within-subject path), FBCCA, TRCA, CH11. CLI: `--window`, `--harmonics`,
`--protocol`, `--data-dir`.

### CCA window-length sweep

```bash
python scripts/sweep_cca_windows.py
```

Sweeps 10 window lengths (0.5–5.0 s) across three configurations:
8-channel canonical, per-subject top-4 (with leakage, kept as the
optimistic upper bound), and per-fold nested top-4 (leakage-free).
Emits `cca_window_sweep.csv`, `cca_window_sweep_nested.csv`,
`cca_window_sweep.png`, and `cca_window_sweep_with_nested.png`.

### CH11 sanity (verifies g.tec's LDA mapping)

```bash
python scripts/verify_ch11.py
```

Brute-forces all 24 permutations of the (LDA class index → stim freq)
mapping per file and aggregate. Confirms the descending mapping
`{1:15, 2:12, 3:10, 4:9}` is the empirically best fit; reports per-file
fire rate and aggregate accuracy.

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
CH11's 49%. The 3.7pp mean gain over CCA concentrates on subject 2
(+10pp), where harmonic recovery rescues the weaker SSVEP response.

**Per-subject decomposition (within-subject LOBO, 3.0s):**

| Configuration | Subject 1 | Subject 2 |
|---|---|---|
| CCA | 1.000 | 0.675 |
| FBCCA (H=2) | 1.000 | 0.900 |
| FBCCA (H=5, methodology note default) | 0.975 | 0.775 |
| TRCA | 0.675 | 0.275 |
| CH11 (on fired trials) | 1.000 (10/40 fired) | 0.828 (29/40 fired) |

Subject 1 is a textbook responder; subject 2 shows the universality
problem documented by Guger et al. 2012. FBCCA's harmonic recovery
narrows the gap from both ends.

**Methodology + verification artifacts:**

- [results/within_subject_evaluation.md](results/within_subject_evaluation.md) — protocol justification (within-subject LOBO per Nakanishi 2018 / Chen 2015) and per-subject decomposition.
- [results/trca_methodology_note.md](results/trca_methodology_note.md) — TRCA negative result analysis. Briefly: 5 trials/class is the floor of Nakanishi's reported regime (saturation at ~11), and the dataset's wide frequency spacing (1–3 Hz) is exactly where CCA's matched-filter optimality dominates.
- [results/channel_selection.md](results/channel_selection.md) — Yuki's per-subject SNR-ranked channel-subset analysis.
- [results/audit_report.md](results/audit_report.md) — end-to-end provenance trace verifying every load-bearing number from raw `.mat` through CSV. Independent re-derivations match published numbers to ≤ ±0.001.
- [results/tables/](results/tables/) and [results/figures/](results/figures/) — CSV/PNG artifacts at 1/2/3/5 s windows for both protocols, plus the window sweep and channel-stability topomaps.

## Pipeline stages → notebooks

| # | Notebook | Owner | Purpose |
|---|---|---|---|
| 01 | `01_data_exploration.ipynb` | shared | full dataset characterization, trial structure, CH11 mapping recovery, PSD plots. **Read this first.** |
| 02 | `02_preprocessing.ipynb` | VT | notch + bandpass + epoch parameter scan. Established the canonical 3–45 Hz bandpass + 50 Hz notch on continuous-before-epoching defaults. CSV outputs in `notebooks/preprocessing_parameter_testing_results/`. |
| 03 | `03_cca_baseline.ipynb` | — | run CCA, accuracy + ITR (stub — CSV pipeline supersedes) |
| 04 | `04_fbcca.ipynb` | — | filter-bank CCA (Chen 2015) (stub — `compare_classifiers.py` supersedes) |
| 05 | `05_trca.ipynb` | — | TRCA (Nakanishi 2018) (stub — `compare_classifiers.py` supersedes) |
| 06 | `06_comparison_and_itr.ipynb` | — | classifier comparison, ITR sweep (stub — `compare_classifiers.py` supersedes) |
| 07 | `07_demo.ipynb` | shared | clean demo for presentation |
| 08 | `08_channel_selection.ipynb` | Yuki | per-subject SNR-ranked channel subsets vs 8-channel baseline; cross-subject channel transfer test. |
| 09 | `09_topomap_vt.ipynb` | VT | topomap visualizations of stimulus-locked response across electrodes; channel-stability heatmaps. |

## `src/ssvep/` modules

| Module | Purpose |
|---|---|
| [io.py](src/ssvep/io.py) | `load_continuous` (raw matrix), `load_mat` (continuous-first preprocessing + epoching), `load_all` (concatenate all files, assign per-file block IDs). Trigger-driven epoching at 0.14 s post-stimulus latency (Chen 2015). Also exports `load_mat_legacy` for the deprecated epoched-with-labels schema. |
| [preprocessing.py](src/ssvep/preprocessing.py) | `filter_continuous` (notch + bandpass on the (11, N) matrix BEFORE epoching, EEG rows only — sample-time/trigger/LDA rows preserved byte-identical). Plus per-trial `notch_filter`, `bandpass`, `epoch`, `baseline_correct` helpers (MNE-backed). |
| [features.py](src/ssvep/features.py) | `cca_reference_signals` (canonical sin/cos basis), `psd` (Welch), `snr_at_freq` (target / flanking-band ratio). |
| [channel_selection.py](src/ssvep/channel_selection.py) | `rank_channels_by_snr` (SNR ranking across stim freqs), `channel_snr_matrix` (per-class per-channel SNR), `sweep_channel_subsets` (LOBO-CV CCA across top-N subsets). |
| [evaluation.py](src/ssvep/evaluation.py) | `accuracy`, `confusion_matrix`, `itr` (Wolpaw bits/min), `leave_one_block_out_cv`. |
| [synthetic.py](src/ssvep/synthetic.py) | `make_synthetic_dataset` — fake SSVEP with per-channel amplitude + phase, white-Gaussian noise at configurable SNR. Mirrors the `load_mat` dict schema. |
| [viz.py](src/ssvep/viz.py) | `plot_psd`, `plot_topomap_at_freq` (bar-plot fallback until MNE montage is set), `plot_spectrogram`, `plot_confusion`. |
| [classifiers/cca.py](src/ssvep/classifiers/cca.py) | `CCAClassifier` — per-trial CCA against canonical sin/cos refs, argmax-correlation predict. `fit` is a no-op. |
| [classifiers/fbcca.py](src/ssvep/classifiers/fbcca.py) | `FBCCAClassifier` — Chen 2015 filter-bank CCA. 5 Cheb I sub-bands, weights `n^-1.25 + 0.25`, squared-correlation aggregation. `fit` is a no-op. |
| [classifiers/trca.py](src/ssvep/classifiers/trca.py) | `TRCAClassifier` — ensemble TRCA (Nakanishi 2018), ported from meegkit (BSD-3, see `LICENSES/`). Per-class generalized eigenvalue solve `S w = λ Q w`; ensemble of per-class spatial filters at predict time. |
| [\_\_init\_\_.py](src/ssvep/__init__.py) | Re-exports all submodules and `RANDOM_SEED = 42`. |

## Tests

```bash
pytest tests/ -q
```

[tests/test_smoke.py](tests/test_smoke.py) covers:

- **Import + plumbing:** every public submodule imports; `RANDOM_SEED` is set.
- **Reference-signal shape:** `cca_reference_signals` returns `(n_classes, 2H, n_samples)` with sin starting at 0, cos at 1.
- **ITR formula edge cases:** perfect accuracy = `log2(N)`, chance = 0, mid-range matches Wolpaw hand-computation.
- **CCA on synthetic** + **LOBO-CV mechanics** (no flaky thresholds).
- **Real-data smoke:** `load_mat` + `load_all` shapes, CH11 sentinel handling, per-trial CH11 accuracy ≥ 0.80.
- **Continuous-first preprocessing:** `filter_continuous` preserves rows 0/9/10 (time, trigger, LDA); `load_mat` with canonical filters puts subject 1 / training 1 in [0.85, 1.0].
- **Composability:** `load_continuous → filter_continuous → epoch_trials` matches `load_mat(...filtered)` byte-for-byte.
- **TRCA on synthetic** (TRCA-friendly inline generator) **+ TRCA on real LOBO** (floor 0.28).
- **FBCCA on synthetic** (≥ 0.75) **+ FBCCA on real LOBO** (≥ 0.65).

Real-data tests auto-skip when `data/raw/` is empty.

## Repo layout

```
br4in_ssvep/
├── README.md                       # this file
├── pyproject.toml                  # setuptools config; package = src/ssvep, requires-python >=3.11
├── environment.yml                 # conda env "br41n-ssvep" (Python 3.11, numpy/scipy/mne/sklearn/...)
├── requirements.txt                # pip equivalent
├── .gitignore                      # ignores data/raw, data/processed, references/, caches
│
├── src/ssvep/                      # canonical pipeline package
│   ├── __init__.py                 # re-exports + RANDOM_SEED = 42
│   ├── io.py                       # load_continuous / load_mat / load_all (continuous schema, trigger-driven epoching)
│   ├── preprocessing.py            # filter_continuous (notch + bandpass before epoching, MNE-backed)
│   ├── features.py                 # cca_reference_signals, psd (Welch), snr_at_freq
│   ├── channel_selection.py        # rank_channels_by_snr, channel_snr_matrix, sweep_channel_subsets
│   ├── evaluation.py               # accuracy, confusion_matrix, itr (Wolpaw), leave_one_block_out_cv
│   ├── synthetic.py                # make_synthetic_dataset (matches load_mat schema)
│   ├── viz.py                      # plot_psd, plot_topomap_at_freq, plot_spectrogram, plot_confusion
│   └── classifiers/
│       ├── __init__.py             # re-exports CCAClassifier, FBCCAClassifier, TRCAClassifier
│       ├── cca.py                  # ✅ Lin 2007 — CCA against canonical sin/cos refs
│       ├── fbcca.py                # ✅ Chen 2015 — 5 Cheb I sub-bands, weighted squared-ρ aggregation
│       └── trca.py                 # ✅ Nakanishi 2018 — ensemble TRCA, port of meegkit BSD-3
│
├── notebooks/                      # 01..09, see "Pipeline stages → notebooks" above
│   ├── 01_data_exploration.ipynb       # full characterization, CH11 mapping recovery
│   ├── 02_preprocessing.ipynb          # VT — bandpass/notch/epoch parameter scan
│   ├── 03_cca_baseline.ipynb           # (stub — superseded by compare_classifiers.py)
│   ├── 04_fbcca.ipynb                  # (stub — superseded)
│   ├── 05_trca.ipynb                   # (stub — superseded)
│   ├── 06_comparison_and_itr.ipynb     # (stub — superseded)
│   ├── 07_demo.ipynb                   # presentation demo
│   ├── 08_channel_selection.ipynb      # Yuki — per-subject SNR-ranked channels
│   ├── 09_topomap_vt.ipynb             # VT — topomap + channel-stability viz
│   └── preprocessing_parameter_testing_results/   # CSV outputs from notebook 02's scan
│       ├── archive_full_scan/                     # full grid (bandpass × notch × window)
│       └── top_outputs/                           # decision tables for the chosen defaults
│
├── scripts/
│   ├── setup.ps1                       # one-command Windows env setup
│   ├── run_baseline.py                 # single-file CCA runner (--mat | --synthetic, ±filter flags)
│   ├── verify_ch11.py                  # 24-permutation CH11 mapping sweep
│   ├── sweep_cca_windows.py            # 10-window × 3-config CCA sweep → CSV + slide-ready PNG
│   └── compare_classifiers.py          # headline harness: CCA / CCA-top4 / FBCCA / TRCA / CH11
│                                       #   --protocol cross_subject|within_subject  --window  --harmonics
│
├── tests/
│   ├── __init__.py
│   └── test_smoke.py                   # imports, refs, ITR, CCA, LOBO, real data, continuous-first, TRCA, FBCCA
│
├── data/
│   ├── README.md                       # provenance, channel layout, recording params, CH11 mapping derivation
│   ├── raw/                            # gitignored — .mat files + Guger 2012 PDF + montage.png
│   └── processed/                      # gitignored — caches, intermediate arrays
│
├── results/
│   ├── audit_report.md                     # end-to-end provenance audit (this audit run)
│   ├── within_subject_evaluation.md        # protocol justification + headline numbers (H=5 FBCCA)
│   ├── trca_methodology_note.md            # TRCA negative-result analysis
│   ├── channel_selection.md                # Yuki's writeup
│   ├── tables/
│   │   ├── comparison.csv                      # cross-subject 3s default
│   │   ├── comparison_{1,2,3,5}s.csv           # cross-subject window variants
│   │   ├── comparison_within_subject.csv       # within-subject (currently 5s — see audit)
│   │   ├── comparison_within_subject_{1,2,3,5}s.csv
│   │   ├── cca_window_sweep.csv                # 10 windows × {A=8ch, B=per-subject top-4 leakage}
│   │   └── cca_window_sweep_nested.csv         # 10 windows × {A=8ch, B-nested=per-fold leakage-free}
│   └── figures/
│       ├── comparison*.png                     # bar charts mirroring the CSVs above
│       ├── cca_window_sweep.png                # 2-line sweep figure
│       ├── cca_window_sweep_with_nested.png    # 3-line sweep figure (slide-ready)
│       ├── channel_snr_heatmap.png             # frequency × channel SNR heatmap (Yuki)
│       ├── channel_snr_ranking.png             # SNR per channel, overall vs per-subject
│       ├── channel_sweep_accuracy.png          # CCA acc vs N channels (Yuki)
│       └── channel_stability_topomap_*.png     # VT — channel-stability topomaps
│
├── LICENSES/
│   └── meegkit-BSD-3.txt               # required attribution for the TRCA port
│
├── presentation/                       # slide assets (gitkept; populate as we draft)
│
└── references/                         # gitignored — third-party reference implementations
    ├── fbcca_eugeneALU/                #   — reference for Chen 2015 FBCCA
    └── meegkit/                        #   — source of the TRCA port (BSD-3)
```

## Team conventions

- **`src/ssvep/` is the canonical pipeline.** Notebooks are exploratory
  and disposable; production code lives in `src/`. If you write a
  function in a notebook and use it twice, promote it to `src/ssvep/`.
- **One owner per stage notebook.** Put your name in the first markdown
  cell. If two people need the same stage, fork as
  `03_cca_baseline_<initials>.ipynb`. Never co-edit a single `.ipynb` —
  Jupyter merge conflicts are unfixable under hackathon time pressure.
- **Clear outputs before saving** for any notebook you commit:
  `Kernel → Restart & Clear Output`. Keeps diffs reviewable.
  (Notebooks 01, 02, 07, 08, 09 are exceptions — their executed
  outputs are part of the deliverable.)
- **Always import as `from ssvep import ...`**. Never `sys.path.insert`.
  This requires `pip install -e .` was run during setup.
- **Use `RANDOM_SEED = 42`** from `ssvep` for any stochastic step so
  teammates see the same numbers.
- Drop data only into `data/raw/`. Everything under `data/` except
  `data/README.md` is gitignored.
- The `references/` directory holds third-party reference
  implementations (meegkit's TRCA, eugeneALU's FBCCA) for cross-check
  during porting; it is **gitignored**. Each teammate clones what they
  need locally. The actual ports live in `src/ssvep/classifiers/` with
  attribution in `LICENSES/`.

## References

- **Guger et al. 2012** — *How many people could use an SSVEP BCI?*
  Frontiers in Neuroscience 6:169. Bundled in `data/raw/`. Describes
  the g.tec hardware/pipeline that produced CH11.
- **Lin et al. 2007** — *Frequency Recognition Based on Canonical
  Correlation Analysis for SSVEP-Based BCIs.* IEEE Trans. Biomed. Eng.
  54(6):1172–1176. The CCA paper.
- **Chen et al. 2015** — *Filter bank canonical correlation analysis
  for implementing a high-speed SSVEP-based brain-computer interface.*
  J. Neural Eng. 12:046008. The FBCCA paper.
- **Nakanishi et al. 2018** — *Enhancing Detection of SSVEPs for a
  High-Speed Brain Speller Using Task-Related Component Analysis.*
  IEEE Trans. Biomed. Eng. 65(1):104–112. The TRCA paper.
- **Wolpaw et al. 1998** — *EEG-based communication: improved accuracy
  by response verification.* IEEE Trans. Rehabil. Eng. 6(3):326–333.
  ITR formula.
