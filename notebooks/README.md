# Notebooks

Pedagogical wrappers around `src/ssvep/` and `scripts/`. None of the
notebooks reimplement the algorithms — they consume the production
APIs and demonstrate how each piece works on real data.

## Reading order

Read in numeric order. Notebooks 03–06 each anchor on the audit-
verified numbers in `results/tables/comparison_*.csv` and will fail
loudly if anything in `src/ssvep/` drifts.

| # | Notebook | Owner | What it demonstrates |
|---|---|---|---|
| 01 | [01_data_exploration.ipynb](01_data_exploration.ipynb) | shared | Raw `(11, N)` `.mat` schema, trigger structure, trial epoching, CH11 → stim-frequency mapping recovery via 24-permutation sweep |
| 02 | [02_preprocessing.ipynb](02_preprocessing.ipynb) | VT | Bandpass / notch / window-length parameter scan; output CSVs in `preprocessing_parameter_testing_results/`; chosen defaults are 3–45 Hz BP + 50 Hz notch on continuous-before-epoching |
| 03 | [03_cca_baseline.ipynb](03_cca_baseline.ipynb) | shared | CCA against canonical sin/cos refs; per-frequency canonical correlation; cross-subject 4-block LOBO = **0.838 ± 0.163** |
| 04 | [04_fbcca.ipynb](04_fbcca.ipynb) | shared | 5-sub-band Chebyshev I filter bank; weights `n⁻¹·²⁵+0.25`; squared-ρ aggregation; LOBO = **0.950 ± 0.061** at 3 s, H=2 |
| 05 | [05_trca.ipynb](05_trca.ipynb) | shared | Generalized eigenvalue solve `S w = λ Q w`; ensemble per-class spatial filters; cross-subject = 0.312, within-subject = 0.475 (subj 1 = 0.675, subj 2 = 0.275); documented negative result with mechanism analysis |
| 06 | [06_comparison_and_itr.ipynb](06_comparison_and_itr.ipynb) | shared | Full classifier ablation; reproduces `results/figures/comparison.png`; Wolpaw ITR sweep across 1/2/3/5 s windows; per-subject decomposition |
| 07 | [07_demo.ipynb](07_demo.ipynb) | shared | Clean presentation demo |
| 08 | [08_channel_selection.ipynb](08_channel_selection.ipynb) | Yuki | Per-subject SNR-ranked channel subsets; cross-subject channel transfer test; subject 2 with own top-4 = 70.0% vs S1's top-4 = 62.5% |
| 09 | [09_topomap_vt.ipynb](09_topomap_vt.ipynb) | VT | Topomap visualizations of stimulus-locked response; channel-stability heatmaps |

All notebooks are committed with executed outputs — read on GitHub
without running.

## What they don't contain

The classifier implementations themselves live in
[src/ssvep/classifiers/](../src/ssvep/classifiers/). The headline
comparison harness lives in
[scripts/compare_classifiers.py](../scripts/compare_classifiers.py).
The notebooks call into these; they do not reimplement them.

For the verification source of truth — every number in every notebook
output traced back to raw `.mat` — see
[../results/audit_report.md](../results/audit_report.md).
