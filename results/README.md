# Results

This directory holds every artifact that backs a number on the slide
deck or in the methodology notes. The verification source of truth is
[audit_report.md](audit_report.md), every load-bearing number was
independently re-derived from the raw `.mat` files and matches the
CSVs to ≤ 0.001.

## Methodology notes

| File | Contains |
|---|---|
| [audit_report.md](audit_report.md) | **Verification source of truth.** End-to-end provenance trace across 14 layers (raw data → loader → preprocessing → epoching → CH11 → reference signals → CCA → channel selection → FBCCA → TRCA → evaluation harness → methodology coherence → figure coherence → coverage). Independent re-derivations using `scipy.io.loadmat`, `scipy.signal`, `numpy`, and `sklearn` directly. |
| [within_subject_evaluation.md](within_subject_evaluation.md) | Why within-subject LOBO (Nakanishi 2017 / Chen 2015 protocol) is the methodologically appropriate primary protocol. Per-subject decomposition. FBCCA's 0.875 ± 0.103 (H=5) headline. |
| [trca_methodology_note.md](trca_methodology_note.md) | TRCA negative-result analysis. Three converging causes: (1) 5 trials/class is below Nakanishi's 11-trial saturation threshold, (2) CCA's matched-filter optimality dominates on widely-spaced stim frequencies, (3) subject 2's inter-session inconsistency is exactly what TRCA needs and lacks. |
| [channel_selection.md](channel_selection.md) | Yuki's per-subject SNR-ranked channel-subset analysis. Subject 1 favors O1/O2/Oz; subject 2 favors PO3/PO4/POz. They share only PO8. |

## Tables (`tables/`)

| File | Description |
|---|---|
| [comparison.csv](tables/comparison.csv) | Cross-subject 4-block LOBO at 3 s default (mirror of `comparison_3s.csv`) |
| [comparison_1s.csv](tables/comparison_1s.csv), [_2s](tables/comparison_2s.csv), [_3s](tables/comparison_3s.csv), [_5s](tables/comparison_5s.csv) | Cross-subject classifier comparison at each window |
| [comparison_within_subject.csv](tables/comparison_within_subject.csv) | Within-subject LOBO (Nakanishi 2017 protocol) currently mirrors the 5 s run; see audit_report Layer 12 |
| [comparison_within_subject_1s.csv](tables/comparison_within_subject_1s.csv), [_2s](tables/comparison_within_subject_2s.csv), [_3s](tables/comparison_within_subject_3s.csv), [_5s](tables/comparison_within_subject_5s.csv) | Within-subject equivalent at each window |
| [cca_window_sweep.csv](tables/cca_window_sweep.csv) | CCA across 10 windows × {Config A = 8-channel, Config B = per-subject top-4 (with leakage as upper-bound reference)} |
| [cca_window_sweep_nested.csv](tables/cca_window_sweep_nested.csv) | CCA across 10 windows × {Config A, Config B-nested = per-fold leakage-free SNR ranking}; includes per-fold channel selections column |

Every `comparison_*.csv` schema: `classifier, accuracy, std, itr_bpm, notes`.
The `notes` column carries per-block accuracies (e.g. `b0=1.00 | b1=1.00 | b2=0.70 | b3=0.65`).

## Figures (`figures/`)

| File | Description |
|---|---|
| [comparison.png](figures/comparison.png) | **Headline figure**. Cross-subject 4-block LOBO at 3 s, 5-classifier bar chart with ITR companion |
| [comparison_1s.png](figures/comparison_1s.png), [_2s](figures/comparison_2s.png), [_3s](figures/comparison_3s.png), [_5s](figures/comparison_5s.png) | Cross-subject at each window |
| [comparison_within_subject.png](figures/comparison_within_subject.png) | Within-subject (Nakanishi 2017 protocol) |
| [comparison_within_subject_1s.png](figures/comparison_within_subject_1s.png), [_2s](figures/comparison_within_subject_2s.png), [_3s](figures/comparison_within_subject_3s.png), [_5s](figures/comparison_within_subject_5s.png) | Within-subject at each window |
| [cca_window_sweep.png](figures/cca_window_sweep.png) | 2-line CCA window sweep (Config A vs Config B-leakage) |
| [cca_window_sweep_with_nested.png](figures/cca_window_sweep_with_nested.png) | 3-line: + Config B-nested (slide-ready) |
| [channel_snr_heatmap.png](figures/channel_snr_heatmap.png) | Frequency × channel SNR heatmap (Yuki) |
| [channel_snr_ranking.png](figures/channel_snr_ranking.png) | SNR per channel, overall vs per-subject (Yuki) |
| [channel_sweep_accuracy.png](figures/channel_sweep_accuracy.png) | CCA accuracy vs N channels (Yuki) |
| [channel_stability_topomap_overall.png](figures/channel_stability_topomap_overall.png), [_subject1](figures/channel_stability_topomap_subject1.png), [_subject2](figures/channel_stability_topomap_subject2.png), [_fullhead](figures/channel_stability_topomap_fullhead.png) | Channel-stability topomaps under nested CV (VT) |

## Reproducing every number

```bash
# All numbers in comparison*.csv:
python scripts/compare_classifiers.py --window 3.0 --harmonics 2
python scripts/compare_classifiers.py --window 3.0 --harmonics 2 --protocol within_subject
# (repeat with --window 1.0 / 2.0 / 5.0 for the other window CSVs)

# All numbers in cca_window_sweep*.csv:
python scripts/sweep_cca_windows.py

# CH11 baseline:
python scripts/verify_ch11.py
```

For independent re-derivation, see the audit scratch scripts archived at `d:/tmp/audit_*.py` or run the verification recipes inline in [audit_report.md](audit_report.md).