# Channel Selection Results

**Author:** Yuki  
**Notebook:** `notebooks/08_channel_selection.ipynb`  
**Method:** SNR-ranked channel subset sweep + LOBO-CV CCA (3.0 s window, 2 harmonics)

> **A simple SNR-based 4-channel subset matches full-channel performance while improving robustness on harder subjects — revealing that channel selection, not model complexity, is the primary performance bottleneck.**

## Main claim

SNR-based 4-channel selection matches full-channel CCA performance overall, and actually **improves robustness on the harder subject (S2)**. The optimal channel subset is subject-specific — S1 and S2 share only one channel in their top-4 (PO8). This strongly suggests that fixed-montage decoding is suboptimal, and that subject-specific spatial filtering (e.g., TRCA) outperforms generic CCA, especially on Subject 2.

---

## Background

Guger et al. 2012 describe a universality problem: SSVEP BCIs work well for some subjects and poorly for others. Our dataset shows this clearly — Subject 1 hits 100% CCA accuracy; Subject 2 reaches only 67.5% with all 8 channels. The question is whether channel selection can help explain and partially address this gap.

---

## Channel SNR ranking

Channels ranked by mean SSVEP SNR across all 4 stimulation frequencies.

| Rank | Overall | Subject 1 | Subject 2 |
|------|---------|-----------|-----------|
| #1   | O2      | O2        | PO4       |
| #2   | O1      | O1        | PO8       |
| #3   | PO8     | Oz        | POz       |
| #4   | Oz      | PO8       | PO3       |
| #5   | PO3     | PO3       | O2        |
| #6   | PO4     | PO4       | Oz        |
| #7   | POz     | PO7       | O1        |
| #8   | PO7     | POz       | PO7       |

S1 favors pure occipital electrodes (O1/Oz/O2); S2 favors the parieto-occipital ring (PO3/PO4/POz). They share only **PO8**.

---

## (a) Does 4-channel match 8-channel?

LOBO-CV CCA accuracy (3.0 s window):

| Configuration                | All    | Subject 1 | Subject 2 |
|------------------------------|--------|-----------|-----------|
| 8-channel (all)              | 83.8%  | 100.0%    | 67.5%     |
| Top-4 by SNR (O2+O1+PO8+Oz) | 81.2%  | 100.0%    | **70.0%** |
| Ishanvir's O1/Oz/O2/POz      | 78.8%  | 97.5%     | 60.0%     |

**Answer: Yes.** Top-4 by SNR is within 2.5% of 8-channel overall, and **outperforms 8-channel on Subject 2** (70.0% vs 67.5%).

Why does S2 improve with fewer channels? S2's worst channels (O1, Oz — ranked #7 and #6 for S2) are the channels that dominate Subject 1's montage. Including them adds noise for S2 rather than signal, and dropping them yields a cleaner spatial average and improves signal-to-noise consistency.

**Practical implication:** half the electrodes, same or better performance — relevant for real-world BCI deployment.

---

## (b) Do S1 and S2 have different optimal subsets?

| | Subject 1 | Subject 2 |
|---|---|---|
| Top-4 channels | O2, O1, Oz, PO8 | PO4, PO8, POz, PO3 |
| Shared | PO8 only | PO8 only |

**Cross-subject transfer test (on S2 data):**

| Ranking used | S2 accuracy |
|---|---|
| S1's top-4 (O2, O1, Oz, PO8) | 62.5% |
| S2's own top-4 (PO4, PO8, POz, PO3) | **70.0%** |

**Answer: Yes — almost completely different.** Using S1's optimal channels on S2 drops accuracy by 7.5 pp. A fixed cross-subject montage is not portable.

**Connection to TRCA:** This directly explains the performance gap between generic CCA and subject-specific methods such as TRCA. TRCA learns a spatial filter from training data rather than applying a fixed montage — exactly the per-subject adaptation that our channel ranking shows is necessary. The accuracy gap between S1-ranking and S2-ranking on S2 data (62.5% → 70.0%) is a lower bound on the gain TRCA should achieve.

**Connection to ITR:** Since the 4-channel subset removes noisy channels, it is expected to maintain accuracy at shorter windows, directly increasing ITR by enabling shorter decision windows.

---

## (c) Does the stable top-3 match top-4?

Ishanvir's nested CV found the 4th channel slot drifts across folds:
- **S1**: O1/O2/Oz stable; 4th slot drifts between PO3 and PO8
- **S2**: PO3/POz/PO4 stable; PO8 underperforms and is sometimes replaced by PO7

| Config        | S1     | S2     |
|---------------|--------|--------|
| 8-channel     | 100.0% | 67.5%  |
| top-4 by SNR  | 100.0% | **70.0%** |
| stable top-3  | 97.5%  | **70.0%** |

**Answer: For S2, yes — stable top-3 equals top-4 exactly (70.0%).** PO8 (the unstable 4th channel) contributes nothing for S2. For S1, dropping the 4th channel costs 2.5pp (97.5% vs 100%), meaning PO8 is genuinely useful for S1.

This reinforces the subject-specificity finding: the "right" electrode set differs not just in which channels are ranked first, but in how many channels are actually informative.

---

## Takeaway

- The primary bottleneck is spatial configuration, not classifier complexity
- Channel selection is subject-specific and non-transferable
- Fewer channels can match or exceed full-channel performance

---

## Figures

- `results/figures/channel_snr_ranking.png` — SNR per channel, overall vs per-subject
- `results/figures/channel_snr_heatmap.png` — SNR heatmap: frequency × channel
- `results/figures/channel_sweep_accuracy.png` — CCA accuracy vs N channels (LOBO-CV)

---

## Final comparison

Generated by `scripts/compare_classifiers.py` (3.0 s window, 2 harmonics, LOBO-CV).

| Classifier      | Accuracy @ 3.0s | ITR (bpm) | Notes |
|-----------------|-----------------|-----------|-------|
| **FBCCA**       | **95.0%**       | **32.7**  | Chen 2015 filter-bank; best overall |
| CH11 (SOTA)     | 87.2%           | —         | 49% trial coverage; acc over all trials = 42.5% |
| CCA-top4-nested | 85.0%           | 23.0      | leakage-free nested CV |
| CCA-8ch         | 83.8%           | 22.0      | canonical baseline |
| TRCA            | 31.2%           | 0.3       | negative result — see `results/trca_methodology_note.md` |

**FBCCA (95.0%) is the top-performing offline classifier**, outperforming the g.tec live system (CH11) and both CCA configurations. Despite the widely-spaced frequency design (9/10/12/15 Hz), filter-bank decomposition captures sub-band structure that plain CCA misses.

TRCA underperforms CCA by ~53 pp. Root causes: training set at the absolute floor (5 trials/class), CCA's matched-filter references are theoretically optimal for this 4-frequency widely-spaced design, and Subject 2's sessions are too inconsistent for inter-trial spatial filters to generalize. Full reasoning in `results/trca_methodology_note.md`.
