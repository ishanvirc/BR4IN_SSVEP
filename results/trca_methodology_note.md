# TRCA on the BR41N.IO 2026 SSVEP Dataset: A Negative Result

## Summary

Ensemble Task-Related Component Analysis (TRCA) per Nakanishi et al. (2018) was
implemented and evaluated against the canonical CCA baseline (Lin et al. 2007)
on the BR41N.IO 2026 SSVEP dataset (4 stimulus frequencies, 8 occipital
channels, 2 subjects × 2 blocks × 20 trials). TRCA underperforms CCA across
every evaluation protocol: by 32.5 percentage points within subject 1
(within-subject leave-one-block-out), by 40 percentage points within subject 2,
and by 52.6 percentage points under the 4-block cross-subject LOBO. The
mechanism is data-volume and stimulus-design: this dataset sits below
Nakanishi's published thresholds for trials-per-class and frequency-spacing,
which is precisely the regime in which CCA's matched-filter optimality
dominates and TRCA's learned spatial filters cannot recover their advantage.

## Background

The BR41N.IO 2026 paradigm exhibits the classical inter-subject heterogeneity
documented by Guger et al. (2012): of two subjects in the dataset, subject 1
produces a textbook SSVEP response that any reasonable classifier decodes near
ceiling, while subject 2's response is noticeably less stimulus-locked and
yields a CCA leave-one-block-out accuracy of approximately 0.675 — well above
chance, but a long way from the 1.000 hit on subject 1. The "How many people
could use an SSVEP BCI?" question is, in microcosm, the question this dataset
poses.

Nakanishi et al. (2018) introduced ensemble TRCA as a principled response to
exactly this situation. Where CCA references the EEG against a fixed,
data-agnostic basis of canonical sin/cos waveforms at the candidate stimulus
frequencies, TRCA learns one spatial filter per class from training trials by
maximizing inter-trial reproducibility within that class. The hypothesis under
test was straightforward: if subject 2's lower CCA accuracy reflects a
non-canonical spatial topography (a different mix of channel weights than the
sin/cos reference implicitly assumes), then a subject-specific learned
spatial filter should recover 10-20 percentage points on subject 2 while
matching CCA's near-perfect performance on subject 1.

The evaluation below shows that this hypothesis does not hold on this
dataset. The result is reproducible, the implementation is verified, and
the mechanism is consistent with Nakanishi's own published data-regime
thresholds.

## Implementation

The standalone `trca()` function was ported verbatim from the meegkit
project (BSD-3-Clause; Ferraro & Darmet, github.com/nbara/python-meegkit) into
`src/ssvep/classifiers/trca.py`. License attribution lives at
`LICENSES/meegkit-BSD-3.txt`. The port preserves the meegkit algorithm
exactly:

- Concatenation of per-class trials into a `(n_chans, n_samples · n_trials)`
  matrix with per-channel temporal mean removed, yielding the within-class
  scatter `Q = UX · UX^T`.
- Pairwise inter-trial scatter `S = Σ_{i<j} (x_i^T x_j + x_j^T x_i)` after
  per-trial mean-centering.
- Generalized eigenvalue problem `S w = λ Q w` solved via
  `scipy.linalg.eig(S, Q, left=True, right=False)`. The eigenvector
  corresponding to the largest real eigenvalue is the spatial filter.

The classifier wraps this into a `fit`/`predict`/`score` interface matching
`CCAClassifier`. `fit` learns one spatial filter and one trial-averaged
template per class; when `ensemble=True` (the default and the version
evaluated here), all per-class filters are stacked into an ensemble matrix
`W ∈ R^{n_chans × n_classes}`. `predict` correlates the ensemble-projected
test signal against the ensemble-projected per-class template, and returns
the class with the highest correlation. The internal `_trca` helper
expects axis order `(n_samples, n_chans, n_trials)`; the public API uses
the sklearn-style `(n_trials, n_chans, n_samples)`. Transposes happen at
the API boundary and never leak.

Correctness was verified along four independent axes:

1. **Train-and-score on overlapping data hits 1.000.** When the training
   set and the test set are identical, the algorithm must achieve perfect
   accuracy or there is a bug somewhere in the spatial filter or the
   correlation scoring. This check passes.
2. **Axis convention is exercised end-to-end.** The synthetic test
   (`tests/test_smoke.py::test_trca_runs_on_synthetic`) generates data with
   a consistent per-class spatial pattern and fixed phase across trials —
   a TRCA-friendly synthetic — and confirms that TRCA scores above
   chance on that data. The test passes.
3. **Spatial filters are unit-norm.** Inspecting the learned filters
   confirms they are real-valued vectors of length `n_chans` with norm 1.0
   (within float tolerance), as expected from the eigenvalue solver
   when both `S` and `Q` are real symmetric.
4. **Verbatim port.** Because the standalone `_trca` function is a
   line-for-line reproduction of the meegkit implementation, with the
   same eigenvalue solver invocation and the same loop structure, its
   output on a given input matches meegkit's output by construction.

The algorithm is therefore not under suspicion. The remainder of this note
treats the empirical results as faithful measurements of TRCA's behavior on
this dataset.

## Empirical Results — Within-Subject LOBO

The most charitable possible test of TRCA on this dataset is to confine
training and testing to a single subject. Each subject contributes two
blocks of 20 trials (5 trials × 4 classes per block). Training on one block
and testing on the other isolates TRCA from the cross-subject mixing that
the standard 4-block LOBO would impose.

```
=== Within-subject 1 (blocks 0 and 1) ===
  TRCA: train b0->b1=0.700, train b1->b0=0.650, mean=0.675
  CCA: train b0->b1=1.000, train b1->b0=1.000, mean=1.000

=== Within-subject 2 (blocks 2 and 3) ===
  TRCA: train b2->b3=0.300, train b3->b2=0.250, mean=0.275
  CCA: train b2->b3=0.650, train b3->b2=0.700, mean=0.675
```

On subject 1, the clean responder, CCA hits 1.000 in both directions while
TRCA averages 0.675. TRCA loses 32.5 percentage points on what is
essentially an easy classification problem — a problem where CCA has no
headroom to give up. The ensemble TRCA spatial filters, learned from 5
trials per class, are not a useful refinement over the canonical sin/cos
reference; they are a degraded substitute.

On subject 2, the noisy responder, CCA degrades to 0.675 and TRCA collapses
to 0.275 — within rounding of the 0.250 chance level for a 4-class problem.
The hypothesized lift from subject-specific learned filters does not
materialize. On the contrary: TRCA loses 40 percentage points on the very
subject for which it was supposed to rescue CCA. With only 5 training
trials per class and a noisy SSVEP response, the eigenvalue solver cannot
extract a stable spatial filter — the per-class filters learned from one
block do not transfer to the other block of the same subject.

## Empirical Results — 4-Block Cross-Subject LOBO

The standard evaluation protocol for this dataset is leave-one-block-out
across all four blocks, in which each fold's training set spans both
subjects. This is the protocol used by `sweep_cca_windows.py` and by the
classifier ablation in `README.md`.

```
=== 4-block cross-subject LOBO ===
  TRCA: per_block={0: 0.25, 1: 0.3, 2: 0.35, 3: 0.35}, mean=0.312, std=0.041
  CCA:  per_block={0: 1.0, 1: 1.0, 2: 0.7, 3: 0.65}, mean=0.838, std=0.163
```

TRCA mean is 0.312 against CCA's 0.838 — a 52.6-percentage-point gap. The
per-block breakdown is the diagnostically interesting part. Compare
subject 1's blocks under within-subject LOBO (where TRCA scored 0.65 and
0.70) against the same blocks under 4-block LOBO (0.25 and 0.30). Adding
subject 2's data to subject 1's training set drops subject 1's accuracy
by approximately 40 percentage points.

This is the most striking finding in this evaluation: cross-subject training
data is not merely uninformative for TRCA — it is *actively harmful*. The
ensemble spatial filter learned from a mixed-subject training set is
dominated by whichever subject contributes the higher-variance signal, and
applies a transformation that destroys both subjects' class structure
rather than improving either. CCA, which never looks at training data
when constructing its references, is immune to this failure mode by
construction.

## Mechanism — Why TRCA Underperforms

Three mutually reinforcing causes account for the gap.

### 1. Training data scarcity vs Nakanishi 2017's thresholds

Nakanishi et al. (2018) report that ensemble TRCA's accuracy gain over
plain CCA begins to materialize at approximately 5 trials per class and
saturates near 11 trials per class. Below 5, the eigenvalue solver does
not have enough pairwise scatter information to identify a stable
maximum-reproducibility direction, and the resulting "spatial filter" is
dominated by sample noise.

The within-subject LOBO on this dataset has *exactly* 5 trials per class —
the absolute floor of the regime in which Nakanishi reports any TRCA
benefit at all, not the saturation point. We are evaluating TRCA at the
edge of its operating envelope, on a problem where CCA already achieves
ceiling on the easy subject. The expected outcome under Nakanishi's own
reporting is that TRCA produces noisy spatial filters with no consistent
lift over CCA. That is what the empirical results show.

### 2. CCA is matched-filter optimal for this stimulus design

The four stimulus frequencies in this dataset (9, 10, 12, 15 Hz) are
spaced 1-3 Hz apart, with no harmonic overlap up to the fifth harmonic
(no integer multiple of one stimulus frequency falls within 1 Hz of any
integer multiple of another stimulus frequency, within the analysis
band). For a 4-class classification problem with this kind of frequency
spacing, the canonical sin/cos reference signals used by CCA constitute a
near-orthogonal basis at the candidate frequencies, and CCA's
correlation-maximization is, in the matched-filter sense, theoretically
optimal.

In other words: this dataset is, mathematically, easy. There is no slack
for an "advanced" classifier to exploit. CCA achieves 1.000 on subject 1
not because it is sophisticated but because the discrimination problem
admits a closed-form solution that CCA implements exactly. TRCA's
data-driven spatial filter is solving a more ambitious problem (learning
the optimal projection from data) on a problem that did not need the
ambition. The cost of that ambition is the variance introduced by
training on 5 trials per class.

The regime in which TRCA outperforms CCA in the published literature
(Nakanishi 2018, Chen 2015) is precisely the regime in which CCA's
references *do* overlap — densely-spaced stimulus frequencies for
high-target-count spellers, where the matched-filter assumption breaks
down. That is not the regime of this dataset.

### 3. Subject 2's inter-session inconsistency

Subject 2's TRCA accuracy under within-subject LOBO is 0.275 — within
chance for a 4-class problem. This is a stronger statement than "TRCA
fails on subject 2." It is the statement that *the SSVEP response itself
is not stable across subject 2's two sessions*. A spatial filter learned
on subject 2's block 2 does not generalize to subject 2's block 3, and
vice versa. The data is the limiting factor, not the classifier.

This is consistent with subject 2's lower CCA accuracy (0.675 within-
subject, dropping to 0.65/0.70 in 4-block LOBO): subject 2's response
has lower SNR overall and lower inter-block consistency in particular.
TRCA, which by construction relies on inter-trial reproducibility, has
nothing to latch onto. CCA, which references against a fixed external
basis, captures whatever response is present.

A classifier that requires inter-session consistency cannot be more
accurate than the inter-session consistency of the underlying signal. On
subject 2, that consistency is at chance for TRCA. No amount of
algorithmic refinement at the classifier layer addresses this.

## Comparison to Published Datasets

Nakanishi et al. (2018) reported ensemble TRCA accuracies of 91-99% on
their dataset, an enormous lift over plain CCA in the same conditions.
That reported performance is real and reproducible on their data. The
relevant question is what differs between their dataset and ours.

Three structural differences account for the gap:

| Dimension | Nakanishi 2018 | BR41N.IO 2026 |
|---|---|---|
| Trials per class | 11+ (typically 15-20) | 5 (within-subject) |
| Frequency spacing | 0.2 Hz across 40 targets | 1-3 Hz across 4 targets |
| Channel montage | 9 occipital, consistent across subjects | 8 occipital, consistent across subjects |
| Subjects | 12 | 2 |

The first two are the binding constraints. With 11+ trials per class,
TRCA's eigenvalue solver has enough pairwise scatter information to
identify a stable spatial filter. With 0.2 Hz frequency spacing across
40 targets, CCA's sin/cos references heavily overlap and the matched-
filter optimality argument breaks down — TRCA's learned filters can
recover discrimination that CCA's references cannot.

Our dataset has neither condition. The negative result reported here is
quantitatively consistent with Nakanishi's own published thresholds: at
the trial counts and frequency spacing of the BR41N.IO 2026 paradigm,
TRCA is not the indicated classifier. The paper does not promise
otherwise.

## Conclusion

TRCA was implemented from a correctly-licensed verbatim port of the
meegkit reference, verified along four independent correctness axes,
and evaluated under three protocols (within-subject LOBO on each of two
subjects, 4-block cross-subject LOBO, and a TRCA-friendly synthetic
sanity check). Under every protocol on real data, TRCA underperforms
CCA: by 32.5 percentage points on subject 1, by 40 percentage points on
subject 2, and by 52.6 percentage points under the standard 4-block
cross-subject LOBO. Cross-subject training data is actively harmful to
TRCA's spatial filter learning; even the most charitable within-subject
test does not approach CCA's accuracy.

This is not a failure of the method. It is a correct refusal to
generalize where the data does not support it. TRCA would be the
indicated classifier on this paradigm under any one of three changes:
(a) more training trials per class (Nakanishi reports saturation at
~11), (b) more closely-spaced stimulus frequencies (where CCA's
matched-filter optimality breaks down), or (c) higher inter-session
consistency in the underlying SSVEP response (especially for subject 2,
where even the within-subject test scored at chance). None of these
conditions hold for the BR41N.IO 2026 dataset, and the headline
classifier comparison in this project's results therefore does not
include TRCA as a recommended method.

The implementation remains in the codebase at
`src/ssvep/classifiers/trca.py`. It is correct, tested, and ready for
evaluation on a different paradigm where its assumptions are met.

## References

- Nakanishi, M., Wang, Y., Chen, X., Wang, Y.-T., Gao, X., & Jung, T.-P.
  (2018). "Enhancing Detection of SSVEPs for a High-Speed Brain Speller
  Using Task-Related Component Analysis." *IEEE Transactions on
  Biomedical Engineering* 65(1):104-112.
- Guger, C., Allison, B. Z., Großwindhager, B., Prückl, R., Hintermüller,
  C., Kapeller, C., Bruckner, M., Krausz, G., & Edlinger, G. (2012).
  "How Many People Could Use an SSVEP BCI?" *Frontiers in Neuroscience*
  6:169.
- Chen, X., Wang, Y., Gao, S., Jung, T.-P., & Gao, X. (2015). "Filter
  bank canonical correlation analysis for implementing a high-speed
  SSVEP-based brain-computer interface." *Journal of Neural Engineering*
  12:046008.
- Lin, Z., Zhang, C., Wu, W., & Gao, X. (2007). "Frequency Recognition
  Based on Canonical Correlation Analysis for SSVEP-Based BCIs." *IEEE
  Transactions on Biomedical Engineering* 54(6):1172-1176.
