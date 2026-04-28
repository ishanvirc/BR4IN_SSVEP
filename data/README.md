# Dataset provenance and structure

`data/raw/` and `data/processed/` are gitignored. This file is the only
thing under `data/` that is committed.

## Dataset name and source

**BR41N.IO 2026 Spring School Hackathon / SSVEP Data Analysis project.**
Provided by g.tec medical engineering GmbH at the kickoff session.
Dataset dropped into `data/raw/` on **2026-04-25**.

## Subjects and sessions

- 2 healthy adult subjects (anonymized as `subject_1`, `subject_2`).
- 2 training sessions per subject: `training_1`, `training_2`.
- 4 `.mat` files total: `subject_{1,2}_fvep_led_training_{1,2}.mat`.

## Hardware

g.USBamp biosignal amplifier (per Guger et al. 2012, "How many people
could use an SSVEP BCI?" bundled in this folder as `How_many_people_could_use_an_SSVEP_BCI.pdf`). Active electrodes,
reference at right earlobe, ground at FPz.

## Channel layout

| Channel | Signal |
|---|---|
| CH1  | Sample time (seconds since recording start) |
| CH2  | PO7 (EEG) |
| CH3  | PO3 (EEG) |
| CH4  | POz (EEG) |
| CH5  | PO4 (EEG) |
| CH6  | PO8 (EEG) |
| CH7  | O1 (EEG) |
| CH8  | Oz (EEG) |
| CH9  | O2 (EEG) |
| CH10 | Trigger (stim freq in Hz; 0 when off) |
| CH11 | LDA classifier output (see "CH11, g.tec LDA Classifier Output" below) |

EEG channel order is the assumed g.tec montage; confirm against
`montage.png` in this folder before relying on CH-name → physical
electrode pairings.

## Recording parameters

- **Sampling rate:** 256 Hz (channel `fs` in the `.mat` file).
- **Recording duration:** ~225.5 s per file (57728 samples).
- **`.mat` schema:** keys are `fs` (int) and `y` (float64 ndarray, shape
  `(11, n_samples)`). Verified via `scipy.io.loadmat(..., simplify_cells=True)`
  on 2026-04-25.

## Stimulation paradigm

- 4 LEDs flickering at **9, 10, 12, 15 Hz**.
- **20 trials per file**, balanced **5 per class** across the 4 frequencies.
- **Trial duration:** 7.36 s (1884 samples). Fixed.
- **Inter-trial gap:** 3.14 s (804 samples). Fixed.
- Total active stim per file: 20 × 7.36 ≈ 147 s.

Trial counts and timings inferred from CH10 transitions; see
`notebooks/01_data_exploration.ipynb` cell 3.

## CH11: g.tec LDA Classifier Output

CH11 contains g.tec's live LDA classifier predictions, recovered via
permutation sweep over the 4! possible class-index → frequency mappings.

**Mapping (descending, not ascending):**

- Class 1 → 15 Hz
- Class 2 → 12 Hz
- Class 3 → 10 Hz
- Class 4 → 9 Hz

**Two accuracy numbers:**

- **Per-sample accuracy: 0.677** across 48,402 scored samples (aggregate,
  best permutation). This number is diluted by sticky/latched values
  carrying forward across trials and by long no-fire stretches; it is
  **not** a fair representation of CH11's quality as a per-trial classifier.
- **Per-trial accuracy: 0.872** across 39 of 80 trials evaluated (last 100
  samples per trial, majority vote, best-perm mapping). This is the
  **BCI-relevant** number, one prediction per trial, mirroring how a real
  speller would use CH11.

**Coverage is the catch:** 41 of 80 trials (51%) have **no** LDA prediction
in the last 100 samples of the trial. The classifier either does not fire
on those trials, or fires earlier and the value is no longer in the window.
See `notebooks/01_data_exploration.ipynb`, cell 13, for the per-file breakdown.

**Per-file split (per-trial accuracy on evaluated trials):**

| File | Per-trial acc | Skipped (no-fire) |
|---|---|---|
| subject_1_fvep_led_training_1 | 4/4 = **100.0%** | 16/20 |
| subject_1_fvep_led_training_2 | 6/6 = **100.0%** | 14/20 |
| subject_2_fvep_led_training_1 | 13/14 = **92.9%** | 6/20 |
| subject_2_fvep_led_training_2 | 11/15 = **73.3%** | 5/20 |

**Strategic implication:** Subject 1 sessions are clean SOTA territory
when CH11 fires; subject 2 sessions show the universality problem
described in Guger et al. 2012. **Coverage and accuracy are both metrics
worth comparing our methods against.**

Reproduce with `python scripts/verify_ch11.py` (per-sample sweep) or
`notebooks/01_data_exploration.ipynb` cell 13 (per-sample + per-trial).

## References

- **Guger et al. 2012**: "How many people could use an SSVEP BCI?". Hardware/paradigm baseline. 
- **Chen et al. 2015** — Filter Bank CCA (FBCCA). To be implemented in `src/ssvep/classifiers/fbcca.py`.
- **Nakanishi et al. 2017** — Task-Related Component Analysis (TRCA). To be implemented in `src/ssvep/classifiers/trca.py`.

## Files NOT in git

The `.mat` files, the Guger 2012 PDF, `montage.png`, and the
`*_result2d.PNG` figures all live under gitignored `data/raw/`. Each
teammate must download them from the hackathon Discord/Drive separately
and drop them into `data/raw/`. This README assumes those files are
already present.

## Provenance log

- **2026-04-25** — Data dropped into `data/raw/` at kickoff. Schema
  confirmed; trial structure (20 × 7.36 s + 3.14 s gap, 5/class)
  characterized; CH11 → LDA mapping recovered (descending freq, 0.677
  aggregate accuracy). See `notebooks/01_data_exploration.ipynb`.
