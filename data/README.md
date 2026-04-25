# data/ — dataset provenance and structure

`data/raw/` and `data/processed/` are gitignored. This file is the only
thing under `data/` that is committed.

## Dataset name and source

**BR41N.IO 2026 Spring School Hackathon — SSVEP Data Analysis project.**
Provided by g.tec medical engineering GmbH at the kickoff session.
Dataset dropped into `data/raw/` on **2026-04-25**.

## Subjects and sessions

- 2 healthy adult subjects (anonymized as `subject_1`, `subject_2`).
- 2 training sessions per subject — `training_1`, `training_2`.
- 4 `.mat` files total: `subject_{1,2}_fvep_led_training_{1,2}.mat`.

## Hardware

g.USBamp biosignal amplifier (per Guger et al. 2012, "How many people
could use an SSVEP BCI?" — bundled in this folder as
`How_many_people_could_use_an_SSVEP_BCI.pdf`). Active electrodes,
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
| CH11 | LDA classifier output (status: see "CH11 status" below) |

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
- **Trial duration:** 7.36 s (1884 samples). Fixed — no jitter.
- **Inter-trial gap:** 3.14 s (804 samples). Fixed — no jitter.
- Total active stim per file: 20 × 7.36 ≈ 147 s.

Trial counts and timings inferred from CH10 transitions; see
`notebooks/01_data_exploration.ipynb` cell 3.

## CH11 status

CH11 is **most likely g.tec's live LDA classifier output, but with partial
coverage and uneven per-subject performance.** Best aggregate accuracy
across all 4 files (after a brute-force sweep over all 24 mappings of LDA
class index → stim frequency) is **0.677**, on 48,402 scored samples
(samples where stim is active *and* CH11 has fired). The recovered
mapping is:

```
{1: 15 Hz, 2: 12 Hz, 3: 10 Hz, 4: 9 Hz}      # descending, not ascending
```

Per-file accuracy varies dramatically:

| File | Best-perm acc | n_scored |
|---|---|---|
| subject_1_fvep_led_training_1 | 0.855 | 5,161 |
| subject_1_fvep_led_training_2 | **0.961** | 5,355 |
| subject_2_fvep_led_training_1 | 0.649 | 18,655 |
| subject_2_fvep_led_training_2 | 0.576 | 19,231 |

Subject 1's runs are near or at SOTA (Guger 2012 reports 95.5% mean across
53 subjects); subject 2's are 30+ points lower. **Investigate latency
offsets and per-trial firing windows (notebook cells 5–6) before treating
CH11 as a SOTA reference.** Reproduce the analysis with
`python scripts/verify_ch11.py`.

## References

- **Guger et al. 2012** — "How many people could use an SSVEP BCI?" —
  hardware/paradigm baseline. PDF: `How_many_people_could_use_an_SSVEP_BCI.pdf`.
- **Chen et al. 2015** — Filter Bank CCA (FBCCA). To be implemented in
  `src/ssvep/classifiers/fbcca.py`.
- **Nakanishi et al. 2017** — Task-Related Component Analysis (TRCA).
  To be implemented in `src/ssvep/classifiers/trca.py`.

## Files NOT in git

The `.mat` files, the Guger 2012 PDF, `montage.png`, and the
`*_result2d.PNG` figures all live under gitignored `data/raw/`. Each
teammate must download them from the hackathon Discord/Drive separately
and drop them into `data/raw/`. This README assumes those files are
already present.

## Conventions

- Drop incoming `.mat` files into `data/raw/` exactly as received.
  **Never rename** the originals — keep them byte-identical to what the
  organizers provided.
- Filtered / epoched arrays go into `data/processed/` with a descriptive
  name (e.g. `S01_bp6-50_notch50_epochs.npz`). Gitignored, so duplicates
  across teammates aren't a problem.

## Provenance log

- **2026-04-25** — Data dropped into `data/raw/` at kickoff. Schema
  confirmed; trial structure (20 × 7.36 s + 3.14 s gap, 5/class)
  characterized; CH11 → LDA mapping recovered (descending freq, 0.677
  aggregate accuracy). See `notebooks/01_data_exploration.ipynb`.
