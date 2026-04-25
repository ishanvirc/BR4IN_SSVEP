# data/ — provenance notes

`data/raw/` and `data/processed/` are gitignored. This file is the only
thing in `data/` that is committed.

Fill in the table below the moment the dataset drops at kickoff so anyone
joining mid-hackathon can orient themselves quickly.

## Dataset facts (fill in after kickoff)

| Field                    | Value |
|--------------------------|-------|
| Dataset name             | _e.g. BR41N.IO 2026 SSVEP_ |
| Source / link            | _URL or "handed out at kickoff"_ |
| Number of subjects       | _N_ |
| File naming convention   | _e.g. `S<id>_<session>.mat`_ |
| Channel count + names    | _e.g. 8: O1, O2, Oz, ..._ |
| Sampling rate (Hz)       | _e.g. 256_ |
| Stimulation frequencies  | _e.g. 7.5, 8.57, 10, 12 Hz_ |
| Trial / window length    | _e.g. 4 s post-stim_ |
| Block / session structure| _trials per block, blocks per subject_ |
| Label scheme             | _0..N-1, mapped to stim_freqs by index?_ |

## `.mat` schema notes

When you first load a file, paste the output of:

```python
from scipy.io import loadmat
m = loadmat("data/raw/<file>.mat", squeeze_me=True, struct_as_record=False, simplify_cells=True)
print(sorted(k for k in m.keys() if not k.startswith("__")))
```

so the team knows which keys carry which fields. Once confirmed, update the
candidate-key tuples at the top of [src/ssvep/io.py](../src/ssvep/io.py).

## Conventions

- Drop incoming `.mat` files into `data/raw/` exactly as received. **Never
  rename** the originals — keep them byte-identical to what the organizers
  provided.
- If you generate filtered / epoched arrays, write them into
  `data/processed/` with a descriptive name (e.g.
  `S01_bp6-50_notch50_epochs.npz`). These are gitignored so duplicates
  across teammates aren't a problem.
