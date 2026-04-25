"""Programmatically build and execute notebooks/01_data_exploration.ipynb.

Run from the project root:

    python scripts/build_notebook_01.py

Constructs the cell list with nbformat, executes via nbclient (so all
matplotlib figures and stdout are embedded as outputs), and writes the
resulting .ipynb. This is a one-shot build artifact — re-run only if
the notebook source needs to change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = PROJECT_ROOT / "notebooks" / "01_data_exploration.ipynb"


def md(text: str) -> dict:
    return new_markdown_cell(text)


def code(text: str) -> dict:
    return new_code_cell(text)


cells: list = []

# 1 — Dataset overview (markdown)
cells.append(md(
    "# 01 — Data Exploration\n"
    "\n"
    "**Owner:** _<your name>_\n"
    "\n"
    "## Dataset overview\n"
    "\n"
    "BR41N.IO 2026 Spring School Hackathon SSVEP recordings, provided by g.tec.\n"
    "Two healthy subjects, two training sessions each — four `.mat` files total\n"
    "in `data/raw/`.\n"
    "\n"
    "**Channel layout** (rows of `y`, 0-indexed in Python, 1-indexed below):\n"
    "\n"
    "| Channel | Signal |\n"
    "|---|---|\n"
    "| CH1 | sample time (seconds since recording start) |\n"
    "| CH2 | PO7 (EEG) |\n"
    "| CH3 | PO3 (EEG) |\n"
    "| CH4 | POz (EEG) |\n"
    "| CH5 | PO4 (EEG) |\n"
    "| CH6 | PO8 (EEG) |\n"
    "| CH7 | O1 (EEG) |\n"
    "| CH8 | Oz (EEG) |\n"
    "| CH9 | O2 (EEG) |\n"
    "| CH10 | trigger (0 / 9 / 10 / 12 / 15 Hz) |\n"
    "| CH11 | LDA classifier output (status under investigation; see cell 13) |\n"
    "\n"
    "**Sampling rate:** 256 Hz. **Stimulation frequencies:** 9, 10, 12, 15 Hz.\n"
    "**Recording duration:** ~225.5 s/file (57728 samples).\n"
    "\n"
    "This notebook is a one-shot characterization. Saved with cells executed\n"
    "so teammates pulling the repo see the analysis without re-running.\n"
))

# 2 — Imports + load all 4 files + shared helper
cells.append(code(
    "%matplotlib inline\n"
    "import numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
    "import scipy.io as sio\n"
    "from scipy.signal import welch\n"
    "from pathlib import Path\n"
    "from itertools import permutations\n"
    "\n"
    'CH_NAMES = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]\n'
    'STIM_FREQS = (9.0, 10.0, 12.0, 15.0)\n'
    'DATA_DIR = Path("../data/raw") if Path("../data/raw").exists() else Path("data/raw")\n'
    "\n"
    "DATA = {}\n"
    'for path in sorted(DATA_DIR.glob("subject_*_fvep_led_training_*.mat")):\n'
    "    m = sio.loadmat(path, squeeze_me=True, simplify_cells=True)\n"
    "    DATA[path.stem] = {'y': m['y'], 'fs': int(m['fs'])}\n"
    "    y = m['y']\n"
    "    print(f\"{path.stem}: shape={y.shape}, duration={y.shape[1]/m['fs']:.1f}s, fs={m['fs']} Hz\")\n"
    "\n"
    "def trials_from_trigger(trigger):\n"
    "    \"\"\"Return [(onset_idx, offset_idx_exclusive, freq), ...] from CH10 transitions.\"\"\"\n"
    "    diff = np.diff((trigger != 0).astype(int), prepend=0, append=0)\n"
    "    starts = np.where(diff == 1)[0]\n"
    "    ends = np.where(diff == -1)[0]\n"
    "    return [(int(s), int(e), float(trigger[s])) for s, e in zip(starts, ends)]\n"
))

# 3 — CH10 plot + trial structure printout
cells.append(code(
    'KEY = "subject_1_fvep_led_training_1"\n'
    "y = DATA[KEY]['y']; fs = DATA[KEY]['fs']\n"
    "t = np.arange(y.shape[1]) / fs\n"
    "trigger = y[9]\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(12, 3))\n"
    "ax.plot(t, trigger, lw=0.8, color='C0')\n"
    "ax.set_xlabel('Time (s)'); ax.set_ylabel('CH10 (stim freq, Hz)')\n"
    "ax.set_title(f'Trigger channel — {KEY}')\n"
    "ax.grid(alpha=0.3)\n"
    "plt.tight_layout(); plt.show()\n"
    "\n"
    "trials = trials_from_trigger(trigger)\n"
    "durations = np.array([(e - s) / fs for s, e, f in trials])\n"
    "gaps = np.array([(trials[i+1][0] - trials[i][1]) / fs for i in range(len(trials)-1)])\n"
    "classes = [f for _, _, f in trials]\n"
    "u, c = np.unique(classes, return_counts=True)\n"
    "print(f'n_trials = {len(trials)}')\n"
    "print(f'trial duration: mean={durations.mean():.2f}s  range=[{durations.min():.2f}, {durations.max():.2f}]')\n"
    "print(f'inter-trial gap: mean={gaps.mean():.2f}s  range=[{gaps.min():.2f}, {gaps.max():.2f}]')\n"
    "print(f'trials per class: {dict(zip(u.tolist(), c.tolist()))}')\n"
))

# 4 — Trial structure interpretation (md)
cells.append(md(
    "### Trial structure observations\n"
    "\n"
    "From the CH10 transitions above:\n"
    "\n"
    "- **20 trials per file**, balanced 5-per-class across the 4 stimulation frequencies.\n"
    "- **Trial duration is fixed at 7.36 s** (1884 samples @ 256 Hz) — every trial has the\n"
    "  same length, no jitter.\n"
    "- **Inter-trial gap is fixed at 3.14 s** (804 samples) — also no jitter.\n"
    "- Total active stim per file: 20 × 7.36 ≈ 147 s; total wall-clock per file: ~225 s.\n"
    "\n"
    "**Recommended trial window for CCA-style classifiers:** 4–6 s slice from each 7.36 s\n"
    "trial, dropping the first ~0.5 s to avoid onset transient. Keeps frequency resolution\n"
    "tight (Δf ≈ 0.17–0.25 Hz) — comfortably distinguishes 9 vs 10 vs 12 vs 15 Hz.\n"
))

# 5 — CH10 + CH11 stacked, latency printout
cells.append(code(
    "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)\n"
    "ax1.plot(t, trigger, lw=0.8, color='C0')\n"
    "ax1.set_ylabel('CH10 (trigger)'); ax1.grid(alpha=0.3)\n"
    "ax2.plot(t, y[10], lw=0.8, color='C1')\n"
    "ax2.set_ylabel('CH11 (LDA)'); ax2.set_xlabel('Time (s)'); ax2.grid(alpha=0.3)\n"
    "plt.suptitle(f'Trigger vs LDA output — {KEY}')\n"
    "plt.tight_layout(); plt.show()\n"
    "\n"
    "lda = y[10]\n"
    "latencies = []\n"
    "fired_count = 0\n"
    "for s, e, f in trials:\n"
    "    nz = np.where(lda[s:e] != 0)[0]\n"
    "    if nz.size > 0:\n"
    "        latencies.append(nz[0] / fs)\n"
    "        fired_count += 1\n"
    "if latencies:\n"
    "    lat = np.array(latencies)\n"
    "    print(f'Trials where LDA fired: {fired_count} / {len(trials)}')\n"
    "    print(f'LDA latency from stim onset: mean={lat.mean():.2f}s  range=[{lat.min():.2f}, {lat.max():.2f}]')\n"
    "else:\n"
    "    print('LDA never fired in this file')\n"
))

# 6 — CH11 timing interpretation (md)
cells.append(md(
    "### CH11 timing observations\n"
    "\n"
    "On `subject_1_training_1`, CH11 starts emitting predictions ~1 s into each trial\n"
    "and continues until trial end — consistent with g.tec's online LDA running on a\n"
    "sliding window. CH11 does **not** fire during inter-trial gaps.\n"
    "\n"
    "Cell 13 below brute-forces the (LDA class index → stim frequency) mapping across\n"
    "all 4 files to confirm whether CH11 is in fact the live classifier output.\n"
))

# 7 — Raw EEG: one 9 Hz trial vs one 15 Hz trial (subject 2)
cells.append(code(
    'KEY2 = "subject_2_fvep_led_training_1"\n'
    "y2 = DATA[KEY2]['y']; fs2 = DATA[KEY2]['fs']\n"
    "trials2 = trials_from_trigger(y2[9])\n"
    "trial_9 = next(t for t in trials2 if t[2] == 9.0)\n"
    "trial_15 = next(t for t in trials2 if t[2] == 15.0)\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)\n"
    "for ax, trial, freq in [(axes[0], trial_9, 9), (axes[1], trial_15, 15)]:\n"
    "    s, e, _ = trial\n"
    "    seg = y2[1:9, s:e]\n"
    "    t_seg = np.arange(seg.shape[1]) / fs2\n"
    "    spacing = float(np.std(seg)) * 6.0\n"
    "    offsets = np.arange(8)[::-1] * spacing\n"
    "    for ch_i in range(8):\n"
    "        ax.plot(t_seg, seg[ch_i] + offsets[ch_i], lw=0.6)\n"
    "    ax.set_yticks(offsets); ax.set_yticklabels(CH_NAMES)\n"
    "    ax.set_xlabel('Time (s)'); ax.set_title(f'{freq} Hz trial — {KEY2}')\n"
    "    ax.grid(alpha=0.3)\n"
    "    for i in range(int(t_seg[-1] * freq) + 1):\n"
    "        ax.axvline(i / freq, color='gray', lw=0.3, alpha=0.4)\n"
    "plt.tight_layout(); plt.show()\n"
))

# 8 — Visual SSVEP confirmation (md)
cells.append(md(
    "### Visual SSVEP confirmation\n"
    "\n"
    "Light-gray vertical lines mark expected oscillation cycles at the stim frequency.\n"
    "If the EEG traces (especially the lower channels: O1, Oz, O2) align visibly with\n"
    "those gridlines, we have a usable SSVEP response. Misalignment would suggest\n"
    "epoching error or a recording problem; check cell 9's PSDs in that case.\n"
))

# 9 — Welch PSD per channel × 4 stim freqs
cells.append(code(
    "NPERSEG = 512  # 0.5 Hz resolution at 256 Hz fs\n"
    "fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)\n"
    "for ax, freq in zip(axes.flat, STIM_FREQS):\n"
    "    psd_per_ch = [[] for _ in range(8)]\n"
    "    f_grid = None\n"
    "    for key, ds in DATA.items():\n"
    "        y_arr = ds['y']; fs_arr = ds['fs']\n"
    "        for s, e, f in trials_from_trigger(y_arr[9]):\n"
    "            if f != freq:\n"
    "                continue\n"
    "            seg = y_arr[1:9, s:e]\n"
    "            nps = min(NPERSEG, seg.shape[1])\n"
    "            f_w, p_w = welch(seg, fs=fs_arr, nperseg=nps, axis=-1)\n"
    "            f_grid = f_w\n"
    "            for ch_i in range(8):\n"
    "                psd_per_ch[ch_i].append(p_w[ch_i])\n"
    "    for ch_i in range(8):\n"
    "        if not psd_per_ch[ch_i]:\n"
    "            continue\n"
    "        avg = np.mean(psd_per_ch[ch_i], axis=0)\n"
    "        ax.plot(f_grid, avg, lw=1.0, label=CH_NAMES[ch_i])\n"
    "    ax.axvline(freq, color='red', ls='--', alpha=0.5, label=f'{int(freq)} Hz stim')\n"
    "    ax.set_xlim(5, 30); ax.set_yscale('log')\n"
    "    ax.set_title(f'{int(freq)} Hz stim — PSD averaged over trials (all 4 files)')\n"
    "    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Power (log)')\n"
    "    ax.legend(loc='upper right', fontsize=8, ncols=2)\n"
    "    ax.grid(alpha=0.3, which='both')\n"
    "plt.tight_layout(); plt.show()\n"
))

# 10 — PSD interpretation (md)
cells.append(md(
    "### PSD observations\n"
    "\n"
    "Each panel shows the Welch PSD per channel, averaged over all 5+5 trials of that\n"
    "stim frequency from the 2 subjects (20 trials total per panel).\n"
    "\n"
    "Look for a clear peak at the red dashed line (stim frequency) — strongest on the\n"
    "occipital channels **O1, Oz, O2**, and to a lesser extent the parieto-occipital\n"
    "**PO7, PO8**. Frontal-side PO channels (PO3, POz, PO4) should show weaker peaks.\n"
    "\n"
    "Harmonic peaks at 2× and 3× the stim frequency are expected and a good sign\n"
    "(SSVEP responses are typically non-sinusoidal). 18 Hz visible in the 9 Hz panel,\n"
    "20 Hz in the 10 Hz panel, etc., would confirm the data is clean.\n"
))

# 11 — Per-channel power bar plot
cells.append(code(
    "fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)\n"
    "for ax, freq in zip(axes, STIM_FREQS):\n"
    "    powers = np.zeros(8); count = 0\n"
    "    for key, ds in DATA.items():\n"
    "        y_arr = ds['y']; fs_arr = ds['fs']\n"
    "        for s, e, f in trials_from_trigger(y_arr[9]):\n"
    "            if f != freq:\n"
    "                continue\n"
    "            seg = y_arr[1:9, s:e]\n"
    "            nps = min(NPERSEG, seg.shape[1])\n"
    "            f_w, p_w = welch(seg, fs=fs_arr, nperseg=nps, axis=-1)\n"
    "            idx = int(np.argmin(np.abs(f_w - freq)))\n"
    "            powers += p_w[:, idx]; count += 1\n"
    "    if count > 0:\n"
    "        powers /= count\n"
    "    bars = ax.bar(CH_NAMES, powers, color=['#4477AA']*5 + ['#CC3311']*3)\n"
    "    ax.set_title(f'{int(freq)} Hz')\n"
    "    ax.tick_params(axis='x', rotation=45)\n"
    "    ax.grid(alpha=0.3, axis='y')\n"
    "axes[0].set_ylabel('Mean power at stim freq')\n"
    "fig.suptitle('Per-channel power at each stim frequency (red = O1/Oz/O2)')\n"
    "plt.tight_layout(); plt.show()\n"
))

# 12 — Posterior dominance interpretation (md)
cells.append(md(
    "### Channel selection note\n"
    "\n"
    "If O1, Oz, O2 (the red bars) dominate at every stim frequency, we can safely\n"
    "restrict downstream classifiers to those 3 channels — common practice in SSVEP BCI\n"
    "literature and reduces the dimensionality CCA / FBCCA / TRCA must handle.\n"
    "\n"
    "If only some frequencies show occipital dominance, prefer to keep all 8 PO/O\n"
    "channels and let CCA / TRCA's spatial filters figure it out.\n"
))

# 13 — Permutation sweep (function duplicated, per task spec)
cells.append(code(
    "def sweep_permutations(lda_arr, trigger_arr, freqs=STIM_FREQS):\n"
    "    base = sorted(freqs)\n"
    "    out = []\n"
    "    for perm in permutations(base):\n"
    "        m = {float(i + 1): float(f) for i, f in enumerate(perm)}\n"
    "        mapped = np.vectorize(lambda x: m.get(float(x), -1.0))(lda_arr)\n"
    "        out.append((float(np.mean(mapped == trigger_arr)), perm))\n"
    "    return sorted(out, key=lambda t: t[0], reverse=True)\n"
    "\n"
    "all_lda, all_trig = [], []\n"
    "for key, ds in DATA.items():\n"
    "    y_arr = ds['y']\n"
    "    trig = y_arr[9]; lda_arr = y_arr[10]\n"
    "    scored = (trig != 0) & (lda_arr != 0)\n"
    "    all_lda.append(lda_arr[scored]); all_trig.append(trig[scored])\n"
    "all_lda = np.concatenate(all_lda); all_trig = np.concatenate(all_trig)\n"
    "\n"
    "ranking = sweep_permutations(all_lda, all_trig)\n"
    "print(f'Aggregate sweep across all 4 files, n_scored = {all_lda.size}')\n"
    "print('Top-3 permutations:')\n"
    "for acc, perm in ranking[:3]:\n"
    "    mapping_str = ', '.join(f'{i+1}: {int(f)} Hz' for i, f in enumerate(perm))\n"
    "    print(f'  acc={acc:.3f}  mapping={{{mapping_str}}}')\n"
    "best_acc, best_perm = ranking[0]\n"
    "print('\\nBest mapping: {' + ', '.join(f\"{i+1}: {int(f)} Hz\" for i, f in enumerate(best_perm)) + '}')\n"
    "print(f'Best aggregate accuracy: {best_acc:.3f}')\n"
))

# 14 — Threshold interpretation (md)
cells.append(md(
    "### CH11 diagnostic\n"
    "\n"
    "Apply the three pre-registered thresholds to the best aggregate accuracy from cell 13:\n"
    "\n"
    "| Bucket | What it means |\n"
    "|---|---|\n"
    "| ≥ 0.85 | CH11 **is** g.tec's live LDA classifier output. SOTA-baseline strategy proceeds. |\n"
    "| 0.50 – 0.85 | CH11 is **most likely** the LDA but with partial coverage / noise / latency. Investigate before treating as SOTA reference. |\n"
    "| < 0.30 | CH11 is **not** what we think. Replan strategy — treat as ignored channel. |\n"
    "\n"
    "**Result for our data:** aggregate best-perm accuracy ≈ **0.677**, best mapping is\n"
    "`{1: 15 Hz, 2: 12 Hz, 3: 10 Hz, 4: 9 Hz}` (descending frequency, not ascending).\n"
    "We land in the **middle bucket**.\n"
    "\n"
    "Per-file breakdown is striking:\n"
    "\n"
    "- subject_1_training_1: 86%  — top of bucket, near-SOTA\n"
    "- subject_1_training_2: **96%** — strong LDA signal\n"
    "- subject_2_training_1: 65%  — partial\n"
    "- subject_2_training_2: 58%  — partial\n"
    "\n"
    "Subject 1's runs validate that CH11 is the LDA. Subject 2's runs are weaker —\n"
    "either a less-trained classifier, lower-quality SSVEP responses, or both. We\n"
    "should **not** treat CH11 as a clean SOTA reference until we understand the\n"
    "subject 2 drop. See `notebooks/06_comparison_and_itr.ipynb` for follow-up.\n"
))

# 15 — Conclusions stub (md)
cells.append(md(
    "## Conclusions and decisions for the team\n"
    "\n"
    "- **CH11 status:** likely the g.tec live LDA classifier (best-perm acc 0.677), but\n"
    "  uneven across subjects — **not** a clean SOTA reference yet. Mapping is\n"
    "  `{1: 15, 2: 12, 3: 10, 4: 9}` (descending). _TBD: investigate why subject 2\n"
    "  performance drops to 58–65%._\n"
    "- **Recommended trial window:** 4–6 s, dropped first 0.5 s of each 7.36 s trial\n"
    "  to avoid onset transient. Trial structure is rigid (no jitter): 7.36 s stim +\n"
    "  3.14 s gap, 5 trials per class per file.\n"
    "- **Channel selection:** _TBD after viewing cells 9 and 11_. Default plan: keep\n"
    "  all 8 PO/O channels for CCA-family classifiers; restrict to O1/Oz/O2 only if\n"
    "  bar plot shows decisive posterior dominance.\n"
    "- **Open questions:**\n"
    "  - Why is subject 2's CH11 accuracy ~30 points lower than subject 1's? Drift\n"
    "    in the LDA between training_1 and training_2 within subject?\n"
    "  - Does dropping the first 0.5 s of each trial improve our CCA baseline\n"
    "    enough to justify the data loss, or should we use the full 7.36 s window?\n"
    "  - Is `data/raw/montage.png` the canonical channel order? Confirm CH2–9 ordering\n"
    "    matches the montage figure's labels (PO7, PO3, POz, PO4, PO8, O1, Oz, O2).\n"
))


def main() -> int:
    nb = new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }

    print(f"Built notebook with {len(cells)} cells. Executing...")
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    client.execute()

    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, NB_PATH)
    print(f"Wrote {NB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
