"""CCA window-length sweep — slide-ready curve for the TRCA comparison.

Runs LOBO-CV CCA across 10 trial-window lengths (0.5 - 5.0 s) in two
configurations:

    Config A — 8-channel canonical
        All 8 EEG channels, canonical-preprocessed pipeline.

    Config B — per-subject top-4 (Yuki's finding)
        Subject 1: ['O2','O1','Oz','PO8']
        Subject 2: ['PO4','PO8','POz','PO3']
        Source: notebooks/08_channel_selection.ipynb (cell 6 SNR ranking).

Outputs
-------
- results/tables/cca_window_sweep.csv  (10 windows x 2 configs = 20 rows)
- results/figures/cca_window_sweep.png (2 panels: accuracy + ITR vs window)

Run from project root:

    python scripts/sweep_cca_windows.py

Methodology note (Config B leakage)
-----------------------------------
Per-subject top-4 channels were selected by Yuki using SNR ranking
computed on each subject's full 40-trial set, with no held-out fold
for the ranking step. The LOBO-CV here uses those rankings as fixed
channel sets, but they were fit on the same data — accuracy is
therefore an optimistic estimate. Treat Config B as a directional
result ("per-subject adaptation matters"), not a clean cross-validated
number. A nested CV pass that re-ranks channels per training fold is
the rigorous follow-up.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Allow running without `pip install -e .`.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ssvep.io import load_all
from ssvep.classifiers import CCAClassifier
from ssvep.evaluation import itr, leave_one_block_out_cv

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


# --- Constants -------------------------------------------------------------

WINDOWS: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
N_CLASSES = 4
ACC_THRESHOLD = 0.80                        # "TRCA must beat" benchmark

# Per-subject top-4 channel selections — see module docstring for source
# and methodology caveat.
TOP4_BY_BLOCK: dict[int, list[str]] = {
    0: ["O2", "O1", "Oz", "PO8"],     # subject 1, training 1
    1: ["O2", "O1", "Oz", "PO8"],     # subject 1, training 2
    2: ["PO4", "PO8", "POz", "PO3"],  # subject 2, training 1
    3: ["PO4", "PO8", "POz", "PO3"],  # subject 2, training 2
}

CSV_PATH = ROOT / "results" / "tables" / "cca_window_sweep.csv"
PNG_PATH = ROOT / "results" / "figures" / "cca_window_sweep.png"


# --- Helpers ---------------------------------------------------------------


def _eval_lobo_8ch(X: np.ndarray, y: np.ndarray, blocks: np.ndarray,
                   stim_freqs: np.ndarray, fs: float) -> dict:
    """Config A: 8-channel canonical, standard LOBO-CV harness."""
    factory = lambda: CCAClassifier(stim_freqs=stim_freqs, fs=fs)
    return leave_one_block_out_cv(X, y, blocks, factory)


def _eval_lobo_top4(X: np.ndarray, y: np.ndarray, blocks: np.ndarray,
                    stim_freqs: np.ndarray, fs: float,
                    ch_names: list[str]) -> dict:
    """Config B: per-subject top-4 channels.

    CCA needs no training data, so each "fold" is just scored on the
    held-out block using that block's subject's channel subset.
    """
    name_to_idx = {n: i for i, n in enumerate(ch_names)}
    per_block: dict[int, float] = {}
    for b in np.unique(blocks):
        idx = [name_to_idx[c] for c in TOP4_BY_BLOCK[int(b)]]
        m = blocks == b
        X_test = X[m][:, idx, :]
        y_test = y[m]
        clf = CCAClassifier(stim_freqs=stim_freqs, fs=fs)
        clf.fit(X_test, y_test)        # no-op for CCA, kept for symmetry
        per_block[int(b)] = float(clf.score(X_test, y_test))
    accs = np.array(list(per_block.values()), dtype=np.float64)
    return {
        "per_block": per_block,
        "mean": float(accs.mean()),
        "std": float(accs.std()),
    }


def _abort(msg: str) -> None:
    print(f"\n[ABORT] {msg}", file=sys.stderr)
    raise SystemExit(2)


def _sanity_check(rows_at_3s: list[dict]) -> None:
    """Verify the 3.0-s sweep matches expectations; abort if it doesn't."""
    by_config = {r["config"]: r for r in rows_at_3s}
    a = by_config["A_8ch"]["mean_acc"]
    b = by_config["B_top4"]["mean_acc"]
    if a < 0.5:
        _abort(f"Config A acc {a:.3f} at 3.0s is implausibly low "
               f"(expected ~0.838); check preprocessing.")
    if b < 0.5:
        _abort(f"Config B acc {b:.3f} at 3.0s is implausibly low "
               f"(expected 0.80-0.90); check channel slicing.")
    if not (0.78 <= a <= 0.88):
        print(f"[warn] Config A 3.0s acc {a:.3f} outside [0.78, 0.88] "
              f"(soft check; continuing).")
    if not (0.78 <= b <= 0.92):
        print(f"[warn] Config B 3.0s acc {b:.3f} outside [0.78, 0.92] "
              f"(soft check; continuing).")


# --- Sweep -----------------------------------------------------------------


def run_sweep() -> list[dict]:
    """Sweep over WINDOWS x {A, B}; return list of result rows."""
    rows: list[dict] = []
    for window_s in WINDOWS:
        print(f"[sweep] window_s={window_s}s ...", end=" ", flush=True)
        ds = load_all(window_s=window_s)  # canonical preprocessing default
        X, y, blocks = ds["X"], ds["y"], ds["blocks"]
        stim_freqs, fs = ds["stim_freqs"], ds["fs"]
        ch_names = ds["ch_names"]

        for cfg_name, eval_fn in [
            ("A_8ch", lambda: _eval_lobo_8ch(X, y, blocks, stim_freqs, fs)),
            ("B_top4", lambda: _eval_lobo_top4(X, y, blocks, stim_freqs, fs, ch_names)),
        ]:
            res = eval_fn()
            rows.append({
                "window_s": float(window_s),
                "config": cfg_name,
                "mean_acc": float(res["mean"]),
                "std_acc": float(res["std"]),
                "itr_bits_per_min": float(itr(res["mean"], N_CLASSES, window_s)),
                "block_0": float(res["per_block"].get(0, float("nan"))),
                "block_1": float(res["per_block"].get(1, float("nan"))),
                "block_2": float(res["per_block"].get(2, float("nan"))),
                "block_3": float(res["per_block"].get(3, float("nan"))),
            })
        print(f"A={rows[-2]['mean_acc']:.3f}  B={rows[-1]['mean_acc']:.3f}")
    return rows


# --- CSV / figure / summary ------------------------------------------------


def write_csv(rows: list[dict]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["window_s", "config", "mean_acc", "std_acc",
                  "itr_bits_per_min", "block_0", "block_1", "block_2", "block_3"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] wrote {CSV_PATH.relative_to(ROOT)}")


def render_figure(rows: list[dict]) -> None:
    a_rows = sorted([r for r in rows if r["config"] == "A_8ch"], key=lambda r: r["window_s"])
    b_rows = sorted([r for r in rows if r["config"] == "B_top4"], key=lambda r: r["window_s"])
    ws = [r["window_s"] for r in a_rows]

    a_mean = np.array([r["mean_acc"] for r in a_rows])
    a_std = np.array([r["std_acc"] for r in a_rows])
    b_mean = np.array([r["mean_acc"] for r in b_rows])
    b_std = np.array([r["std_acc"] for r in b_rows])

    a_itr = np.array([r["itr_bits_per_min"] for r in a_rows])
    b_itr = np.array([r["itr_bits_per_min"] for r in b_rows])

    fig, (ax_acc, ax_itr) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: accuracy ---
    # Per-block thin lines for Config A. Distinguish subjects by linestyle
    # (S1 solid, S2 dashed) so the subject-1-vs-subject-2 split is visible
    # in the plot itself; only one legend entry per subject group.
    for b_id in (0, 1):
        curve = np.array([r[f"block_{b_id}"] for r in a_rows])
        ax_acc.plot(ws, curve, color="C0", alpha=0.35, lw=0.8, ls="-",
                    label="Config A per-block (S1)" if b_id == 0 else None)
    for b_id in (2, 3):
        curve = np.array([r[f"block_{b_id}"] for r in a_rows])
        ax_acc.plot(ws, curve, color="C0", alpha=0.35, lw=0.8, ls="--",
                    label="Config A per-block (S2)" if b_id == 2 else None)

    # Bands + main lines.
    ax_acc.fill_between(ws, a_mean - a_std, a_mean + a_std,
                        color="C0", alpha=0.20)
    ax_acc.plot(ws, a_mean, color="C0", lw=2.0, marker="o",
                label="Config A — 8-channel canonical")
    ax_acc.fill_between(ws, b_mean - b_std, b_mean + b_std,
                        color="C1", alpha=0.20)
    ax_acc.plot(ws, b_mean, color="C1", lw=2.0, marker="s",
                label="Config B — per-subject top-4")

    ax_acc.axhline(1.0 / N_CLASSES, color="gray", ls="--", lw=1.0,
                   label=f"chance (1/{N_CLASSES})")

    ax_acc.set_xlabel("Window length (s)")
    ax_acc.set_ylabel("LOBO-CV accuracy")
    ax_acc.set_title("CCA accuracy vs window length (LOBO across 4 blocks)")
    ax_acc.set_xticks(ws)
    ax_acc.set_xticklabels([f"{w:g}" for w in ws])
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(loc="upper left", fontsize=9)

    # --- Right: ITR ---
    ax_itr.plot(ws, a_itr, color="C0", lw=2.0, marker="o",
                label="Config A — 8-channel canonical")
    ax_itr.plot(ws, b_itr, color="C1", lw=2.0, marker="s",
                label="Config B — per-subject top-4")

    # Mark peaks.
    for itrs, color, ws_arr, label in [
        (a_itr, "C0", ws, "A"), (b_itr, "C1", ws, "B"),
    ]:
        peak_idx = int(np.argmax(itrs))
        peak_w, peak_v = ws_arr[peak_idx], itrs[peak_idx]
        ax_itr.axvline(peak_w, color=color, ls=":", lw=1.0, alpha=0.6)
        # Offset annotation slightly above the peak so it doesn't overlap
        # the line; vertical offset depends on which config to keep them
        # distinct.
        y_off = 1.0 if label == "A" else 4.0
        ax_itr.annotate(
            f"{peak_w:g}s, {peak_v:.1f} bpm",
            xy=(peak_w, peak_v),
            xytext=(peak_w + 0.15, peak_v + y_off),
            color=color,
            fontsize=9,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.7),
        )

    ax_itr.set_xlabel("Window length (s)")
    ax_itr.set_ylabel("ITR (bits / min)")
    ax_itr.set_title(f"CCA ITR vs window length (Wolpaw, N={N_CLASSES})")
    ax_itr.set_xticks(ws)
    ax_itr.set_xticklabels([f"{w:g}" for w in ws])
    ax_itr.set_ylim(bottom=0.0)
    ax_itr.grid(alpha=0.3)
    ax_itr.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "CCA window-length sweep — comparison floor for TRCA",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=140)
    plt.close(fig)
    print(f"[fig] wrote {PNG_PATH.relative_to(ROOT)}")


def _shortest_window_above(rows: list[dict], threshold: float) -> tuple[float, float] | None:
    """Return (window_s, mean_acc) of the shortest window with mean_acc >= threshold."""
    candidates = [r for r in rows if r["mean_acc"] >= threshold]
    if not candidates:
        return None
    best = min(candidates, key=lambda r: r["window_s"])
    return best["window_s"], best["mean_acc"]


def print_summary(rows: list[dict]) -> None:
    a_rows = [r for r in rows if r["config"] == "A_8ch"]
    b_rows = [r for r in rows if r["config"] == "B_top4"]

    a_peak = max(a_rows, key=lambda r: r["itr_bits_per_min"])
    b_peak = max(b_rows, key=lambda r: r["itr_bits_per_min"])

    a_thresh = _shortest_window_above(a_rows, ACC_THRESHOLD)
    b_thresh = _shortest_window_above(b_rows, ACC_THRESHOLD)

    print("\n=== CCA Window Sweep — summary ===")
    print("\nConfig A (8-channel canonical):")
    print(f"  peak ITR: {a_peak['itr_bits_per_min']:.1f} bits/min "
          f"@ window={a_peak['window_s']:g}s (acc={a_peak['mean_acc']:.3f})")
    if a_thresh is None:
        print(f"  shortest window with mean acc >= {ACC_THRESHOLD:.2f}: "
              f"none in this sweep")
    else:
        w, acc = a_thresh
        print(f"  shortest window with mean acc >= {ACC_THRESHOLD:.2f}: "
              f"{w:g}s (acc={acc:.3f})")

    print("\nConfig B (per-subject top-4):")
    print(f"  peak ITR: {b_peak['itr_bits_per_min']:.1f} bits/min "
          f"@ window={b_peak['window_s']:g}s (acc={b_peak['mean_acc']:.3f})")
    if b_thresh is None:
        print(f"  shortest window with mean acc >= {ACC_THRESHOLD:.2f}: "
              f"none in this sweep")
    else:
        w, acc = b_thresh
        print(f"  shortest window with mean acc >= {ACC_THRESHOLD:.2f}: "
              f"{w:g}s (acc={acc:.3f})")

    # --- Templated interpretation (uses real numbers, not filler) ---
    print("\nInterpretation:")
    if a_thresh is None and b_thresh is None:
        print("  Neither config achieves 0.80 in this sweep — TRCA's "
              "benchmark needs re-thinking before the slide goes up.")
    else:
        winner = "B" if b_peak["itr_bits_per_min"] > a_peak["itr_bits_per_min"] else "A"
        loser = "A" if winner == "B" else "B"
        winner_peak = b_peak if winner == "B" else a_peak
        loser_peak = a_peak if winner == "B" else b_peak
        delta = winner_peak["itr_bits_per_min"] - loser_peak["itr_bits_per_min"]

        # Is Config B's lift consistent across windows or 3.0-s only?
        a_3 = next(r for r in a_rows if r["window_s"] == 3.0)["mean_acc"]
        b_3 = next(r for r in b_rows if r["window_s"] == 3.0)["mean_acc"]
        deltas = [bi["mean_acc"] - ai["mean_acc"]
                  for ai, bi in zip(a_rows, b_rows)]
        consistent = all(d >= -0.01 for d in deltas) and (b_3 >= a_3)

        if winner == "B":
            print(f"  Config B wins peak ITR ({winner_peak['itr_bits_per_min']:.1f} vs "
                  f"{loser_peak['itr_bits_per_min']:.1f} bpm; +{delta:.1f}). "
                  f"Per-subject channel adaptation pays off in the comparison")
        else:
            print(f"  Config A wins peak ITR ({winner_peak['itr_bits_per_min']:.1f} vs "
                  f"{loser_peak['itr_bits_per_min']:.1f} bpm; +{delta:.1f}). "
                  f"The 8-channel baseline is hard to beat once windows shorten")
        if consistent:
            print(f"  — and the lift is consistent: Config B >= Config A across all "
                  f"sampled windows (3.0s split: A={a_3:.3f}, B={b_3:.3f}).")
        else:
            mixed = sum(1 for d in deltas if d > 0)
            print(f"  — but the lift is mixed: Config B beats A on "
                  f"{mixed}/{len(deltas)} windows (3.0s split: A={a_3:.3f}, B={b_3:.3f}).")
        if a_thresh and b_thresh:
            a_w, _ = a_thresh
            b_w, _ = b_thresh
            shorter = "B" if b_w < a_w else ("A" if a_w < b_w else "tie")
            if shorter == "tie":
                print(f"  Both configs cross 0.80 at the same window ({a_w:g}s) — "
                      f"that's the 'TRCA must beat' benchmark for the slide.")
            else:
                short_w = b_w if shorter == "B" else a_w
                long_w = a_w if shorter == "B" else b_w
                print(f"  Shortest window with acc >= 0.80: Config {shorter} = {short_w:g}s "
                      f"(vs Config {'A' if shorter=='B' else 'B'} = {long_w:g}s). "
                      f"Use {short_w:g}s as the TRCA benchmark.")
        else:
            single = "A" if a_thresh else "B"
            w, acc = a_thresh if single == "A" else b_thresh
            print(f"  Only Config {single} crosses 0.80 (at {w:g}s, acc={acc:.3f}). "
                  f"Use that as the TRCA benchmark.")


def main() -> int:
    print(f"[sweep] {len(WINDOWS)} windows x 2 configs = {len(WINDOWS)*2} evaluations")
    rows = run_sweep()
    rows_at_3s = [r for r in rows if r["window_s"] == 3.0]
    _sanity_check(rows_at_3s)
    write_csv(rows)
    render_figure(rows)
    print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
