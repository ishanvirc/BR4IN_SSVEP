"""CCA window-length sweep — slide-ready curve for the TRCA comparison.

Runs LOBO-CV CCA across 10 trial-window lengths (0.5 - 5.0 s) in three
configurations:

    Config A — 8-channel canonical
        All 8 EEG channels, canonical-preprocessed pipeline.

    Config B — per-subject top-4 (LEAKAGE)
        Subject 1: ['O2','O1','Oz','PO8']
        Subject 2: ['PO4','PO8','POz','PO3']
        Source: notebooks/08_channel_selection.ipynb (cell 6 SNR ranking).
        Channels were chosen using all 40 of each subject's trials,
        then scored via LOBO on the same trials → leakage.

    Config B-nested — per-subject top-4 (nested CV)
        SNR ranking is recomputed PER FOLD on training-only data:
        when a block is held out, the ranking uses only the same
        subject's OTHER block (the one not held out). Each fold's
        channels were never observed during ranking.

Outputs (all four artifacts emitted on every run; idempotent)
-------------------------------------------------------------
- results/tables/cca_window_sweep.csv             (A + B-leakage,  20 rows, 9 cols)
- results/tables/cca_window_sweep_nested.csv      (A + B-nested,   20 rows, 10 cols)
- results/figures/cca_window_sweep.png            (2-line, original layout)
- results/figures/cca_window_sweep_with_nested.png (3-line, supersedes the above for the slide)

Run from project root:

    python scripts/sweep_cca_windows.py

Per-fold subject-training data (Config B-nested)
------------------------------------------------
With 4 blocks and 2 blocks per subject, each fold's subject-training
set is exactly ONE block (20 trials):

    Held-out block  | Subject  | Subject's training (for ranking)
    ----------------|----------|---------------------------------
    0 (subj 1)      | subj 1   | block 1
    1 (subj 1)      | subj 1   | block 0
    2 (subj 2)      | subj 2   | block 3
    3 (subj 2)      | subj 2   | block 2

Per-fold rankings on 20 trials are noisy — that's *why* we're doing it.
The leakage version "averages out" noise by using all 40 trials.

Asymmetry footnote
------------------
The Config B (LEAKAGE) channels are Yuki's fixed lists derived at
window_s=3.0 and applied verbatim across ALL sweep windows. The Config
B-nested rankings are recomputed per fold at EACH window length, which
gives the nested config slight latitude to pick window-specific
channels that the leakage config does not have. Locking nested to a
single window length would defeat the point of nested CV at multiple
windows; the asymmetry is left in and documented here.
"""

from __future__ import annotations

import csv
import json
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
from ssvep.channel_selection import rank_channels_by_snr

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

# Block -> subject id, used by Config B-nested to find the same-subject
# training block when one block is held out.
SUBJECT_OF_BLOCK: dict[int, int] = {0: 1, 1: 1, 2: 2, 3: 2}

CSV_PATH = ROOT / "results" / "tables" / "cca_window_sweep.csv"
NESTED_CSV_PATH = ROOT / "results" / "tables" / "cca_window_sweep_nested.csv"
PNG_PATH = ROOT / "results" / "figures" / "cca_window_sweep.png"
COMBINED_PNG_PATH = ROOT / "results" / "figures" / "cca_window_sweep_with_nested.png"


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


def _eval_lobo_top4_nested(X: np.ndarray, y: np.ndarray, blocks: np.ndarray,
                           stim_freqs: np.ndarray, fs: float,
                           ch_names: list[str]) -> dict:
    """Config B-nested: per-fold SNR ranking on training-only data.

    For each held-out block, recompute the SNR ranking using only the
    same subject's OTHER block, then take the top-4 channels and score
    the held-out block. The channels used were never observed during
    ranking — leakage closed.
    """
    name_at = {i: n for i, n in enumerate(ch_names)}
    per_block: dict[int, float] = {}
    channels_used: dict[int, list[str]] = {}
    for b_raw in np.unique(blocks):
        b = int(b_raw)
        # Same-subject training blocks (the OTHER block from this subject).
        same_subj = SUBJECT_OF_BLOCK[b]
        train_block_ids = [bb for bb, sub in SUBJECT_OF_BLOCK.items()
                           if sub == same_subj and bb != b]
        train_mask = np.isin(blocks, train_block_ids)
        if not train_mask.any():
            raise RuntimeError(f"No same-subject training data for block {b}")
        # Rank channels on training data at the current window length.
        _snr, ranking = rank_channels_by_snr(
            X[train_mask], y[train_mask], fs, stim_freqs
        )
        top4_idx = [int(i) for i in ranking[:4]]
        channels_used[b] = [name_at[i] for i in top4_idx]
        # Score held-out block using these channels.
        m = blocks == b
        X_test = X[m][:, top4_idx, :]
        y_test = y[m]
        clf = CCAClassifier(stim_freqs=stim_freqs, fs=fs)
        clf.fit(X_test, y_test)        # no-op for CCA, kept for symmetry
        per_block[b] = float(clf.score(X_test, y_test))
    accs = np.array(list(per_block.values()), dtype=np.float64)
    return {
        "per_block": per_block,
        "mean": float(accs.mean()),
        "std": float(accs.std()),
        "channels_used": channels_used,
    }


def _abort(msg: str) -> None:
    print(f"\n[ABORT] {msg}", file=sys.stderr)
    raise SystemExit(2)


def _sanity_check(rows_at_3s: list[dict]) -> None:
    """Verify the 3.0-s sweep matches expectations; abort if it doesn't."""
    by_config = {r["config"]: r for r in rows_at_3s}
    a = by_config["A_8ch"]["mean_acc"]
    b = by_config["B_top4"]["mean_acc"]
    bn = by_config["B_top4_nested"]["mean_acc"]
    if a < 0.5:
        _abort(f"Config A acc {a:.3f} at 3.0s is implausibly low "
               f"(expected ~0.838); check preprocessing.")
    if b < 0.5:
        _abort(f"Config B acc {b:.3f} at 3.0s is implausibly low "
               f"(expected 0.80-0.90); check channel slicing.")
    if bn < 0.65:
        _abort(f"Config B-nested acc {bn:.3f} at 3.0s is implausibly low "
               f"(expected >= 0.65); check ranking/slicing logic.")
    if not (0.78 <= a <= 0.88):
        print(f"[warn] Config A 3.0s acc {a:.3f} outside [0.78, 0.88] "
              f"(soft check; continuing).")
    if not (0.78 <= b <= 0.92):
        print(f"[warn] Config B 3.0s acc {b:.3f} outside [0.78, 0.92] "
              f"(soft check; continuing).")
    if bn > 0.90:
        print(f"[warn] Config B-nested 3.0s acc {bn:.3f} exceeds 0.90 — "
              f"unusual, since nested should be <= leakage by construction; "
              f"investigate.")


# --- Sweep -----------------------------------------------------------------


def run_sweep() -> list[dict]:
    """Sweep over WINDOWS x {A, B-leakage, B-nested}; return list of result rows."""
    rows: list[dict] = []
    for window_s in WINDOWS:
        print(f"[sweep] window_s={window_s}s ...", end=" ", flush=True)
        ds = load_all(window_s=window_s)  # canonical preprocessing default
        X, y, blocks = ds["X"], ds["y"], ds["blocks"]
        stim_freqs, fs = ds["stim_freqs"], ds["fs"]
        ch_names = ds["ch_names"]

        evaluators = [
            ("A_8ch",          lambda: _eval_lobo_8ch(X, y, blocks, stim_freqs, fs)),
            ("B_top4",         lambda: _eval_lobo_top4(X, y, blocks, stim_freqs, fs, ch_names)),
            ("B_top4_nested",  lambda: _eval_lobo_top4_nested(X, y, blocks, stim_freqs, fs, ch_names)),
        ]
        for cfg_name, eval_fn in evaluators:
            res = eval_fn()
            row = {
                "window_s": float(window_s),
                "config": cfg_name,
                "mean_acc": float(res["mean"]),
                "std_acc": float(res["std"]),
                "itr_bits_per_min": float(itr(res["mean"], N_CLASSES, window_s)),
                "block_0": float(res["per_block"].get(0, float("nan"))),
                "block_1": float(res["per_block"].get(1, float("nan"))),
                "block_2": float(res["per_block"].get(2, float("nan"))),
                "block_3": float(res["per_block"].get(3, float("nan"))),
            }
            if cfg_name == "B_top4_nested":
                # Channel sets used per fold, ordered by block id 0..3.
                cu = res.get("channels_used", {})
                row["channels_used_per_block"] = [
                    cu.get(b, []) for b in (0, 1, 2, 3)
                ]
            rows.append(row)
        # Three rows added per window — print in order.
        print(f"A={rows[-3]['mean_acc']:.3f}  "
              f"B_leak={rows[-2]['mean_acc']:.3f}  "
              f"B_nest={rows[-1]['mean_acc']:.3f}")
    return rows


# --- CSV / figure / summary ------------------------------------------------


_BASE_FIELDS = ["window_s", "config", "mean_acc", "std_acc",
                "itr_bits_per_min", "block_0", "block_1", "block_2", "block_3"]


def write_csv_leakage(rows: list[dict]) -> None:
    """Existing CSV: A + B-leakage rows, identical schema to the original."""
    out_rows = [r for r in rows if r["config"] in ("A_8ch", "B_top4")]
    out_rows = [{k: r[k] for k in _BASE_FIELDS} for r in out_rows]
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_BASE_FIELDS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"[csv] wrote {CSV_PATH.relative_to(ROOT)}")


def write_csv_nested(rows: list[dict]) -> None:
    """New CSV: A + B-nested rows, with per-fold channel column."""
    fieldnames = _BASE_FIELDS + ["channels_used_per_block"]
    out_rows = []
    for r in rows:
        if r["config"] not in ("A_8ch", "B_top4_nested"):
            continue
        row = {k: r[k] for k in _BASE_FIELDS}
        # Config A has no channel selection → null. Config B-nested →
        # list of 4 channel-name lists (one per fold), JSON-encoded so
        # csv quoting handles it cleanly.
        chs = r.get("channels_used_per_block")
        row["channels_used_per_block"] = json.dumps(chs) if chs is not None else "null"
        out_rows.append(row)
    NESTED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NESTED_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"[csv] wrote {NESTED_CSV_PATH.relative_to(ROOT)}")


def render_original_figure(rows: list[dict]) -> None:
    """Existing 2-line figure (Config A + Config B-leakage). Idempotent re-emit."""
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


def render_combined_figure(rows: list[dict]) -> None:
    """3-line figure: A + B-leakage + B-nested. Slide-ready."""
    a_rows  = sorted([r for r in rows if r["config"] == "A_8ch"],         key=lambda r: r["window_s"])
    b_rows  = sorted([r for r in rows if r["config"] == "B_top4"],        key=lambda r: r["window_s"])
    bn_rows = sorted([r for r in rows if r["config"] == "B_top4_nested"], key=lambda r: r["window_s"])
    ws = [r["window_s"] for r in a_rows]

    a_mean = np.array([r["mean_acc"] for r in a_rows])
    a_std  = np.array([r["std_acc"]  for r in a_rows])
    b_mean = np.array([r["mean_acc"] for r in b_rows])
    b_std  = np.array([r["std_acc"]  for r in b_rows])
    bn_mean = np.array([r["mean_acc"] for r in bn_rows])
    bn_std  = np.array([r["std_acc"]  for r in bn_rows])

    a_itr  = np.array([r["itr_bits_per_min"] for r in a_rows])
    b_itr  = np.array([r["itr_bits_per_min"] for r in b_rows])
    bn_itr = np.array([r["itr_bits_per_min"] for r in bn_rows])

    fig, (ax_acc, ax_itr) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: accuracy ---
    # Three main lines + std bands. No per-block thin lines (clutter).
    for mean, std, color, marker, label in [
        (a_mean,  a_std,  "C0", "o", "Config A — 8-channel canonical"),
        (b_mean,  b_std,  "C1", "s", "Config B — per-subject top-4 (LEAKAGE)"),
        (bn_mean, bn_std, "C2", "^", "Config B-nested — per-subject top-4 (nested CV)"),
    ]:
        ax_acc.fill_between(ws, mean - std, mean + std, color=color, alpha=0.18)
        ax_acc.plot(ws, mean, color=color, lw=2.0, marker=marker, label=label)

    ax_acc.axhline(1.0 / N_CLASSES, color="gray", ls="--", lw=1.0,
                   label=f"chance (1/{N_CLASSES})")

    ax_acc.set_xlabel("Window length (s)")
    ax_acc.set_ylabel("LOBO-CV accuracy")
    ax_acc.set_title("CCA accuracy vs window length (LOBO across 4 blocks)")
    ax_acc.set_xticks(ws)
    ax_acc.set_xticklabels([f"{w:g}" for w in ws])
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(loc="lower right", fontsize=9)

    # --- Right: ITR ---
    ax_itr.plot(ws, a_itr,  color="C0", lw=2.0, marker="o",
                label="Config A — 8-channel canonical")
    ax_itr.plot(ws, b_itr,  color="C1", lw=2.0, marker="s",
                label="Config B — per-subject top-4 (LEAKAGE)")
    ax_itr.plot(ws, bn_itr, color="C2", lw=2.0, marker="^",
                label="Config B-nested — per-subject top-4 (nested CV)")

    # Peak annotations for the slide-relevant lines (A + B-nested).
    for itrs, color, label, y_off in [
        (a_itr,  "C0", "A",        1.0),
        (bn_itr, "C2", "B-nested", 4.0),
    ]:
        peak_idx = int(np.argmax(itrs))
        peak_w, peak_v = ws[peak_idx], itrs[peak_idx]
        ax_itr.axvline(peak_w, color=color, ls=":", lw=1.0, alpha=0.6)
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
        "CCA window-length sweep — leakage vs nested CV",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    COMBINED_PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(COMBINED_PNG_PATH, dpi=140)
    plt.close(fig)
    print(f"[fig] wrote {COMBINED_PNG_PATH.relative_to(ROOT)}")


def _shortest_window_above(rows: list[dict], threshold: float) -> tuple[float, float] | None:
    """Return (window_s, mean_acc) of the shortest window with mean_acc >= threshold."""
    candidates = [r for r in rows if r["mean_acc"] >= threshold]
    if not candidates:
        return None
    best = min(candidates, key=lambda r: r["window_s"])
    return best["window_s"], best["mean_acc"]


def _row_summary(rows: list[dict]) -> tuple[dict, tuple[float, float] | None, float]:
    """Return (peak_itr_row, threshold_or_none, acc_at_3s)."""
    peak = max(rows, key=lambda r: r["itr_bits_per_min"])
    thresh = _shortest_window_above(rows, ACC_THRESHOLD)
    acc_3 = next(r for r in rows if r["window_s"] == 3.0)["mean_acc"]
    return peak, thresh, acc_3


def _format_threshold(thresh: tuple[float, float] | None) -> str:
    if thresh is None:
        return "none"
    w, acc = thresh
    return f"{w:g}s (acc={acc:.3f})"


def print_summary(rows: list[dict]) -> None:
    a_rows  = [r for r in rows if r["config"] == "A_8ch"]
    b_rows  = [r for r in rows if r["config"] == "B_top4"]
    bn_rows = [r for r in rows if r["config"] == "B_top4_nested"]

    a_peak,  a_thresh,  a_3  = _row_summary(a_rows)
    b_peak,  b_thresh,  b_3  = _row_summary(b_rows)
    bn_peak, bn_thresh, bn_3 = _row_summary(bn_rows)

    # --- Comparison table ---
    print("\n=== CCA Window Sweep — leakage vs nested-CV comparison ===\n")
    header = f"{'Config':<18}{'Peak ITR':>11}{'@ Window':>11}{'3.0s acc':>11}{'>=0.80 window':>20}"
    print(header)
    print("-" * len(header))
    for label, peak, thresh, acc_3 in [
        ("A (8-ch)",     a_peak,  a_thresh,  a_3),
        ("B (leakage)",  b_peak,  b_thresh,  b_3),
        ("B (nested)",   bn_peak, bn_thresh, bn_3),
    ]:
        thresh_str = "none" if thresh is None else f"{thresh[0]:g}s (acc={thresh[1]:.3f})"
        print(f"{label:<18}"
              f"{peak['itr_bits_per_min']:>8.1f} bpm"
              f"{peak['window_s']:>9g}s"
              f"{acc_3:>11.3f}"
              f"{thresh_str:>20}")

    # --- Channel stability diagnostic (Config B-nested) ---
    print("\nChannel stability (B-nested) — counts across all 10 windows × "
          "2 same-subject folds:")
    # Subject -> { channel_name -> count }
    counts: dict[int, dict[str, int]] = {1: {}, 2: {}}
    for r in bn_rows:
        chs_per_block = r.get("channels_used_per_block")
        if not chs_per_block:
            continue
        for b_id, chs in enumerate(chs_per_block):
            subj = SUBJECT_OF_BLOCK[b_id]
            for c in chs:
                counts[subj][c] = counts[subj].get(c, 0) + 1
    # Render per subject. Use the canonical CH order for stable layout.
    canonical_order = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]
    max_count = len(WINDOWS) * 2  # 10 windows × 2 folds per subject
    for subj in (1, 2):
        print(f"  Subject {subj} (max={max_count}):")
        line1 = "    " + "   ".join(f"{c}: {counts[subj].get(c, 0):>2d}"
                                    for c in canonical_order[:4])
        line2 = "    " + "   ".join(f"{c}: {counts[subj].get(c, 0):>2d}"
                                    for c in canonical_order[4:])
        print(line1)
        print(line2)

    # --- Templated interpretation ---
    print("\nInterpretation:")
    delta_3 = bn_3 - b_3                               # nested vs leakage at 3.0s
    delta_vs_a = bn_3 - a_3                            # nested vs A baseline at 3.0s
    abs_delta_3 = abs(delta_3)

    if delta_3 > 0.01:
        print(f"  Nested CV (3.0s acc {bn_3:.3f}) is HIGHER than leakage "
              f"({b_3:.3f}) — unusual, since leakage should >= nested by "
              f"construction. Likely the bottleneck is subject 2's underlying "
              f"SSVEP variance, not channel choice; investigate before quoting.")
    elif abs_delta_3 <= 0.03:
        print(f"  Nested CV (3.0s acc {bn_3:.3f}) is within 3pp of leakage "
              f"({b_3:.3f}). The leakage premium was small; per-subject "
              f"channel selection holds up under nested CV.")
    elif abs_delta_3 <= 0.10:
        survives = bn_3 > a_3
        print(f"  Nested CV (3.0s acc {bn_3:.3f}) is {abs_delta_3*100:.1f}pp "
              f"below leakage ({b_3:.3f}). Leakage inflated the result modestly; "
              f"the per-subject lift over Config A "
              f"({'survives' if survives else 'disappears'} "
              f"(nested vs A = {delta_vs_a:+.3f}).")
    else:
        print(f"  Nested CV (3.0s acc {bn_3:.3f}) is {abs_delta_3*100:.1f}pp "
              f"below leakage ({b_3:.3f}). Nested CV substantially erodes "
              f"Config B; the channels look subject-specific on aggregate "
              f"data but the per-fold pick is too noisy.")

    # Per-subject channel stability summary (top channel per subject).
    for subj in (1, 2):
        if not counts[subj]:
            continue
        top_chs = sorted(counts[subj].items(), key=lambda kv: kv[1], reverse=True)
        top_str = ", ".join(f"{c}({n})" for c, n in top_chs[:4])
        print(f"  Subject {subj} most-picked channels: {top_str} (max={max_count}).")

    # TRCA benchmark line (use nested where available; fall back to A).
    bench = bn_thresh or a_thresh
    bench_label = "B-nested" if bn_thresh else "A"
    if bench is not None:
        print(f"  TRCA benchmark for the slide: Config {bench_label} crosses 0.80 "
              f"at {bench[0]:g}s (acc={bench[1]:.3f}).")
    else:
        print("  Neither A nor B-nested crosses 0.80 — TRCA's benchmark needs "
              "re-thinking before the slide goes up.")


def main() -> int:
    print(f"[sweep] {len(WINDOWS)} windows x 3 configs = {len(WINDOWS)*3} evaluations")
    rows = run_sweep()
    rows_at_3s = [r for r in rows if r["window_s"] == 3.0]
    _sanity_check(rows_at_3s)
    write_csv_leakage(rows)
    write_csv_nested(rows)
    render_original_figure(rows)
    render_combined_figure(rows)
    print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
