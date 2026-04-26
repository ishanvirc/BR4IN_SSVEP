"""Final classifier comparison harness — run this once TRCA lands to get the results table.

This script runs every classifier side-by-side at a chosen window length
and writes a slide-ready results table + bar chart. It is the single source
of truth for the final comparison in the presentation.

To run (from project root):

    python scripts/compare_classifiers.py              # default: 3.0s window
    python scripts/compare_classifiers.py --window 2.0
    python scripts/compare_classifiers.py --window 3.0 --harmonics 3

Classifiers
-----------
  CCA-8ch         : CCA on all 8 channels (canonical baseline)
  CCA-top4-nested : CCA on per-subject top-4 channels, SNR-ranked per fold
                    (no data leakage — channel ranking uses only training data)
  FBCCA           : Filter-Bank CCA (Chen 2015) — runs when implemented,
                    skips with "pending" if still a stub
  TRCA            : Task-Related Component Analysis (Nakanishi 2017) — same
  CH11            : g.tec live LDA classifier output (SOTA reference).
                    Note: fires on only ~49% of trials; accuracy reported on
                    fired trials only and across all trials separately.

Outputs (written on every run, idempotent)
------------------------------------------
  results/tables/comparison.csv   — one row per classifier (accuracy, std, ITR)
  results/figures/comparison.png  — accuracy + ITR bar chart, slide-ready
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ssvep.io import load_all
from ssvep.classifiers import CCAClassifier, FBCCAClassifier, TRCAClassifier
from ssvep.evaluation import leave_one_block_out_cv, itr, accuracy
from ssvep.channel_selection import rank_channels_by_snr

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_CLASSES = 4
SUBJECT_OF_BLOCK: dict[int, int] = {0: 1, 1: 1, 2: 2, 3: 2}

OUT_CSV = ROOT / "results" / "tables" / "comparison.csv"
OUT_PNG = ROOT / "results" / "figures" / "comparison.png"

# ---------------------------------------------------------------------------
# Per-classifier evaluation functions
# ---------------------------------------------------------------------------


def eval_cca_8ch(X, y, blocks, stim_freqs, fs, n_harmonics):
    """CCA on all 8 channels — standard LOBO-CV."""
    factory = lambda: CCAClassifier(stim_freqs=stim_freqs, fs=fs, n_harmonics=n_harmonics)
    return leave_one_block_out_cv(X, y, blocks, factory)


def eval_cca_top4_nested(X, y, blocks, stim_freqs, fs, n_harmonics, ch_names):
    """CCA with per-fold SNR-ranked top-4 channels (leakage-free)."""
    name_at = {i: n for i, n in enumerate(ch_names)}
    per_block: dict[int, float] = {}
    for b_raw in np.unique(blocks):
        b = int(b_raw)
        same_subj = SUBJECT_OF_BLOCK[b]
        train_ids = [bb for bb, s in SUBJECT_OF_BLOCK.items() if s == same_subj and bb != b]
        train_mask = np.isin(blocks, train_ids)
        _, ranking = rank_channels_by_snr(X[train_mask], y[train_mask], fs, stim_freqs)
        top4_idx = [int(i) for i in ranking[:4]]
        test_mask = blocks == b
        X_test = X[test_mask][:, top4_idx, :]
        clf = CCAClassifier(stim_freqs=stim_freqs, fs=fs, n_harmonics=n_harmonics)
        clf.fit(X_test, y[test_mask])
        per_block[b] = float(clf.score(X_test, y[test_mask]))
    accs = np.array(list(per_block.values()))
    return {"per_block": per_block, "mean": float(accs.mean()), "std": float(accs.std())}


def eval_fbcca(X, y, blocks, stim_freqs, fs, n_harmonics):
    """FBCCA — skips gracefully if stub."""
    def factory():
        return FBCCAClassifier(stim_freqs=stim_freqs, fs=fs, num_harmonics=n_harmonics)
    try:
        return leave_one_block_out_cv(X, y, blocks, factory)
    except NotImplementedError:
        return None


def eval_trca(X, y, blocks, stim_freqs, fs):
    """TRCA — skips gracefully if stub."""
    def factory():
        return TRCAClassifier(stim_freqs=stim_freqs, fs=fs)
    try:
        return leave_one_block_out_cv(X, y, blocks, factory)
    except NotImplementedError:
        return None


def eval_ch11(ds):
    """CH11 (g.tec live LDA) — coverage + per-trial accuracy."""
    y_true = ds["y"]
    ch11 = ds["ch11_pred"]
    fired = ch11 >= 0
    n_total = len(y_true)
    n_fired = int(fired.sum())
    acc = float((ch11[fired] == y_true[fired]).mean()) if n_fired > 0 else 0.0
    coverage = n_fired / n_total
    # Effective accuracy over all trials (treat no-fire as wrong)
    acc_all = float((ch11[fired] == y_true[fired]).sum() / n_total)
    return {
        "mean": acc,            # accuracy on fired trials only
        "acc_all": acc_all,     # accuracy across all trials
        "coverage": coverage,
        "n_fired": n_fired,
        "n_total": n_total,
        "std": float("nan"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Compare all classifiers side-by-side.")
    parser.add_argument("--window", type=float, default=3.0,
                        help="Analysis window length in seconds (default: 3.0)")
    parser.add_argument("--harmonics", type=int, default=2,
                        help="Number of CCA harmonics (default: 2)")
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "data" / "raw"),
                        help="Path to data/raw directory")
    args = parser.parse_args()

    window_s = args.window
    n_harmonics = args.harmonics
    data_dir = args.data_dir

    print(f"Loading data (window={window_s}s) ...")
    ds = load_all(data_dir, window_s=window_s)
    X, y, blocks = ds["X"], ds["y"], ds["blocks"]
    stim_freqs = ds["stim_freqs"]
    ch_names = ds["ch_names"]
    fs = ds["fs"]
    print(f"  {X.shape[0]} trials, {X.shape[1]} channels, {X.shape[2]} samples @ {fs} Hz")

    # --- Run classifiers ---------------------------------------------------
    configs = []

    print("\nCCA-8ch ...")
    r = eval_cca_8ch(X, y, blocks, stim_freqs, fs, n_harmonics)
    configs.append(("CCA-8ch", r))

    print("CCA-top4-nested ...")
    r = eval_cca_top4_nested(X, y, blocks, stim_freqs, fs, n_harmonics, ch_names)
    configs.append(("CCA-top4-nested", r))

    print("FBCCA ...")
    r = eval_fbcca(X, y, blocks, stim_freqs, fs, n_harmonics)
    configs.append(("FBCCA", r))

    print("TRCA ...")
    r = eval_trca(X, y, blocks, stim_freqs, fs)
    configs.append(("TRCA", r))

    print("CH11 (g.tec LDA) ...")
    r = eval_ch11(ds)
    configs.append(("CH11", r))

    # --- Print table -------------------------------------------------------
    print(f"\n{'Classifier':20} {'Acc (LOBO)':>12} {'Std':>6} {'ITR (bpm)':>10} {'Notes'}")
    print("-" * 75)
    rows = []
    for name, res in configs:
        if res is None:
            print(f"  {name:18} {'pending':>12}")
            rows.append({"classifier": name, "accuracy": "", "std": "",
                         "itr_bpm": "", "notes": "not implemented"})
            continue

        acc = res["mean"]
        std = res.get("std", float("nan"))

        if name == "CH11":
            itr_val = float("nan")
            notes = (f"fired {res['n_fired']}/{res['n_total']} trials "
                     f"({res['coverage']:.0%} coverage); "
                     f"acc_all={res['acc_all']:.3f}")
        else:
            itr_val = itr(acc, N_CLASSES, window_s)
            notes = " | ".join(f"b{b}={v:.2f}" for b, v in res["per_block"].items())

        std_str = f"{std:.3f}" if not np.isnan(std) else "n/a"
        itr_str = f"{itr_val:.1f}" if not np.isnan(itr_val) else "n/a"
        print(f"  {name:18} {acc:>12.3f} {std_str:>6} {itr_str:>10}   {notes}")
        rows.append({"classifier": name, "accuracy": f"{acc:.3f}", "std": std_str,
                     "itr_bpm": itr_str, "notes": notes})

    # --- Save CSV ----------------------------------------------------------
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["classifier", "accuracy", "std", "itr_bpm", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved: {OUT_CSV.relative_to(ROOT)}")

    # --- Plot --------------------------------------------------------------
    implemented = [(n, r) for n, r in configs if r is not None]
    names  = [n for n, _ in implemented]
    accs   = [r["mean"] for _, r in implemented]
    stds   = [r.get("std", 0.0) for _, r in implemented]
    itrs   = [itr(r["mean"], N_CLASSES, window_s)
              if n != "CH11" else float("nan")
              for n, r in implemented]

    # CH11 bar is acc on fired trials only; mark with lighter color
    colors = ["steelblue" if n != "CH11" else "lightcoral" for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    bars = ax1.bar(names, accs, color=colors, yerr=[s if not np.isnan(s) else 0 for s in stds],
                   capsize=4, alpha=0.85)
    ax1.axhline(0.80, color="red", linestyle="--", linewidth=1.2, label="0.80 target")
    ax1.set_ylabel("LOBO-CV accuracy")
    ax1.set_title(f"Accuracy @ {window_s}s window")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{acc:.1%}", ha="center", va="bottom", fontsize=9)

    # ITR
    itr_vals = [v if not np.isnan(v) else 0 for v in itrs]
    itr_colors = ["steelblue" if n != "CH11" else "lightcoral" for n in names]
    bars2 = ax2.bar(names, itr_vals, color=itr_colors, alpha=0.85)
    ax2.set_ylabel("ITR (bits/min, Wolpaw)")
    ax2.set_title(f"ITR @ {window_s}s window")
    for bar, val, name in zip(bars2, itr_vals, names):
        label = f"{val:.1f}" if name != "CH11" else "n/a"
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 label, ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"Classifier comparison — {window_s}s window, {n_harmonics} harmonics",
                 fontsize=13, fontweight="bold")

    caption = (
        "Error bars: ±1 SD over LOBO-CV folds (n=4).\n"
        "CH11 (coral): accuracy on fired trials only (87.2%, 49% coverage; all-trial = 42.5%).\n"
        "No error bar for CH11 (single observation).\n"
        "CH11 ITR not defined (no prediction on 51% of trials)."
    )
    fig.text(0.5, -0.04, caption, ha="center", va="top", fontsize=8,
             color="dimgray", wrap=True,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.6))

    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {OUT_PNG.relative_to(ROOT)}")
    plt.show()


if __name__ == "__main__":
    main()
