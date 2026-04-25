"""CLI entry point: load -> preprocess -> CCA -> metrics.

Usage:

    python scripts/run_baseline.py --synthetic
    python scripts/run_baseline.py --mat data/raw/                                         # canonical default
    python scripts/run_baseline.py --mat data/raw/<file>.mat [--bandpass LOW HIGH] [--notch HZ]
    python scripts/run_baseline.py --mat data/raw/<file>.mat --no-filter                   # raw opt-out

By default, real-data loads apply 3-45 Hz bandpass + 50 Hz notch on
the continuous signal before epoching (per VT's preprocessing scan).
Override with explicit --bandpass / --notch, or escape with --no-filter.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow `python scripts/run_baseline.py ...` even without `pip install -e .`.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ssvep import io, evaluation
from ssvep.classifiers import CCAClassifier
from ssvep.synthetic import make_synthetic_dataset


# Library defaults (kept here so the banner can describe them).
_DEFAULT_BANDPASS = (3.0, 45.0)
_DEFAULT_NOTCH = 50.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SSVEP CCA baseline")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--mat", type=str,
                     help="Path to a single .mat file, or a directory (loads all matching .mat files via load_all)")
    src.add_argument("--synthetic", action="store_true",
                     help="Generate synthetic SSVEP data instead of loading .mat")

    p.add_argument("--bandpass", nargs=2, type=float, metavar=("LOW", "HIGH"),
                   default=None,
                   help="Custom bandpass low/high in Hz, e.g. --bandpass 6 50. "
                        "Default (when omitted) is canonical 3-45 Hz.")
    p.add_argument("--notch", type=float, default=None,
                   help="Custom notch frequency in Hz, e.g. --notch 60. "
                        "Default (when omitted) is canonical 50 Hz.")
    p.add_argument("--no-filter", action="store_true",
                   help="Skip continuous-first filtering entirely (raw epoched output). "
                        "Sanity-baseline use only.")
    p.add_argument("--harmonics", type=int, default=2,
                   help="Number of harmonics for CCA reference signals")
    p.add_argument("--window-s", type=float, default=None,
                   help="Trial window length in seconds (defaults to n_samples / fs)")
    return p.parse_args()


def _resolve_filter_kwargs(args: argparse.Namespace) -> tuple[dict, str]:
    """Return (kwargs_for_load_mat, banner_text) describing the preproc config."""
    if args.no_filter:
        return {"filter": False}, "[preproc] none (raw epoched, --no-filter)"

    custom = args.bandpass is not None or args.notch is not None
    bp = tuple(args.bandpass) if args.bandpass is not None else _DEFAULT_BANDPASS
    notch = args.notch if args.notch is not None else _DEFAULT_NOTCH
    kwargs: dict = {"l_freq": bp[0], "h_freq": bp[1], "notch_hz": notch}

    label = "custom" if custom else "canonical default"
    banner = f"[preproc] bandpass={bp[0]:g}-{bp[1]:g} Hz, notch={notch:g} Hz ({label})"
    return kwargs, banner


def main() -> int:
    args = parse_args()

    if args.synthetic:
        print("[preproc] n/a (synthetic data)")
        print("[load] generating synthetic dataset")
        ds = make_synthetic_dataset()
    else:
        filter_kwargs, banner = _resolve_filter_kwargs(args)
        print(banner)
        mat_path = Path(args.mat)
        if mat_path.is_dir():
            print(f"[load] {mat_path} (directory -> load_all, cross-file LOBO)")
            ds = io.load_all(mat_path, **filter_kwargs)
        else:
            print(f"[load] {mat_path}")
            ds = io.load_mat(mat_path, **filter_kwargs)

    X = ds["X"]
    y = ds["y"]
    fs = ds["fs"]
    stim_freqs = ds["stim_freqs"]
    blocks = ds["blocks"]

    print(f"[load] X={X.shape}  y={y.shape}  fs={fs}  freqs={stim_freqs}")

    factory = lambda: CCAClassifier(stim_freqs=stim_freqs, fs=fs, n_harmonics=args.harmonics)

    if blocks is not None and np.unique(blocks).size >= 2:
        print(f"[cv] leave-one-block-out across {np.unique(blocks).size} blocks")
        result = evaluation.leave_one_block_out_cv(X, y, blocks, factory)
        for b, acc in result["per_block"].items():
            print(f"  block {b}: acc = {acc:.3f}")
        print(f"[cv] mean acc = {result['mean']:.3f} ± {result['std']:.3f}")
        overall_acc = result["mean"]
    else:
        print("[cv] no blocks → single-shot evaluation on all trials")
        clf = factory()
        clf.fit(X, y)
        overall_acc = clf.score(X, y)
        print(f"[cv] acc = {overall_acc:.3f}")

    n_classes = int(stim_freqs.size)
    window_s = args.window_s if args.window_s is not None else float(X.shape[-1]) / fs
    bits_per_min = evaluation.itr(overall_acc, n_classes=n_classes, window_s=window_s)
    print(f"[itr] window={window_s:.2f}s  N={n_classes}  acc={overall_acc:.3f}  → {bits_per_min:.2f} bits/min")

    # Confusion (single-shot, full data)
    clf = factory()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    cm = evaluation.confusion_matrix(y, y_pred)
    print("[confusion]")
    print(cm)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
