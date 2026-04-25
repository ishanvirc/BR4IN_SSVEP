"""Quick check: is CH11 g.tec's live LDA classifier?

Loads the four .mat files, computes CH11 predictions vs CH10 trigger
labels (only at samples where stimulus is active), reports accuracy
per file and overall. If accuracy lands near Guger 2012's 95.5%,
CH11 is the SOTA reference and we restructure the scaffold around it.

Also brute-forces all 24 permutations of the (LDA class index → stim
frequency) mapping, since g.tec systems map class indices to whichever
LED order the experimenter wired — not necessarily ascending frequency.
"""
from __future__ import annotations
import sys
from itertools import permutations
from pathlib import Path
import numpy as np
import scipy.io as sio

# Windows consoles default to cp1252, which can't encode some Unicode
# punctuation in the existing prints. Reconfigure stdout to UTF-8 so the
# script runs cleanly in PowerShell / cmd as well as Unix terminals.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

DATA_DIR = Path("data/raw")
STIM_FREQS = (9, 10, 12, 15)


def load_continuous(path: Path) -> np.ndarray:
    """Return the 11-row continuous matrix from a .mat file."""
    mat = sio.loadmat(path, squeeze_me=True, simplify_cells=True)
    flat = {k: v for k, v in mat.items() if not k.startswith("__")}
    # If wrapped in a single struct, descend.
    if len(flat) == 1:
        only = next(iter(flat.values()))
        if isinstance(only, dict):
            flat = only
    # Find the (11, N) or (N, 11) array.
    for v in flat.values():
        arr = np.asarray(v)
        if arr.ndim == 2 and 11 in arr.shape:
            return arr if arr.shape[0] == 11 else arr.T
    raise ValueError(f"No (11, N) array found in {path.name}; keys={list(flat.keys())}")


def _load_scored(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (trigger, lda_out, scored_mask) for a single .mat file.

    scored_mask selects samples where stim is active AND CH11 has fired.
    Used by both ``score_file`` and the cross-file permutation sweep so
    they share the same definition of "what to score."
    """
    data = load_continuous(path)
    trigger = data[9]
    lda_out = data[10]
    scored = (trigger != 0) & (lda_out != 0)
    return trigger, lda_out, scored


def sweep_permutations(
    lda_arr: np.ndarray,
    trigger_arr: np.ndarray,
) -> list[tuple[float, tuple[float, ...]]]:
    """Brute-force every (LDA class → stim freq) mapping.

    Parameters
    ----------
    lda_arr     : 1-D array of LDA outputs (already filtered to scored samples).
    trigger_arr : 1-D array of stim frequencies (same filter, same length).

    Returns
    -------
    list of (accuracy, perm) sorted descending. ``perm`` is a tuple of
    stim frequencies in canonical class-index order: perm[0] is the
    frequency mapped to LDA class index 1, perm[1] to class index 2, etc.
    """
    base_freqs = sorted(STIM_FREQS)
    results: list[tuple[float, tuple[float, ...]]] = []
    for perm in permutations(base_freqs):
        mapping = {float(i + 1): float(f) for i, f in enumerate(perm)}
        mapped = np.vectorize(lambda x: mapping.get(float(x), -1.0))(lda_arr)
        acc = float(np.mean(mapped == trigger_arr))
        results.append((acc, tuple(float(f) for f in perm)))
    return sorted(results, key=lambda t: t[0], reverse=True)


def score_file(path: Path) -> dict:
    trigger, lda_out, scored = _load_scored(path)
    active = trigger != 0
    n_active = int(active.sum())
    n_scored = int(scored.sum())
    lda_unique = sorted(set(np.unique(lda_out[scored]))) if n_scored else []

    if n_scored == 0:
        return {
            "file": path.name,
            "n_active_samples": n_active,
            "n_scored_samples": 0,
            "trigger_unique": sorted(set(np.unique(trigger[active]))),
            "lda_unique_nonzero": [],
            "direct_accuracy": float("nan"),
            "mapped_accuracy": float("nan"),
        }

    # Canonical mapping: LDA class i (1-indexed) ↔ i-th lowest stim freq.
    # Applied unconditionally — files where CH11 emitted only a subset of
    # classes are still scoreable.
    sorted_freqs = sorted(STIM_FREQS)
    mapping = {float(i + 1): float(f) for i, f in enumerate(sorted_freqs)}
    mapped_lookup = np.vectorize(lambda x: mapping.get(float(x), -1.0))
    direct_acc = float(np.mean(lda_out[scored] == trigger[scored]))
    mapped_acc = float(np.mean(mapped_lookup(lda_out[scored]) == trigger[scored]))

    return {
        "file": path.name,
        "n_active_samples": n_active,
        "n_scored_samples": n_scored,
        "trigger_unique": sorted(set(np.unique(trigger[active]))),
        "lda_unique_nonzero": lda_unique,
        "direct_accuracy": direct_acc,
        "mapped_accuracy": mapped_acc,
    }


def _format_perm(perm: tuple[float, ...]) -> str:
    return "{" + ", ".join(f"{i+1}: {f:g} Hz" for i, f in enumerate(perm)) + "}"


def main() -> int:
    files = sorted(DATA_DIR.glob("subject_*_fvep_led_training_*.mat"))
    if not files:
        print(f"No .mat files found in {DATA_DIR}")
        return 1

    # Existing per-file direct/mapped output (preserved for comparison).
    accs: list[float] = []
    for p in files:
        r = score_file(p)
        print(r)
        best = max(
            v for v in (r.get("direct_accuracy"), r.get("mapped_accuracy"))
            if v is not None and not np.isnan(v)
        )
        accs.append(best)
    print(f"\n=== Mean best-of-{{direct, mapped}} accuracy: {np.mean(accs):.3f} ===")
    print("Guger 2012 reports 95.5% mean across 53 subjects on identical hardware.")
    print("If our number is 0.85–0.99, CH11 is the SOTA reference — proceed with restructure.")
    print("If 0.50–0.80, it might be partial/early-trial LDA — restructure with caveats.")
    print("If <0.30 or =1.00 exactly, something is off — investigate before restructuring.")

    # Permutation sweep — per file, then aggregate across all files.
    print("\n" + "=" * 64)
    print("Permutation sweep (24 mappings of LDA class index -> stim freq)")
    print("=" * 64)
    aggregate_lda: list[np.ndarray] = []
    aggregate_trig: list[np.ndarray] = []
    for p in files:
        trigger, lda_out, scored = _load_scored(p)
        if not scored.any():
            print(f"\n--- {p.name}: no scored samples, skipping sweep")
            continue
        lda_scored = lda_out[scored]
        trig_scored = trigger[scored]
        ranking = sweep_permutations(lda_scored, trig_scored)
        print(f"\n--- {p.name}  (n_scored={scored.sum()})")
        for acc, perm in ranking[:3]:
            print(f"  acc={acc:.3f}  mapping={_format_perm(perm)}")
        aggregate_lda.append(lda_scored)
        aggregate_trig.append(trig_scored)

    if aggregate_lda:
        all_lda = np.concatenate(aggregate_lda)
        all_trig = np.concatenate(aggregate_trig)
        ranking = sweep_permutations(all_lda, all_trig)
        best_acc, best_perm = ranking[0]
        print("\n" + "-" * 64)
        print(f"Aggregate sweep across {len(aggregate_lda)} file(s), n_scored={all_lda.size}")
        print("-" * 64)
        for acc, perm in ranking[:3]:
            print(f"  acc={acc:.3f}  mapping={_format_perm(perm)}")
        print(f"\nBest mapping: {_format_perm(best_perm)}")
        print(f"Best aggregate accuracy: {best_acc:.3f}")
        if best_acc >= 0.85:
            verdict = ">= 0.85 → CH11 is g.tec's live LDA. SOTA-baseline strategy proceeds."
        elif best_acc >= 0.50:
            verdict = "0.50–0.85 → CH11 is likely LDA but partial/noisy/latency-misaligned. Investigate."
        elif best_acc >= 0.30:
            verdict = "0.30–0.50 → ambiguous. CH11's identity unresolved."
        else:
            verdict = "< 0.30 → CH11 is probably not what we think it is. Replan."
        print(f"Diagnostic bucket: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())