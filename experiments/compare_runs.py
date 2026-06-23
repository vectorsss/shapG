"""Compare two benchmark result folders method-by-method.

Generalizes the old ``compare_shapley_modes.py`` (which diffed masking vs
retraining): given any two ``results/<run>/`` directories produced by
``run_benchmark.py``, report for every shared method:

  * ranking agreement  — Overlap@K of the top-K feature rankings
  * S-slope delta       — backward/forward KPI slope, absolute and relative
  * speedup             — run-A time / run-B time

Usage:
    python compare_runs.py results/run_A/ results/run_B/
    python compare_runs.py results/run_A/ results/run_B/ --top-k 5 --kpi forward
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_rankings(run_dir):
    """method -> [features sorted by descending Shapley value]."""
    df = pd.read_csv(run_dir / "shapley_values.csv")
    rankings = {}
    for method, sub in df.groupby("method"):
        ordered = sub.sort_values("shapley_value", ascending=False)
        rankings[method] = list(ordered["node"].astype(str))
    return rankings


def _load_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def _overlap_at_k(a, b, k):
    return len(set(a[:k]) & set(b[:k])) / k if k else 0.0


def main():
    p = argparse.ArgumentParser(description="Compare two benchmark result folders.")
    p.add_argument("run_a", type=str, help="First results/<run>/ directory")
    p.add_argument("run_b", type=str, help="Second results/<run>/ directory")
    p.add_argument("--top-k", type=int, default=10, help="K for Overlap@K")
    p.add_argument(
        "--kpi",
        choices=["backward", "forward"],
        default="forward",
        help="Which KPI slope to compare",
    )
    args = p.parse_args()

    run_a, run_b = Path(args.run_a), Path(args.run_b)
    rank_a, rank_b = _load_rankings(run_a), _load_rankings(run_b)
    time_a = _load_json(run_a / "timing.json")
    time_b = _load_json(run_b / "timing.json")
    kpi_file = f"kpi_{args.kpi}.json"
    kpi_a = _load_json(run_a / kpi_file)
    kpi_b = _load_json(run_b / kpi_file)

    shared = [m for m in rank_a if m in rank_b]
    print(f"A = {run_a}")
    print(f"B = {run_b}")
    print(f"{len(shared)} shared methods | Overlap@{args.top_k} | {args.kpi} S-slope\n")
    header = (
        f"{'method':32s}{'Ovlp@K':>8s}"
        f"{'S_A':>9s}{'S_B':>9s}{'dS':>9s}{'dS%':>8s}{'speedup':>9s}"
    )
    print(header)
    print("-" * len(header))

    for m in shared:
        ovl = _overlap_at_k(rank_a[m], rank_b[m], args.top_k)
        sa = kpi_a.get(m, {}).get("slope")
        sb = kpi_b.get(m, {}).get("slope")
        ta, tb = time_a.get(m), time_b.get(m)
        if sa is not None and sb is not None:
            ds = sb - sa
            dsp = (ds / abs(sa) * 100) if sa else float("nan")
            s_cols = f"{sa:>9.4f}{sb:>9.4f}{ds:>+9.4f}{dsp:>+7.1f}%"
        else:
            s_cols = f"{'-':>9s}{'-':>9s}{'-':>9s}{'-':>8s}"
        speed = f"{ta / tb:>8.2f}x" if ta and tb else f"{'-':>9s}"
        print(f"{m:32s}{ovl:>8.2f}{s_cols}{speed}")


if __name__ == "__main__":
    main()
