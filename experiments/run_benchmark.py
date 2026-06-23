"""
Config-driven benchmark for Shapley value computation methods.

Usage:
    python run_benchmark.py configs/comparison_housing.yaml
    python run_benchmark.py configs/comparison_housing.yaml --only shapg cis
    python run_benchmark.py configs/comparison_housing.yaml --resume results/previous_run/
    python run_benchmark.py configs/comparison_housing.yaml --resume results/previous_run/ --only
    python run_benchmark.py configs/comparison_housing.yaml --resume results/previous_run/ --tidy
"""

import argparse
from datetime import datetime
from pathlib import Path

from benchmark import (
    build_char_func,
    build_graphs,
    build_model,
    compute_forward_kpi,
    compute_kpi,
    get_metric_fn,
    load_config,
    load_dataset,
    load_previous_results,
    plot_backward_kpi,
    plot_forward_kpi,
    plot_shapley_values,
    plot_timing,
    run_explainers,
    save_results,
    to_feature_ranking,
)

from sklearn.model_selection import train_test_split


def _compute_valid_labels(cfg):
    """Compute all valid method labels from config."""
    graph_names = list(cfg.get("graphs", {}).keys())
    valid_labels = set()
    for exp_name, ecfg in cfg["explainers"].items():
        if ecfg.get("graph_dependent", True):
            for gn in graph_names:
                valid_labels.add(f"{exp_name} | {gn}")
        else:
            valid_labels.add(exp_name)
    return valid_labels


def _tidy_results(valid_labels, *dicts):
    """Remove keys not in valid_labels from all dicts. Returns list of removed keys."""
    removed = [k for k in dicts[0] if k not in valid_labels]
    if removed:
        print(f"  --tidy: removing {len(removed)} stale methods: {removed}")
        for k in removed:
            for d in dicts:
                d.pop(k, None)
    return removed


def _print_summary(explainer_results, kpi_results, forward_kpi_results):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label, res in explainer_results.items():
        bwd = kpi_results.get(label, {}).get("slope")
        fwd = forward_kpi_results.get(label, {}).get("slope")
        bwd_str = f"  bwd_S={bwd:.4f}" if bwd is not None else ""
        fwd_str = f"  fwd_S={fwd:.4f}" if fwd is not None else ""
        print(f"  {label:40s}  time={res['time']:.2f}s{bwd_str}{fwd_str}")
    for ref in ("Greedy Best", "Greedy Worst", "Random"):
        ref_data = forward_kpi_results.get(ref, {})
        if "time" in ref_data:
            print(f"  {ref + ' (reference curve)':40s}  time={ref_data['time']:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Run Shapley value benchmark from config."
    )
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--output", type=str, default=None, help="Override output directory"
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Only run these explainer names (e.g. --only shapg cis). "
        "Use --only without args to load and replot only.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Load previous results from this directory, only re-run --only methods",
    )
    parser.add_argument(
        "--tidy",
        action="store_true",
        default=False,
        help="Remove results for methods not in the current config (requires --resume)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    # --only with no arguments = replot only (requires --resume)
    replot_only = args.only is not None and len(args.only) == 0

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    elif args.resume:
        output_dir = Path(args.resume)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_stem = config_path.stem
        output_dir = Path(__file__).parent / "results" / f"{config_stem}_{timestamp}"

    print(f"Config:  {config_path}")
    print(f"Output:  {output_dir}")

    # ── Replot-only mode ─────────────────────────────────────────────────────
    if replot_only:
        if not args.resume:
            print(
                "ERROR: --only (replot) requires --resume to specify the results directory."
            )
            return
        print("\n  Replot-only mode: loading all results from disk...")
        previous_results, prev_bwd_kpi, prev_fwd_kpi, full_metric = (
            load_previous_results(Path(args.resume))
        )

        X, y = load_dataset(cfg)
        model = build_model(cfg)
        metric_name = get_metric_fn(model, cfg)[1]

        if args.tidy:
            valid_labels = _compute_valid_labels(cfg)
            removed = _tidy_results(
                valid_labels, previous_results, prev_bwd_kpi, prev_fwd_kpi
            )
            if removed:
                save_results(
                    output_dir,
                    previous_results,
                    prev_bwd_kpi,
                    prev_fwd_kpi,
                    cfg,
                    full_metric=full_metric,
                )
                print("  Tidied results saved.")

        plot_timing(previous_results, output_dir, cfg)
        plot_backward_kpi(prev_bwd_kpi, metric_name, output_dir, cfg)
        plot_forward_kpi(prev_fwd_kpi, metric_name, full_metric, output_dir, cfg)
        plot_shapley_values(previous_results, X.columns, output_dir, cfg)
        print("\n  Plots regenerated.")
        _print_summary(previous_results, prev_bwd_kpi, prev_fwd_kpi)
        return

    # ── Normal mode ──────────────────────────────────────────────────────────

    # --- Load data ---
    print("\n[1/6] Loading dataset...")
    X, y = load_dataset(cfg)
    print(f"  {X.shape[0]} samples, {X.shape[1]} features")

    # --- Build model & split ---
    print("\n[2/6] Building model and characteristic function...")
    model = build_model(cfg)
    split_cfg = cfg["model"]["train_test_split"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
    )
    char_func = build_char_func(cfg, model, X_train, X_test, y_train, y_test, X, y)

    # --- Build graphs ---
    # Only ShapG-style graph_dependent explainers use more than the first graph;
    # graph_dependent: false explainers all run on the first graph. So if none of
    # the explainers being run is graph_dependent, skip building the rest (the
    # rank_deletion graphs in particular are expensive to construct).
    print("\n[3/6] Building graphs...")
    all_graph_names = list(cfg["graphs"].keys())
    run_names = set(args.only) if args.only else set(cfg["explainers"].keys())
    needs_all_graphs = any(
        cfg["explainers"].get(n, {}).get("graph_dependent", True) for n in run_names
    )
    graph_names = all_graph_names if needs_all_graphs else all_graph_names[:1]
    graphs = build_graphs(cfg, X, y, graph_names=graph_names)

    # --- Resume previous results ---
    previous_results = {}
    prev_bwd_kpi, prev_fwd_kpi = {}, {}
    prev_full_metric = None
    changed_methods = set()
    if args.resume:
        print(f"\n  Resuming from {args.resume}")
        (
            previous_results,
            prev_bwd_kpi,
            prev_fwd_kpi,
            prev_full_metric,
        ) = load_previous_results(Path(args.resume))

    # --- Run explainers ---
    print("\n[4/6] Running explainers...")
    new_results = run_explainers(cfg, char_func, graphs, X, y, model, only=args.only)
    changed_methods = set(new_results.keys())

    # Merge: new results override previous
    explainer_results = {**previous_results, **new_results}

    # --- Tidy: remove results not in current config ---
    if args.tidy:
        valid_labels = _compute_valid_labels(cfg)
        _tidy_results(valid_labels, explainer_results, prev_bwd_kpi, prev_fwd_kpi)

    # --- Feature rankings ---
    feature_rankings = {
        label: to_feature_ranking(res["values"], X.columns)
        for label, res in explainer_results.items()
    }

    # --- Backward KPI ---
    if changed_methods and prev_bwd_kpi:
        recompute_rankings = {
            k: v for k, v in feature_rankings.items() if k in changed_methods
        }
        unchanged_rankings = {
            k: v for k, v in feature_rankings.items() if k not in changed_methods
        }
        print(
            f"\n[5/6] Evaluating backward KPI (recomputing {len(recompute_rankings)} changed methods)..."
        )
        if recompute_rankings:
            new_bwd, metric_name = compute_kpi(cfg, X, y, recompute_rankings)
        else:
            new_bwd, metric_name = {}, get_metric_fn(model, cfg)[1]
        kpi_results = {
            k: prev_bwd_kpi[k] for k in unchanged_rankings if k in prev_bwd_kpi
        }
        kpi_results.update(new_bwd)
    else:
        print("\n[5/6] Evaluating backward KPI (feature dropping)...")
        kpi_results, metric_name = compute_kpi(cfg, X, y, feature_rankings)

    # --- Forward KPI ---
    if changed_methods and prev_fwd_kpi:
        recompute_rankings = {
            k: v for k, v in feature_rankings.items() if k in changed_methods
        }
        print(
            f"\n[6/6] Evaluating forward KPI (recomputing {len(recompute_rankings)} changed methods)..."
        )
        has_cached_refs = all(
            r in prev_fwd_kpi for r in ("Greedy Best", "Greedy Worst", "Random")
        )
        if recompute_rankings:
            full_fwd, full_metric, _ = compute_forward_kpi(
                cfg,
                X,
                y,
                recompute_rankings,
                skip_references=has_cached_refs,
            )
        else:
            full_fwd, full_metric = {}, None
        forward_kpi_results = {}
        for k in feature_rankings:
            if k in full_fwd:
                forward_kpi_results[k] = full_fwd[k]
            elif k in prev_fwd_kpi:
                forward_kpi_results[k] = prev_fwd_kpi[k]
        for ref in ("Greedy Best", "Greedy Worst", "Random"):
            if ref in full_fwd:
                forward_kpi_results[ref] = full_fwd[ref]
            elif ref in prev_fwd_kpi:
                forward_kpi_results[ref] = prev_fwd_kpi[ref]
        if full_metric is None:
            full_metric = prev_full_metric
    else:
        print("\n[6/6] Evaluating forward KPI (feature addition)...")
        forward_kpi_results, full_metric, _ = compute_forward_kpi(
            cfg, X, y, feature_rankings
        )

    # --- Save ---
    save_results(
        output_dir,
        explainer_results,
        kpi_results,
        forward_kpi_results,
        cfg,
        full_metric=full_metric,
    )

    # --- Plots ---
    plot_timing(explainer_results, output_dir, cfg)
    plot_backward_kpi(kpi_results, metric_name, output_dir, cfg)
    plot_forward_kpi(forward_kpi_results, metric_name, full_metric, output_dir, cfg)
    plot_shapley_values(explainer_results, X.columns, output_dir, cfg)

    # --- Summary ---
    _print_summary(explainer_results, kpi_results, forward_kpi_results)


if __name__ == "__main__":
    main()
