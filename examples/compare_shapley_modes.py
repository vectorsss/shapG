#!/usr/bin/env python3
"""
Compare Shapley Modes: Retraining vs Masking

This script compares the impact of using retraining vs masking during Shapley value
calculation, while keeping the KPI mode constant (using retraining for fair comparison).

The comparison helps answer the research question:
  "Does retraining during Shapley calculation improve feature importance accuracy?"

Usage:
  python compare_shapley_modes.py --dataset housing --imputation mean --kpi-mode retrain
  python compare_shapley_modes.py --dataset housing --imputation mean --kpi-mode mask

Output:
  - Comparison tables printed to terminal
  - Comparative CSV files saved to output/comparisons/
  - Analysis summary with statistical metrics
"""

import argparse
import pandas as pd
from pathlib import Path
import numpy as np


def load_csv_results(dataset, imputation, shapley_mode, kpi_mode, output_dir='output'):
    """
    Load all CSV results for a given configuration.

    Returns:
        Dict with 'rankings', 'time', and 'kpi_metrics' DataFrames
    """
    output_path = Path(output_dir)

    results = {}

    # Load rankings
    rankings_file = output_path / f"rankings_{dataset}_{imputation}_shapley-{shapley_mode}_kpi-{kpi_mode}.csv"
    if rankings_file.exists():
        results['rankings'] = pd.read_csv(rankings_file)
    else:
        print(f"Warning: {rankings_file} not found")
        results['rankings'] = None

    # Load time comparison
    time_file = output_path / f"time_{dataset}_{imputation}_shapley-{shapley_mode}_kpi-{kpi_mode}.csv"
    if time_file.exists():
        results['time'] = pd.read_csv(time_file)
    else:
        print(f"Warning: {time_file} not found")
        results['time'] = None

    # Load KPI metrics
    kpi_file = output_path / f"kpi_metrics_{dataset}_{imputation}_shapley-{shapley_mode}_kpi-{kpi_mode}.csv"
    if kpi_file.exists():
        results['kpi_metrics'] = pd.read_csv(kpi_file)
    else:
        print(f"Warning: {kpi_file} not found")
        results['kpi_metrics'] = None

    return results


def calculate_ranking_agreement(rankings1, rankings2, top_k=10):
    """
    Calculate ranking agreement metrics between two ranking methods.

    Metrics:
      - Overlap@K: How many features appear in both top-K lists
      - Rank correlation: Spearman correlation of rankings
      - Position difference: Average absolute difference in positions

    Returns:
        Dict with agreement metrics
    """
    if rankings1 is None or rankings2 is None:
        return None

    metrics = {}

    # Get top-K features from each ranking
    top_k = min(top_k, len(rankings1), len(rankings2))

    set1 = set(rankings1[:top_k])
    set2 = set(rankings2[:top_k])

    # Overlap@K
    overlap = len(set1 & set2)
    metrics['overlap@K'] = overlap
    metrics['overlap_ratio'] = overlap / top_k

    # Calculate position differences for overlapping features
    position_diffs = []
    for feature in set1 & set2:
        pos1 = rankings1.index(feature) if feature in rankings1 else None
        pos2 = rankings2.index(feature) if feature in rankings2 else None
        if pos1 is not None and pos2 is not None:
            position_diffs.append(abs(pos1 - pos2))

    metrics['avg_position_diff'] = np.mean(position_diffs) if position_diffs else None
    metrics['max_position_diff'] = np.max(position_diffs) if position_diffs else None

    return metrics


def compare_rankings(mask_results, retrain_results, top_k=10):
    """
    Compare feature rankings between masking and retraining modes.

    Returns:
        DataFrame with comparison metrics for each method
    """
    mask_rankings = mask_results['rankings']
    retrain_rankings = retrain_results['rankings']

    if mask_rankings is None or retrain_rankings is None:
        print("Error: Cannot compare rankings - data not found")
        return None

    # Get all methods (exclude 'Top-N' column)
    methods = [col for col in mask_rankings.columns if col != 'Top-N']

    comparison_data = []

    for method in methods:
        mask_list = mask_rankings[method].dropna().tolist()
        retrain_list = retrain_rankings[method].dropna().tolist()

        # Calculate agreement metrics
        agreement = calculate_ranking_agreement(mask_list, retrain_list, top_k)

        if agreement:
            comparison_data.append({
                'Method': method,
                f'Overlap@{top_k}': agreement['overlap@K'],
                'Overlap Ratio': f"{agreement['overlap_ratio']:.2%}",
                'Avg Position Diff': f"{agreement['avg_position_diff']:.2f}" if agreement['avg_position_diff'] else 'N/A',
                'Max Position Diff': agreement['max_position_diff'] if agreement['max_position_diff'] else 'N/A'
            })

    return pd.DataFrame(comparison_data)


def compare_kpi_metrics(mask_results, retrain_results):
    """
    Compare KPI metrics (S values and metric sequences) between modes.

    Returns:
        DataFrame with side-by-side comparison
    """
    mask_kpi = mask_results['kpi_metrics']
    retrain_kpi = retrain_results['kpi_metrics']

    if mask_kpi is None or retrain_kpi is None:
        print("Error: Cannot compare KPI metrics - data not found")
        return None

    # Merge on Method
    comparison = pd.merge(
        mask_kpi[['Method', 'Weighted Slope (S)']],
        retrain_kpi[['Method', 'Weighted Slope (S)']],
        on='Method',
        suffixes=(' (Mask)', ' (Retrain)')
    )

    # Calculate difference and relative difference
    comparison['S Difference'] = (
        comparison['Weighted Slope (S) (Retrain)'] -
        comparison['Weighted Slope (S) (Mask)']
    )

    comparison['S Relative Change'] = (
        comparison['S Difference'] / comparison['Weighted Slope (S) (Mask)'] * 100
    )

    # Add interpretation
    comparison['Better Mode'] = comparison['S Difference'].apply(
        lambda x: 'Retrain' if x > 0 else ('Mask' if x < 0 else 'Same')
    )

    return comparison


def compare_execution_time(mask_results, retrain_results):
    """
    Compare execution times between modes.

    Returns:
        DataFrame with side-by-side time comparison
    """
    mask_time = mask_results['time']
    retrain_time = retrain_results['time']

    if mask_time is None or retrain_time is None:
        print("Error: Cannot compare execution times - data not found")
        return None

    # Merge on Method
    comparison = pd.merge(
        mask_time[['Method', 'Time (seconds)']],
        retrain_time[['Method', 'Time (seconds)']],
        on='Method',
        suffixes=(' (Mask)', ' (Retrain)')
    )

    # Calculate speedup
    comparison['Speedup (Mask/Retrain)'] = (
        comparison['Time (seconds) (Retrain)'] /
        comparison['Time (seconds) (Mask)']
    )

    return comparison


def print_summary_statistics(comparison_df, metric_name):
    """Print summary statistics for a comparison."""
    print(f"\n{metric_name} Summary Statistics:")
    print("=" * 80)

    if comparison_df is None or len(comparison_df) == 0:
        print("No data available")
        return

    print(comparison_df.to_string(index=False))
    print()


def save_comparison_results(dataset, imputation, kpi_mode,
                           ranking_comparison, kpi_comparison, time_comparison,
                           output_dir='output/comparisons'):
    """Save all comparison results to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_name = f"shapley_comparison_{dataset}_{imputation}_kpi-{kpi_mode}"

    saved_files = []

    if ranking_comparison is not None:
        ranking_file = output_path / f"{base_name}_rankings.csv"
        ranking_comparison.to_csv(ranking_file, index=False)
        saved_files.append(ranking_file)
        print(f"✓ Saved ranking comparison: {ranking_file}")

    if kpi_comparison is not None:
        kpi_file = output_path / f"{base_name}_kpi.csv"
        kpi_comparison.to_csv(kpi_file, index=False)
        saved_files.append(kpi_file)
        print(f"✓ Saved KPI comparison: {kpi_file}")

    if time_comparison is not None:
        time_file = output_path / f"{base_name}_time.csv"
        time_comparison.to_csv(time_file, index=False)
        saved_files.append(time_file)
        print(f"✓ Saved time comparison: {time_file}")

    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description='Compare Shapley modes (masking vs retraining)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare with KPI in retrain mode (recommended for fair comparison)
  python compare_shapley_modes.py --dataset housing --imputation mean --kpi-mode retrain

  # Compare with KPI in mask mode
  python compare_shapley_modes.py --dataset housing --imputation mean --kpi-mode mask

  # Compare all imputation strategies
  for imp in mean zero permutation; do
    python compare_shapley_modes.py --dataset housing --imputation $imp --kpi-mode retrain
  done
        """
    )

    parser.add_argument('--dataset', default='housing',
                       help='Dataset name (default: housing)')
    parser.add_argument('--imputation', default='mean',
                       choices=['mean', 'zero', 'permutation'],
                       help='Imputation strategy (default: mean)')
    parser.add_argument('--kpi-mode', default='retrain',
                       choices=['mask', 'retrain'],
                       help='KPI mode to use for comparison (default: retrain)')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of top features to compare (default: 10)')
    parser.add_argument('--output-dir', default='output',
                       help='Directory containing CSV files (default: output)')

    args = parser.parse_args()

    print("=" * 80)
    print("SHAPLEY MODE COMPARISON: Masking vs Retraining")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Imputation: {args.imputation}")
    print(f"  KPI Mode: {args.kpi_mode} (held constant)")
    print(f"  Top-K: {args.top_k}")
    print()

    # Load results for both Shapley modes
    print("Loading results...")
    print("-" * 80)

    print("\n[1/2] Loading Shapley MASK mode results...")
    mask_results = load_csv_results(
        args.dataset, args.imputation, 'mask', args.kpi_mode, args.output_dir
    )

    print("\n[2/2] Loading Shapley RETRAIN mode results...")
    retrain_results = load_csv_results(
        args.dataset, args.imputation, 'retrain', args.kpi_mode, args.output_dir
    )

    # Perform comparisons
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    # 1. Compare rankings
    print("\n" + "-" * 80)
    print(f"1. FEATURE RANKING AGREEMENT (Top-{args.top_k})")
    print("-" * 80)
    print("\nThis shows how similar the feature rankings are between the two Shapley modes.")
    print("Higher overlap = more agreement between methods")

    ranking_comparison = compare_rankings(mask_results, retrain_results, args.top_k)
    print_summary_statistics(ranking_comparison, "Ranking Agreement")

    # 2. Compare KPI metrics (S values)
    print("\n" + "-" * 80)
    print("2. KPI METRICS COMPARISON (Weighted Slope S)")
    print("-" * 80)
    print("\nThis shows how retraining during Shapley calculation affects feature")
    print("importance quality (higher S = better feature importance ranking).")

    kpi_comparison = compare_kpi_metrics(mask_results, retrain_results)
    print_summary_statistics(kpi_comparison, "KPI Metrics (S Values)")

    if kpi_comparison is not None and len(kpi_comparison) > 0:
        print("\nInterpretation:")
        better_with_retrain = (kpi_comparison['Better Mode'] == 'Retrain').sum()
        better_with_mask = (kpi_comparison['Better Mode'] == 'Mask').sum()
        same = (kpi_comparison['Better Mode'] == 'Same').sum()
        total = len(kpi_comparison)

        print(f"  - Methods improved with retraining: {better_with_retrain}/{total} ({better_with_retrain/total*100:.1f}%)")
        print(f"  - Methods better with masking: {better_with_mask}/{total} ({better_with_mask/total*100:.1f}%)")
        print(f"  - No difference: {same}/{total}")

        avg_relative_change = kpi_comparison['S Relative Change'].mean()
        print(f"\n  Average S value change: {avg_relative_change:+.2f}%")

        if avg_relative_change > 1:
            print(f"  → Retraining during Shapley calculation improves accuracy by ~{avg_relative_change:.1f}% on average")
        elif avg_relative_change < -1:
            print(f"  → Masking during Shapley calculation is better by ~{abs(avg_relative_change):.1f}% on average")
        else:
            print(f"  → No significant difference between modes")

    # 3. Compare execution time
    print("\n" + "-" * 80)
    print("3. EXECUTION TIME COMPARISON")
    print("-" * 80)
    print("\nThis shows the computational cost of retraining vs masking.")

    time_comparison = compare_execution_time(mask_results, retrain_results)
    print_summary_statistics(time_comparison, "Execution Time")

    if time_comparison is not None and len(time_comparison) > 0:
        avg_speedup = time_comparison['Speedup (Mask/Retrain)'].mean()
        print(f"\nAverage speedup (masking vs retraining): {avg_speedup:.1f}x")
        print(f"→ Masking is ~{avg_speedup:.0f}x faster than retraining on average")

    # Save results
    print("\n" + "=" * 80)
    print("SAVING COMPARISON RESULTS")
    print("=" * 80)

    saved_files = save_comparison_results(
        args.dataset, args.imputation, args.kpi_mode,
        ranking_comparison, kpi_comparison, time_comparison
    )

    print(f"\n✓ Saved {len(saved_files)} comparison files to output/comparisons/")

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nCompared Shapley modes (mask vs retrain) for:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Imputation: {args.imputation}")
    print(f"  KPI Mode: {args.kpi_mode}")
    print()
    print("Key Question: Does retraining during Shapley calculation improve accuracy?")
    print()

    if kpi_comparison is not None and len(kpi_comparison) > 0:
        avg_change = kpi_comparison['S Relative Change'].mean()
        if avg_change > 1:
            print(f"Answer: YES - Retraining improves S values by {avg_change:.1f}% on average")
        elif avg_change < -1:
            print(f"Answer: NO - Masking performs {abs(avg_change):.1f}% better on average")
        else:
            print(f"Answer: MINIMAL DIFFERENCE - Only {abs(avg_change):.2f}% difference")

        if time_comparison is not None:
            speedup = time_comparison['Speedup (Mask/Retrain)'].mean()
            print(f"\nTrade-off: Masking is {speedup:.0f}x faster but {'less' if avg_change > 0 else 'more'} accurate")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
