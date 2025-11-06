"""
Standalone plotting script for benchmark results.

This script loads cached benchmark results and creates publication-quality plots:
1. KPI comparison plot (feature importance evaluation)
2. Time comparison histogram (algorithm execution time)

Features:
- Excludes ImprovedQRCS and ImprovedBlockQRCS from plots
- Uses histogram for time comparison (vertical bars)
- Loads data from cache files
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional

try:
    import gnuplot_style as gp
    gp.use("all")
except ImportError:
    print("gnuplot_style not found, using default matplotlib style")


def load_benchmark_cache(dataset_name='housing'):
    """
    Load unified cache containing all benchmark results.

    Parameters:
    - dataset_name: Name of the dataset (e.g., 'housing', 'h1n1')

    Returns:
    - Dictionary with all cached data (Shapley values, timing, and KPI results)
    """
    cache_file = f"{dataset_name}_shapley_cache.pkl"

    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache file {cache_file} not found. Run benchmark first.")

    print(f"Loading unified cache from {cache_file}...")
    with open(cache_file, 'rb') as f:
        cached_data = pickle.load(f)

    # Validate cache contents
    time_results = cached_data.get('time_results', {})
    kpi_results = cached_data.get('kpi_results', {})

    print(f"  Shapley methods: {len(time_results)}")
    print(f"  KPI methods: {len(kpi_results)}")

    return cached_data


def filter_methods(data_dict, exclude_methods=None):
    """
    Filter out specific methods from results.

    Parameters:
    - data_dict: Dictionary to filter
    - exclude_methods: List of method names to exclude

    Returns:
    - Filtered dictionary
    """
    if exclude_methods is None:
        exclude_methods = ['ImprovedQRCS', 'ImprovedBlockQRCS']

    filtered = {}
    for key, value in data_dict.items():
        # Check if key should be excluded
        should_exclude = False
        for exclude in exclude_methods:
            if exclude in key:
                should_exclude = True
                break

        if not should_exclude:
            filtered[key] = value

    return filtered


def plot_kpi_comparison(results: Dict, filename: str = 'kpi_comparison.png',
                        metric_name: str = '$R^2$', exclude_methods=None):
    """
    Plot KPI comparison from cached results.

    Parameters:
    - results: Dictionary with KPI results
    - filename: Output filename
    - metric_name: Name of the metric for y-axis label
    - exclude_methods: List of methods to exclude
    """
    # Filter out unwanted methods
    results = filter_methods(results, exclude_methods)

    plt.figure(figsize=(12, 8))

    for method, data in results.items():
        label = f'{method} $S$={data["Slope"]:.4f}'
        plt.plot(
            range(len(data['Metrics'])),
            data['Metrics'],
            label=label,
            alpha=0.6,
        )

    plt.xlabel('Number of Features Dropped', fontsize=13, fontweight='bold')
    plt.ylabel(metric_name, fontsize=13, fontweight='bold')
    plt.title(f'Feature Importance Comparison ({metric_name})',
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved KPI comparison plot to {filename}")
    plt.close()


def plot_time_comparison_histogram(time_results: Dict, filename: str = 'time_comparison.png',
                                   exclude_methods=None):
    """
    Plot time comparison as a histogram (vertical bars).

    Parameters:
    - time_results: Dictionary mapping algorithm names to execution times
    - filename: Output filename
    - exclude_methods: List of methods to exclude
    """
    # Filter out unwanted methods
    time_results = filter_methods(time_results, exclude_methods)

    # Sort by time for better visualization
    sorted_items = sorted(time_results.items(), key=lambda x: x[1])
    algorithms = [item[0] for item in sorted_items]
    times = [item[1] for item in sorted_items]

    # Create histogram (vertical bars)
    plt.figure(figsize=(14, 8))

    # Use position-based x-axis
    x_pos = np.arange(len(algorithms))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(algorithms)))

    bars = plt.bar(x_pos, times, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1.5)

    # Add value labels on top of bars
    for i, (bar, time_val) in enumerate(zip(bars, times)):
        plt.text(i, time_val, f'{time_val:.2f}s',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Set x-axis labels
    plt.xticks(x_pos, algorithms, rotation=45, ha='right', fontsize=11)
    plt.ylabel('Execution Time (seconds)', fontsize=13, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=13, fontweight='bold')
    plt.title('Algorithm Execution Time Comparison', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved time comparison histogram to {filename}")
    plt.close()


def plot_time_comparison_log_scale(time_results: Dict, filename: str = 'time_comparison_log.png',
                                   exclude_methods=None):
    """
    Plot time comparison with log scale (useful when times vary greatly).

    Parameters:
    - time_results: Dictionary mapping algorithm names to execution times
    - filename: Output filename
    - exclude_methods: List of methods to exclude
    """
    # Filter out unwanted methods
    time_results = filter_methods(time_results, exclude_methods)

    # Sort by time
    sorted_items = sorted(time_results.items(), key=lambda x: x[1])
    algorithms = [item[0] for item in sorted_items]
    times = [item[1] for item in sorted_items]

    # Create histogram with log scale
    plt.figure(figsize=(14, 8))

    x_pos = np.arange(len(algorithms))
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(algorithms)))

    bars = plt.bar(x_pos, times, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, time_val) in enumerate(zip(bars, times)):
        plt.text(i, time_val, f'{time_val:.2f}s',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.xticks(x_pos, algorithms, rotation=45, ha='right', fontsize=11)
    plt.ylabel('Execution Time (seconds, log scale)', fontsize=13, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=13, fontweight='bold')
    plt.title('Algorithm Execution Time Comparison (Log Scale)',
              fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.3, linestyle='--', which='both')
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved log-scale time comparison to {filename}")
    plt.close()


def create_summary_table(time_results: Dict, kpi_results: Dict,
                        output_file: str = 'benchmark_summary.csv',
                        exclude_methods=None):
    """
    Create a summary table with execution times and KPI slopes.

    Parameters:
    - time_results: Dictionary with timing data
    - kpi_results: Dictionary with KPI data
    - output_file: Output CSV filename
    - exclude_methods: List of methods to exclude
    """
    # Filter both dictionaries
    time_results = filter_methods(time_results, exclude_methods)
    kpi_results = filter_methods(kpi_results, exclude_methods)

    # Create summary data
    summary_data = []
    for method in time_results.keys():
        row = {
            'Algorithm': method,
            'Execution Time (s)': time_results.get(method, 0),
            'KPI Slope': kpi_results.get(method, {}).get('Slope', 0) if method in kpi_results else 0
        }
        summary_data.append(row)

    # Create DataFrame and sort by execution time
    df = pd.DataFrame(summary_data)
    df = df.sort_values('Execution Time (s)')

    # Save to CSV
    df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"\nSaved summary table to {output_file}")
    print("\nSummary Table:")
    print(df.to_string(index=False))

    return df


def main():
    """
    Main function to generate all plots.
    """
    print("=" * 70)
    print("BENCHMARK RESULTS PLOTTING SCRIPT")
    print("=" * 70)

    # Configuration
    dataset_name = 'housing'
    exclude_methods = ['ImprovedQRCS', 'ImprovedBlockQRCS']

    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Excluding: {', '.join(exclude_methods)}")

    # Load cached data from unified cache file
    print("\n" + "=" * 70)
    print("Loading cached data...")
    print("=" * 70)

    try:
        cached_data = load_benchmark_cache(dataset_name)
        time_results = cached_data.get('time_results', {})
        kpi_results = cached_data.get('kpi_results', {})

        if not kpi_results:
            print("\nWarning: No KPI results found in cache. Skipping KPI comparison plot.")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run benchmark_qr_new_api.py first to generate cache file.")
        return

    # Generate plots
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70)

    # 1. KPI Comparison Plot (if KPI results available)
    if kpi_results:
        print("\n1. Creating KPI comparison plot...")
        plot_kpi_comparison(
            kpi_results,
            filename=f'{dataset_name}_kpi_comparison.png',
            metric_name='$R^2$',
            exclude_methods=exclude_methods
        )
    else:
        print("\n1. Skipping KPI comparison plot (no KPI results)")

    # 2. Time Comparison Histogram
    print("\n2. Creating time comparison histogram...")
    plot_time_comparison_histogram(
        time_results,
        filename=f'{dataset_name}_time_histogram.png',
        exclude_methods=exclude_methods
    )

    # 3. Time Comparison with Log Scale (optional, useful for large time differences)
    print("\n3. Creating log-scale time comparison...")
    plot_time_comparison_log_scale(
        time_results,
        filename=f'{dataset_name}_time_histogram_log.png',
        exclude_methods=exclude_methods
    )

    # 4. Summary Table
    print("\n4. Creating summary table...")
    create_summary_table(
        time_results,
        kpi_results,
        output_file=f'{dataset_name}_benchmark_summary.csv',
        exclude_methods=exclude_methods
    )

    print("\n" + "=" * 70)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print("\nOutput files:")
    print(f"  - {dataset_name}_kpi_comparison.png")
    print(f"  - {dataset_name}_time_histogram.png")
    print(f"  - {dataset_name}_time_histogram_log.png")
    print(f"  - {dataset_name}_benchmark_summary.csv")


if __name__ == "__main__":
    main()
