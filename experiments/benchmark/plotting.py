"""All plotting functions."""

import matplotlib.pyplot as plt
import numpy as np

from .explainers import node_to_feature
from .labels import SKIP_METHODS, display as _display

# Optional plotting style
try:
    import gnuplot_style as gp

    gp.use("all", cycle_mode="zip")
except ImportError:
    pass


def _get_plot_cfg(cfg):
    pcfg = cfg.get("plot", {})
    return pcfg.get("figsize", [12, 8]), pcfg.get("dpi", 300)


def plot_timing(explainer_results, output_dir, cfg):
    figsize, dpi = _get_plot_cfg(cfg)
    labels = list(explainer_results.keys())
    display_labels = [_display(l) for l in labels]
    times = [explainer_results[l]["time"] for l in labels]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(display_labels, times)
    ax.set_xlabel("Time (s)")
    ax.set_title("Computation Time")
    for bar, t in zip(bars, times):
        ax.text(
            bar.get_width() + 0.01 * max(times),
            bar.get_y() + bar.get_height() / 2,
            f"{t:.2f}s",
            va="center",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(output_dir / "timing.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_backward_kpi(kpi_results, metric_name, output_dir, cfg):
    if not kpi_results:
        return
    figsize, dpi = _get_plot_cfg(cfg)
    fig, ax = plt.subplots(figsize=figsize)
    for method, data in kpi_results.items():
        if method in SKIP_METHODS:
            continue
        ax.plot(
            range(len(data["metrics"])),
            data["metrics"],
            label=f"{_display(method)} (S={data['slope']:.4f})",
        )
    ax.set_xlabel("Features Dropped", fontsize=14)
    ax.set_ylabel(metric_name, fontsize=14)
    ax.set_title("Backward KPI: Feature Dropping", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "kpi_backward.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()


def _plot_forward_on_ax(
    ax,
    forward_kpi_results,
    metric_name,
    full_metric,
    xlim=None,
    ylim=None,
):
    """Plot forward KPI curves on a given axes."""
    reference_styles = {
        "Greedy Best": {
            "color": "black",
            "linestyle": "--",
            "marker": "x",
            "linewidth": 2,
            "alpha": 0.8,
        },
    }

    for method, data in forward_kpi_results.items():
        if method in SKIP_METHODS:
            continue
        metrics = data["metrics"]
        max_val = max(metrics)
        max_step = metrics.index(max_val)
        exceed_str = ""
        if full_metric is not None:
            exceed_step = next(
                (i for i, m in enumerate(metrics) if m >= full_metric), None
            )
            exceed_str = (
                f", ≥full@{exceed_step}" if exceed_step is not None else ", <full"
            )
        disp = _display(method)
        label = (
            f"{disp} (S={data['slope']:.4f}, max={max_val:.4f}@{max_step}{exceed_str})"
        )
        if method in reference_styles:
            ax.plot(
                range(len(metrics)), metrics, label=label, **reference_styles[method]
            )
        else:
            ax.plot(range(len(metrics)), metrics, label=label, alpha=0.6)

    if full_metric is not None:
        ax.axhline(
            y=full_metric,
            color="blue",
            linestyle="-.",
            linewidth=1.5,
            alpha=0.7,
            label=f"Full model ({full_metric:.4f})",
        )

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    ax.set_xlabel("Number of Features Added", fontsize=14)
    ax.set_ylabel(metric_name, fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)


def plot_forward_kpi(forward_kpi_results, metric_name, full_metric, output_dir, cfg):
    if not forward_kpi_results:
        return
    figsize, dpi = _get_plot_cfg(cfg)
    max_x = max(
        len(d["metrics"]) - 1
        for m, d in forward_kpi_results.items()
        if m not in SKIP_METHODS
    )

    fig, ax = plt.subplots(figsize=figsize)
    _plot_forward_on_ax(ax, forward_kpi_results, metric_name, full_metric)
    ax.set_xticks(range(0, max_x + 1))
    ax.set_title("Forward KPI: Feature Addition", fontsize=16)
    ax.legend(fontsize=12, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_dir / "kpi_forward.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()


_MARKERS = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]


def plot_efficiency(fwd_data, timing_data, metric_name, full_metric, output_dir, cfg):
    """Scatter: x = computation time (s, log scale), y = 1 / peak performance.

    Each method gets a unique tab10 color and marker; labels are drawn near points.
    """
    _, dpi = _get_plot_cfg(cfg)
    fig, ax = plt.subplots(figsize=(8, 4))

    # Collect (time, 1/peak, display_label) for all methods
    points = []
    for method, data in fwd_data.items():
        if method in SKIP_METHODS:
            continue
        if method in timing_data:
            t = timing_data[method]
        elif "time" in data:
            t = data["time"]
        else:
            continue
        peak = max(data["metrics"])
        if peak <= 0:
            continue
        points.append((t, 1.0 / peak, _display(method)))

    if not points:
        return

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(points), 2)))

    for i, (x_val, y_val, label) in enumerate(points):
        ax.scatter(
            x_val,
            y_val,
            label=label,
            color=colors[i % len(colors)],
            marker=_MARKERS[i % len(_MARKERS)],
            edgecolors="black",
            s=100,
            zorder=4,
        )

    if full_metric is not None and full_metric > 0:
        ax.axhline(
            1.0 / full_metric,
            color="blue",
            linestyle="-.",
            linewidth=1.2,
            alpha=0.7,
            label=f"1/full ({1.0/full_metric:.4f})",
        )

    # Set log scale last so no subsequent call can override it
    ax.set_xscale("log")
    ax.set_xlabel("Time Cost (s)", fontsize=11)
    ax.set_ylabel(f"$1/$peak {metric_name}", fontsize=11)
    ax.set_title("Efficiency: Time vs. Peak Performance", fontsize=13)
    ax.grid(alpha=0.3)

    # Legend outside below figure with box
    ncol = min(len(points) + 1, 5)
    ax.legend(
        title="Methods",
        fontsize=7,
        title_fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=ncol,
        frameon=True,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "efficiency.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_shapley_values(explainer_results, columns, output_dir, cfg):
    figsize, dpi = _get_plot_cfg(cfg)
    fig, ax = plt.subplots(figsize=figsize)

    methods = list(explainer_results.keys())
    n_methods = len(methods)
    bar_width = 0.8 / n_methods

    all_features = set()
    for res in explainer_results.values():
        for node in res["values"]:
            feat = node_to_feature(node, columns)
            if feat:
                all_features.add(feat)
    all_features = sorted(all_features)
    x = np.arange(len(all_features))

    for i, method in enumerate(methods):
        vals = []
        for feat in all_features:
            found = False
            for node, v in explainer_results[method]["values"].items():
                if node_to_feature(node, columns) == feat:
                    vals.append(v)
                    found = True
                    break
            if not found:
                vals.append(0.0)
        ax.bar(x + i * bar_width, vals, bar_width, label=_display(method))

    ax.set_xticks(x + bar_width * (n_methods - 1) / 2)
    ax.set_xticklabels(all_features, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Shapley Value")
    ax.set_title("Shapley Values Comparison")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "shapley_values.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()
