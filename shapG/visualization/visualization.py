"""
Visualization module for feature importance and Shapley values.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class FeatureImportanceVisualizer:
    """Visualizer for feature importance scores."""

    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (10, 6)):
        """Initialize visualizer.

        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        if style in mplstyle.available:
            plt.style.use(style)

    def plot_importance(
        self,
        importance_scores: Union[Dict[int, float], pd.Series],
        feature_names: Optional[List[str]] = None,
        top_n: Optional[int] = None,
        figsize: Optional[Tuple[int, int]] = None,
        color: str = "#1f77b4",
        title: str = "Feature Importance",
        xlabel: str = "Importance Score",
        ylabel: str = "Features",
        show_values: bool = True,
        value_format: str = "{:.3f}",
        sort: bool = True,
        ax: Optional[Axes] = None,
        show_plot: bool = True,
        filename: Optional[str] = None,
    ) -> Tuple[Figure, Axes]:
        """Plot feature importance as horizontal bar chart.

        Args:
            importance_scores: Dictionary or Series of importance scores
            feature_names: Optional feature names
            top_n: Show only top N features
            figsize: Figure size
            color: Bar color
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            show_values: Whether to show values on bars
            value_format: Format string for values
            sort: Whether to sort by importance
            ax: Optional axes to plot on

        Returns:
            Figure and axes objects
        """
        # Convert to DataFrame for easier handling
        if isinstance(importance_scores, dict):
            df = pd.DataFrame.from_dict(
                importance_scores, orient="index", columns=["importance"]
            )
        else:
            df = pd.DataFrame(importance_scores, columns=["importance"])

        # Add feature names
        if feature_names:
            df["feature"] = [
                feature_names[i] if i < len(feature_names) else f"Feature {i}"
                for i in df.index
            ]
        else:
            df["feature"] = [f"Feature {i}" for i in df.index]

        # Sort by importance
        if sort:
            df = df.sort_values("importance", ascending=True)

        # Select top N
        if top_n and len(df) > top_n:
            df = df.tail(top_n)

        # Create plot
        if ax is None:
            actual_figsize = figsize if figsize is not None else self.figsize
            fig, ax = plt.subplots(figsize=actual_figsize)
        else:
            fig = ax.figure

        # Create horizontal bar plot
        bars = ax.barh(df["feature"], df["importance"], color=color)

        # Add value labels
        if show_values:
            for bar in bars:
                width = bar.get_width()
                label = value_format.format(width)
                ax.text(
                    width,
                    bar.get_y() + bar.get_height() / 2,
                    label,
                    ha="left",
                    va="center",
                    fontsize=9,
                )

        # Labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # Grid
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        # Handle file saving and showing
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
        if show_plot:
            plt.show()

        return fig, ax

    def plot_comparison(
        self,
        importance_dict: Dict[str, Union[Dict[int, float], pd.Series]],
        feature_names: Optional[List[str]] = None,
        top_n: Optional[int] = 10,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Feature Importance Comparison",
        colors: Optional[List[str]] = None,
        show_plot: bool = True,
        filename: Optional[str] = None,
    ) -> Tuple[Figure, Axes]:
        """Plot comparison of multiple importance scores.

        Args:
            importance_dict: Dictionary of method names to importance scores
            feature_names: Optional feature names
            top_n: Show only top N features
            figsize: Figure size
            title: Plot title
            colors: Optional list of colors for each method

        Returns:
            Figure and axes objects
        """
        # Convert all to DataFrames
        dfs = {}
        all_features = set()

        for method, scores in importance_dict.items():
            if isinstance(scores, dict):
                df = pd.DataFrame.from_dict(scores, orient="index", columns=[method])
            else:
                df = pd.DataFrame(scores, columns=[method])
            dfs[method] = df
            all_features.update(df.index)

        # Combine all methods
        # Sort features handling mixed types
        try:
            # Try to sort normally
            sorted_features = sorted(all_features)
        except TypeError:
            # If mixed types, convert all to strings and sort
            sorted_features = sorted(all_features, key=str)
        combined = pd.DataFrame(index=sorted_features)
        for method, df in dfs.items():
            combined[method] = df[method]

        # Fill NaN with 0
        combined = combined.fillna(0)

        # Add feature names
        if feature_names:
            # Handle both integer indices and string feature names
            feature_labels = []
            for idx in combined.index:
                if isinstance(idx, int) and idx < len(feature_names):
                    feature_labels.append(feature_names[idx])
                elif isinstance(idx, str):
                    # If it's already a string (feature name), use it directly
                    feature_labels.append(idx)
                else:
                    feature_labels.append(f"Feature {idx}")
            combined["feature"] = feature_labels
        else:
            combined["feature"] = [str(i) for i in combined.index]

        # Get top features by mean importance
        combined["mean_importance"] = combined[list(importance_dict.keys())].mean(
            axis=1
        )
        combined = combined.nlargest(top_n, "mean_importance")

        # Create plot
        actual_figsize = figsize if figsize is not None else (12, 8)
        fig, ax = plt.subplots(figsize=actual_figsize)

        # Set up bar positions
        n_methods = len(importance_dict)
        n_features = len(combined)
        bar_width = 0.8 / n_methods
        positions = np.arange(n_features)

        # Default colors
        if colors is None:
            colors = plt.cm.Set3(np.linspace(0, 1, n_methods))

        # Plot bars for each method
        for i, method in enumerate(importance_dict.keys()):
            pos = positions + i * bar_width
            ax.barh(pos, combined[method], bar_width, label=method, color=colors[i])

        # Customize plot
        ax.set_yticks(positions + bar_width * (n_methods - 1) / 2)
        ax.set_yticklabels(combined["feature"])
        ax.set_xlabel("Importance Score")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        # Handle file saving and showing
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
        if show_plot:
            plt.show()

        return fig, ax

    def plot_heatmap(
        self,
        importance_matrix: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        sample_names: Optional[List[str]] = None,
        row_labels: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        cmap: str = "RdBu_r",
        title: str = "Feature Importance Heatmap",
        show_values: bool = False,
        value_format: str = "{:.2f}",
        show_plot: bool = True,
        filename: Optional[str] = None,
    ) -> Tuple[Figure, Axes]:
        """Plot importance scores as heatmap.

        Args:
            importance_matrix: 2D array of importance scores (samples x features)
            feature_names: Optional feature names
            sample_names: Optional sample names
            figsize: Figure size
            cmap: Colormap
            title: Plot title
            show_values: Whether to show values in cells
            value_format: Format string for values

        Returns:
            Figure and axes objects
        """
        actual_figsize = figsize if figsize is not None else (12, 8)
        fig, ax = plt.subplots(figsize=actual_figsize)

        # Convert to array if needed
        if isinstance(importance_matrix, pd.DataFrame):
            data = importance_matrix.values
            if feature_names is None:
                feature_names = list(importance_matrix.columns)
            if sample_names is None:
                sample_names = list(importance_matrix.index)
        else:
            data = importance_matrix

        # Use row_labels and col_labels if provided (they override feature_names and sample_names)
        if row_labels is not None:
            sample_names = row_labels
        if col_labels is not None:
            feature_names = col_labels

        # Create heatmap
        im = ax.imshow(data, cmap=cmap, aspect="auto")

        # Set ticks
        if feature_names:
            ax.set_xticks(np.arange(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=45, ha="right")
        if sample_names:
            ax.set_yticks(np.arange(len(sample_names)))
            ax.set_yticklabels(sample_names)

        # Add values
        if show_values:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    text = ax.text(
                        j,
                        i,
                        value_format.format(data[i, j]),
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )

        # Colorbar
        plt.colorbar(im, ax=ax)

        # Labels
        ax.set_xlabel("Features")
        ax.set_ylabel("Samples")
        ax.set_title(title)

        plt.tight_layout()

        # Handle file saving and showing
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
        if show_plot:
            plt.show()

        return fig, ax

    @staticmethod
    def plot_network(
        graph: Any,  # nx.Graph
        importance_scores: Optional[Dict[int, float]] = None,
        node_values: Optional[Dict[int, float]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        node_size_factor: float = 1000,
        edge_width_factor: float = 2,
        cmap: str = "coolwarm",
        title: str = "Feature Importance Network",
        layout: str = "spring",
        show_plot: bool = True,
        filename: Optional[str] = None,
    ) -> Tuple[Figure, Axes]:
        """Plot importance scores on network graph.

        Args:
            graph: NetworkX graph
            importance_scores: Node importance scores
            figsize: Figure size
            node_size_factor: Factor for node sizes
            edge_width_factor: Factor for edge widths
            cmap: Colormap for nodes
            title: Plot title
            layout: Graph layout algorithm

        Returns:
            Figure and axes objects
        """
        import networkx as nx

        # Handle node_values parameter (alias for importance_scores)
        scores = node_values if node_values is not None else importance_scores
        if scores is None:
            scores = {node: 0 for node in graph.nodes()}

        actual_figsize = figsize if figsize is not None else (12, 8)
        fig, ax = plt.subplots(figsize=actual_figsize)

        # Get layout
        if layout == "spring":
            pos = nx.spring_layout(graph)
        elif layout == "circular":
            pos = nx.circular_layout(graph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(graph)
        else:
            pos = nx.spring_layout(graph)

        # Node sizes and colors based on importance
        node_sizes = [
            abs(scores.get(node, 0)) * node_size_factor for node in graph.nodes()
        ]
        node_colors = [scores.get(node, 0) for node in graph.nodes()]

        # Edge widths based on weights
        edge_widths = []
        for u, v in graph.edges():
            weight = graph[u][v].get("weight", 1)
            edge_widths.append(weight * edge_width_factor)

        # Draw network
        score_values = list(scores.values()) if scores else [0]
        vmin, vmax = min(score_values), max(score_values)

        nx.draw_networkx_nodes(
            graph,
            pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )

        nx.draw_networkx_edges(graph, pos, width=edge_widths, alpha=0.3, ax=ax)

        nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Importance Score")

        ax.set_title(title)
        ax.axis("off")

        plt.tight_layout()

        # Handle file saving and showing
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
        if show_plot:
            plt.show()

        return fig, ax


# Convenience function for backward compatibility
def plot_shapley_values(
    shapley_values: Union[Dict[int, float], pd.Series],
    feature_names: Optional[List[str]] = None,
    top_n: int = 10,
    style: str = "seaborn-v0_8",
    figsize: Tuple[int, int] = (10, 6),
    color: str = "#1f77b4",
    show_values: bool = True,
    show_plot: bool = True,
    title: str = "Shapley Values",
    graph: Optional[Any] = None,
    layout: str = "spring",
    node_size: int = 300,
    font_size: int = 10,
    filename: Optional[str] = None,
) -> Optional[Tuple[Figure, Axes]]:
    """Plot Shapley values (backward compatibility function).

    Args:
        shapley_values: Shapley values to plot
        feature_names: Optional feature names
        top_n: Number of top features to show
        style: Matplotlib style
        figsize: Figure size
        color: Bar color
        show_values: Whether to show values on bars
        show_plot: Whether to display the plot

    Returns:
        Figure and axes if show_plot is False, None otherwise
    """
    if graph is not None:
        # Use network plot
        fig, ax = FeatureImportanceVisualizer.plot_network(
            graph=graph,
            node_values=shapley_values,
            figsize=figsize,
            title=title,
            layout=layout,
            node_size_factor=node_size,
            show_plot=show_plot,
            filename=filename,
        )
    else:
        # Use bar chart
        visualizer = FeatureImportanceVisualizer(style, figsize)
        fig, ax = visualizer.plot_importance(
            shapley_values,
            feature_names=feature_names,
            top_n=top_n,
            figsize=figsize,
            color=color,
            title=title,
            xlabel="Shapley Value",
            show_values=show_values,
            show_plot=show_plot,
            filename=filename,
        )

    return (fig, ax)
