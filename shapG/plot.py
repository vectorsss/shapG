import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Union, Optional, Any

def plot(
    shapley_values: Dict[Any, float], 
    top_n: int = 10, 
    style: str = 'seaborn-v0_8',
    file_name: Optional[str] = None,
    title: str = 'Top Shapley Values',
    figsize: tuple = (8, 6),
    color: str = '#1f77b4',
    show_values: bool = True,
    value_format: str = '{:.2f}',
    show_plot: bool = True
):
    """Plot the Shapley values as a horizontal bar chart.

    Args:
        shapley_values (Dict[Any, float]): Dictionary of Shapley values.
        top_n (int, optional): Number of top values to display. Defaults to 10.
        style (str, optional): Matplotlib style. Defaults to 'seaborn-v0_8'.
        file_name (Optional[str], optional): If provided, save figure to this file. Defaults to None.
        title (str, optional): Chart title. Defaults to 'Top Shapley Values'.
        figsize (tuple, optional): Figure size as (width, height). Defaults to (8, 6).
        color (str, optional): Bar color. Defaults to '#1f77b4'.
        show_values (bool, optional): Whether to display values next to bars. Defaults to True.
        value_format (str, optional): Format string for displayed values. Defaults to '{:.2f}'.
    """
    # Sort values in descending order
    sorted_values = sorted(shapley_values.items(), key=lambda item: item[1], reverse=True)
    
    # Select top-n values
    if len(sorted_values) > top_n:
        sorted_values = sorted_values[:top_n]
    
    # Unpack nodes and values
    nodes, values = zip(*sorted_values)
    
    # Convert node labels to strings for display
    node_labels = [str(node) for node in nodes]
    
    # Set the plot style
    plt.style.use(style)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bars
    y_pos = np.arange(len(nodes))
    bars = ax.barh(y_pos, values, align='center', color=color)
    
    # Configure axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(node_labels)
    ax.invert_yaxis()  # Display from top to bottom
    ax.set_xlabel('Shapley Value')
    ax.set_title(title)
    
    # Add grid lines for readability
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Display values next to bars
    if show_values:
        # Determine appropriate offset based on max value
        max_val = max(values)
        offset = max_val * 0.01
        
        for i, bar in enumerate(bars):
            value = values[i]
            ax.text(
                value + offset, 
                bar.get_y() + bar.get_height()/2, 
                value_format.format(value),
                va='center'
            )
    
    # Save to file if specified
    if file_name is not None:
        plt.savefig(file_name, dpi=300)
    if show_plot:
        plt.show()
    # return fig, ax for further customization
    return fig, ax

if __name__ == '__main__':
    shapley_values = {
        0: 0.1,
        1: 0.2,
        2: 0.3,
        3: 0.4,
        4: 0.5,
        5: 0.6,
        6: 0.7,
        7: 0.8,
        8: 0.9,
        9: 1.0,
    }
    fig, ax = plot(shapley_values, show_plot=False)
    ax.set_xlabel("Nodes Importance", fontsize=14)  # Change x-axis label
    ax.set_ylabel("Nodes", fontsize=14)               # Add/change y-axis label
    ax.spines['top'].set_visible(False)               # Remove top border
    ax.spines['right'].set_visible(False)             # Remove right border
    ax.set_title('Top 10 Shapley Values', fontsize=16)    # Change title font size
    # show the plot
    plt.tight_layout()
    plt.show()