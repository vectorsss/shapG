# ShapG: a fast and exactly approach to approximate the Shaple value on graph

## Installation

```bash
pip install shapG
```

## Features

- Exact Shapley value computation for small to medium-sized graphs
- Fast approximate Shapley value computation for large graphs using local search
- Visualization tools for Shapley values
- Utility functions for graph generation and analysis

## Quick Start

```python
import networkx as nx
from shapG.shapley import shapG, graph_generator
from shapG.plot import plot

# Generate a random graph
G = graph_generator(n_nodes=10, density=0.5)

# Compute approximate Shapley values
shapley_values = shapG(G, depth=1, m=15)

# Visualize the results
plot(shapley_values, top_n=10)
```

## Advanced Usage

### Custom Characteristic Function

You can define a custom characteristic function:

```python
def my_coalition_function(G, S):
    # Your custom characteristic function
    subgraph = G.subgraph(S)
    return nx.density(subgraph) * len(S)

# Use your custom function
shapley_values = shapG(G, f=my_coalition_function)
```

More example, see `./examples`.

### Customizing the Plot

```python
fig, ax = plot(
    shapley_values,
    top_n=5,                    # Show only top 5 values
    style='seaborn-v0_8',       # Matplotlib style
    file_name="shapley.eps",    # Save to file
    title="Node Importance",    # Custom title
    figsize=(10, 6),            # Figure size (width, height)
    color="#2E86C1",            # Bar color
    show_values=True,           # Show values next to bars
    value_format="{:.4f}",      # Format for displayed values
    show_plot: bool = False      # Show plot
)

# Further customize the plot using matplotlib objects
ax.set_xlabel("Contribution Score", fontsize=14)
plt.show()
```

## License

MIT License

## Citation

If you find this code useful in your research, please consider citing:

1. For centralities measures:
      ```
      @article{zhao2024centralitymeasuresopiniondynamics,
            title={Centrality measures and opinion dynamics in two-layer networks with replica nodes}, 
            author={Chi Zhao and Elena Parilina},
            year={2024},
            eprint={2406.18780},
            archivePrefix={arXiv},
            primaryClass={physics.soc-ph},
            journal={arXiv preprint arXiv:2406.18780},
            url={https://arxiv.org/abs/2406.18780}, 
      }
      ```
2. For the new method for explanable ai:
      ```
      @article{ZHAO2025110409,
            title = {ShapG: New feature importance method based on the Shapley value},
            journal = {Engineering Applications of Artificial Intelligence},
            volume = {148},
            pages = {110409},
            year = {2025},
            issn = {0952-1976},
            doi = {https://doi.org/10.1016/j.engappai.2025.110409},
            url = {https://www.sciencedirect.com/science/article/pii/S0952197625004099},
            author = {Chi Zhao and Jing Liu and Elena Parilina},
      }
      ```