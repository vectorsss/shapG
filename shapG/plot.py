import matplotlib.pyplot as plt
import numpy as np

def plot(shapley_values, top_n=10, file_name=None):
    """plot the shapley values

    Args:
        shapley_values (dict): dictionary of shapley values
        top_n (int, optional): if dictionary is too long, only plot top-n. Defaults to 10.
    """
    # sort in DESC order
    sorted_values = sorted(shapley_values.items(), key=lambda item: item[1], reverse=True)
    
    # select top-n
    if len(sorted_values) > top_n:
        sorted_values = sorted_values[:top_n]
    
    nodes, values = zip(*sorted_values)
    
    # set gnuplot style
    plt.style.use('seaborn')
    
    # plot the bar chart
    fig, ax = plt.subplots()
    y_pos = np.arange(len(nodes))
    
    ax.barh(y_pos, values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(nodes)
    ax.invert_yaxis()  # 从上到下排列
    ax.set_xlabel('Shapley Value')
    ax.set_title('Top Shapley Values')
    
    # add the label
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f'{v:.2f}', va='center')
    if file_name is not None:
        plt.savefig(file_name, format='eps', dpi=300)
    plt.show()
    
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
    plot(shapley_values)