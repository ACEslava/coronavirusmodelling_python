import networkx as nx
import matplotlib.pyplot as plt
from network2tikz import plot

while True:
    graph = nx.generators.random_graphs.fast_gnp_random_graph(5,0.5)
    labels = {0:1, 1:2, 2:3, 3:4, 4:5}
    graph = nx.relabel_nodes(graph, labels)
    plt.plot()
    nx.draw(graph, with_labels=True, font_weight='bold')
    plt.show()
    choice = int(input())
    if choice == 1:
        plot(graph, "graph.tex")
    else:
        continue