import networkx as nx
from network2tikz import plot

SimGraph = nx.complete_graph(14)
plot(SimGraph, 'test.tex')