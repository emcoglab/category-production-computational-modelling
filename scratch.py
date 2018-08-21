from numpy import ones, eye, multiply
from numpy.random import random_sample

from model.graph import Graph

graph_size = 3

wide_graph = Graph.from_distance_matrix(
    length_granularity=1,
    # everything 10 away from everything else
    distance_matrix=multiply(10 * (ones((graph_size, graph_size)) - eye(graph_size)), (0.5 + 0.5 * random_sample())),
    ignore_edges_longer_than=5,
    keep_at_least_n_edges=1
)
print(wide_graph.edge_lengths)
for node in wide_graph.nodes:
    print(len(list(wide_graph.incident_edges(node))))
