from numpy import ones, eye

from model.graph import Graph

graph_size = 20
wide_graph = Graph.from_distance_matrix(
    length_granularity=1,
    # everything 10 away from everything else
    distance_matrix=10 * (ones((graph_size, graph_size)) - eye(graph_size)),
    ignore_edges_longer_than=5,
    keep_at_least_n_edges=3
)
for node in wide_graph.nodes:
    print(len(list(wide_graph.incident_edges(node))))
