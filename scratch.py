from model.graph import Graph
from tests.test_materials.metadata import test_graph_file_path

wide_graph: Graph = Graph.load_from_edgelist(file_path=test_graph_file_path, ignore_edges_longer_than=3, keep_at_least_n_edges=2)
print(wide_graph.edge_lengths)
for node in wide_graph.nodes:
    print(len(list(wide_graph.incident_edges(node))))
