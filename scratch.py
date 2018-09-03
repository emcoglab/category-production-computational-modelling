from model.graph import Graph

importance_pruned_graph = Graph.load_from_edgelist_with_importance_pruning(
    file_path="/Users/caiwingfield/code/spreading_activation/tests/test_materials/test_graph_importance.edgelist",
    ignore_edges_with_importance_greater_than=50)
n_edges = len(importance_pruned_graph.edges)

print("Done")
