from numpy import array

from model.graph import Graph

orphan_graph = Graph.from_adjacency_matrix(
            adjacency_matrix=array([
                [0, 1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ])
        )
print(orphan_graph.has_orphaned_nodes())
