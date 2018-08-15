"""
===========================
For working with graphs.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2018
---------------------------
"""

import logging
import os
from collections import namedtuple, defaultdict
from typing import Dict, Set, Tuple, Iterator, DefaultDict

from numpy.core.multiarray import ndarray
from numpy.core.umath import ceil

logger = logging.getLogger()

Node = int
EdgeData = namedtuple('EdgeData', ['length'])


class Edge(frozenset):
    def __init__(self, seq=()):
        assert len(seq) == 2
        frozenset.__init__(seq)

    @property
    def nodes(self) -> Tuple[Node, Node]:
        return tuple(self)


class GraphError(Exception):
    pass


class Graph:
    """
    This is a fragile class that needs to be made more robust.
    Right now Graph.nodes and Graph.edges are Dicts that can be modified at will.
    They should be
    """

    # TODO: Make it more robust by protecting dictionaries from editing outside of the add_* methods.

    def __init__(self, nodes: Set[Node] = None, edges: Dict[Edge, EdgeData] = None):
        self.nodes: Set[Node] = set()
        self.edge_data: Dict[Edge, EdgeData] = dict()
        # Node-keyed dict of sets of incident edges
        self._incident_edges: DefaultDict[Node, Set[Edge]] = defaultdict(set)

        if nodes is not None:
            for node in nodes:
                self.add_node(node)
        if edges is not None:
            for edge, edge_data in edges.items():
                self.add_edge(edge, edge_data)

    @property
    def edges(self):
        return self.edge_data.keys()

    def add_edge(self, edge: Edge, edge_data: EdgeData = None):
        # Check if edge already added
        if edge in self.edges:
            raise GraphError(f"Edge {edge} already exists!")
        # Add endpoint nodes
        for node in edge:
            if node not in self.nodes:
                self.add_node(node)
        # Add edge
        self.edge_data[edge] = edge_data
        # Add incident edges information
        nodes = list(edge)
        self._incident_edges[nodes[0]].add(edge)
        self._incident_edges[nodes[1]].add(edge)

    def add_node(self, node: Node):
        if node not in self.nodes:
            self.nodes.add(node)

    def incident_edges(self, node: Node) -> Iterator[Edge]:
        """The edges which have `node` as an endpoint."""
        for edge in self._incident_edges[node]:
            yield edge

    def neighbourhood(self, node: Node) -> Iterator[Node]:
        """The nodes which are connected to `node` by exactly one edge."""
        assert node in self._incident_edges.keys()
        for edge in self._incident_edges[node]:
            for n in edge:
                # Skip the source node
                if n == node:
                    continue
                yield n

    # region IO

    def save_as_edgelist(self, file_path: str):
        """Saves a Graph as an edgelist. Disconnected nodes will not be included."""
        with open(file_path, mode="w", encoding="utf-8") as edgelist_file:
            for edge, edge_data in self.edge_data.items():
                n1, n2 = sorted(edge)
                length = int(edge_data.length)
                edgelist_file.write(f"{Node(n1)} {Node(n2)} {length}\n")

    @classmethod
    def load_from_edgelist(cls, file_path: str) -> 'Graph':
        """Loads a Graph from an edgelist file."""
        graph = cls()
        with open(file_path, mode="r", encoding="utf-8") as edgelist_file:
            for line in edgelist_file:
                n1, n2, length = line.split()
                graph.add_edge(Edge((Node(n1), Node(n2))), EdgeData(length=int(length)))
        return graph

    @classmethod
    def from_distance_matrix(cls,
                             distance_matrix: ndarray,
                             length_granularity: int,
                             prune_connections_longer_than: int = None) -> 'Graph':
        """
        Produces a Graph of the correct format to underlie a TemporalSpreadingActivation.

        Nodes will be numbered according to the row/column indices of weight_matrix (and so can
        be relabelled accordingly).

        Distances will be converted to weights using x ↦ 1-x.

        Distances will be converted to integer lengths using the supplied scaling factor.

        :param distance_matrix:
        A symmetric distance matrix in numpy format.
        :param length_granularity:
        Distances will be scaled into integer connection lengths using this granularity scaling factor.
        Whether to use weights on the edges.
        If True, distances will be converted to weights using x ↦ 1-x.
            (This means it's only suitable for things like cosine and correlation distances, not Euclidean.)
        If False, all edges get the same weight.
        :param prune_connections_longer_than:
        (Optional.) If provided and not None: Any connections with lengths (strictly) longer than this will be severed.
        :return:
        A Graph of the correct format.
        """

        graph = cls()

        n_nodes = distance_matrix.shape[0]

        for n1 in range(0, n_nodes):
            for n2 in range(n1 + 1, n_nodes):
                distance = distance_matrix[n1, n2]
                length = int(ceil(distance * length_granularity))
                # Skip the edge if we're pruning and it's too long
                if (prune_connections_longer_than is not None) and (length > prune_connections_longer_than):
                    continue
                # Add the edge
                graph.add_edge(Edge((n1, n2)), EdgeData(length=length))

        return graph

    # endregion IO

    # region conversion

    def as_networkx_graph(self):
        """Converts the Graph into a NetworkX Graph."""
        import networkx
        g = networkx.Graph()
        for edge, edge_data in self.edge_data.items():
            g.add_edge(*edge.nodes, length=edge_data.length)
        return g

    # endregion conversion


def save_edgelist_from_distance_matrix(file_path: str,
                                       distance_matrix: ndarray,
                                       length_granularity: int,
                                       prune_connections_longer_than: int = None):
    """
    Saves a graph of the correct form to underlie a TemporalSpreadingActivation.
    Saved as a networkx-compatible edgelist format.

    This can be loaded using `load_graph`.

    It is often faster (and more memory efficient) to save this way than building the graph and then saving it.

    :param file_path:
    :param distance_matrix:
    :param length_granularity:
    :param prune_connections_longer_than:
    :return:
    """

    temp_file_path = file_path + ".incomplete"

    # Log progress every time we reach a percentage milestone
    # Record here the most recently logged milestone
    logged_percent_milestone = 0

    with open(temp_file_path, mode="w", encoding="utf8") as temp_file:
        for i in range(0, distance_matrix.shape[0]):
            # Log progress
            percent_done = int(ceil(100 * i / distance_matrix.shape[0]))
            if (percent_done % 10 == 0) and (percent_done > logged_percent_milestone):
                logger.info(f"\t{percent_done}% done")
                logged_percent_milestone = percent_done
            for j in range(i + 1, distance_matrix.shape[1]):
                distance = distance_matrix[i, j]
                length = int(ceil(distance * length_granularity))
                if (prune_connections_longer_than is not None) and (length > prune_connections_longer_than):
                    continue
                # Write edge to file
                temp_file.write(f"{i} {j} {length}\n")
    os.rename(temp_file_path, file_path)
