#!/Users/cai/Applications/miniconda3/bin/python
"""
===========================
Look for orphaned nodes in pruned graphs.
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
import argparse
import sys
from os import path

from numpy import linspace

from framework.cli.lookups import get_corpus_from_name, get_model_from_params
from framework.cognitive_model.ldm.corpus.indexing import FreqDist
from framework.cognitive_model.ldm.model.count import CountVectorModel
from framework.cognitive_model.ldm.utils.maths import DistanceType
from framework.cognitive_model.linguistic_propagator import _load_labels_from_corpus
from framework.cognitive_model.utils.logging import logger
from framework.cognitive_model.graph import Graph
from framework.cognitive_model.utils.maths import nearest_value_at_quantile
from framework.cognitive_model.preferences import Preferences


def main(n_words: int, length_factor: int, corpus_name: str, distance_type_name: str, model_name: str, radius: int):

    corpus = get_corpus_from_name(corpus_name)
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    distance_type = DistanceType.from_name(distance_type_name)
    distributional_model: CountVectorModel = get_model_from_params(corpus, freq_dist, model_name, radius)

    graph_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words length {length_factor}.edgelist"

    # Load node relabelling dictionary
    logger.info(f"Loading node labels")
    node_labelling_dictionary = _load_labels_from_corpus(corpus, n_words)

    # Load the full graph
    logger.info(f"Loading graph from {graph_file_name}")
    graph = Graph.load_from_edgelist(path.join(Preferences.graphs_dir, graph_file_name))
    logger.info(f"Graph has {len(graph.edges):,} edges")

    # Prune by length quantile
    logger.info("Length quantile pruning\n")

    for i, top_quantile in enumerate(linspace(0.0, 1.0, 11)):
        # Invert the quantiles so q of 0.1 gives TOP 10%
        pruning_length = nearest_value_at_quantile([length for edge, length in graph.edge_lengths], 1 - top_quantile)
        logger.info(f"Pruning longest {int(100*top_quantile)}% of edges (anything longer than {pruning_length}).")
        graph.prune_longest_edges_by_quantile(top_quantile)
        logger.info(f"Graph has {len(graph.edges):,} edges")
        logger.info(f"Graph has {len(graph.nodes):,} nodes")
        if graph.has_orphaned_nodes():
            orphaned_nodes = graph.orphaned_nodes()
            orphaned_node_labels = sorted(list(node_labelling_dictionary[node] for node in orphaned_nodes))
            logger.info(f"Graph has {len(orphaned_nodes)} orphaned nodes "
                        f"({len(orphaned_nodes)/len(graph.nodes):.2}% of total): "
                        f"{', '.join(orphaned_node_labels)}")
        else:
            logger.info("Graph has no orphaned nodes")
        logger.info("")
    logger.info("")

    # Prune by importance
    logger.info("Importance pruning\n")

    for importance in [0, 19, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        logger.info(f"Pruning at importance level {importance}")
        importance_pruned_graph = Graph.load_from_edgelist_with_importance_pruning(path.join(Preferences.graphs_dir, graph_file_name),
                                                                                   ignore_edges_with_importance_greater_than=importance,
                                                                                   keep_at_least_n_edges=0)
        logger.info(f"Graph has {len(importance_pruned_graph.edges):,} edges")
        logger.info(f"Graph has {len(importance_pruned_graph.nodes):,} nodes")
        if importance_pruned_graph.has_orphaned_nodes():
            orphaned_nodes = importance_pruned_graph.orphaned_nodes()
            orphaned_node_labels = sorted(list(node_labelling_dictionary[node] for node in orphaned_nodes))
            logger.info(f"Graph has {len(orphaned_nodes)} orphaned nodes "
                        f"({len(orphaned_nodes)/len(graph.nodes):.2}% of total): "
                        f"{', '.join(orphaned_node_labels)}")
        else:
            logger.info("Graph has no orphaned nodes")
        logger.info("")
    logger.info("")


if __name__ == '__main__':
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Run temporal spreading activation on a graph.")
    parser.add_argument("-l", "--length_factor", required=True, type=int)
    parser.add_argument("-c", "--corpus_name", required=True, type=str)
    parser.add_argument("-d", "--distance_type", required=True, type=str)
    parser.add_argument("-m", "--model_name", required=True, type=str)
    parser.add_argument("-r", "--radius", required=True, type=int)
    parser.add_argument("-w", "--words", type=int, required=True, help="The number of words to use from the corpus. (Top n words.)")

    args = parser.parse_args()

    main(length_factor=args.length_factor,
         corpus_name=args.corpus_name,
         model_name=args.model_name,
         radius=args.radius,
         distance_type_name=args.distance_type,
         n_words=args.words)

    logger.info("Done!")
