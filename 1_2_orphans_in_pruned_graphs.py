"""
===========================
Compute pruning thresholds for graphs of different sizes.
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
import json
import logging
import sys
from os import path

from numpy import linspace

from ldm.core.corpus.indexing import FreqDist
from ldm.core.model.count import LogCoOccurrenceCountModel
from ldm.core.utils.maths import DistanceType
from ldm.preferences.preferences import Preferences as CorpusPreferences
from model.graph import Graph
from preferences import Preferences

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


# Results DataFrame column names
RESPONSE = "Response"
NODE_ID = "Node ID"
ACTIVATION = "Activation"
TICK_ON_WHICH_ACTIVATED = "Tick on which activated"


def main(n_words: int):

    length_factor = 1_000

    corpus = CorpusPreferences.source_corpus_metas.bbc
    distance_type = DistanceType.cosine
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    distributional_model = LogCoOccurrenceCountModel(corpus, window_radius=5, freq_dist=freq_dist)

    graph_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words length {length_factor}.edgelist"

    # Load node relabelling dictionary
    logger.info(f"Loading node labels")
    # TODO: this is duplicated code and can be refactored out in to a library function
    # TODO: in fact, it SHOULD be
    with open(path.join(Preferences.graphs_dir, f"{corpus.name} {n_words} words.nodelabels"), mode="r",
              encoding="utf-8") as nrd_file:
        node_relabelling_dictionary_json = json.load(nrd_file)
    node_relabelling_dictionary = dict()
    for k, v in node_relabelling_dictionary_json.items():
        node_relabelling_dictionary[int(k)] = v

    # Load the full graph
    logger.info(f"Loading graph from {graph_file_name}")
    graph = Graph.load_from_edgelist(path.join(Preferences.graphs_dir, graph_file_name))
    logger.info(f"Graph has {len(graph.edges):,} edges")

    # Prune by quantile
    for i, top_quantile in enumerate(linspace(0.0, 1.0, 11)):
        # Invert the quantiles so q of 0.1 gives TOP 10%
        pruning_length = graph.edge_length_quantile(1-top_quantile)
        logger.info(f"Pruning longest {int(100*top_quantile)}% of edges (anything longer than {pruning_length}).")
        graph.prune_longest_edges_by_quantile(top_quantile)
        logger.info(f"Graph has {len(graph.edges):,} edges")
        if graph.has_orphaned_nodes():
            orphaned_nodes = graph.orphaned_nodes()
            orphaned_node_labels = sorted(list(node_relabelling_dictionary[node] for node in orphaned_nodes))
            logger.info(f"Graph has {len(orphaned_nodes)} orphaned nodes "
                        f"({len(orphaned_nodes)/len(graph.nodes):.2}% of total): "
                        f"{', '.join(orphaned_node_labels)}")
        else:
            logger.info("Graph has no orphaned nodes")
        logger.info("")
    logger.info("")


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Run temporal spreading activation on a graph.")
    parser.add_argument("n_words", type=int, help="The number of words to use from the corpus. (Top n words.)",
                        nargs='?', default='1000')
    args = parser.parse_args()

    main(n_words=args.n_words)
    logger.info("Done!")
