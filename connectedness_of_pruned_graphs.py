"""
===========================
Check the connectedness of pruned graphs.
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
import json
import logging
import sys
from os import path

from ldm.core.corpus.indexing import FreqDist
from ldm.core.model.count import LogCoOccurrenceCountModel
from ldm.core.utils.maths import DistanceType
from ldm.preferences.preferences import Preferences as CorpusPreferences
from model.graph import Graph
from preferences import Preferences

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"

NODE_COUNTS = [
    1_000,
    3_000,
    5_000,
    10_000,
    15_000,
    20_000,
    25_000,
    30_000,
    35_000,
    40_000,
]
LENGTH_FACTOR = 1_000
TOP_QUANTILES = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]


def main(node_count_i):

    # Load distributional model
    corpus = CorpusPreferences.source_corpus_metas.bbc
    distance_type = DistanceType.cosine
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    distributional_model = LogCoOccurrenceCountModel(corpus, window_radius=5, freq_dist=freq_dist)

    n_words = NODE_COUNTS[node_count_i]

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
    graph_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words length {LENGTH_FACTOR}.edgelist"
    logger.info(f"Loading graph from {graph_file_name}")
    graph = Graph.load_from_edgelist(path.join(Preferences.graphs_dir, graph_file_name))
    logger.info(f"Graph has {len(graph.edges):,} edges")

    # Prune by percentage
    # Invert the quantiles so q of 0.1 gives TOP 10%
    pruning_lengths = [graph.edge_length_quantile(1-q) for q in TOP_QUANTILES]

    for i, pruning_length in enumerate(pruning_lengths):
        top_quantile = TOP_QUANTILES[i]
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
    # Take the given index, else do them all
    if len(sys.argv) > 1:
        main(int(sys.argv[1])-1)
    else:
        for i in range(len(NODE_COUNTS)):
            main(i)
    logger.info("Done!")
