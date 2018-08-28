"""
===========================
Look for disconnections in pruned graphs.
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


def main(n_words: int):

    length_factor = 1_000

    corpus = CorpusPreferences.source_corpus_metas.bbc
    distance_type = DistanceType.cosine
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    distributional_model = LogCoOccurrenceCountModel(corpus, window_radius=5, freq_dist=freq_dist)

    graph_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words length {length_factor}.edgelist"

    # Load the full graph
    keep_at_least = 10
    logger.info(f"Loading graph from {graph_file_name}, pruning edges longer than {pruning_length}, but keeping at least {keep_at_least} edges per node")
    graph = Graph.load_from_edgelist(path.join(Preferences.graphs_dir, graph_file_name),
                                     ignore_edges_longer_than=pruning_length,
                                     keep_at_least_n_edges=keep_at_least)
    logger.info(f"Graph has {len(graph.edges):,} edges")

    if graph.has_orphaned_nodes():
        logger.info("Graph has orphaned nodes.")
    else:
        logger.info("Graph does not have orphaned nodes")
    if graph.is_connected():
        logger.info("Graph is connected")
    else:
        logger.info("Graph is not connected")
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
