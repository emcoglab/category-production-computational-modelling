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
import logging
import sys
from os import path

from matplotlib import pyplot
from seaborn import distplot

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
    # 3_000,
    # 5_000,
    # 10_000,
    # 15_000,
    # 20_000,
    # 25_000,
    # 30_000,
    # 35_000,
    # 40_000,
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


def main():

    # Load distributional model
    corpus = CorpusPreferences.source_corpus_metas.bbc
    distance_type = DistanceType.cosine
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    distributional_model = LogCoOccurrenceCountModel(corpus, window_radius=5, freq_dist=freq_dist)

    for n_words in NODE_COUNTS:

        # Load the full graph
        graph_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words length {LENGTH_FACTOR}.edgelist"
        logger.info(f"Loading graph from {graph_file_name}")
        graph = Graph.load_from_edgelist(path.join(Preferences.graphs_dir, graph_file_name))
        logger.info(f"Graph has {len(graph.edges):,} edges")

        f = pyplot.figure()
        distplot([graph.edge_data[edge].length for edge in graph.edges])
        f.savefig("/Users/caiwingfield/Desktop/fig.png")

        # Prune by percentage
        # Invert the quantiles so q of 0.1 gives TOP 10%
        pruning_lengths = [graph.edge_length_quantile(1-q) for q in TOP_QUANTILES]

        for i, pruning_length in enumerate(pruning_lengths):
            top_quantile = TOP_QUANTILES[i]
            logger.info(f"Pruning longest {int(100*top_quantile)}% of edges (anything longer than {pruning_length}).")
            graph.prune_longest_edges_by_quantile(top_quantile)
            logger.info(f"Graph has {len(graph.edges):,} edges")
            if graph.has_orphaned_nodes():
                logger.info("Graph has orphaned nodes")
            else:
                logger.info("Graph has no orphaned nodes")
            if graph.is_connected():
                logger.info("Graph is connected")
            else:
                logger.info("Graph is disconnected")
                break
            logger.info("")
        logger.info("")


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
