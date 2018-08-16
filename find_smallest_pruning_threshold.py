"""
===========================
Finds the smallest pruning threshold estimate for a graph.
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
from collections import defaultdict
from os import path

import numpy
from matplotlib import pyplot
from scipy.stats import percentileofscore
from seaborn import distplot

from ldm.core.corpus.indexing import FreqDist
from ldm.core.model.count import LogCoOccurrenceCountModel
from ldm.core.utils.maths import DistanceType
from ldm.preferences.preferences import Preferences as CorpusPreferences
from model.graph import edge_data_from_edgelist
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


def main(node_count_i):

    # Load distributional model
    corpus = CorpusPreferences.source_corpus_metas.bbc
    distance_type = DistanceType.cosine
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    distributional_model = LogCoOccurrenceCountModel(corpus, window_radius=5, freq_dist=freq_dist)

    n_words = NODE_COUNTS[node_count_i]

    # Load the full graph
    graph_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words length {LENGTH_FACTOR}.edgelist"

    min_edge_lengths = defaultdict(lambda: numpy.inf)
    lengths = []

    for edge, edge_data in edge_data_from_edgelist(path.join(Preferences.graphs_dir, graph_file_name)):
        n1, n2 = edge.nodes
        length = edge_data.length
        lengths.append(length)
        min_edge_lengths[n1] = min(min_edge_lengths[n1], length)
        min_edge_lengths[n2] = min(min_edge_lengths[n2], length)

    f = pyplot.figure()
    distplot(lengths)
    f.savefig(f"/Users/caiwingfield/Desktop/length_distributions_{n_words}.png")
    pyplot.close(f)

    f = pyplot.figure()
    distplot([length for node, length in min_edge_lengths.items()])
    f.savefig(f"/Users/caiwingfield/Desktop/min_length_distributions_{n_words}.png")
    pyplot.close(f)

    max_min_length = -numpy.inf
    for node, min_length in min_edge_lengths.items():
        max_min_length = max(max_min_length, min_length)

    # Check where this falls in the distribution of edge lengths
    centile = percentileofscore(lengths, max_min_length)

    # Display the max of those
    logger.info(f"Max min-incident-edge length for {n_words:,}-node graph with length factor {LENGTH_FACTOR:,}: {max_min_length}")
    logger.info(f"That's at the {centile:.2f}% percentile among all edge lengths.")


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
