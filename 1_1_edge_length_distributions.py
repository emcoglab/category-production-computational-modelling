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
import logging
import sys
from collections import defaultdict
from os import path

from matplotlib import pyplot
from numpy import inf
from seaborn import distplot

from ldm.core.corpus.indexing import FreqDist
from ldm.core.model.count import LogCoOccurrenceCountModel
from ldm.core.utils.maths import DistanceType
from ldm.preferences.preferences import Preferences as CorpusPreferences
from model.graph import iter_edges_from_edgelist
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

    # We want to over-prune isolated nodes and under-prune highly accessible nodes, so that we end up pruning approx the
    # target fraction of edges.  We classify a node's "accessibility" as its minimum edge length.

    edge_lengths_from_node = defaultdict(list)
    accessibility = defaultdict(lambda: inf)
    for edge, length in iter_edges_from_edgelist(path.join(Preferences.graphs_dir, graph_file_name)):
        n1, n2 = edge.nodes
        accessibility[n1] = min(accessibility[n1], length)
        accessibility[n2] = min(accessibility[n2], length)
        edge_lengths_from_node[n1].append(length)
        edge_lengths_from_node[n2].append(length)

    f = pyplot.figure()
    distplot([length for edge, lengths in edge_lengths_from_node.items() for length in lengths])
    f.savefig(path.join(Preferences.figures_dir, "length distributions", f"length_distributions_{n_words}.png"))
    pyplot.close(f)

    f = pyplot.figure()
    distplot([length for node, length in accessibility.items()])
    f.savefig(path.join(Preferences.figures_dir, "length distributions", f"min_length_distributions_{n_words}.png"))
    pyplot.close(f)


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Run temporal spreading activation on a graph.")
    parser.add_argument("n_words", type=int, help="The number of words to use from the corpus. (Top n words.)",
                        nargs='?', default='1000')
    args = parser.parse_args()

    main(n_words=args.n_words)
    logger.info("Done!")
