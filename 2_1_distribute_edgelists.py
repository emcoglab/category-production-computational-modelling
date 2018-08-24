"""
===========================
Distribute edgelists.
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
from os import path, mkdir

from numpy import linspace

from ldm.core.corpus.indexing import FreqDist
from ldm.core.model.count import LogCoOccurrenceCountModel
from ldm.core.utils.maths import DistanceType
from ldm.preferences.preferences import Preferences as CorpusPreferences
from model.graph import edge_length_quantile, iter_edges_from_edgelist
from preferences import Preferences

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def main(n_words: int):
    logger.info("")

    length_factor = 1_000

    corpus = CorpusPreferences.source_corpus_metas.bbc
    distance_type = DistanceType.cosine
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    distributional_model = LogCoOccurrenceCountModel(corpus, window_radius=5, freq_dist=freq_dist)

    graph_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words length {length_factor}.edgelist"

    for i, (edge, length) in enumerate(iter_edges_from_edgelist(path.join(Preferences.graphs_dir, graph_file_name))):
        # Write the edge n1→n2 into the edgelist for each of n1 and n2
        n1, n2 = edge.nodes
        for n in [n1, n2]:
            node_dist_filename = f"n{n}_edge_lengths.edgelist"
            node_dist_dir = path.join(Preferences.node_distributions_dir, f"{distributional_model.name} {distance_type.name} {n_words} words length {length_factor}")
            if not path.isdir(node_dist_dir):
                mkdir(node_dist_dir)
            with open(path.join(node_dist_dir, node_dist_filename), mode="a", encoding="utf-8") as node_file:
                node_file.write(f"{n1} {n2} {length}\n")

        # Occasional logging
        if i % 1_000 == 0:
            logger.info(f"Done {i:,} edges.")


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Run temporal spreading activation on a graph.")
    parser.add_argument("n_words", type=int, help="The number of words to use from the corpus. (Top n words.)",
                        nargs='?', default='1000')
    args = parser.parse_args()

    main(n_words=args.n_words)
    logger.info("Done!")
