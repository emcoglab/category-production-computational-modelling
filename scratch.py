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
import json
import logging
import sys
from os import path

from ldm.corpus.indexing import FreqDist
from ldm.model.ngram import PPMINgramModel
from ldm.preferences.preferences import Preferences as CorpusPreferences
from ldm.utils.maths import DistanceType
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
    distributional_model = PPMINgramModel(corpus, window_radius=5, freq_dist=freq_dist)

    graph_file_name = f"{distributional_model.name} {n_words} words length {length_factor}.edgelist"

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
    logger.info(f"Graph has {len(graph.nodes):,} nodes")


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Run temporal spreading activation on a graph.")
    parser.add_argument("n_words", type=int, help="The number of words to use from the corpus. (Top n words.)",
                        nargs='?', default='40000')
    args = parser.parse_args()

    main(n_words=args.n_words)
    logger.info("Done!")
