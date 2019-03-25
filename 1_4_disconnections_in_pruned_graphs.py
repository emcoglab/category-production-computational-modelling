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

from pandas import DataFrame

from cli.lookups import get_corpus_from_name, get_model_from_params
from ldm.corpus.indexing import FreqDist
from ldm.model.count import CountVectorModel
from ldm.utils.maths import DistanceType
from model.graph import Graph
from preferences import Preferences

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def main(n_words: int, prune_top_percentile: int, length_factor: int, corpus_name: str, distance_type_name: str, model_name: str, radius: int):

    corpus = get_corpus_from_name(corpus_name)
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    distance_type = DistanceType.from_name(distance_type_name)
    distributional_model: CountVectorModel = get_model_from_params(corpus, freq_dist, model_name, radius)

    graph_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words length {length_factor}.edgelist"
    quantile_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words length {length_factor} edge length quantiles.csv"
    quantile_data = DataFrame.from_csv(path.join(Preferences.graphs_dir, quantile_file_name), header=0, index_col=None)
    if prune_top_percentile is not None:
        pruning_length = quantile_data[
            # Use 1 - so that smallest top quantiles get converted to longest edges
            quantile_data["Top quantile"] == 1 - (prune_top_percentile / 100)
        ]["Pruning length"].iloc[0]
    else:
        pruning_length = None

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
    parser.add_argument("n_words", type=int, help="The number of words to use from the corpus. (Top n words.)")
    parser.add_argument("length_factor", type=int, help="The length factor.")
    parser.add_argument("corpus", type=str, help="The corpus.")
    parser.add_argument("distance_type", type=str, help="The distance type.")
    parser.add_argument("model", type=str, help="The model.")
    parser.add_argument("radius", type=int, help="The radius.")
    parser.add_argument("prune_top_percentile", type=int, help="Prune this percent of edges, starting with the top",
                        nargs="?", default=None)
    args = parser.parse_args()

    main(args.n_words, args.prune_top_percentile, args.length_factor, args.corpus, args.distance_type, args.model, args.radius)
    logger.info("Done!")
