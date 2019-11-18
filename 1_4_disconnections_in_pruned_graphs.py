#!/Users/cai/Applications/miniconda3/bin/python
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
from model.graph import Graph, log_graph_topology
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
    log_graph_topology(graph)


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Run temporal spreading activation on a graph.")

    parser.add_argument("-l", "--length_factor", required=True, type=int)
    parser.add_argument("-c", "--corpus_name", required=True, type=str)
    parser.add_argument("-d", "--distance_type", required=True, type=str)
    parser.add_argument("-m", "--model_name", required=True, type=str)
    parser.add_argument("-r", "--radius", required=True, type=int)
    parser.add_argument("-w", "--words", type=int, required=True, help="The number of words to use from the corpus. (Top n words.)")
    parser.add_argument("-p", "--prune_percent", required=False, type=int, help="The percentage of longest edges to prune from the graph.", default=None)

    args = parser.parse_args()

    main(length_factor=args.length_factor,
         corpus_name=args.corpus_name,
         model_name=args.model_name,
         radius=args.radius,
         distance_type_name=args.distance_type,
         n_words=args.words,
         prune_top_percentile=args.prune_percent)
    args = parser.parse_args()

    logger.info("Done!")
