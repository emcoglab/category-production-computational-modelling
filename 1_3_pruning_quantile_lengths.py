"""
===========================
Edge lengths by quantile and graph size.
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
from ldm.model.count import LogCoOccurrenceCountModel
from ldm.utils.maths import DistanceType
from ldm.preferences.preferences import Preferences as CorpusPreferences
from model.graph import edge_length_quantile, iter_edges_from_edgelist
from preferences import Preferences

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def main(n_words: int, length_factor: int, corpus_name: str, distance_type_name: str, model_name: str, radius: int):
    logger.info("")

    corpus = get_corpus_from_name(corpus_name)
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    distance_type = DistanceType.from_name(distance_type_name)
    distributional_model: CountVectorModel = get_model_from_params(corpus, freq_dist, model_name, radius)

    graph_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words length {length_factor}.edgelist"
    quantile_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words length {length_factor} edge length quantiles.csv"

    data = []
    # Prune by quantile
    for i, top_quantile in enumerate([.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]):
        pruning_length = edge_length_quantile(
            [length for _edge, length in iter_edges_from_edgelist(path.join(Preferences.graphs_dir, graph_file_name))],
            top_quantile)
        logger.info(f"Edges above the {int(100*top_quantile)}% percentile are those longer than {pruning_length}).")
        data.append((top_quantile, pruning_length))
    # Save the data
    DataFrame(data, columns=["Top quantile", "Pruning length"]).to_csv(
        path.join(Preferences.graphs_dir, quantile_file_name),
        header=True,
        index=False
    )


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
    args = parser.parse_args()

    main(args.n_words, args.length_factor, args.corpus, args.distance_type, args.model, args.radius)
    logger.info("Done!")
