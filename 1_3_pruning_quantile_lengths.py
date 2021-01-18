#!/Users/cai/Applications/miniconda3/bin/python
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
import sys
from os import path

from pandas import DataFrame

from framework.cli.lookups import get_corpus_from_name, get_model_from_params
from framework.cognitive_model.ldm.corpus.indexing import FreqDist
from framework.cognitive_model.ldm.utils.maths import DistanceType
from framework.cognitive_model.utils.logging import logger
from framework.cognitive_model.graph import iter_edges_from_edgelist
from framework.cognitive_model.utils.maths import nearest_value_at_quantile
from framework.cognitive_model.preferences.preferences import Preferences


def main(n_words: int, length_factor: int, corpus_name: str, distance_type_name: str, model_name: str, radius: int):
    logger.info("")

    corpus = get_corpus_from_name(corpus_name)
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    distance_type = DistanceType.from_name(distance_type_name)
    distributional_model = get_model_from_params(corpus, freq_dist, model_name, radius)

    graph_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words length {length_factor}.edgelist"
    quantile_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words length {length_factor} edge length quantiles.csv"

    data = []
    # Prune by quantile
    for i, top_quantile in enumerate([.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]):
        pruning_length = nearest_value_at_quantile(
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

    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Run temporal spreading activation on a graph.")

    parser.add_argument("-l", "--length_factor", required=True, type=int)
    parser.add_argument("-c", "--corpus_name", required=True, type=str)
    parser.add_argument("-d", "--distance_type", required=True, type=str)
    parser.add_argument("-m", "--model_name", required=True, type=str)
    parser.add_argument("-r", "--radius", required=True, type=int)
    parser.add_argument("-w", "--words", type=int, required=True,
                        help="The number of words to use from the corpus. (Top n words.)")

    args = parser.parse_args()

    main(length_factor=args.length_factor,
         corpus_name=args.corpus_name,
         model_name=args.model_name,
         radius=args.radius,
         distance_type_name=args.distance_type,
         n_words=args.words)

    logger.info("Done!")
