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
from collections import defaultdict, Counter
from os import path

from matplotlib import pyplot, ticker
from numpy import inf
from seaborn import distplot, barplot

from ldm.corpus.indexing import FreqDist
from ldm.model.base import DistributionalSemanticModel
from ldm.model.count import LogCoOccurrenceCountModel
from ldm.preferences.preferences import Preferences as CorpusPreferences
from ldm.utils.maths import DistanceType
from model.graph import iter_edges_from_edgelist
from preferences import Preferences

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def main_linguistic(n_words: int, length_factor: int):

    corpus = CorpusPreferences.source_corpus_metas.bbc
    distance_type = DistanceType.cosine
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    distributional_model = LogCoOccurrenceCountModel(corpus, window_radius=5, freq_dist=freq_dist)

    if distributional_model.model_type.metatype is DistributionalSemanticModel.MetaType.count:
        graph_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words length {length_factor}.edgelist"
    elif distributional_model.model_type.metatype is DistributionalSemanticModel.MetaType.ngram:
        graph_file_name = f"{distributional_model.name} {n_words} words length {length_factor}.edgelist"
    else:
        raise NotImplementedError()

    # We want to over-prune isolated nodes and under-prune highly accessible nodes, so that we end up pruning approx the
    # target fraction of edges.

    edge_lengths_from_node = defaultdict(list)
    min_edge_length = defaultdict(lambda: inf)
    n_edges_considered = 0
    for edge, length in iter_edges_from_edgelist(path.join(Preferences.graphs_dir, graph_file_name)):
        n1, n2 = edge
        min_edge_length[n1] = min(min_edge_length[n1], length)
        min_edge_length[n2] = min(min_edge_length[n2], length)
        edge_lengths_from_node[n1].append(length)
        edge_lengths_from_node[n2].append(length)

        n_edges_considered += 1
        if n_edges_considered % 1_000_000 == 0:
            logger.info(f"Considered {n_edges_considered:,} edges...")

    f = pyplot.figure()
    distplot([length for _node, lengths in edge_lengths_from_node.items() for length in lengths])
    f.savefig(path.join(Preferences.figures_dir,
                        "length distributions",
                        f"length_distributions_[{distributional_model.name}]_length_{length_factor}_{n_words} words.png"))
    pyplot.close(f)

    f = pyplot.figure()
    distplot([length for _node, length in min_edge_length.items()])
    f.savefig(path.join(Preferences.figures_dir,
                        "length distributions",
                        f"min_length_distributions_[{distributional_model.name}]_length_{length_factor}_{n_words} words.png"))
    pyplot.close(f)


def main_sensorimotor(length_factor: int, distance_type_name: str):

    distance_type = DistanceType.from_name(distance_type_name)

    edgelist_filename = f"sensorimotor {distance_type.name} distance length {length_factor}.edgelist"

    edge_lengths = Counter()
    min_edge_length = defaultdict(lambda: inf)
    n_edges_considered = 0
    for edge, length in iter_edges_from_edgelist(path.join(Preferences.graphs_dir, edgelist_filename)):
        n1, n2 = edge
        min_edge_length[n1] = min(min_edge_length[n1], length)
        min_edge_length[n2] = min(min_edge_length[n2], length)
        edge_lengths[length] += 1

        n_edges_considered += 1
        if n_edges_considered % 1_000_000 == 0:
            logger.info(f"Considered {n_edges_considered:,} edges...")

    # add zero intermediate counts to edge lengths
    edge_lengths.update({
        l: 0
        for l in range(min(edge_lengths.keys()), max(edge_lengths.keys()))
    })

    f = pyplot.figure()
    edge_lengths_sorted = sorted([(length, count) for length, count in edge_lengths.items()], key=lambda tup: tup[0])
    barplot(x=[length for length, count in edge_lengths_sorted],
            y=[count for length, count in edge_lengths_sorted])
    ax = pyplot.gca()
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=50))
    f.savefig(path.join(Preferences.figures_dir,
                        "length distributions",
                        f"length_distributions_sensorimotor_length_{length_factor}_{distance_type.name} words.png"))
    pyplot.close(f)

    f = pyplot.figure()
    distplot([length for node, length in min_edge_length.items()])
    f.savefig(path.join(Preferences.figures_dir,
                        "length distributions",
                        f"min_length_distributions_sensorimotor_length_{length_factor}_{distance_type.name} words.png"))
    pyplot.close(f)

    logger.info(f"Edge lengths fall in range ["
                f"{min(edge_lengths.keys())}, "
                f"{max(edge_lengths.keys())}]")
    logger.info(f"Node-minimum edge lengths fall in range ["
                f"{min([length for node, length in min_edge_length.items()])}, "
                f"{max([length for node, length in min_edge_length.items()])}]")


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Run temporal spreading activation on a graph.")

    mode_subparser = parser.add_subparsers(dest="mode")
    mode_subparser.required = True
    mode_sensorimotor_parser = mode_subparser.add_parser("sensorimotor")
    mode_linguistic_parser = mode_subparser.add_parser("linguistic")

    mode_linguistic_parser.add_argument("-w", "--words", type=int, required=True,
                                        help="The number of words to use from the corpus. (Top n words.)")
    mode_sensorimotor_parser.add_argument("-d", "--distance_type", required=True, type=str)
    for mp in [mode_sensorimotor_parser, mode_linguistic_parser]:
        mp.add_argument("-l", "--length_factor", required=True, type=int)

    args = parser.parse_args()

    if args.mode == "sensorimotor":
        main_sensorimotor(length_factor=args.length_factor, distance_type_name=args.distance_type)
    elif args.mode == "linguistic":
        main_linguistic(n_words=args.words, length_factor=args.length_factor)
    else:
        raise NotImplementedError()
    logger.info("Done!")
