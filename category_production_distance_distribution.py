"""
===========================
Distribution of pairwise distances in category production data.
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

import sys
from os import path

import numpy
from matplotlib import pyplot
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances

from category_production.category_production import CategoryProduction
from ldm.corpus.indexing import FreqDist, TokenIndex
from ldm.model.count import LogCoOccurrenceCountModel
from ldm.utils.maths import DistanceType
from ldm.preferences.preferences import Preferences as CorpusPreferences

from preferences import Preferences
from model.utils.logging import logger


def main():
    corpus = CorpusPreferences.source_corpus_metas.bbc
    window_radius = 5
    distance_type = DistanceType.cosine
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    token_index = TokenIndex.from_freqdist_ranks(freq_dist)
    model_vocab = token_index.tokens
    model = LogCoOccurrenceCountModel(corpus, window_radius, freq_dist)
    model.train(memory_map=True)

    category_production_vocabulary = CategoryProduction().vocabulary_single_word
    category_production_ldm_ids = sorted([token_index.token2id[w]
                                          for w in category_production_vocabulary
                                          if w in model_vocab])

    # Compute pairwise distances between vocabulary words
    d = pairwise_distances(model.matrix.tocsr()[category_production_ldm_ids], metric=distance_type.name, n_jobs=-1)
    numpy.fill_diagonal(d, 0)
    # UTV
    d = squareform(d)

    # Aggregate distances in histogram

    # Define histogram parameters
    data_min = 0  # min possible distance
    data_max = 1  # max possible distance
    n_bins = 250  # number of bins

    bins = numpy.linspace(data_min, data_max, n_bins)

    h, _ = numpy.histogram(d, bins)

    # Save histogram
    bar_width = 1 * (bins[1] - bins[0])
    bar_centres = (bins[:-1] + bins[1:]) / 2
    f, a = pyplot.subplots()
    a.bar(bar_centres, h, align="center", width=bar_width)
    fig_name = f"distance distribution for category production words [{model.name} {distance_type.name}].png"
    f.savefig(path.join(Preferences.output_dir, fig_name))


if __name__ == '__main__':
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
