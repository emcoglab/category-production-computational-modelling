"""
===========================
Runs the example of spreading activation on a language model.
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
from itertools import combinations

from corpus_analysis.core.corpus.indexing import TokenIndexDictionary, FreqDist
from corpus_analysis.core.model.count import LogCoOccurrenceCountModel
from corpus_analysis.core.utils.maths import DistanceType
from corpus_analysis.preferences.preferences import Preferences as CorpusPreferences

from spreading_activation import SpreadingActivation

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def main():

    top_n = 300

    corpus_meta = CorpusPreferences.source_corpus_metas[0]
    freq_dist = FreqDist.load(corpus_meta.freq_dist_path)
    distributional_model = LogCoOccurrenceCountModel(
        corpus_meta,
        window_radius=1,
        token_indices=TokenIndexDictionary.from_freqdist(freq_dist))

    distributional_model.train(memory_map=True)

    word_list = set([word for word, _ in freq_dist.most_common(top_n)])

    # Initialise graph

    logger.info("Building graph...")

    sa = SpreadingActivation(decay_factor=0.01, firing_threshold=0.5)
    for i, word_pair in enumerate(combinations(word_list, 2)):
        word_1, word_2 = word_pair
        sa.add_edge(word_1, word_2,
                    # cosine similarity is 1- cosine distance
                    1-distributional_model.distance_between(word_1, word_2, DistanceType.cosine))
        if i % 1000 == 0:
            logger.info(f"\tAdded {i} word pairs ({word_1} and {word_2})")
    sa.freeze()

    # Pick initial node

    initial_word = "school"
    logger.info(f"Activating initial node {initial_word}")
    sa.activate_node(initial_word)

    logger.info("Running spreading activation")
    for i in range(1, 5):
        logger.info(f"Step {i}:")
        sa.spread_once()


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
