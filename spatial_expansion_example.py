"""
===========================
Example of tandem model function
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2019
---------------------------
"""
from itertools import islice
from logging import getLogger

from numpy import array

from cli.lookups import get_corpus_from_name
from ldm.corpus.indexing import FreqDist, TokenIndex
from ldm.utils.maths import DistanceType
from model.points_in_space import PointsInSpace
from model.temporal_spatial_expansion import TemporalSpatialExpansion
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms

logger = getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def main(n_words, corpus_name, expansion_rate, max_radius, decay_median, decay_shape, decay_threshold, run_for_ticks, bailout):

    sm_norms = SensorimotorNorms()

    # Use corpus to select top words
    corpus = get_corpus_from_name(corpus_name)
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    filtered_words = list(islice((word for word in freq_dist.iter_most_common_tokens() if sm_norms.has_word(word)), n_words))

    # Relabelling dictionaries

    # maps words to their indices within the corpus
    corpus_index = TokenIndex.from_freqdist_ranks(freq_dist)

    # maps corpus ids to point ids


    # maps points-ids to words
    point_label_dictionary =


    sensorimotor_component: TemporalSpatialExpansion = TemporalSpatialExpansion(
        points_in_space=PointsInSpace(data_matrix=array(sm_norms.matrix_for_words(filtered_words))),
        item_labelling_dictionary=,
        expansion_rate=expansion_rate,
        max_radius=max_radius,
        distance_type=DistanceType.cosine,
        decay_median=decay_median,
        decay_shape=decay_shape,
        decay_threshold=decay_threshold,
    )

    sensorimotor_component.activate_item_with_label("school", 1.0)

    for tick in range(run_for_ticks):
        sensorimotor_component.tick()
        if sensorimotor_component.n_suprathreshold_items() > bailout:
            logger.warning("Bailout!")
            break


if __name__ == '__main__':

    main(n_words=3_000,
         corpus_name="bbc",
         expansion_rate=,
         max_radius=,
         decay_median=,
         decay_shape=,
         decay_threshold=,
         run_for_ticks=1_000,
         bailout=1_000)
    logger.info("Done!")
