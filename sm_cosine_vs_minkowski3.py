"""
===========================
Compare cosine and Minkowski3 distances for modelling category production data.
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

import logging
from collections import defaultdict
from os import path
from typing import Dict, DefaultDict

from numpy import array, nan
from pandas import DataFrame

from ldm.utils.maths import DistanceType, distance
from category_production.category_production import CategoryProduction
from category_production.category_production import ColNames as CPColNames
from preferences import Preferences
from sensorimotor_norms.exceptions import WordNotInNormsError
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"

_COSINE_DISTANCE = "Cosine distance"
_MINKOWSKI_DISTANCE = "Minkowski-3 distance"


def main():
    sensorimotor_norms = SensorimotorNorms()
    category_production = CategoryProduction()

    main_dataframe: DataFrame = category_production.data.copy()

    # category -> sm_response -> distance
    cosine_distances: Dict[str, DefaultDict[str, float]] = dict()
    minkowski_distances: Dict[str, DefaultDict[str, float]] = dict()
    for category in category_production.category_labels:
        cosine_distances[category] = defaultdict(lambda: nan)
        minkowski_distances[category] = defaultdict(lambda: nan)

        sm_category = category_production.apply_sensorimotor_substitution(category)

        try:
            category_sm_vector = array(sensorimotor_norms.vector_for_word(sm_category))
        except WordNotInNormsError:
            continue

        logger.info(f"Category: {category}")

        for sm_response in category_production.responses_for_category(category, use_sensorimotor_responses=True,
                                                                      single_word_only=True):

            try:
                response_sm_vector = array(sensorimotor_norms.vector_for_word(sm_response))
            except WordNotInNormsError:
                continue

            cosine_distances[category][sm_response] = distance(category_sm_vector, response_sm_vector,
                                                               DistanceType.cosine)
            minkowski_distances[category][sm_response] = distance(category_sm_vector, response_sm_vector,
                                                                  DistanceType.Minkowski3)

    main_dataframe[_COSINE_DISTANCE] = main_dataframe.apply(
        lambda row: cosine_distances[row[CPColNames.Category]][row[CPColNames.ResponseSensorimotor]],
        axis=1)
    main_dataframe[_MINKOWSKI_DISTANCE] = main_dataframe.apply(
        lambda row: minkowski_distances[row[CPColNames.Category]][row[CPColNames.ResponseSensorimotor]],
        axis=1)

    main_dataframe.to_csv(path.join(Preferences.results_dir,
                                    "Category production fit sensorimotor",
                                    "item-level data (cosine vs Minkowski-3).csv"))


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)

    main()
