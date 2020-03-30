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

from model.utils.logging import logger

_COSINE_DISTANCE = "Cosine distance"
_MINKOWSKI_DISTANCE = "Minkowski-3 distance"


def main():
    sensorimotor_norms = SensorimotorNorms()
    category_production = CategoryProduction()

    main_dataframe: DataFrame = category_production.data.copy()

    # category -> sm_response -> distance
    cosine_distances: Dict[str, DefaultDict[str, float]] = dict()
    minkowski_distances: Dict[str, DefaultDict[str, float]] = dict()
    for sm_category in category_production.category_labels_sensorimotor:
        cosine_distances[sm_category] = defaultdict(lambda: nan)
        minkowski_distances[sm_category] = defaultdict(lambda: nan)

        try:
            category_sm_vector = array(sensorimotor_norms.vector_for_word(sm_category))
        except WordNotInNormsError:
            continue

        logger.info(f"Category: {sm_category}")

        for sm_response in category_production.responses_for_category(sm_category,
                                                                      use_sensorimotor=True,
                                                                      single_word_only=True):

            try:
                response_sm_vector = array(sensorimotor_norms.vector_for_word(sm_response))
            except WordNotInNormsError:
                continue

            cosine_distances[sm_category][sm_response] = distance(category_sm_vector, response_sm_vector,
                                                                  DistanceType.cosine)
            minkowski_distances[sm_category][sm_response] = distance(category_sm_vector, response_sm_vector,
                                                                     DistanceType.Minkowski3)

    main_dataframe[_COSINE_DISTANCE] = main_dataframe.apply(
        lambda row: cosine_distances[row[CPColNames.CategorySensorimotor]][row[CPColNames.ResponseSensorimotor]],
        axis=1)
    main_dataframe[_MINKOWSKI_DISTANCE] = main_dataframe.apply(
        lambda row: minkowski_distances[row[CPColNames.CategorySensorimotor]][row[CPColNames.ResponseSensorimotor]],
        axis=1)

    main_dataframe.to_csv(path.join(Preferences.results_dir,
                                    "Category production fit sensorimotor",
                                    "item-level data (cosine vs Minkowski-3).csv"),
                          index=False)


if __name__ == '__main__':


    main()
