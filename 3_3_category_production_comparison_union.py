#!/Users/cai/Applications/miniconda3/bin/python
"""
===========================
Compare model to Briony's category production actual responses.

Pass the parent location to a bunch of results.
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

import argparse
import logging
import sys
from os import path
from typing import Optional

from numpy import nan, array
from pandas import DataFrame

from category_production.category_production import ColNames as CPColNames
from evaluation.category_production import exclude_idiosyncratic_responses, add_predictor_column_model_hit, \
    add_predictor_column_production_proportion, add_rfop_column, add_rmr_column, add_model_predictor_columns, \
    CATEGORY_PRODUCTION, get_model_ttfas_for_category_sensorimotor, save_item_level_data, save_hitrate_summary_tables, \
    get_model_ttfas_for_category_linguistic, get_n_words_from_path_linguistic, \
    get_firing_threshold_from_path_linguistic, ModelType
from evaluation.column_names import TTFA, MODEL_HIT
from ldm.utils.maths import DistanceType, distance
from preferences import Preferences
from sensorimotor_norms.exceptions import WordNotInNormsError
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"

sensorimotor_norms = SensorimotorNorms()

distance_column = f"{DistanceType.Minkowski3.name} distance"


def main(input_results_dir_sensorimotor: str,
         input_results_dir_linguistic: str,
         linguistic_cat: Optional[int] = None):

    # Set defaults
    if linguistic_cat is None:
        # If no CAT provided, use the FT
        linguistic_cat = get_firing_threshold_from_path_linguistic(input_results_dir_linguistic)
        logger.info(f"No CAT provided, using FT instead ({linguistic_cat})")

    logger.info(path.basename(f"{input_results_dir_sensorimotor}, {input_results_dir_linguistic}"))

    main_data = compile_model_data(input_results_dir_sensorimotor, input_results_dir_linguistic, linguistic_cat)
    process_one_model_output(main_data, input_results_dir_sensorimotor, input_results_dir_linguistic)


def compile_model_data(input_results_dir_sensorimotor, input_results_dir_linguistic: str, conscious_access_threshold) -> DataFrame:

    n_words = get_n_words_from_path_linguistic(input_results_dir_linguistic)

    # Main dataframe holds category production data and model response data
    main_data: DataFrame = CATEGORY_PRODUCTION.data.copy()

    main_data.rename(columns={col_name: f"Precomputed {col_name}"
                              for col_name in ['Sensorimotor.distance.cosine.additive', 'Linguistic.PPMI']},
                     inplace=True)
    main_data = exclude_idiosyncratic_responses(main_data)

    add_predictor_column_sensorimotor_distance(main_data)

    # Sensorimotor TTFA and model hit
    add_model_predictor_columns(main_data,
                                {category: get_model_ttfas_for_category_sensorimotor(category, input_results_dir_sensorimotor)
                               for category in CATEGORY_PRODUCTION.category_labels_sensorimotor},
                                model_type=ModelType.linguistic)
    add_predictor_column_model_hit(main_data)
    main_data.rename(columns={
        TTFA: f"{TTFA} sensorimotor",
        MODEL_HIT: f"{MODEL_HIT} sensorimotor"
    }, inplace=True)

    # Linguistic TTFA and model hit
    add_model_predictor_columns(main_data,
                                {category: get_model_ttfas_for_category_linguistic(category, input_results_dir_linguistic, n_words, conscious_access_threshold)
                               for category in CATEGORY_PRODUCTION.category_labels},
                                model_type=ModelType.sensorimotor)
    add_predictor_column_model_hit(main_data)
    main_data.rename(columns={
        TTFA: f"{TTFA} linguistic",
        MODEL_HIT: f"{MODEL_HIT} linguistic"
    }, inplace=True)

    # Combined model hit
    main_data[MODEL_HIT] = main_data[f"{MODEL_HIT} sensorimotor"] | main_data[f"{MODEL_HIT} linguistic"]

    add_predictor_column_production_proportion(main_data)
    add_rfop_column(main_data, model_type=ModelType.combined_set_union)
    add_rmr_column(main_data)

    return main_data


def process_one_model_output(main_data: DataFrame,
                             input_results_dir_sensorimotor: str,
                             input_results_dir_linguistic: str):
    save_item_level_data(main_data, path.join(Preferences.results_dir,
                                              f"Category production fit naïve combined",
                                              # TODO: deal with these super-long filenames
                                              # f"item-level data ({path.basename(input_results_dir_sensorimotor)}; {path.basename(input_results_dir_linguistic)}).csv"
                                              f"item-level data (combined test).csv"
                                              ))

    hitrate_fit_rfop, hitrate_fit_rmr = save_hitrate_summary_tables(
        # f"{path.basename(input_results_dir_sensorimotor)}; {path.basename(input_results_dir_linguistic)}",
        "combined test",
        main_data,
        ModelType.combined_set_union, conscious_access_threshold=None)


def add_predictor_column_sensorimotor_distance(main_data):
    """Mutates `main_data`."""
    logger.info("Adding distance column")
    main_data[distance_column] = main_data.apply(get_sensorimotor_distance_minkowski3, axis=1)


def get_sensorimotor_distance_minkowski3(row):
    try:
        category_vector = array(sensorimotor_norms.vector_for_word(row[CPColNames.CategorySensorimotor]))
        response_vector = array(sensorimotor_norms.vector_for_word(row[CPColNames.ResponseSensorimotor]))
        return distance(category_vector, response_vector, DistanceType.Minkowski3)
    except WordNotInNormsError:
        return nan


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Compare spreading activation results with Category Production data.")
    parser.add_argument("sensorimotor_path", type=str, help="The path in which to find the sensorimotor results.")
    parser.add_argument("linguistic_path", type=str, help="The path in which to find the linguistic results.")
    parser.add_argument("cat", type=float, nargs="?", default=None,
                        help="The conscious-access threshold."
                             " Omit to use CAT = firing threshold.")
    args = parser.parse_args()

    main(args.sensorimotor_path, args.linguistic_path, args.cat)

    logger.info("Done!")
