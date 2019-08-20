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
from glob import glob
from os import path

from numpy import nan, array
from pandas import DataFrame

from ldm.utils.maths import DistanceType, distance
from category_production.category_production import ColNames as CPColNames
from preferences import Preferences
from sensorimotor_norms.config.config import Config as SMConfig; SMConfig(use_config_overrides_from_file=path.join(path.dirname(__file__), "sm_config.yaml"))
from sensorimotor_norms.exceptions import WordNotInNormsError
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms

from evaluation.category_production import exclude_idiosyncratic_responses, add_predictor_column_model_hit, \
    add_predictor_column_production_proportion, add_rfop_column, add_rmr_column, add_predictor_column_ttfa, \
    CATEGORY_PRODUCTION, get_model_ttfas_for_category_sensorimotor, save_item_level_data, save_hitrate_summary_tables, \
    save_model_performance_stats, drop_missing_data
from evaluation.column_names import CATEGORY_AVAILABLE

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"

sensorimotor_norms = SensorimotorNorms()

distance_column = f"{DistanceType.Minkowski3.name} distance"


def main(input_results_dir: str,
         single_model: bool,
         min_first_rank_freq: int = None):

    # Set defaults
    min_first_rank_freq = 1 if min_first_rank_freq is None else min_first_rank_freq

    if single_model:
        model_output_dirs = [input_results_dir]
    else:
        model_output_dirs = glob(path.join(input_results_dir, "Category production traces "))

    for model_output_dir in model_output_dirs:
        main_data = compile_model_data(model_output_dir)
        process_one_model_output(main_data, model_output_dir, min_first_rank_freq)


def compile_model_data(input_results_dir: str) -> DataFrame:

    # Main dataframe holds category production data and model response data
    main_data: DataFrame = CATEGORY_PRODUCTION.data.copy()

    main_data.rename(columns={col_name: f"Precomputed {col_name}"
                              for col_name in ['Sensorimotor.distance.cosine.additive', 'Linguistic.PPMI']},
                     inplace=True)
    main_data = exclude_idiosyncratic_responses(main_data)

    add_predictor_column_sensorimotor_distance(main_data)
    add_predictor_column_ttfa(main_data,
                              {category: get_model_ttfas_for_category_sensorimotor(category, input_results_dir)
                               for category in CATEGORY_PRODUCTION.category_labels_sensorimotor},
                              sensorimotor=True)
    add_predictor_column_model_hit(main_data)
    add_predictor_column_category_available_in_norms(main_data)

    add_predictor_column_production_proportion(main_data)
    add_rfop_column(main_data)
    add_rmr_column(main_data)

    return main_data


def process_one_model_output(main_data: DataFrame, input_results_dir: str, min_first_rank_freq: int):
    save_item_level_data(main_data, path.join(Preferences.results_dir,
                                              f"Category production fit sensorimotor",
                                              f"item-level data ({path.basename(input_results_dir)}).csv"))
    hitrate_stats = save_hitrate_summary_tables(input_results_dir, main_data, sensorimotor=True)

    drop_missing_data(main_data, distance_column)

    save_model_performance_stats(
        main_data,
        results_dir=input_results_dir,
        min_first_rank_freq=min_first_rank_freq,
        **hitrate_stats,
        sensorimotor=True,
    )


def add_predictor_column_sensorimotor_distance(main_data):
    """Mutates `main_data`."""
    logger.info("Adding distance column")
    main_data[distance_column] = main_data.apply(get_sensorimotor_distance_minkowski3, axis=1)


def add_predictor_column_category_available_in_norms(main_data):
    """Mutates `main_data`."""
    logger.info("Adding category availability column")
    main_data[CATEGORY_AVAILABLE] = main_data.apply(lambda row: sensorimotor_norms.has_word(row[CPColNames.CategorySensorimotor]), axis=1)


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
    parser.add_argument("path", type=str, help="The path in which to find the results.")
    parser.add_argument("--single-model", action="store_true",
                        help="If specified, `path` will be interpreted to be the dir for a single model's output; "
                             "otherwise `path` will be interpreted to contain many models' output dirs.")
    parser.add_argument("min_frf", type=int, nargs="?", default=None,
                        help="The minimum FRF required for zRT and FRF correlations.")
    args = parser.parse_args()

    main(args.path, args.single_model, args.min_frf)

    logger.info("Done!")
