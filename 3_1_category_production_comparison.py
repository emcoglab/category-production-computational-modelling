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
2018
---------------------------
"""

import argparse
import logging
import sys
from glob import glob
from os import path

from pandas import DataFrame

from category_production.category_production import ColNames as CPColNames
from evaluation.category_production import get_n_words_from_path_linguistic, get_model_ttfas_for_category_linguistic, \
    available_categories, exclude_idiosyncratic_responses, add_predictor_column_model_hit, \
    add_predictor_column_production_proportion, add_rfop_column, add_rmr_column, CATEGORY_PRODUCTION, \
    add_predictor_column_ttfa, save_item_level_data, save_hitrate_summary_tables, save_model_performance_stats
from evaluation.column_names import TTFA, CATEGORY_AVAILABLE
from preferences import Preferences

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def main(input_results_dir: str, single_model: bool, conscious_access_threshold: float, min_first_rank_freq: int = None):

    # Set defaults
    min_first_rank_freq = 1 if min_first_rank_freq is None else min_first_rank_freq

    if single_model:
        model_output_dirs = [input_results_dir]
    else:
        model_output_dirs = glob(path.join(input_results_dir, "Category production traces "))

    for model_output_dir in model_output_dirs:
        main_data = compile_model_data(model_output_dir, conscious_access_threshold)
        process_one_model_output(main_data, model_output_dir, conscious_access_threshold, min_first_rank_freq)


def compile_model_data(input_results_dir: str, conscious_access_threshold) -> DataFrame:

    n_words = get_n_words_from_path_linguistic(input_results_dir)

    # Main dataframe holds category production data and model response data
    main_data: DataFrame = CATEGORY_PRODUCTION.data.copy()

    main_data.rename(columns={col_name: f"Precomputed {col_name}"
                              for col_name in ['Sensorimotor.distance.cosine.additive', 'Linguistic.PPMI']},
                     inplace=True)
    main_data = exclude_idiosyncratic_responses(main_data)

    add_predictor_column_ttfa(main_data,
                              {category: get_model_ttfas_for_category_linguistic(category, input_results_dir, n_words, conscious_access_threshold)
                               for category in CATEGORY_PRODUCTION.category_labels},
                              sensorimotor=False)
    add_predictor_column_model_hit(main_data)
    add_predictor_column_category_available_to_model(main_data, input_results_dir)

    add_predictor_column_production_proportion(main_data)
    add_rfop_column(main_data)
    add_rmr_column(main_data)

    main_data.dropna(inplace=True, how='any', subset=[TTFA])
    # Now we can convert TTFAs to ints as there won't be null values
    main_data[TTFA] = main_data[TTFA].astype(int)

    return main_data


def process_one_model_output(main_data: DataFrame,
                             input_results_dir: str,
                             conscious_access_threshold: float,
                             min_first_rank_freq: int):
    save_item_level_data(main_data, path.join(Preferences.results_dir,
                                              "Category production fit",
                                              f"item-level data ({path.basename(input_results_dir)}) "
                                              f"CAT={conscious_access_threshold}.csv"))
    hitrate_stats = save_hitrate_summary_tables(input_results_dir, main_data, sensorimotor=False)
    save_model_performance_stats(
        main_data,
        results_dir=input_results_dir,
        min_first_rank_freq=min_first_rank_freq,
        **hitrate_stats,
        sensorimotor=False,
    )


def add_predictor_column_category_available_to_model(main_data, input_results_dir):
    """Mutates `main_data`."""
    logger.info("Adding category availability column")
    main_data[CATEGORY_AVAILABLE] = main_data.apply(lambda row: row[CPColNames.Category] in available_categories(input_results_dir), axis=1)


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Compare spreading activation results with Category Production data.")
    parser.add_argument("path", type=str, help="The path in which to find the results.")
    parser.add_argument("cat", type=float, help="The conscious-access threshold.")
    parser.add_argument("min_frf", type=int, nargs="?", default=None,
                        help="The minimum FRF required for zRT and FRF correlations.")
    parser.add_argument("--single-model", action="store_true",
                        help="If specified, `path` will be interpreted to be the dir for a single model's output; "
                             "otherwise `path` will be interpreted to contain many models' output dirs.")
    args = parser.parse_args()

    main(args.path, args.single_model, args.cat, args.min_frf)

    logger.info("Done!")
