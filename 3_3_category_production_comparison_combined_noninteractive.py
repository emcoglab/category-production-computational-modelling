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
from pathlib import Path
from typing import Optional

from numpy import nan, array
from pandas import DataFrame

from category_production.category_production import ColNames as CPColNames
from evaluation.category_production import add_model_predictor_columns, CATEGORY_PRODUCTION, \
    get_model_ttfas_for_category_sensorimotor, save_item_level_data, save_hitrate_summary_tables, \
    get_model_ttfas_for_category_linguistic, get_n_words_from_path_linguistic, save_hitrate_graphs, \
    get_firing_threshold_from_path_linguistic, ModelType, prepare_category_production_data, \
    drop_missing_data_to_add_types, save_model_performance_stats, hitrate_within_sd_of_hitrate_mean_frac
from evaluation.column_names import TTFA, MODEL_HIT
from ldm.utils.maths import DistanceType, distance
from preferences import Preferences
from sensorimotor_norms.exceptions import WordNotInNormsError
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"

SN = SensorimotorNorms()

distance_column = f"{DistanceType.Minkowski3.name} distance"


def main(input_results_dir_sensorimotor: str,
         input_results_dir_linguistic: str,
         linguistic_cat: Optional[int] = None):

    if linguistic_cat is None:
        ft = get_firing_threshold_from_path_linguistic(input_results_dir_linguistic)
        logger.info(f"No CAT provided, using FT instead ({ft})")
        this_linguistic_cat = ft
    else:
        this_linguistic_cat = linguistic_cat

    logger.info(path.basename(f"{input_results_dir_sensorimotor}, {input_results_dir_linguistic}"))

    main_data = compile_model_data(input_results_dir_linguistic, input_results_dir_sensorimotor, this_linguistic_cat)

    process_one_model_output(main_data,
                             ModelType.combined_set_union,
                             input_results_dir_linguistic,
                             input_results_dir_sensorimotor,
                             None,  # TODO: min-frf
                             linguistic_cat
                             )


def compile_model_data(input_results_dir_linguistic: str, input_results_dir_sensorimotor: str, conscious_access_threshold) -> DataFrame:

    n_words = get_n_words_from_path_linguistic(input_results_dir_linguistic)

    # Main dataframe holds category production data and model response data
    main_data = prepare_category_production_data(ModelType.combined_set_union)

    # Linguistic TTFA and model hit
    add_model_predictor_columns(main_data, model_type=ModelType.linguistic,
                                ttfas={
                                    category: get_model_ttfas_for_category_sensorimotor(category, input_results_dir_sensorimotor)
                                    for category in CATEGORY_PRODUCTION.category_labels_sensorimotor})
    main_data.rename(columns={
        TTFA: f"{TTFA} linguistic",
        MODEL_HIT: f"{MODEL_HIT} linguistic"
    }, inplace=True)

    # Sensorimotor TTFA and model hit
    add_model_predictor_columns(main_data, model_type=ModelType.sensorimotor,
                                ttfas={
                                    category: get_model_ttfas_for_category_linguistic(category, input_results_dir_linguistic, n_words, conscious_access_threshold)
                                    for category in CATEGORY_PRODUCTION.category_labels})
    main_data.rename(columns={
        TTFA: f"{TTFA} sensorimotor",
        MODEL_HIT: f"{MODEL_HIT} sensorimotor"
    }, inplace=True)

    # Combined model columns
    main_data[TTFA] = main_data[[f"{TTFA} linguistic", f"{TTFA} sensorimotor"]].min(axis=1)
    main_data[MODEL_HIT] = main_data[f"{MODEL_HIT} sensorimotor"] | main_data[f"{MODEL_HIT} linguistic"]

    return main_data


def process_one_model_output(main_data: DataFrame,
                             model_type: ModelType,
                             input_results_dir_linguistic: str,
                             input_results_dir_sensorimotor: str,
                             min_first_rank_freq: Optional[int],
                             conscious_access_threshold: Optional[float],
                             ):
    assert model_type == ModelType.combined_set_union
    input_results_dir_linguistic, input_results_dir_sensorimotor = Path(input_results_dir_linguistic), Path(input_results_dir_sensorimotor)
    # model_identifier = f"{input_results_dir_linguistic.parent.name} {input_results_dir_linguistic.name} — " \
    #                    f"{input_results_dir_sensorimotor.parent.name} {input_results_dir_sensorimotor.name}"
    # TODO: this makes a "name too long" error
    model_identifier = "combined test"
    output_dir = f"Category production fit {model_type.name}"
    save_item_level_data(main_data, path.join(Preferences.results_dir,
                                              output_dir,
                                              f"item-level data ({model_identifier})"
                                              + (f" CAT={conscious_access_threshold}" if conscious_access_threshold is not None else "") +
                                              ".csv"))

    if conscious_access_threshold is not None:
        file_suffix = f"({model_identifier}) CAT={conscious_access_threshold}"
    else:
        file_suffix = f"({model_identifier})"

    hitrates_per_rpf, hitrates_per_rmr = save_hitrate_summary_tables(main_data, model_type, file_suffix)

    # Compute hitrate fits
    hitrate_fit_rpf_hr = hitrate_within_sd_of_hitrate_mean_frac(hitrates_per_rpf)
    hitrate_fit_rmr_hr = hitrate_within_sd_of_hitrate_mean_frac(hitrates_per_rmr)

    drop_missing_data_to_add_types(main_data, {TTFA: int})

    save_model_performance_stats(
        main_data,
        model_identifier=model_identifier,
        results_dir=None,
        min_first_rank_freq=min_first_rank_freq,
        hitrate_fit_rpf_hr=hitrate_fit_rpf_hr,
        hitrate_fit_rmr_hr=hitrate_fit_rmr_hr,
        model_type=model_type,
        conscious_access_threshold=conscious_access_threshold,
    )

    save_hitrate_graphs(hitrates_per_rpf, hitrates_per_rmr, model_type, file_suffix)


def add_predictor_column_sensorimotor_distance(main_data):
    """Mutates `main_data`."""
    logger.info("Adding distance column")
    main_data[distance_column] = main_data.apply(get_sensorimotor_distance_minkowski3, axis=1)


def get_sensorimotor_distance_minkowski3(row):
    try:
        category_vector = array(SN.vector_for_word(row[CPColNames.CategorySensorimotor]))
        response_vector = array(SN.vector_for_word(row[CPColNames.ResponseSensorimotor]))
        return distance(category_vector, response_vector, DistanceType.Minkowski3)
    except WordNotInNormsError:
        return nan


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Compare spreading activation results with Category Production data.")
    parser.add_argument("linguistic_path", type=str, help="The path in which to find the linguistic results.")
    parser.add_argument("sensorimotor_path", type=str, help="The path in which to find the sensorimotor results.")
    parser.add_argument("cat", type=float, nargs="?", default=None,
                        help="The conscious-access threshold."
                             " Omit to use CAT = firing threshold.")
    args = parser.parse_args()

    main(args.sensorimotor_path, args.linguistic_path, args.cat)

    logger.info("Done!")
