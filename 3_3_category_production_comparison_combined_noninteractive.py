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

from numpy import Inf
from pandas import DataFrame

from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms
from category_production.category_production import ColNames as CPColNames, CategoryProduction

from evaluation.category_production import add_ttfa_column, ModelType, save_hitrate_graphs, \
    get_model_ttfas_for_category_sensorimotor, get_hitrate_summary_tables, get_model_ttfas_for_category_linguistic, \
    get_n_words_from_path_linguistic, hitrate_within_sd_of_hitrate_mean_frac, \
    get_firing_threshold_from_path_linguistic, prepare_category_production_data
from evaluation.column_names import TTFA

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"

SN = SensorimotorNorms()
CP = CategoryProduction()

MODEL_TYPE = ModelType.combined_noninteractive

# The number of members of a category to produce when computing TTFA scale ratios
N_MEMBERS = 3

# Additional TTFA column names for parts of combined model
TTFA_LINGUISTIC          = f"{TTFA} linguistic"
TTFA_SENSORIMOTOR        = f"{TTFA} sensorimotor"
TTFA_SENSORIMOTOR_SCALED = f"{TTFA} sensorimotor scaled"
TTFA_COMBINED            = f"{TTFA} combined"


def main(input_results_dir_sensorimotor: str,
         input_results_dir_linguistic: str,
         linguistic_cat: Optional[int] = None):

    logger.info(path.basename(f"{input_results_dir_sensorimotor}, {input_results_dir_linguistic}"))

    # region Process args

    if linguistic_cat is None:
        ft = get_firing_threshold_from_path_linguistic(input_results_dir_linguistic)
        logger.info(f"No CAT provided, using FT instead ({ft})")
        this_linguistic_cat = ft
    else:
        this_linguistic_cat = linguistic_cat

    # TODO: this makes a "name too long" error
    # input_results_dir_linguistic, input_results_dir_sensorimotor = Path(input_results_dir_linguistic), Path(
    #     input_results_dir_sensorimotor)
    # model_identifier = f"{input_results_dir_linguistic.parent.name} {input_results_dir_linguistic.name} — " \
    #                    f"{input_results_dir_sensorimotor.parent.name} {input_results_dir_sensorimotor.name}"
    model_identifier = "combined test"
    output_dir = f"Category production fit {MODEL_TYPE.name}"

    if this_linguistic_cat is not None:
        file_suffix = f"({model_identifier}) CAT={this_linguistic_cat}"
    else:
        file_suffix = f"({model_identifier})"

    # endregion -------------------

    # region Compile individual model component data

    n_words = get_n_words_from_path_linguistic(input_results_dir_linguistic)

    # Main dataframe holds category production data and model response data
    main_data: DataFrame = prepare_category_production_data(ModelType.combined_noninteractive)

    # Linguistic TTFAs
    linguistic_ttfas = {
        category: get_model_ttfas_for_category_linguistic(category, input_results_dir_linguistic, n_words, this_linguistic_cat)
        for category in CP.category_labels_sensorimotor
    }
    add_ttfa_column(main_data, model_type=ModelType.linguistic, ttfas=linguistic_ttfas)
    main_data.rename(columns={TTFA: TTFA_LINGUISTIC}, inplace=True)

    # Sensorimotor TTFAs
    sensorimotor_ttfas = {
        category: get_model_ttfas_for_category_sensorimotor(category, input_results_dir_sensorimotor)
        for category in CP.category_labels
    }
    add_ttfa_column(main_data, model_type=ModelType.sensorimotor, ttfas=sensorimotor_ttfas)
    main_data.rename(columns={TTFA: TTFA_SENSORIMOTOR}, inplace=True)

    # endregion -------------------

    # region Find mean TTFA required for first 3 members

    mean_ttfa_linguistic = (
        main_data
            .dropna(subset=[TTFA_LINGUISTIC])
            .sort_values(by=TTFA_LINGUISTIC, ascending=True)
            .groupby(CPColNames.Category, sort=False)
            .head(N_MEMBERS)
            .groupby(CPColNames.Category, sort=False)
            .max()
        [TTFA_LINGUISTIC]
        .mean()
    )
    mean_ttfa_sensorimotor = (
        main_data
            .dropna(subset=[TTFA_SENSORIMOTOR])
            .sort_values(by=TTFA_SENSORIMOTOR, ascending=True)
            .groupby(CPColNames.CategorySensorimotor, sort=False)
            .head(N_MEMBERS)
            .groupby(CPColNames.CategorySensorimotor, sort=False)
            .max()
        [TTFA_SENSORIMOTOR]
        .mean()
    )

    # endregion -------------------

    # region Scale sensorimotor TTFAs to achieve 1:1 ratio

    ratio = mean_ttfa_sensorimotor / mean_ttfa_linguistic
    main_data[TTFA_SENSORIMOTOR_SCALED] = main_data[TTFA_SENSORIMOTOR] / ratio

    # endregion -------------------

    # region Combined model columns

    main_data[TTFA_COMBINED] = main_data[[TTFA_LINGUISTIC, TTFA_SENSORIMOTOR_SCALED]].min(axis=1)

    # endregion -------------------

    # region Find TTFA cut-off for best fit with participant data

    hitrates_per_rpf, hitrates_per_rmr = get_hitrate_summary_tables(main_data, model_type)
    save_hitrate_summary_tables(hitrates_per_rmr, hitrates_per_rpf, model_type, file_suffix)
    hitrate_fit_rpf = hitrate_within_sd_of_hitrate_mean_frac(hitrates_per_rpf)
    hitrate_fit_rmr = hitrate_within_sd_of_hitrate_mean_frac(hitrates_per_rmr)



    # endregion -------------------

    # region Process model output



    # endregion -------------------


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
