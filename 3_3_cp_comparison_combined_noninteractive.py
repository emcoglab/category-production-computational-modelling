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

from matplotlib import pyplot
from numpy import ceil, asarray, savetxt, array
from pandas import DataFrame

from preferences import Preferences
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms
from category_production.category_production import ColNames as CPColNames, CategoryProduction

from evaluation.category_production import add_ttfa_column, ModelType, save_hitrate_graphs, \
    get_model_ttfas_for_category_sensorimotor, get_hitrate_summary_tables, get_model_ttfas_for_category_linguistic, \
    get_n_words_from_path_linguistic, frac_within_sd_of_hitrate_mean, \
    get_firing_threshold_from_path_linguistic, prepare_category_production_data
from evaluation.column_names import TTFA, MODEL_HIT

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

    if this_linguistic_cat is not None:
        file_suffix = f"({model_identifier}) CAT={this_linguistic_cat}"
    else:
        file_suffix = f"({model_identifier})"

    # endregion -------------------

    # region Compile individual model component data

    # Main dataframe holds category production data and model response data
    main_data: DataFrame = prepare_category_production_data(ModelType.combined_noninteractive)

    # Linguistic TTFAs
    n_words = get_n_words_from_path_linguistic(input_results_dir_linguistic)
    linguistic_ttfas = {
        category: get_model_ttfas_for_category_linguistic(category, input_results_dir_linguistic, n_words, this_linguistic_cat)
        for category in CP.category_labels
    }
    add_ttfa_column(main_data, model_type=ModelType.linguistic, ttfas=linguistic_ttfas)
    main_data.rename(columns={TTFA: TTFA_LINGUISTIC}, inplace=True)

    # Sensorimotor TTFAs
    sensorimotor_ttfas = {
        category: get_model_ttfas_for_category_sensorimotor(category, input_results_dir_sensorimotor)
        for category in CP.category_labels_sensorimotor
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

    ratio = mean_ttfa_linguistic / mean_ttfa_sensorimotor
    logger.info(f"Sensorimotor TTFAs *= {ratio}")
    main_data[TTFA_SENSORIMOTOR_SCALED] = main_data[TTFA_SENSORIMOTOR] * ratio

    # endregion -------------------

    # region Combined model columns

    main_data[TTFA_COMBINED] = main_data[[TTFA_LINGUISTIC, TTFA_SENSORIMOTOR_SCALED]].min(axis=1)

    # endregion -------------------

    # region Find TTFA cut-off for best fit with participant data

    max_ttfa = int(ceil(max(main_data[TTFA_LINGUISTIC].max(), main_data[TTFA_SENSORIMOTOR_SCALED].max())))

    combined_hitrates_rmr, combined_hitrates_rpf = [], []
    for ttfa_cutoff in range(max_ttfa + 1):
        cut_data = apply_cutoff(main_data, TTFA_COMBINED, ttfa_cutoff)
        hrs_rpf, hrs_rmr = get_hitrate_summary_tables(cut_data, MODEL_TYPE)
        combined_hitrates_rmr.append(frac_within_sd_of_hitrate_mean(hrs_rmr))
        combined_hitrates_rpf.append(frac_within_sd_of_hitrate_mean(hrs_rpf))

    # Convert to arrays so I can do quick argmax
    combined_hitrates_rmr: array = asarray(combined_hitrates_rmr)
    combined_hitrates_rpf: array = asarray(combined_hitrates_rpf)

    # Optimum cutoffs for each stat
    combined_rmr_ttfa_cutoff = combined_hitrates_rmr.argmax()
    combined_rpf_ttfa_cutoff = combined_hitrates_rpf.argmax()

    # endregion -------------------

    # region Graph cutoff-by-fit

    # Save values
    savetxt(path.join(Preferences.results_dir, MODEL_TYPE.model_output_dirname, "rmr cutoff.csv"),
            combined_hitrates_rmr, delimiter=",")
    savetxt(path.join(Preferences.results_dir, MODEL_TYPE.model_output_dirname, "rpf cutoff.csv"),
            combined_hitrates_rpf, delimiter=",")

    # RMR graph
    pyplot.plot(combined_hitrates_rmr)
    pyplot.ylim((0, 1))
    pyplot.xlabel("TTFA cutoff")
    pyplot.ylabel("Fraction of hitrates within 1SD of participant mean")
    pyplot.title("Noninteractive combined fits (RMR)")
    pyplot.savefig(path.join(Preferences.figures_dir, MODEL_TYPE.figures_dirname, "rmr fits by cutoff.png"))
    pyplot.clf()
    pyplot.cla()
    pyplot.close()

    # RPF graph
    pyplot.plot(combined_hitrates_rpf)
    pyplot.ylim((0, 1))
    pyplot.xlabel("TTFA cutoff")
    pyplot.ylabel("Fraction of hitrates within 1SD of participant mean")
    pyplot.title("Noninteractive combined fits (RPF)")
    pyplot.savefig(path.join(Preferences.figures_dir, MODEL_TYPE.figures_dirname, "rpf fits by cutoff.png"))
    pyplot.clf()
    pyplot.cla()
    pyplot.close()

    # endregion -----------------

    # region Save optimal graphs

    # Combined (rmr-optimal)
    cut_data = apply_cutoff(main_data, TTFA_COMBINED, combined_rmr_ttfa_cutoff)
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(cut_data, MODEL_TYPE)
    save_hitrate_graphs(hrs_rpf, hrs_rmr, MODEL_TYPE, file_suffix + f" rmr-optimal ({combined_rmr_ttfa_cutoff})")
    logger.info(f"rmr-optimal ({combined_rmr_ttfa_cutoff}) rmr fit: {frac_within_sd_of_hitrate_mean(hrs_rmr)}")
    logger.info(f"rmr-optimal ({combined_rmr_ttfa_cutoff}) rpf fit: {frac_within_sd_of_hitrate_mean(hrs_rpf)}")

    # Combined (rpf-optimal)
    cut_data = apply_cutoff(main_data, TTFA_COMBINED, combined_rpf_ttfa_cutoff)
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(cut_data, MODEL_TYPE)
    save_hitrate_graphs(hrs_rpf, hrs_rmr, MODEL_TYPE, file_suffix + f" rpf-optimal ({combined_rpf_ttfa_cutoff})")
    logger.info(f"rpf-optimal ({combined_rpf_ttfa_cutoff}) rmr fit: {frac_within_sd_of_hitrate_mean(hrs_rmr)}")
    logger.info(f"rpf-optimal ({combined_rpf_ttfa_cutoff}) rpf fit: {frac_within_sd_of_hitrate_mean(hrs_rpf)}")

    # endregion -----------------

    # region Apply cutoff to individual components

    cut_data = apply_cutoff(main_data, TTFA_LINGUISTIC, combined_rmr_ttfa_cutoff)
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(cut_data, MODEL_TYPE)
    save_hitrate_graphs(hrs_rpf, hrs_rmr, MODEL_TYPE, file_suffix + f" rmr-optimal linguistic ({combined_rmr_ttfa_cutoff})")
    logger.info(f"rmr-optimal ling ({combined_rmr_ttfa_cutoff}) rmr fit: {frac_within_sd_of_hitrate_mean(hrs_rmr)}")
    logger.info(f"rmr-optimal ling ({combined_rmr_ttfa_cutoff}) rpf fit: {frac_within_sd_of_hitrate_mean(hrs_rpf)}")

    cut_data = apply_cutoff(main_data, TTFA_SENSORIMOTOR_SCALED, combined_rmr_ttfa_cutoff)
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(cut_data, MODEL_TYPE)
    save_hitrate_graphs(hrs_rpf, hrs_rmr, MODEL_TYPE, file_suffix + f" rmr-optimal sensorimotor ({combined_rmr_ttfa_cutoff})")
    logger.info(f"rmr-optimal sm ({combined_rmr_ttfa_cutoff}) rmr fit: {frac_within_sd_of_hitrate_mean(hrs_rmr)}")
    logger.info(f"rmr-optimal sm ({combined_rmr_ttfa_cutoff}) rpf fit: {frac_within_sd_of_hitrate_mean(hrs_rpf)}")

    cut_data = apply_cutoff(main_data, TTFA_LINGUISTIC, combined_rpf_ttfa_cutoff)
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(cut_data, MODEL_TYPE)
    save_hitrate_graphs(hrs_rpf, hrs_rmr, MODEL_TYPE, file_suffix + f" rpf-optimal linguistic ({combined_rpf_ttfa_cutoff})")
    logger.info(f"rpf-optimal ling ({combined_rpf_ttfa_cutoff}) rmr fit: {frac_within_sd_of_hitrate_mean(hrs_rmr)}")
    logger.info(f"rpf-optimal ling ({combined_rpf_ttfa_cutoff}) rpf fit: {frac_within_sd_of_hitrate_mean(hrs_rpf)}")

    cut_data = apply_cutoff(main_data, TTFA_SENSORIMOTOR_SCALED, combined_rpf_ttfa_cutoff)
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(cut_data, MODEL_TYPE)
    save_hitrate_graphs(hrs_rpf, hrs_rmr, MODEL_TYPE, file_suffix + f" rpf-optimal sensorimotor ({combined_rpf_ttfa_cutoff})")
    logger.info(f"rpf-optimal sm ({combined_rpf_ttfa_cutoff}) rmr fit: {frac_within_sd_of_hitrate_mean(hrs_rmr)}")
    logger.info(f"rpf-optimal sm ({combined_rpf_ttfa_cutoff}) rpf fit: {frac_within_sd_of_hitrate_mean(hrs_rpf)}")

    # endregion -------------------


def apply_cutoff(data, ttfa_column, ttfa_cutoff):
    """Adds a cut-off `MODEL_HIT` column to a copy of `data`."""
    cut_data = data.copy()
    cut_data[MODEL_HIT] = cut_data[ttfa_column] < ttfa_cutoff
    cut_data.fillna(value={MODEL_HIT: False}, inplace=True)
    return cut_data


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
