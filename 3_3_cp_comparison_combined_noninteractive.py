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
from os import path, makedirs
from pathlib import Path
from typing import Optional

from matplotlib import pyplot
from numpy import ceil, savetxt, array
from pandas import DataFrame

from preferences import Preferences
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms
from category_production.category_production import ColNames as CPColNames, CategoryProduction

from evaluation.category_production import add_ttfa_column, ModelType, save_hitrate_graphs, \
    get_model_ttfas_for_category_sensorimotor, get_hitrate_summary_tables, get_model_ttfas_for_category_linguistic, \
    get_n_words_from_path_linguistic, frac_within_sd_of_hitrate_mean, \
    get_firing_threshold_from_path_linguistic, prepare_category_production_data
from evaluation.column_names import TTFA, MODEL_HIT, MODEL_HITRATE, PARTICIPANT_HITRATE_All_f

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "1%Y-%m-%d %H:%M:%S"

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

    logger.info("")
    logger.info(path.basename(f"{input_results_dir_sensorimotor}, {input_results_dir_linguistic}"))

    # region Process args

    if linguistic_cat is None:
        ft = get_firing_threshold_from_path_linguistic(input_results_dir_linguistic)
        logger.info(f"No CAT provided, using FT instead ({ft})")
        this_linguistic_cat = ft
    else:
        this_linguistic_cat = linguistic_cat

    input_results_dir_linguistic   = Path(input_results_dir_linguistic)
    input_results_dir_sensorimotor = Path(input_results_dir_sensorimotor)
    # Organise by linguistic and sensorimotor model names
    evaluation_save_dir = path.join(Preferences.results_dir, MODEL_TYPE.model_output_dirname,
                                    f"{input_results_dir_linguistic.parent.name} {input_results_dir_linguistic.name}",
                                    f"{input_results_dir_sensorimotor.parent.name} {input_results_dir_sensorimotor.name}")
    figures_dir = path.join(Preferences.figures_dir, MODEL_TYPE.figures_dirname,
                            f"{input_results_dir_linguistic.parent.name} {input_results_dir_linguistic.name}",
                            f"{input_results_dir_sensorimotor.parent.name} {input_results_dir_sensorimotor.name}")
    makedirs(evaluation_save_dir, exist_ok=True)
    makedirs(figures_dir, exist_ok=True)
    model_identifier = MODEL_TYPE.name

    file_suffix = f"({model_identifier}) CAT={this_linguistic_cat}"

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
    logger.info(f"Sensorimotor TTFAs *= {ratio:.4f}")
    main_data[TTFA_SENSORIMOTOR_SCALED] = main_data[TTFA_SENSORIMOTOR] * ratio

    # endregion -------------------

    # region Combined model columns

    main_data[TTFA_COMBINED] = main_data[[TTFA_LINGUISTIC, TTFA_SENSORIMOTOR_SCALED]].min(axis=1)

    # endregion -------------------

    # region Find TTFA cut-off for best fit with participant data

    max_ttfa = int(ceil(max(main_data[TTFA_LINGUISTIC].max(), main_data[TTFA_SENSORIMOTOR_SCALED].max())))

    combined_hitrates_rmr, combined_hitrates_rpf = [], []
    for ttfa_cutoff in range(max_ttfa + 1):
        cutoff_data = apply_cutoff(main_data, TTFA_COMBINED, ttfa_cutoff)
        hrs_rpf, hrs_rmr = get_hitrate_summary_tables(cutoff_data, MODEL_TYPE)
        combined_hitrates_rmr.append(frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=True))
        combined_hitrates_rpf.append(frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=True))

    # Convert to arrays so I can do quick argmax
    combined_hitrates_rmr = array(combined_hitrates_rmr)
    combined_hitrates_rpf = array(combined_hitrates_rpf)

    # endregion -------------------

    # region Graph cutoff-by-fit

    # Save values (ignore erroneous inferred type check errors)
    # noinspection PyTypeChecker
    savetxt(path.join(evaluation_save_dir, "rmr cutoff.csv"), combined_hitrates_rmr, delimiter=",")
    # noinspection PyTypeChecker
    savetxt(path.join(evaluation_save_dir, "rpf cutoff.csv"), combined_hitrates_rpf, delimiter=",")

    # RMR graph
    pyplot.plot(combined_hitrates_rmr)
    pyplot.ylim((0, 1))
    pyplot.xlabel("TTFA cutoff")
    pyplot.ylabel("Fraction of hitrates within 1SD of participant mean")
    pyplot.title("Noninteractive combined fits (RMR)")
    pyplot.savefig(path.join(figures_dir, "rmr fits by cutoff.png"))
    pyplot.clf()
    pyplot.cla()
    pyplot.close()

    # RPF graph
    pyplot.plot(combined_hitrates_rpf)
    pyplot.ylim((0, 1))
    pyplot.xlabel("TTFA cutoff")
    pyplot.ylabel("Fraction of hitrates within 1SD of participant mean")
    pyplot.title("Noninteractive combined fits (RPF)")
    pyplot.savefig(path.join(figures_dir, "rpf fits by cutoff.png"))
    pyplot.clf()
    pyplot.cla()
    pyplot.close()

    # endregion -----------------

    # region Compute cut-off points

    # Optimum cutoffs for each stat
    combined_rmr_ttfa_cutoff = combined_hitrates_rmr.argmax()
    combined_rpf_ttfa_cutoff = combined_hitrates_rpf.argmax()

    # Find the first point at which the two stats become equal (or flipped)
    # Start at 1 to skip the first point where everything may be 0
    start = 1
    rmr_is_smaller = combined_hitrates_rmr[start] < combined_hitrates_rpf[start]
    balanced_cut_off = 0  # shouldn't be necessary, but it keeps the static analysis happy
    for balanced_cut_off, (hr_rmr, hr_rpf) in enumerate(zip(combined_hitrates_rmr, combined_hitrates_rpf)):
        if balanced_cut_off < start:
            continue
        if (hr_rmr < hr_rpf) != rmr_is_smaller:
            break

    # endregion -----------------

    # region Save optimal graphs

    # Combined (balanced)
    cutoff_data = apply_cutoff(main_data, TTFA_COMBINED, balanced_cut_off)
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(cutoff_data, MODEL_TYPE)
    save_hitrate_graphs(hrs_rpf, hrs_rmr, MODEL_TYPE, file_suffix + f" balanced combined ({balanced_cut_off})", figures_dir=figures_dir)
    logger.info(f"combined ({balanced_cut_off}) rmr fit: {frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")
    logger.info(f"combined ({balanced_cut_off}) rpf fit: {frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")

    # Combined (rmr-optimal)
    cutoff_data = apply_cutoff(main_data, TTFA_COMBINED, combined_rmr_ttfa_cutoff)
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(cutoff_data, MODEL_TYPE)
    save_hitrate_graphs(hrs_rpf, hrs_rmr, MODEL_TYPE, file_suffix + f" rmr-optimal combined ({combined_rmr_ttfa_cutoff})", figures_dir=figures_dir)
    logger.info(f"rmr-optimal ({combined_rmr_ttfa_cutoff}) rmr fit: {frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")
    logger.info(f"rmr-optimal ({combined_rmr_ttfa_cutoff}) rpf fit: {frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")

    # Combined (rpf-optimal)
    cutoff_data = apply_cutoff(main_data, TTFA_COMBINED, combined_rpf_ttfa_cutoff)
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(cutoff_data, MODEL_TYPE)
    save_hitrate_graphs(hrs_rpf, hrs_rmr, MODEL_TYPE, file_suffix + f" rpf-optimal combined ({combined_rpf_ttfa_cutoff})", figures_dir=figures_dir)
    logger.info(f"rpf-optimal ({combined_rpf_ttfa_cutoff}) rmr fit: {frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")
    logger.info(f"rpf-optimal ({combined_rpf_ttfa_cutoff}) rpf fit: {frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")

    # endregion -----------------

    # region Apply cutoff to individual components

    # balanced
    cutoff_data = apply_cutoff(main_data, TTFA_LINGUISTIC, balanced_cut_off)
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(cutoff_data, MODEL_TYPE)
    save_hitrate_graphs(hrs_rpf, hrs_rmr, MODEL_TYPE, file_suffix + f" balanced linguistic ({balanced_cut_off})", figures_dir=figures_dir)
    logger.info(f"balanced ling ({balanced_cut_off}) rmr fit: {frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")
    logger.info(f"balanced ling ({balanced_cut_off}) rpf fit: {frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")

    cutoff_data = apply_cutoff(main_data, TTFA_SENSORIMOTOR_SCALED, balanced_cut_off)
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(cutoff_data, MODEL_TYPE)
    save_hitrate_graphs(hrs_rpf, hrs_rmr, MODEL_TYPE, file_suffix + f" balanced sensorimotor ({balanced_cut_off})", figures_dir=figures_dir)
    logger.info(f"balanced sm ({balanced_cut_off}) rmr fit: {frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")
    logger.info(f"balanced sm ({balanced_cut_off}) rpf fit: {frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")

    # rmr optimal
    cutoff_data = apply_cutoff(main_data, TTFA_LINGUISTIC, combined_rmr_ttfa_cutoff)
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(cutoff_data, MODEL_TYPE)
    save_hitrate_graphs(hrs_rpf, hrs_rmr, MODEL_TYPE, file_suffix + f" rmr-optimal linguistic ({combined_rmr_ttfa_cutoff})", figures_dir=figures_dir)
    logger.info(f"rmr-optimal ling ({combined_rmr_ttfa_cutoff}) rmr fit: {frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")
    logger.info(f"rmr-optimal ling ({combined_rmr_ttfa_cutoff}) rpf fit: {frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")

    cutoff_data = apply_cutoff(main_data, TTFA_SENSORIMOTOR_SCALED, combined_rmr_ttfa_cutoff)
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(cutoff_data, MODEL_TYPE)
    save_hitrate_graphs(hrs_rpf, hrs_rmr, MODEL_TYPE, file_suffix + f" rmr-optimal sensorimotor ({combined_rmr_ttfa_cutoff})", figures_dir=figures_dir)
    logger.info(f"rmr-optimal sm ({combined_rmr_ttfa_cutoff}) rmr fit: {frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")
    logger.info(f"rmr-optimal sm ({combined_rmr_ttfa_cutoff}) rpf fit: {frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")

    # rpf optimal
    cutoff_data = apply_cutoff(main_data, TTFA_LINGUISTIC, combined_rpf_ttfa_cutoff)
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(cutoff_data, MODEL_TYPE)
    save_hitrate_graphs(hrs_rpf, hrs_rmr, MODEL_TYPE, file_suffix + f" rpf-optimal linguistic ({combined_rpf_ttfa_cutoff})", figures_dir=figures_dir)
    logger.info(f"rpf-optimal ling ({combined_rpf_ttfa_cutoff}) rmr fit: {frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")
    logger.info(f"rpf-optimal ling ({combined_rpf_ttfa_cutoff}) rpf fit: {frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")

    cutoff_data = apply_cutoff(main_data, TTFA_SENSORIMOTOR_SCALED, combined_rpf_ttfa_cutoff)
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(cutoff_data, MODEL_TYPE)
    save_hitrate_graphs(hrs_rpf, hrs_rmr, MODEL_TYPE, file_suffix + f" rpf-optimal sensorimotor ({combined_rpf_ttfa_cutoff})", figures_dir=figures_dir)
    logger.info(f"rpf-optimal sm ({combined_rpf_ttfa_cutoff}) rmr fit: {frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")
    logger.info(f"rpf-optimal sm ({combined_rpf_ttfa_cutoff}) rpf fit: {frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")

    # endregion -------------------

    # region Participant hitrate %s

    # TODO: we're using cutoff_data here even though the cutting isn't required because get_summary_tables (called by
    #  get_hitrate_summary_tables) requires MODEL_HIT to be a column.  In an ideal world the addition of model hit/rate
    #  data would be separated from the application of participant hit/rate data.
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(cutoff_data, MODEL_TYPE)
    participant_hitrates_rmr = array([
        frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=PARTICIPANT_HITRATE_All_f.format(p), only_before_sd_includes_0=True)
        for p in CP.participants
    ])
    logger.info(f"Mean participant hitrate % (RMR) {participant_hitrates_rmr.mean()}\n{participant_hitrates_rmr} head only")
    participant_hitrates_rpf = array([
        frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=PARTICIPANT_HITRATE_All_f.format(p), only_before_sd_includes_0=True)
        for p in CP.participants
    ])
    logger.info(f"Mean participant hitrate % (RPF) {participant_hitrates_rpf.mean()}\n{participant_hitrates_rpf} head only")

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
