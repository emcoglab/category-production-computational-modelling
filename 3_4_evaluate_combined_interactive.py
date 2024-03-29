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
2021
---------------------------
"""

import sys
from copy import deepcopy
from logging import getLogger, basicConfig, INFO
from pathlib import Path
from typing import Optional

from matplotlib import pyplot
from numpy import savetxt, array
from pandas import DataFrame

from framework.category_production.category_production import ColNames as CPColNames
from framework.cli.job import InteractiveCombinedJobSpec
from framework.cognitive_model.utils.maths import cm_to_inches
from framework.evaluation.category_production import add_ttfa_column, ModelType, \
    get_hitrate_summary_tables, frac_within_sd_of_hitrate_mean, prepare_category_production_data, \
    save_hitrate_graphs, apply_ttfa_cutoff, CP_INSTANCE,\
    get_model_ttfas_and_components_for_category_combined_interactive, add_component_column
from framework.evaluation.column_names import TTFA, MODEL_HITRATE, PARTICIPANT_HITRATE_All_f, PRODUCTION_PROPORTION, \
    RANKED_PRODUCTION_FREQUENCY, ROUNDED_MEAN_RANK, COMPONENT

logger = getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "1%Y-%m-%d %H:%M:%S"

# Paths
root_input_dir = Path("/Volumes/Big Data/spreading activation model/Model output/Category production")


def prepare_main_dataframe(spec: InteractiveCombinedJobSpec, filter_events: Optional[str], accessible_set_hits: bool) -> DataFrame:
    from framework.cognitive_model.basic_types import Component

    # Main dataframe holds category production data and model response data
    main_data: DataFrame = prepare_category_production_data(ModelType.combined_interactive)

    logger.info("Loading model TTFAs")
    ttfa_dict = dict()
    components_dict = dict()
    for category in CP_INSTANCE.category_labels:
        results_dir = Path(root_input_dir, spec.output_location_relative())
        if filter_events is not None:
            results_dir = Path(results_dir.parent, results_dir.name + f" only {filter_events}")
        ttfas, components = get_model_ttfas_and_components_for_category_combined_interactive(
            category=category,
            results_dir=results_dir,
            require_buffer_entry=not accessible_set_hits,
            require_activations_in_component=Component.linguistic,
        )

        ttfa_dict[category] = ttfas
    add_ttfa_column(main_data, model_type=ModelType.combined_interactive, ttfas=ttfa_dict)
    add_component_column(main_data, model_type=ModelType.combined_interactive, components=components_dict)

    return main_data


def save_participant_hitrates(data):
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(data)
    for head_only in [True, False]:
        head_message = "head only" if head_only else "whole graph"
        participant_hitrates_rmr: array = array([
            frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=PARTICIPANT_HITRATE_All_f.format(p),
                                           only_before_sd_includes_0=head_only)
            for p in CP_INSTANCE.participants
        ])
        logger.info(
            f"Mean participant hitrate % (RMR) {participant_hitrates_rmr.mean()} (sd {participant_hitrates_rmr.std()}; "
            f"range {participant_hitrates_rmr.min()}–{participant_hitrates_rmr.max()}) {head_message}")
        participant_hitrates_rpf = array([
            frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=PARTICIPANT_HITRATE_All_f.format(p),
                                           only_before_sd_includes_0=head_only)
            for p in CP_INSTANCE.participants
        ])
        logger.info(
            f"Mean participant hitrate % (RPF) {participant_hitrates_rpf.mean()} (sd {participant_hitrates_rpf.std()}; "
            f"range {participant_hitrates_rpf.min()}–{participant_hitrates_rpf.max()}) {head_message}")


def explore_ttfa_cutoffs(main_data, output_dir):

    combined_hitrates_rmr, combined_hitrates_rpf = [], []

    max_ttfa = main_data[TTFA].max()
    for ttfa_cutoff in range(max_ttfa + 1):
        logger.info(f"Testing cutoff at {ttfa_cutoff}")
        cutoff_data_rpf = apply_ttfa_cutoff(main_data, TTFA, ttfa_cutoff)
        hrs_rpf, hrs_rmr = get_hitrate_summary_tables(cutoff_data_rpf)
        combined_hitrates_rmr.append(
            frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=True))
        combined_hitrates_rpf.append(
            frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=True))

    # Convert to arrays so I can do quick argmax
    combined_hitrates_rmr = array(combined_hitrates_rmr)
    combined_hitrates_rpf = array(combined_hitrates_rpf)

    # Save values
    # (and ignore erroneous inferred type check errors)
    # noinspection PyTypeChecker
    savetxt(Path(output_dir, "MR cutoff.csv"), combined_hitrates_rmr, delimiter=",")
    # noinspection PyTypeChecker
    savetxt(Path(output_dir, "PF cutoff.csv"), combined_hitrates_rpf, delimiter=",")
    # Optimum cutoffs for each stat
    combined_rmr_ttfa_cutoff = combined_hitrates_rmr.argmax()
    combined_rpf_ttfa_cutoff = combined_hitrates_rpf.argmax()

    # Graph cut-of-by-fit
    graph_cutoff_by_fit(combined_hitrates_rmr, combined_hitrates_rpf, output_dir)

    # Save optimal graphs

    # Combined (rmr-optimal)
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(
        apply_ttfa_cutoff(main_data, TTFA, combined_rmr_ttfa_cutoff))
    logger.info(f"rmr-optimal ({combined_rmr_ttfa_cutoff}) rmr fit: {frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")
    logger.info(f"rmr-optimal ({combined_rmr_ttfa_cutoff}) rpf fit: {frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")
    # Combined (rpf-optimal)
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(
        apply_ttfa_cutoff(main_data, TTFA, combined_rpf_ttfa_cutoff))
    logger.info(f"rpf-optimal ({combined_rpf_ttfa_cutoff}) rmr fit: {frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")
    logger.info(f"rpf-optimal ({combined_rpf_ttfa_cutoff}) rpf fit: {frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")


def fit_data_at_cutoff(main_data, output_dir, manual_cut_off: int):

    logger.info(f"Testing cutoff at {manual_cut_off}")

    cutoff_data = apply_ttfa_cutoff(main_data, TTFA, manual_cut_off)
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(cutoff_data)

    save_hitrate_graphs(hrs_rpf, hrs_rmr, file_suffix=f"({manual_cut_off})", figures_dir=output_dir)


def graph_cutoff_by_fit(combined_hitrates_rmr, combined_hitrates_rpf, output_dir):
    cutoff_graph_data = DataFrame()
    cutoff_graph_data["MR"] = combined_hitrates_rmr
    cutoff_graph_data["PF"] = combined_hitrates_rpf
    pyplot.plot(combined_hitrates_rmr, label="MR", zorder=10)
    pyplot.plot(combined_hitrates_rpf, label="PF", zorder=10)
    # pyplot.xlim((100, 800))
    pyplot.ylim((0, 1))
    pyplot.xlabel("TTFA cutoff")
    pyplot.ylabel("Fraction of hit rates within 1SD of participant mean")
    pyplot.title("Interactive combined fits")
    graph_filename = Path(output_dir, "rmr & rpf fits by cutoff.png")
    pyplot.legend()
    pyplot.gcf().set_size_inches(cm_to_inches(15), cm_to_inches(10))
    pyplot.savefig(graph_filename, dpi=600)
    pyplot.clf()
    pyplot.cla()
    pyplot.close()


def main(spec: InteractiveCombinedJobSpec, manual_cut_off: Optional[int] = None, filter_events: Optional[str] = None,
         accessible_set_hits: bool = False):

    main_data = prepare_main_dataframe(spec=spec, filter_events=filter_events, accessible_set_hits=accessible_set_hits)

    model_output_dir = Path(root_input_dir, spec.output_location_relative())
    if filter_events is not None:
        model_output_dir = Path(model_output_dir.parent, model_output_dir.name + f" only {filter_events}")
    if accessible_set_hits:
        evaluation_output_dir = Path(model_output_dir, " Evaluation (accessible set hits)")
    else:
        evaluation_output_dir = Path(model_output_dir, " Evaluation")
    evaluation_output_dir.mkdir(exist_ok=True)

    if manual_cut_off is None:
        explore_ttfa_cutoffs(main_data, evaluation_output_dir)
    else:
        fit_data_at_cutoff(main_data, evaluation_output_dir, manual_cut_off)

    # Save final main dataframe
    with open(Path(evaluation_output_dir, f"main model data.csv"), mode="w", encoding="utf-8") as main_data_output_file:
        main_data[[
            # Select only relevant columns for output
            CPColNames.Category,
            CPColNames.Response,
            CPColNames.CategorySensorimotor,
            CPColNames.ResponseSensorimotor,
            CPColNames.ProductionFrequency,
            CPColNames.MeanRank,
            CPColNames.FirstRankFrequency,
            CPColNames.MeanRT,
            CPColNames.MeanZRT,
            PRODUCTION_PROPORTION,
            RANKED_PRODUCTION_FREQUENCY,
            ROUNDED_MEAN_RANK,
            TTFA,
            COMPONENT,
        ]].to_csv(main_data_output_file, index=False)


if __name__ == '__main__':
    basicConfig(format=logger_format, datefmt=logger_dateformat, level=INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    loaded_specs = InteractiveCombinedJobSpec.load_multiple(
        Path(Path(__file__).parent,
             "job_specifications/2022-05-24 longer runs more ccas.yaml"))
    systematic_cca_test = True

    if systematic_cca_test:
        ccas = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
        specs = []
        s: InteractiveCombinedJobSpec
        for s in loaded_specs:
            for cca in ccas:
                spec = deepcopy(s)
                spec.cross_component_attenuation = cca
                specs.append(spec)
    else:
        specs = loaded_specs

    for spec in specs:
        main(spec=spec, filter_events="accessible_set", accessible_set_hits=False)

    logger.info("Done!")
