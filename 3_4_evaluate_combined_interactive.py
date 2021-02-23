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
from logging import getLogger, basicConfig, INFO
from pathlib import Path

from matplotlib import pyplot
from numpy import savetxt, array
from pandas import DataFrame

from framework.category_production.category_production import ColNames as CPColNames
from framework.cli.job import InteractiveCombinedJobSpec
from framework.cognitive_model.utils.maths import cm_to_inches
from framework.evaluation.category_production import add_ttfa_column, ModelType, \
    get_hitrate_summary_tables, frac_within_sd_of_hitrate_mean, prepare_category_production_data, \
    get_hitrate_variance, save_hitrate_graphs, apply_ttfa_cutoff, \
    CP_INSTANCE, save_hitrate_summary_tables, get_model_ttfas_and_components_for_category_combined_interactive, \
    add_component_column
from framework.evaluation.column_names import TTFA, MODEL_HITRATE, PARTICIPANT_HITRATE_All_f, PRODUCTION_PROPORTION, \
    RANKED_PRODUCTION_FREQUENCY, ROUNDED_MEAN_RANK, COMPONENT

logger = getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "1%Y-%m-%d %H:%M:%S"

# Paths
root_input_dir = Path("/Volumes/Big Data/spreading activation model/Model output/Category production")
root_output_dir = Path("/Volumes/Big Data/spreading activation model/Evaluation/Interactive")


def prepare_main_dataframe(spec: InteractiveCombinedJobSpec) -> DataFrame:
    from framework.cognitive_model.basic_types import Component

    # Main dataframe holds category production data and model response data
    main_data: DataFrame = prepare_category_production_data(ModelType.combined_interactive)

    logger.info("Loading model TTFAs")
    ttfa_dict = dict()
    components_dict = dict()
    for category in CP_INSTANCE.category_labels:
        ttfas, components = get_model_ttfas_and_components_for_category_combined_interactive(
            category=category,
            results_dir=Path(root_input_dir, spec.output_location_relative()),
            require_buffer_entry=True,
            require_activations_in_component=None,
        )

        ttfa_dict[category] = ttfas
    add_ttfa_column(main_data, model_type=ModelType.combined_interactive, ttfas=ttfa_dict)
    add_component_column(main_data, model_type=ModelType.combined_interactive, components=components_dict)

    return main_data


def save_participant_hitrates(data):
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(data, ModelType.combined_noninteractive)
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


def process_model_output(data: DataFrame, model_type: ModelType, cutoff: int, output_dir):
    cutoff_data = apply_ttfa_cutoff(data, ttfa_column=TTFA, ttfa_cutoff=cutoff)

    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(cutoff_data, model_type)
    save_hitrate_summary_tables(hrs_rmr, hrs_rpf, model_type,
                                file_suffix=f"{model_type.name} coregistered cutoff={cutoff}",
                                output_dir=output_dir)
    save_hitrate_graphs(hrs_rpf, hrs_rmr,
                        file_suffix=f" {model_type.name} cutoff={cutoff}",
                        figures_dir=output_dir)
    logger.info(f"Hitrate fits for {model_type.name} model")
    logger.info(f"cutoff={cutoff} rmr fit: {frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")
    logger.info(f"cutoff={cutoff} rpf fit: {frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")
    get_hitrate_variance(cutoff_data).to_csv(Path(output_dir, f"{model_type.name} hitrate variance.csv"),
                                             na_rep="")
    logger.info("Computing participant hitrates")
    save_participant_hitrates(cutoff_data)


def explore_ttfa_cutoffs(main_data, output_dir):

    combined_hitrates_rmr, combined_hitrates_rpf = [], []

    max_ttfa = main_data[TTFA].max()
    for ttfa_cutoff in range(max_ttfa + 1):
        logger.info(f"Testing cutoff at {ttfa_cutoff}")
        cutoff_data_rpf = apply_ttfa_cutoff(main_data, TTFA, ttfa_cutoff)
        hrs_rpf, hrs_rmr = get_hitrate_summary_tables(cutoff_data_rpf, ModelType.combined_noninteractive)
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
        apply_ttfa_cutoff(main_data, TTFA, combined_rmr_ttfa_cutoff),
        ModelType.combined_noninteractive)
    logger.info(f"rmr-optimal ({combined_rmr_ttfa_cutoff}) rmr fit: {frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")
    logger.info(f"rmr-optimal ({combined_rmr_ttfa_cutoff}) rpf fit: {frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")
    # Combined (rpf-optimal)
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(
        apply_ttfa_cutoff(main_data, TTFA, combined_rpf_ttfa_cutoff),
        ModelType.combined_noninteractive)
    logger.info(f"rpf-optimal ({combined_rpf_ttfa_cutoff}) rmr fit: {frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")
    logger.info(f"rpf-optimal ({combined_rpf_ttfa_cutoff}) rpf fit: {frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")


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


def main(spec: InteractiveCombinedJobSpec):

    main_data = prepare_main_dataframe(spec=spec)

    output_dir = Path(root_output_dir, spec.shorthand)
    output_dir.mkdir(exist_ok=True)

    # Process separate and one-hop models
    # process_model_output(main_data, model_type=ModelType.combined_interactive, cutoff=)

    explore_ttfa_cutoffs(main_data, output_dir)

    # Save final main dataframe
    with open(Path(output_dir, f"main model data.csv"), mode="w", encoding="utf-8") as main_data_output_file:
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

    for spec in InteractiveCombinedJobSpec.load_multiple(Path(Path(__file__).parent,
                                                              "job_specifications/job_interactive_testing.yaml")):
        main(spec=spec)

    logger.info("Done!")
