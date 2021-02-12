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
import sys
from logging import getLogger, basicConfig, INFO
from pathlib import Path
from typing import Dict

from matplotlib import pyplot
from numpy import savetxt, array
from pandas import DataFrame

from framework.category_production.category_production import ColNames as CPColNames
from framework.cli.job import InteractiveCombinedJobSpec
from framework.cognitive_model.utils.maths import cm_to_inches
from framework.evaluation.category_production import add_ttfa_column, ModelType, \
    get_hitrate_summary_tables, frac_within_sd_of_hitrate_mean, prepare_category_production_data, \
    get_hitrate_variance, save_hitrate_graphs, apply_ttfa_cutoff, \
    CP_INSTANCE, save_hitrate_summary_tables, get_model_ttfas_for_category_combined_interactive
from framework.evaluation.column_names import TTFA, MODEL_HITRATE, PARTICIPANT_HITRATE_All_f, PRODUCTION_PROPORTION, \
    RANKED_PRODUCTION_FREQUENCY, ROUNDED_MEAN_RANK

logger = getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "1%Y-%m-%d %H:%M:%S"


# Additional TTFA column names for parts of combined model
COMPONENT_ACTIVATED      = f"Activated in component"

# Paths
root_input_dir = Path("/Users/caiwingfield/Box Sync/LANGBOOT Project/Manuscripts/Category Production - Full Paper/Modelling data and results/Model output per category")
root_output_dir = Path("/Users/caiwingfield/Box Sync/LANGBOOT Project/Manuscripts/Category Production - Full Paper/Modelling data and results/Model hitrates")

input_dirs: Dict[ModelType, Path] = {
    ModelType.sensorimotor:         Path(root_input_dir, "Sensorimotor 0.9.6/Minkowski-3 length 100 att Prevalence; max-r 1.5; n-decay-m 5.0; n-decay-σ 0.9; as-θ 0.15; as-cap 3,000; buff-θ 0.35; buff-cap 10; run-for 2000; bail None"),
    ModelType.sensorimotor_one_hop: Path(root_input_dir, "Sensorimotor one-hop 0.9.6/Minkowski-3 length 100 att Prevalence; max-r 1.5; n-decay-m 5.0; n-decay-σ 0.9; as-θ 0.15; as-cap 3,000; buff-θ 0.35; buff-cap 10; run-for None; bail None"),
    ModelType.linguistic:           Path(root_input_dir, "Linguistic 0.9.6/PPMI n-gram (BBC), r=5 40,000 words, length 10; firing-θ 0.45; n-decay-f 0.99; e-decay-sd 15.0; as-θ 0.0; as-cap None; imp-prune-θ 0.05; run-for 2000; bail 20000"),
    ModelType.linguistic_one_hop:   Path(root_input_dir, "Linguistic one-hop 0.9.6/PPMI n-gram (BBC), r=5 40,000 words, length 10; firing-θ 0.45; n-decay-f 0.99; e-decay-sd 15.0; as-θ 0.0; as-cap None; imp-prune-θ 0.05; run-for None; bail None"),
}

output_dirs: Dict[ModelType, Path] = {
    ModelType.sensorimotor:            Path(root_output_dir, ModelType.sensorimotor.name),
    ModelType.sensorimotor_one_hop:    Path(root_output_dir, ModelType.sensorimotor_one_hop.name),
    ModelType.linguistic:              Path(root_output_dir, ModelType.linguistic.name),
    ModelType.linguistic_one_hop:      Path(root_output_dir, ModelType.linguistic_one_hop.name),
    ModelType.combined_noninteractive: Path(root_output_dir, ModelType.combined_noninteractive.name),
}


def prepare_main_dataframe(spec: InteractiveCombinedJobSpec) -> DataFrame:

    input_dir = spec.output_location_relative()

    # Main dataframe holds category production data and model response data
    main_data: DataFrame = prepare_category_production_data(ModelType.combined_interactive)

    # TODO: get params from spec, then build the path from that using the same code as the model itself, rather than
    #  trying to reverse-engineer it

    # Linguistic TTFAs
    n_words = spec.linguistic_spec.n_words

    ttfas = {
        category: get_model_ttfas_for_category_combined_interactive(category=category, results_dir=root_input_dir)
        for category in CP_INSTANCE.category_labels
    }
    add_ttfa_column(main_data, model_type=ModelType.combined_interactive, ttfas=ttfas)

    return main_data


def process_model_output(data: DataFrame, model_type: ModelType, cutoff: int):
    cutoff_data = apply_ttfa_cutoff(data, ttfa_column=TTFA, ttfa_cutoff=cutoff)

    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(cutoff_data, model_type)
    save_hitrate_summary_tables(hrs_rmr, hrs_rpf, model_type,
                                file_suffix=f"{model_type.name} coregistered cutoff={cutoff}",
                                output_dir=output_dirs[model_type])
    save_hitrate_graphs(hrs_rpf, hrs_rmr,
                        file_suffix=f" {model_type.name} cutoff={cutoff}",
                        figures_dir=output_dirs[model_type])
    logger.info(f"Hitrate fits for {model_type.name} model")
    logger.info(f"cutoff={cutoff} rmr fit: {frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")
    logger.info(f"cutoff={cutoff} rpf fit: {frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")
    get_hitrate_variance(cutoff_data).to_csv(Path(output_dirs[model_type], f"{model_type.name} hitrate variance.csv"),
                                             na_rep="")
    logger.info("Computing participant hitrates")
    save_participant_hitrates(cutoff_data)


def save_participant_hitrates(data):
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(data, ModelType.combined_noninteractive)
    for head_only in [True, False]:
        head_message = "head only" if head_only else "whole graph"
        participant_hitrates_rmr = array([
            frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=PARTICIPANT_HITRATE_All_f.format(p),
                                           only_before_sd_includes_0=head_only)
            for p in CP_INSTANCE.participants
        ])
        logger.info(
            f"Mean participant hitrate % (RMR) {participant_hitrates_rmr.mean()} (sd {participant_hitrates_rmr.std()}; range {participant_hitrates_rmr.min()}–{participant_hitrates_rmr.max()}) {head_message}")
        participant_hitrates_rpf = array([
            frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=PARTICIPANT_HITRATE_All_f.format(p),
                                           only_before_sd_includes_0=head_only)
            for p in CP_INSTANCE.participants
        ])
        logger.info(
            f"Mean participant hitrate % (RPF) {participant_hitrates_rpf.mean()} (sd {participant_hitrates_rpf.std()}; range {participant_hitrates_rpf.min()}–{participant_hitrates_rpf.max()}) {head_message}")


def explore_ttfa_cutoffs(main_data):

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
    savetxt(Path(root_output_dir, "MR cutoff.csv"), combined_hitrates_rmr, delimiter=",")
    # noinspection PyTypeChecker
    savetxt(Path(root_output_dir, "PF cutoff.csv"), combined_hitrates_rpf, delimiter=",")
    # Optimum cutoffs for each stat
    combined_rmr_ttfa_cutoff = combined_hitrates_rmr.argmax()
    combined_rpf_ttfa_cutoff = combined_hitrates_rpf.argmax()

    # Graph cut-of-by-fit

    graph_cutoff_by_fit(combined_hitrates_rmr, combined_hitrates_rpf)

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


def graph_cutoff_by_fit(combined_hitrates_rmr, combined_hitrates_rpf):
    cutoff_graph_data = DataFrame()
    cutoff_graph_data["MR"] = combined_hitrates_rmr
    cutoff_graph_data["PF"] = combined_hitrates_rpf
    pyplot.plot(combined_hitrates_rmr, label="MR", zorder=10)
    pyplot.plot(combined_hitrates_rpf, label="PF", zorder=10)
    pyplot.xlim((100, 800))
    pyplot.ylim((0, 1))
    pyplot.xlabel("TTFA cutoff")
    pyplot.ylabel("Fraction of hit rates within 1SD of participant mean")
    pyplot.title("Noninteractive combined fits")
    graph_filename = Path(root_output_dir, "rmr & rpf fits by cutoff.png")
    pyplot.legend()
    pyplot.gcf().set_size_inches(cm_to_inches(15), cm_to_inches(10))
    pyplot.savefig(graph_filename, dpi=600)
    pyplot.clf()
    pyplot.cla()
    pyplot.close()


def main():

    main_data = prepare_main_dataframe()

    # Process separate and one-hop models
    process_model_output(main_data, model_type=ModelType.combined_interactive)

    explore_ttfa_cutoffs(main_data)

    # Save final main dataframe
    with open(Path(root_output_dir, "main model data.csv"), mode="w", encoding="utf-8") as main_data_output_file:
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
            COMPONENT_ACTIVATED,
        ]].to_csv(main_data_output_file, index=False)


if __name__ == '__main__':
    basicConfig(format=logger_format, datefmt=logger_dateformat, level=INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main()

    logger.info("Done!")
