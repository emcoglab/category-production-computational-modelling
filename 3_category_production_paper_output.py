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

# TODO: This is temporary to create some differently formatted graphs. But eventually it should be refactored into an
#  "output" project, just like with LDM.

import argparse
from logging import getLogger, basicConfig, INFO
import sys
from pathlib import Path
from typing import Optional, Dict

from matplotlib import pyplot
from numpy import ceil, savetxt, array
from pandas import DataFrame

from category_production.category_production import ColNames as CPColNames, CategoryProduction
from evaluation.category_production import add_ttfa_column, ModelType, \
    get_model_ttfas_for_category_sensorimotor, get_hitrate_summary_tables, get_model_ttfas_for_category_linguistic, \
    get_n_words_from_path_linguistic, frac_within_sd_of_hitrate_mean, \
    get_firing_threshold_from_path_linguistic, prepare_category_production_data, get_hitrate_variance, \
    save_hitrate_graphs, add_model_hit_column, \
    process_one_model_output, apply_ttfa_cutoff
from evaluation.column_names import TTFA, MODEL_HITRATE, PARTICIPANT_HITRATE_All_f
from model.utils.maths import cm_to_inches
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms

logger = getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "1%Y-%m-%d %H:%M:%S"


SN = SensorimotorNorms()
CP = CategoryProduction()

# The number of members of a category to produce when computing TTFA scale ratios
N_MEMBERS_COREGISTRATION = 3

# Additional TTFA column names for parts of combined model
TTFA_LINGUISTIC          = f"{TTFA} linguistic"
TTFA_SENSORIMOTOR        = f"{TTFA} sensorimotor"
TTFA_SENSORIMOTOR_SCALED = f"{TTFA} sensorimotor scaled"
TTFA_COMBINED            = f"{TTFA} combined"
COMPONENT_ACTIVATED      = f"activated in component"

TTFA_COLUMNS_FOR_CUTOFF: Dict[ModelType, str] = {
    ModelType.sensorimotor:            TTFA_SENSORIMOTOR_SCALED,
    ModelType.linguistic:              TTFA_LINGUISTIC,
    ModelType.combined_noninteractive: TTFA_COMBINED,
}

# Paths
root_input_dir = Path("/Users/cai/Box Sync/LANGBOOT Project/Manuscripts/Category Production - Full Paper/Data/Model output")
root_output_dir = Path("/Users/cai/Box Sync/LANGBOOT Project/Manuscripts/Category Production - Full Paper/Data/Results")

input_dirs: Dict[ModelType, Path] = {
    ModelType.sensorimotor:         Path(root_input_dir, "Sensorimotor 0.7/Minkowski-3 length 100 attenuate Prevalence/max-r 150; n-decay-median 500.0; n-decay-sigma 0.9; as-θ 0.3; as-cap 3,000; buff-θ 0.7; buff-cap 10; run-for 10000; bail None"),
    ModelType.sensorimotor_one_hop: Path(root_input_dir, "Sensorimotor one-hop 0.7/Minkowski-3 length 100 attenuate Prevalence/max-r 150; n-decay-median 500.0; n-decay-sigma 0.9; as-θ 0.3; as-cap 3,000; buff-θ 0.7; buff-cap 10"),
    ModelType.linguistic:           Path(root_input_dir, "Linguistic 0.7/PPMI n-gram (BBC), r=5 40,000 words, length 10/firing-θ 0.9; n-decay-f 0.99; e-decay-sd 15.0; imp-prune-θ 0.05; run-for 3000; bail 20000"),
    ModelType.linguistic_one_hop:   Path(root_input_dir, "Linguistic one-hop 0.7/PPMI n-gram (BBC), r=5 40,000 words, length 10/firing-θ 0.9; n-decay-f 0.99; e-decay-sd 15.0; imp-prune-θ 0.05"),
}

output_dirs: Dict[ModelType, Path] = {
    ModelType.sensorimotor:            Path(root_output_dir, ModelType.sensorimotor.name),
    ModelType.sensorimotor_one_hop:    Path(root_output_dir, ModelType.sensorimotor_one_hop.name),
    ModelType.linguistic:              Path(root_output_dir, ModelType.linguistic.name),
    ModelType.linguistic_one_hop:      Path(root_output_dir, ModelType.linguistic_one_hop.name),
    ModelType.combined_noninteractive: Path(root_output_dir, ModelType.combined_noninteractive.name),
}


def prepare_main_dataframe() -> DataFrame:

    # Not using CAT; just use FT
    this_linguistic_cat = get_firing_threshold_from_path_linguistic(input_dirs[ModelType.linguistic])

    # Main dataframe holds category production data and model response data
    main_data: DataFrame = prepare_category_production_data(ModelType.combined_noninteractive)
    # Linguistic TTFAs
    n_words = get_n_words_from_path_linguistic(input_dirs[ModelType.linguistic])
    linguistic_ttfas = {
        category: get_model_ttfas_for_category_linguistic(category, input_dirs[ModelType.linguistic],
                                                          n_words, this_linguistic_cat)
        for category in CP.category_labels
    }
    add_ttfa_column(main_data, model_type=ModelType.linguistic, ttfas=linguistic_ttfas)
    main_data.rename(columns={TTFA: TTFA_LINGUISTIC}, inplace=True)

    # Sensorimotor TTFAs
    sensorimotor_ttfas = {
        category: get_model_ttfas_for_category_sensorimotor(category, input_dirs[ModelType.sensorimotor])
        for category in CP.category_labels_sensorimotor
    }
    add_ttfa_column(main_data, model_type=ModelType.sensorimotor, ttfas=sensorimotor_ttfas)
    main_data.rename(columns={TTFA: TTFA_SENSORIMOTOR}, inplace=True)

    return main_data


def apply_coregistration(main_data: DataFrame):
    mean_ttfa_linguistic = (
        main_data
            .dropna(subset=[TTFA_LINGUISTIC])
            .sort_values(by=TTFA_LINGUISTIC, ascending=True)
            .groupby(CPColNames.Category, sort=False)
            .head(N_MEMBERS_COREGISTRATION)
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
            .head(N_MEMBERS_COREGISTRATION)
            .groupby(CPColNames.CategorySensorimotor, sort=False)
            .max()
        [TTFA_SENSORIMOTOR]
            .mean()
    )

    # region Scale sensorimotor TTFAs to achieve 1:1 ratio

    ratio = mean_ttfa_linguistic / mean_ttfa_sensorimotor
    logger.info(f"Sensorimotor TTFAs *= {ratio:.4f}")
    main_data[TTFA_SENSORIMOTOR_SCALED] = main_data[TTFA_SENSORIMOTOR] * ratio


def combine_components(main_data):
    main_data[TTFA_COMBINED] = main_data[[TTFA_LINGUISTIC, TTFA_SENSORIMOTOR_SCALED]].min(axis=1)
    main_data[COMPONENT_ACTIVATED] = main_data[[TTFA_LINGUISTIC, TTFA_SENSORIMOTOR_SCALED]].idxmin(axis=1)


def process_original_model_output(data: DataFrame, model_type: ModelType):

    local_data = data.copy()

    if model_type in {ModelType.linguistic, ModelType.linguistic_one_hop}:
        n_words = get_n_words_from_path_linguistic(input_dirs[model_type])
        ttfas = {
            category: get_model_ttfas_for_category_linguistic(
                category, input_dirs[model_type], n_words,
                conscious_access_threshold=get_firing_threshold_from_path_linguistic(input_dirs[model_type]))
            for category in CP.category_labels
        }
    elif model_type in {ModelType.sensorimotor, ModelType.sensorimotor_one_hop}:
        ttfas = {
            category: get_model_ttfas_for_category_sensorimotor(category, input_dirs[model_type])
            for category in CP.category_labels_sensorimotor
        }
    else:
        raise NotImplementedError()

    add_ttfa_column(local_data, model_type=model_type, ttfas=ttfas)
    add_model_hit_column(local_data)

    process_one_model_output(main_data=local_data,
                             model_type=model_type,
                             input_results_dir=input_dirs[model_type],
                             min_first_rank_freq=None,
                             conscious_access_threshold=None,
                             manual_ttfa_cutoff=None,
                             stats_save_path=output_dirs[model_type],
                             figures_save_path=output_dirs[model_type])


def process_coregistered_model_output(data: DataFrame, model_type: ModelType, cutoff: int):
    cutoff_data = apply_ttfa_cutoff(data,
                                    ttfa_column=TTFA_COLUMNS_FOR_CUTOFF[model_type],
                                    ttfa_cutoff=cutoff)
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(cutoff_data, model_type)
    save_hitrate_graphs(hrs_rpf, hrs_rmr,
                        file_suffix=f" {model_type.name} cutoff={cutoff}",
                        figures_dir=output_dirs[model_type])
    logger.info(f"cutoff={cutoff} rmr fit: {frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")
    logger.info(f"cutoff={cutoff} rpf fit: {frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")
    get_hitrate_variance(cutoff_data).to_csv(Path(output_dirs[model_type], "linguistic hitrate variance.csv"),
                                             na_rep="")
    save_participant_hitrates(cutoff_data)


def explore_ttfa_cutoffs(main_data):

    combined_hitrates_rmr, combined_hitrates_rpf = [], []

    max_ttfa = int(ceil(max(main_data[TTFA_LINGUISTIC].max(), main_data[TTFA_SENSORIMOTOR_SCALED].max())))
    for ttfa_cutoff in range(max_ttfa + 1):
        logger.info(f"Testing cutoff at {ttfa_cutoff}")
        cutoff_data_rpf = apply_ttfa_cutoff(main_data, TTFA_COMBINED, ttfa_cutoff)
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
        apply_ttfa_cutoff(main_data, TTFA_COMBINED, combined_rmr_ttfa_cutoff),
        ModelType.combined_noninteractive)
    logger.info(
        f"rmr-optimal ({combined_rmr_ttfa_cutoff}) rmr fit: {frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")
    logger.info(
        f"rmr-optimal ({combined_rmr_ttfa_cutoff}) rpf fit: {frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")
    # Combined (rpf-optimal)
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(
        apply_ttfa_cutoff(main_data, TTFA_COMBINED, combined_rpf_ttfa_cutoff),
        ModelType.combined_noninteractive)
    logger.info(
        f"rpf-optimal ({combined_rpf_ttfa_cutoff}) rmr fit: {frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")
    logger.info(
        f"rpf-optimal ({combined_rpf_ttfa_cutoff}) rpf fit: {frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")


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


def save_participant_hitrates(data):
    hrs_rpf, hrs_rmr = get_hitrate_summary_tables(data, ModelType.combined_noninteractive)
    participant_hitrates_rmr = array([
        frac_within_sd_of_hitrate_mean(hrs_rmr, test_column=PARTICIPANT_HITRATE_All_f.format(p),
                                       only_before_sd_includes_0=True)
        for p in CP.participants
    ])
    logger.info(f"Mean participant hitrate % (RMR) {participant_hitrates_rmr.mean()} (range {participant_hitrates_rmr.min()}–{participant_hitrates_rmr.max()}) head only")
    participant_hitrates_rpf = array([
        frac_within_sd_of_hitrate_mean(hrs_rpf, test_column=PARTICIPANT_HITRATE_All_f.format(p),
                                       only_before_sd_includes_0=True)
        for p in CP.participants
    ])
    logger.info(f"Mean participant hitrate % (RPF) {participant_hitrates_rpf.mean()} (range {participant_hitrates_rpf.min()}–{participant_hitrates_rpf.max()}) head only")


def main(manual_ttfa_cutoff: Optional[int] = None):

    main_data = prepare_main_dataframe()

    apply_coregistration(main_data)

    combine_components(main_data)

    # Process separate and one-hop models
    for model_type in [ModelType.sensorimotor, ModelType.sensorimotor_one_hop,
                       ModelType.linguistic, ModelType.linguistic_one_hop]:
        process_original_model_output(main_data, model_type=model_type)

    if manual_ttfa_cutoff is None:
        explore_ttfa_cutoffs(main_data)

    else:
        # Cut-offs only apply to separate and combined components
        for model_type in [ModelType.sensorimotor, ModelType.linguistic,
                           ModelType.combined_noninteractive]:
            # Apply cutoffs to joint and original data
            process_coregistered_model_output(main_data, model_type=model_type, cutoff=manual_ttfa_cutoff)


if __name__ == '__main__':
    basicConfig(format=logger_format, datefmt=logger_dateformat, level=INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-cut-off", type=int, default=None)
    args = parser.parse_args()

    main(args.manual_cut_off)

    logger.info("Done!")
