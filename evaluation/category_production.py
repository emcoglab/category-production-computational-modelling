"""
===========================
Evaluating cognitive models against Category Production data.
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


import re
import logging
from collections import defaultdict
from enum import Enum, auto
from glob import glob
from math import floor
from os import path, listdir
from pathlib import Path
from typing import DefaultDict, Dict, Set, List, Optional

from matplotlib import pyplot
from numpy import nan
from pandas import DataFrame, read_csv, isna, Series

from category_production.category_production import CategoryProduction, ColNames as CPColNames
from ldm.corpus.tokenising import modified_word_tokenize
from model.graph_propagation import GraphPropagation
from model.basic_types import ActivationValue
from model.utils.exceptions import ParseError
from evaluation.column_names import *
from preferences import Preferences

logger = logging.getLogger(__name__)


_CP = CategoryProduction()

# Each participant saw 39 categories
CATEGORIES_PER_PARTICIPANT = 39
# Each category was seen by 20 participants
PARTICIPANTS_PER_CATEGORY = 20
# Participants were separated into 3 equipollent sets, each of which saw disjoint equipollent sets of categories.
PARTICIPANT_SETS = 3
TOTAL_PARTICIPANTS = PARTICIPANTS_PER_CATEGORY * PARTICIPANT_SETS
TOTAL_CATEGORIES = CATEGORIES_PER_PARTICIPANT * PARTICIPANT_SETS
assert (TOTAL_CATEGORIES == len(_CP.category_labels))


class ModelType(Enum):
    """Represents the type of model being used in the comparison."""
    # Linguistic models
    linguistic = auto()
    linguistic_distance_only = auto()
    linguistic_one_hop = auto()
    # Sensorimotor models
    sensorimotor = auto()
    sensorimotor_distance_only = auto()
    sensorimotor_one_hop = auto()
    # Combined models
    combined_noninteractive = auto()
    combined_full_tandem = auto()

    @property
    def name(self) -> str:
        if self == ModelType.sensorimotor_distance_only:
            return "sensorimotor distance-only"
        elif self == ModelType.linguistic_distance_only:
            return "linguistic distance-only"
        elif self == ModelType.linguistic_one_hop:
            return "linguistic one-hop"
        elif self == ModelType.sensorimotor_one_hop:
            return "sensorimotor one-hop"
        else:
            return super(ModelType, self).name.replace("_", " ")

    @property
    def figures_dirname(self) -> str:
        return f"hitrates {self.name}"

    @property
    def model_output_dirname(self) -> str:
        return f"Category production fit {self.name}"


class ParticipantSummaryType(Enum):
    """Represents a way to summarise participant data."""
    # Individual hitrates of participants (all responses)
    individual_hitrates = auto()
    # mean and sd of hitrates over categories
    hitrates_mean_sd    = auto()


def get_n_words_from_path_linguistic(results_dir_path: str) -> int:
    """
    Gets the number of words from a path storing results.
    :param results_dir_path:
    :return: n_words: int
    """
    dir_name = path.basename(path.dirname(results_dir_path))
    words_match = re.match(re.compile(r".* (?P<n_words>[0-9,]+) words,.*"), dir_name)
    if words_match:
        # remove the comma and parse as int
        n_words = int(words_match.group("n_words").replace(",", ""))
        return n_words
    else:
        raise ParseError(f"Could not parse number of words from {dir_name}.")


def get_firing_threshold_from_path_linguistic(results_dir_path: str) -> ActivationValue:
    """
    Gets the firing threshold from a path storing the results.
    :param results_dir_path:
    :return: firing_threshold: ActivationValue
    """
    dir_name = path.basename(results_dir_path)
    ft_match = re.match(re.compile(r"firing-θ (?P<firing_threshold>[0-9.]+);"), dir_name)
    if ft_match:
        ft = ActivationValue(ft_match.group("firing_threshold"))
        return ft
    else:
        raise ParseError(f"Could not parse firing threshold from {dir_name}.")


def available_categories(results_dir_path: str) -> List[str]:
    """
    Gets the list of available categories from a path storing results.
    A category is available iff there is a results file for it.
    """
    response_files = listdir(results_dir_path)
    category_name_re = re.compile(r"responses_(?P<category_name>[a-z ]+)(_.+)?\.csv")
    categories = []
    for response_file in response_files:
        category_name_match = re.match(category_name_re, response_file)
        if category_name_match:
            categories.append(category_name_match.group("category_name"))
    return categories


def get_model_ttfas_for_category_linguistic(category: str,
                                            results_dir: str,
                                            n_words: int,
                                            conscious_access_threshold: float) -> DefaultDict[str, int]:
    """
    Dictionary of
        response -> time to first activation
    for the specified category.

    DefaultDict gives nans where response not found

    :param category:
    :param results_dir:
    :param n_words:
    :param conscious_access_threshold:
    :return:
    """

    # Try to load model response
    try:
        model_responses_path = path.join(results_dir, f"responses_{category}_{n_words:,}.csv")
        with open(model_responses_path, mode="r", encoding="utf-8") as model_responses_file:
            model_responses: DataFrame = read_csv(model_responses_file, header=0, comment="#", index_col=False)

        consciously_active_data = model_responses[model_responses[ACTIVATION] >= conscious_access_threshold]\
            .sort_values(by=TICK_ON_WHICH_ACTIVATED)

        ttfas = consciously_active_data\
            .groupby(RESPONSE)\
            .first()[[TICK_ON_WHICH_ACTIVATED]]\
            .to_dict('dict')[TICK_ON_WHICH_ACTIVATED]

        return defaultdict(lambda: nan, ttfas)

    # If the category wasn't found, there are no TTFAs
    except FileNotFoundError:
        return defaultdict(lambda: nan)


def get_model_ttfas_for_category_sensorimotor(category: str, results_dir: str) -> Dict[str, int]:
    """
    Dictionary of
        response -> time to first activation
    for the specified category.

    DefaultDict gives nans where response not found

    :param category:
    :param results_dir:
    :return:
    """

    # Try to load model response
    try:
        model_responses_path = path.join(results_dir, f"responses_{category}.csv")
        with open(model_responses_path, mode="r", encoding="utf-8") as model_responses_file:
            # TODO: should have unified csv i/o code to ensure datatypes are kept consistent
            #  this doesn't actually affect the results, but it would make text files smaller and cleaner to skip the
            #  ".00000"s on fields which should be ints, etc.
            model_responses: DataFrame = read_csv(model_responses_file, header=0, comment="#", index_col=False,
                                                  dtype={ITEM_ENTERED_BUFFER: bool, TICK_ON_WHICH_ACTIVATED: int})

        in_buffer_data = model_responses[model_responses[ITEM_ENTERED_BUFFER] == True] \
            .sort_values(by=TICK_ON_WHICH_ACTIVATED)

        ttfas: Dict = in_buffer_data \
            .groupby(RESPONSE) \
            .first()[[TICK_ON_WHICH_ACTIVATED]] \
            .to_dict('dict')[TICK_ON_WHICH_ACTIVATED]

        return defaultdict(lambda: nan, ttfas)

    # If the category wasn't found, there are no TTFAs
    except FileNotFoundError:
        return defaultdict(lambda: nan)


def get_model_unique_responses_sensorimotor(category: str, results_dir: str) -> Set[str]:
    """
    Set of unique responses for the specified category.
    """

    # Try to load model response
    try:
        model_responses_path = path.join(
            results_dir,
            f"responses_{category}.csv")
        with open(model_responses_path, mode="r", encoding="utf-8") as model_responses_file:
            return set(row[RESPONSE]
                       for _i, row in read_csv(model_responses_file, header=0, comment="#", index_col=False).iterrows())

    # If the category wasn't found, there are no responses
    except FileNotFoundError:
        return set()


def exclude_idiosyncratic_responses(main_data) -> DataFrame:
    return main_data[main_data[CPColNames.ProductionFrequency] > 1]


def add_predictor_column_production_proportion(main_data):
    """Mutates `main_data`."""
    logger.info("Adding production proportion column")
    # Production proportion is the number of times a response was given,
    # divided by the number of participants who gave it
    main_data[PRODUCTION_PROPORTION] = main_data.apply(lambda row: row[CPColNames.ProductionFrequency] / PARTICIPANTS_PER_CATEGORY, axis=1)


def category_response_col_names_for_model_type(model_type):
    if model_type in [ModelType.linguistic, ModelType.linguistic_one_hop, ModelType.linguistic_distance_only]:
        c, r = CPColNames.Category, CPColNames.Response
    elif model_type in [ModelType.sensorimotor, ModelType.sensorimotor_one_hop, ModelType.sensorimotor_distance_only]:
        c, r = CPColNames.CategorySensorimotor, CPColNames.ResponseSensorimotor
    elif model_type == ModelType.combined_noninteractive:
        # We could use either here
        c, r = CPColNames.Category, CPColNames.Response
    else:
        raise NotImplementedError()
    return c, r


def add_rpf_column(main_data: DataFrame, model_type: ModelType):
    """Mutates `main_data`."""
    logger.info("Adding RPF column")
    specific_category_column, _ = category_response_col_names_for_model_type(model_type)
    main_data[RANKED_PRODUCTION_FREQUENCY] = (
        main_data
        # Within each category
        .groupby(specific_category_column)
        # Rank the responses according to production frequency
        [CPColNames.ProductionFrequency]
        # Descending, so largest production frequency is lowest rank
        .rank(ascending=False,
              # For ties, order alphabetically (i.e. pseudorandomly (?))
              method='first')
    )


def add_rmr_column(main_data):
    """Mutates `main_data`."""
    logger.info("Adding RMR column")
    main_data[ROUNDED_MEAN_RANK] = main_data.apply(lambda row: floor(row[CPColNames.MeanRank]), axis=1)


def add_ttfa_column(main_data, ttfas: Dict[str, Dict[str, int]], model_type: ModelType):
    """Mutates `main_data`."""
    # TODO: this function's signature is a bit of a mess... the ttfas dict should probably be built in here
    logger.info("Adding TTFA column")

    def get_min_ttfa_for_multiword_responses(row) -> int:
        """
        Helper function to convert a row in the output into a ttfa when the response is formed either of a single or
        multiple norms terms.
        """

        category_column, response_column = category_response_col_names_for_model_type(model_type)
        c, r = row[category_column], row[response_column]

        # If the category is not found in the dictionary, it was not accessed by the model so no TTFA will be present
        # for any response.
        try:
            # response -> TTFA
            c_ttfas: Dict[str, int] = ttfas[c]
        except KeyError:
            return nan

        # If the response was directly found, we can return it
        if r in c_ttfas:
            return c_ttfas[r]

        # Otherwise, try to break the response into components and find any one of them
        else:
            r_ttfas = [c_ttfas[w]
                       for w in modified_word_tokenize(r)
                       if (w not in _CP.ignored_words)
                       and (w in c_ttfas)]

            # The multi-word response is said to be activated the first time any one of its constituent words are
            if len(r_ttfas) > 1:
                return min(r_ttfas)
            # If none of the constituent words have a ttfa, we have no ttfa for the multiword term
            else:
                return nan

    main_data[TTFA] = main_data.apply(get_min_ttfa_for_multiword_responses, axis=1)


def add_model_hit_column(main_data):
    """Mutates `main_data`."""
    logger.info("Adding model hit column")
    main_data[MODEL_HIT] = main_data.apply(lambda row: not isna(row[TTFA]), axis=1)


def save_item_level_data(main_data: DataFrame, save_path):
    main_data.to_csv(save_path, index=False)


def save_hitrate_summary_figure(summary_table, x_selector, fig_title, fig_name,
                                model_type: ModelType, summarise_participants_by: ParticipantSummaryType):
    """Save a summary table as a figure."""

    # add participant bounds
    if summarise_participants_by == ParticipantSummaryType.individual_hitrates:
        for participant in _CP.participants:
            pyplot.plot(summary_table.reset_index()[x_selector],
                        summary_table[PARTICIPANT_HITRATE_All_f.format(participant)],
                        linewidth=0.4, linestyle="-", color="b", alpha=0.4)
            pyplot.ylabel("hitrate")
    elif summarise_participants_by == ParticipantSummaryType.hitrates_mean_sd:
        pyplot.fill_between(x=summary_table.reset_index()[x_selector],
                            y1=summary_table['Hitrate Mean'] - summary_table[
                                'Hitrate SD'],
                            y2=summary_table['Hitrate Mean'] + summary_table[
                                'Hitrate SD'])
        pyplot.scatter(x=summary_table.reset_index()[x_selector],
                       y=summary_table['Hitrate Mean'])
        pyplot.ylabel("hitrate")
    else:
        raise NotImplementedError()

    # add model performance
    pyplot.scatter(x=summary_table.reset_index()[x_selector],
                   y=summary_table[MODEL_HITRATE],
                   marker="o", color="g")

    pyplot.ylim((0, None))

    pyplot.title(fig_title)
    pyplot.xlabel(x_selector)

    if summarise_participants_by == ParticipantSummaryType.individual_hitrates:
        filename = f"{fig_name} traces.png"
    elif summarise_participants_by == ParticipantSummaryType.hitrates_mean_sd:
        filename = f"{fig_name} hitrate-sd.png"
    else:
        raise NotImplementedError()

    pyplot.savefig(
        path.join(Preferences.figures_dir,
                  model_type.figures_dirname,
                  filename))

    pyplot.clf()
    pyplot.cla()
    pyplot.close()


def get_hitrate_summary_tables(main_data: DataFrame, model_type: ModelType):

    hitrates_per_rpf = get_summary_table(main_data, RANKED_PRODUCTION_FREQUENCY)
    category_column, response_column = category_response_col_names_for_model_type(model_type)
    # For RPF we will truncate the table at mean + 2SD (over categories) items
    n_items_mean = (
        main_data[[category_column, response_column]]
        .groupby(category_column)
        .count()[response_column]
        .mean())
    n_items_sd = (
        main_data[[category_column, response_column]]
        .groupby(category_column)
        .count()[response_column]
        .std())
    hitrates_per_rpf = hitrates_per_rpf[
        hitrates_per_rpf[RANKED_PRODUCTION_FREQUENCY] < n_items_mean + (2 * n_items_sd)]

    hitrates_per_rmr = get_summary_table(main_data, ROUNDED_MEAN_RANK)

    return hitrates_per_rpf, hitrates_per_rmr


def save_hitrate_summary_tables(hitrates_per_rmr, hitrates_per_rpf, model_type, file_suffix):
    # Save summary tables
    base_dir = path.join(Preferences.results_dir, model_type.model_output_dirname)
    hitrates_per_rpf.to_csv(path.join(base_dir,
                                      f"Production proportion per rank frequency of production {file_suffix}.csv"),
                            index=False)
    hitrates_per_rmr.to_csv(path.join(base_dir,
                                      f"Production proportion per rounded mean rank {file_suffix}.csv"),
                            index=False)


def save_hitrate_graphs(hitrates_per_rpf, hitrates_per_rmr, model_type, file_suffix):

    # rpf sd region
    save_hitrate_summary_figure(summary_table=hitrates_per_rpf,
                                x_selector=RANKED_PRODUCTION_FREQUENCY,
                                fig_title="Hitrate per RPF",
                                fig_name=f"hitrate per RPF {file_suffix}",
                                model_type=model_type,
                                summarise_participants_by=ParticipantSummaryType.hitrates_mean_sd)
    # rpf traces
    save_hitrate_summary_figure(summary_table=hitrates_per_rpf,
                                x_selector=RANKED_PRODUCTION_FREQUENCY,
                                fig_title="Hitrate per RPF",
                                fig_name=f"hitrate per RPF {file_suffix}",
                                model_type=model_type,
                                summarise_participants_by=ParticipantSummaryType.individual_hitrates)
    # rmr sd region
    save_hitrate_summary_figure(summary_table=hitrates_per_rmr,
                                x_selector=ROUNDED_MEAN_RANK,
                                fig_title="Hitrate per RMR",
                                fig_name=f"hitrate per RMR {file_suffix}",
                                model_type=model_type,
                                summarise_participants_by=ParticipantSummaryType.hitrates_mean_sd)
    # rmr traces
    save_hitrate_summary_figure(summary_table=hitrates_per_rmr,
                                x_selector=ROUNDED_MEAN_RANK,
                                fig_title="Hitrate per RMR",
                                fig_name=f"hitrate per RMR {file_suffix}",
                                model_type=model_type,
                                summarise_participants_by=ParticipantSummaryType.individual_hitrates)


def process_one_model_output(main_data: DataFrame,
                             model_type: ModelType,
                             input_results_dir: str,
                             min_first_rank_freq: Optional[int],
                             conscious_access_threshold: Optional[float],
                             ):
    assert model_type in [ModelType.linguistic, ModelType.sensorimotor, ModelType.linguistic_one_hop, ModelType.sensorimotor_one_hop]
    input_results_path = Path(input_results_dir)
    model_identifier = f"{input_results_path.parent.name} {input_results_path.name}"
    save_item_level_data(main_data, path.join(Preferences.results_dir,
                                              model_type.model_output_dirname,
                                              f"item-level data ({model_identifier})"
                                              + (f" CAT={conscious_access_threshold}" if conscious_access_threshold is not None else "") +
                                              ".csv"))

    if conscious_access_threshold is not None:
        file_suffix = f"({model_identifier}) CAT={conscious_access_threshold}"
    else:
        file_suffix = f"({model_identifier})"

    hitrates_per_rpf, hitrates_per_rmr = get_hitrate_summary_tables(main_data, model_type)
    save_hitrate_summary_tables(hitrates_per_rmr, hitrates_per_rpf, model_type, file_suffix)

    # Compute hitrate fits
    # TODO: these names are whack
    hitrate_fit_rpf_hr = hitrate_within_sd_of_hitrate_mean_frac(hitrates_per_rpf)
    hitrate_fit_rmr_hr = hitrate_within_sd_of_hitrate_mean_frac(hitrates_per_rmr)

    drop_missing_data_to_add_types(main_data, {TTFA: int})

    save_model_performance_stats(
        main_data,
        model_identifier=model_identifier,
        results_dir=input_results_dir,
        min_first_rank_freq=min_first_rank_freq,
        hitrate_fit_rpf_hr=hitrate_fit_rpf_hr,
        hitrate_fit_rmr_hr=hitrate_fit_rmr_hr,
        model_type=model_type,
        conscious_access_threshold=conscious_access_threshold,
    )

    save_hitrate_graphs(hitrates_per_rpf, hitrates_per_rmr, model_type, file_suffix)


def process_one_model_output_distance_only(main_data: DataFrame,
                                           model_type: ModelType,
                                           input_results_dir: str,
                                           min_first_rank_freq: Optional[int],
                                           ):
    assert model_type in [ModelType.linguistic_distance_only, ModelType.sensorimotor_distance_only]
    input_results_path = Path(input_results_dir)
    model_identifier = f"{input_results_path.parent.name} {input_results_path.name}"
    output_dir = f"Category production fit {model_type.name}"
    save_item_level_data(main_data, path.join(Preferences.results_dir,
                                              output_dir,
                                              f"item-level data ({model_identifier}).csv"))

    file_suffix = f"({model_identifier})"

    hitrates_per_rpf, hitrates_per_rmr = get_hitrate_summary_tables(main_data, model_type)

    # Compute hitrate fits
    hitrate_fit_rpf_hr = hitrate_within_sd_of_hitrate_mean_frac(hitrates_per_rpf)
    hitrate_fit_rmr_hr = hitrate_within_sd_of_hitrate_mean_frac(hitrates_per_rmr)

    save_model_performance_stats(
        main_data,
        model_identifier=model_identifier,
        results_dir=input_results_dir,
        min_first_rank_freq=min_first_rank_freq,
        hitrate_fit_rpf_hr=hitrate_fit_rpf_hr,
        hitrate_fit_rmr_hr=hitrate_fit_rmr_hr,
        model_type=model_type,
        conscious_access_threshold=None,
    )

    save_hitrate_graphs(hitrates_per_rpf, hitrates_per_rmr, model_type, file_suffix)


def save_model_performance_stats(main_dataframe,
                                 results_dir,
                                 model_identifier: str,
                                 min_first_rank_freq: Optional[int],
                                 hitrate_fit_rpf_hr,
                                 hitrate_fit_rmr_hr,
                                 model_type: ModelType,
                                 conscious_access_threshold: Optional[float],
                                 ):

    # Build output dir
    if conscious_access_threshold is not None:
        filename_suffix = f" CAT={conscious_access_threshold}"
    else:
        filename_suffix = ""
    overall_stats_output_path = path.join(
        Preferences.results_dir,
        model_type.model_output_dirname,
        f"model_effectiveness_overall ({model_identifier}){filename_suffix}.csv")

    df_dict = dict()

    if model_type in [ModelType.linguistic, ModelType.sensorimotor]:
        # Only spreading-activation models have specs, and produce TTFAs from which correlation stats are generated
        df_dict.update(GraphPropagation.load_model_spec(results_dir))
        df_dict.update(get_correlation_stats(main_dataframe, min_first_rank_freq, model_type=model_type))

    df_dict.update({
        "Hitrate within SD of HR mean (RPF)": hitrate_fit_rpf_hr,
        "Hitrate within SD of HR mean (RMR)": hitrate_fit_rmr_hr,
    })

    if conscious_access_threshold is not None:
        df_dict[CAT] = conscious_access_threshold

    model_performance_data: DataFrame = DataFrame.from_records([df_dict])
    model_performance_data.to_csv(overall_stats_output_path,
                                  # As of Python 3.7, dictionary keys are ordered by insertion,
                                  # so we should automatically get a consistent order for stacking,
                                  columns=list(df_dict.keys()) + [CAT],
                                  index=False)


def get_correlation_stats(correlation_dataframe, min_first_rank_freq, model_type: ModelType):
    # frf vs ttfa
    corr_frf_vs_ttfa = correlation_dataframe[CPColNames.FirstRankFrequency].corr(correlation_dataframe[TTFA],
                                                                                 method='pearson')
    # The Pearson's correlation over all categories between average first-response RT (for responses with
    # first-rank frequency ≥4) and the time to the first activation (TTFA) within the model.
    first_rank_frequent_data = correlation_dataframe[
        correlation_dataframe[CPColNames.FirstRankFrequency] >= min_first_rank_freq]
    n_first_rank_frequent = first_rank_frequent_data.shape[0]
    first_rank_frequent_corr_rt_vs_ttfa = first_rank_frequent_data[CPColNames.MeanZRT].corr(
        first_rank_frequent_data[TTFA], method='pearson')
    # Pearson's correlation between production frequency and ttfa
    corr_prodfreq_vs_ttfa = correlation_dataframe[CPColNames.ProductionFrequency].corr(correlation_dataframe[TTFA],
                                                                                       method='pearson')
    corr_meanrank_vs_ttfa = correlation_dataframe[CPColNames.MeanRank].corr(correlation_dataframe[TTFA],
                                                                            method='pearson')
    # Save correlation and hitrate stats
    if model_type == ModelType.sensorimotor:
        available_pairs = set(correlation_dataframe[[CPColNames.CategorySensorimotor, CPColNames.ResponseSensorimotor]]
                              .groupby([CPColNames.CategorySensorimotor, CPColNames.ResponseSensorimotor])
                              .groups.keys())
    elif model_type == ModelType.linguistic:
        available_pairs = set(correlation_dataframe[[CPColNames.Category, CPColNames.Response]]
                              .groupby([CPColNames.Category, CPColNames.Response])
                              .groups.keys())
    else:
        raise NotImplementedError()

    return {
        "FRF corr (-)": corr_frf_vs_ttfa,
        "FRF N": n_first_rank_frequent,
        f"zRT corr (+; FRF≥{min_first_rank_freq})": first_rank_frequent_corr_rt_vs_ttfa,
        "zRT N": n_first_rank_frequent,
        "ProdFreq corr (-)": corr_prodfreq_vs_ttfa,
        "ProdFreq N": len(available_pairs),
        "MeanRank corr (+)": corr_meanrank_vs_ttfa,
        "Mean Rank N": len(available_pairs),
    }


def drop_missing_data_to_add_types(main_data: DataFrame, type_dict: Dict):
    """
    Drops rows of `main_data` with missing data in columns given by the keys of `type_dict`, so that the datatypes from
    values of `type_dict` can be successfully applied.

    :param main_data:
    :param type_dict:
        column_name -> column_datatype
    :return:
    :side effects: Mutates `main_data`.
    """
    main_data.dropna(inplace=True, how='any', subset=list(type_dict.keys()))
    # Now we can convert columns to appropriate types as there won't be null values
    for c, t in type_dict.items():
        main_data[c] = main_data[c].astype(t)


def hitrate_within_sd_of_hitrate_mean_frac(df: DataFrame) -> DataFrame:
    # When the model hitrate is within one SD of the hitrate mean
    within = Series(
        (df[MODEL_HITRATE] > df["Hitrate Mean"] - df["Hitrate SD"])
        & (df[MODEL_HITRATE] < df["Hitrate Mean"] + df["Hitrate SD"]))
    # The fraction of times this happens
    return within.aggregate('mean')


def get_summary_table(main_dataframe, groupby_column):
    """Summarise main dataframe by aggregating production proportion by the stated `groupby_column` column."""
    df = DataFrame()

    # Individual participant columns
    for participant in _CP.participants:
        df[PARTICIPANT_HITRATE_All_f.format(participant)] = (
            main_dataframe
            [main_dataframe[PARTICIPANT_SAW_CATEGORY_f.format(participant)] == True]
            [[groupby_column, PARTICIPANT_RESPONSE_HIT_f.format(participant)]].astype(float)
            .groupby(groupby_column)
            .sum()[PARTICIPANT_RESPONSE_HIT_f.format(participant)]
            / CATEGORIES_PER_PARTICIPANT)

    # Participant summary columns: hitrate
    df['Hitrate Mean'] = df[[PARTICIPANT_HITRATE_All_f.format(p) for p in _CP.participants]].mean(axis=1)
    df['Hitrate SD'] = df[[PARTICIPANT_HITRATE_All_f.format(p) for p in _CP.participants]].std(axis=1)

    # Model columns
    df[MODEL_HITRATE] = (
        main_dataframe[[groupby_column, MODEL_HIT]].astype(float)
        .groupby(groupby_column)
        .sum()[MODEL_HIT]
        / TOTAL_CATEGORIES
    )

    # Forget rows with nans
    df = df.dropna().reset_index()

    return df


def find_output_dirs(root_dir: str):
    """Finds all model-output dirs within a specified root directory."""
    # Find ` model_spec.yaml` files. Then each lives alongside the model output, so we return the containing dirs
    return [
        path.dirname(model_spec_location)
        for model_spec_location in glob(
            path.join(root_dir, "**", " model_spec.yaml"), recursive=True)
    ]


def prepare_category_production_data(model_type: ModelType) -> DataFrame:
    # Main dataframe holds category production data and model response data
    main_data: DataFrame = _CP.data.copy()

    # Some distances have been precomputed, so we label them as such
    main_data.rename(columns={col_name: f"Precomputed {col_name}"
                              for col_name in [
                                  'Sensorimotor.distance.cosine.additive',
                                  'Sensorimotor.distance.Minkowski.3.additive',
                                  'Linguistic.PPMI',
                              ]},
                     inplace=True)
    main_data = exclude_idiosyncratic_responses(main_data)

    add_predictor_column_production_proportion(main_data)
    add_rpf_column(main_data, model_type)
    add_rmr_column(main_data)

    return main_data
