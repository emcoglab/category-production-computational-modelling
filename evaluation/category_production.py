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
from model.utils.maths import t_confidence_interval
from preferences import Preferences

logger = logging.getLogger(__name__)


CATEGORY_PRODUCTION = CategoryProduction(use_cache=True)

# Each participant saw 39 categories
CATEGORIES_PER_PARTICIPANT = 39
# Each category was seen by 20 participants
PARTICIPANTS_PER_CATEGORY = 20
# Participants were separated into 3 equipollent sets, each of which saw disjoint equipollent sets of categories.
PARTICIPANT_SETS = 3
TOTAL_PARTICIPANTS = PARTICIPANTS_PER_CATEGORY * PARTICIPANT_SETS
TOTAL_CATEGORIES = CATEGORIES_PER_PARTICIPANT * PARTICIPANT_SETS
assert (TOTAL_CATEGORIES == len(CATEGORY_PRODUCTION.category_labels))


class ModelType(Enum):
    """Represents the type of model being used in the comparison."""
    naïve_sensorimotor = auto()
    naïve_linguistic = auto()
    sensorimotor = auto()
    linguistic = auto()
    # Combining sensorimotor and linguistic in a naïve way; i.e. set union of activated items
    combined_set_union = auto()
    # Full tandem model
    tandem = auto()

    @property
    def name(self) -> str:
        return super(ModelType, self).name.replace("_", " ")


class ParticipantSummaryType(Enum):
    """Represents a way to summarise participant data."""
    # Production proportion of responses (mean and sd over categories)
    production_proportion_mean_sd  = auto()
    # Individual hitrates of participants (producible responses only)
    individual_hitrates_producible = auto()
    # Individual hitrates of participants (all responses)
    individual_hitrates_all        = auto()


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


def add_rfop_column(main_data, model_type: ModelType):
    """Mutates `main_data`."""
    logger.info("Adding RFoP column")
    if model_type in [ModelType.linguistic, ModelType.naïve_linguistic]:
        specific_category_column = CPColNames.Category
    elif model_type in [ModelType.sensorimotor, ModelType.naïve_sensorimotor]:
        specific_category_column = CPColNames.CategorySensorimotor
    elif model_type == ModelType.combined_set_union:
        # We could use either here
        specific_category_column = CPColNames.Category
    else:
        raise NotImplementedError()
    main_data[RANK_FREQUENCY_OF_PRODUCTION] = (
        main_data
        # Within each category
        .groupby(specific_category_column)
        # Rank the responses according to production frequency
        [CPColNames.ProductionFrequency]
        .rank(ascending=False,
              # For ties, order alphabetically (i.e. pseudorandomly (?))
              method='first'))


def add_rmr_column(main_data):
    """Mutates `main_data`."""
    logger.info("Adding RMR column")
    main_data[ROUNDED_MEAN_RANK] = main_data.apply(lambda row: floor(row[CPColNames.MeanRank]), axis=1)


def add_model_predictor_columns(main_data, ttfas: Dict[str, Dict[str, int]], model_type: ModelType):
    """Mutates `main_data`."""
    # TODO: this function's signature is a bit of a mess... the ttfas dict should probably be built in here
    logger.info("Adding TTFA column")

    def get_min_ttfa_for_multiword_responses(row) -> int:
        """
        Helper function to convert a row in the output into a ttfa when the response is formed either of a single or
        multiple norms terms.
        """

        if model_type == ModelType.sensorimotor:
            c = row[CPColNames.CategorySensorimotor]
            r = row[CPColNames.ResponseSensorimotor]
        elif model_type == ModelType.linguistic:
            c = row[CPColNames.Category]
            r = row[CPColNames.Response]
        else:
            raise NotImplementedError()

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
                       if (w not in CATEGORY_PRODUCTION.ignored_words)
                       and (w in c_ttfas)]

            # The multi-word response is said to be activated the first time any one of its constituent words are
            if len(r_ttfas) > 1:
                return min(r_ttfas)
            # If none of the constituent words have a ttfa, we have no ttfa for the multiword term
            else:
                return nan

    main_data[TTFA] = main_data.apply(get_min_ttfa_for_multiword_responses, axis=1)

    logger.info("Adding model hit column")
    main_data[MODEL_HIT] = main_data.apply(lambda row: not isna(row[TTFA]), axis=1)


def save_item_level_data(main_data: DataFrame, save_path):
    main_data.to_csv(save_path, index=False)


def save_hitrate_summary_figure(summary_table, x_selector, fig_title, fig_name,
                                model_type: ModelType, summarise_participants_by: ParticipantSummaryType):
    """Save a summary table as a figure."""

    # add participant bounds
    if summarise_participants_by == ParticipantSummaryType.production_proportion_mean_sd:
        pyplot.fill_between(x=summary_table.reset_index()[x_selector],
                            y1=summary_table[PRODUCTION_PROPORTION + ' Mean'] - summary_table[
                                PRODUCTION_PROPORTION + ' SD'],
                            y2=summary_table[PRODUCTION_PROPORTION + ' Mean'] + summary_table[
                                PRODUCTION_PROPORTION + ' SD'])
        pyplot.scatter(x=summary_table.reset_index()[x_selector],
                       y=summary_table[PRODUCTION_PROPORTION + ' Mean'])
        pyplot.ylabel("Production proportion / hitrate")
    elif summarise_participants_by == ParticipantSummaryType.individual_hitrates_producible:
        for participant in CATEGORY_PRODUCTION.participants:
            pyplot.plot(summary_table.reset_index()[x_selector],
                        summary_table[PARTICIPANT_HITRATE_PRODUCIBLE_f.format(participant)],
                        linewidth=0.4, linestyle="-", color="b", alpha=0.4)
            pyplot.ylabel("hitrate")
    elif summarise_participants_by == ParticipantSummaryType.individual_hitrates_all:
        for participant in CATEGORY_PRODUCTION.participants:
            pyplot.plot(summary_table.reset_index()[x_selector],
                        summary_table[PARTICIPANT_HITRATE_All_f.format(participant)],
                        linewidth=0.4, linestyle="-", color="b", alpha=0.4)
            pyplot.ylabel("hitrate")
    else:
        raise NotImplementedError()

    # add model performance
    if summarise_participants_by in {ParticipantSummaryType.production_proportion_mean_sd,
                                     ParticipantSummaryType.individual_hitrates_all}:
        pyplot.scatter(x=summary_table.reset_index()[x_selector],
                       y=summary_table[MODEL_HITRATE_ALL],
                       marker="o", color="g")
    elif summarise_participants_by == ParticipantSummaryType.individual_hitrates_producible:
        pyplot.scatter(x=summary_table.reset_index()[x_selector],
                       y=summary_table[MODEL_HITRATE_PRODUCIBLE],
                       marker="o", color="g")
    else:
        raise NotImplementedError()

    pyplot.ylim((0, None))

    pyplot.title(fig_title)
    pyplot.xlabel(x_selector)

    if model_type == ModelType.sensorimotor:
        figures_dir = "hitrates sensorimotor"
    elif model_type == ModelType.linguistic:
        figures_dir = "hitrates linguistic"
    elif model_type == ModelType.combined_set_union:
        figures_dir = "hitrates combined set union"
    elif model_type == ModelType.naïve_linguistic:
        figures_dir = "hitrates naïve linguistic"
    elif model_type == ModelType.naïve_sensorimotor:
        figures_dir = "hitrates naïve sensorimotor"
    else:
        raise NotImplementedError()

    if summarise_participants_by == ParticipantSummaryType.production_proportion_mean_sd:
        filename = f"{fig_name} sd.png"
    elif summarise_participants_by == ParticipantSummaryType.individual_hitrates_producible:
        filename = f"{fig_name} traces (producible).png"
    elif summarise_participants_by == ParticipantSummaryType.individual_hitrates_all:
        filename = f"{fig_name} traces (all).png"
    else:
        raise NotImplementedError()

    pyplot.savefig(
        path.join(Preferences.figures_dir,
                  figures_dir,
                  filename))

    pyplot.clf()
    pyplot.cla()
    pyplot.close()


def save_hitrate_summary_tables(model_identifier_string: str, main_data: DataFrame, model_type: ModelType,
                                conscious_access_threshold: Optional[float]):

    hitrates_per_rfop = get_summary_table(main_data, RANK_FREQUENCY_OF_PRODUCTION)
    # For FROP we will truncate the table at mean + 2SD (over categories) items
    if model_type in [ModelType.linguistic, ModelType.naïve_linguistic]:
        n_items_mean = (
            main_data[[CPColNames.Category, CPColNames.Response]]
            .groupby(CPColNames.Category)
            .count()[CPColNames.Response]
            .mean())
        n_items_sd = (
            main_data[[CPColNames.Category, CPColNames.Response]]
            .groupby(CPColNames.Category)
            .count()[CPColNames.Response]
            .std())
    elif model_type in [ModelType.sensorimotor, ModelType.naïve_sensorimotor]:
        n_items_mean = (
            main_data[[CPColNames.CategorySensorimotor, CPColNames.ResponseSensorimotor]]
            .groupby(CPColNames.CategorySensorimotor)
            .count()[CPColNames.ResponseSensorimotor]
            .mean())
        n_items_sd = (
            main_data[[CPColNames.CategorySensorimotor, CPColNames.ResponseSensorimotor]]
            .groupby(CPColNames.CategorySensorimotor)
            .count()[CPColNames.ResponseSensorimotor]
            .std())
    else:
        raise NotImplementedError()
    hitrates_per_rfop = hitrates_per_rfop[
        hitrates_per_rfop[RANK_FREQUENCY_OF_PRODUCTION] < n_items_mean + (2 * n_items_sd)]

    hitrates_per_rmr = get_summary_table(main_data, ROUNDED_MEAN_RANK)

    # Compute hitrate fits
    hitrate_fit_rfop = hitrate_within_sd_of_mean_frac(hitrates_per_rfop)
    hitrate_fit_rmr = hitrate_within_sd_of_mean_frac(hitrates_per_rmr)

    # Save summary tables
    if model_type == ModelType.sensorimotor:
        base_dir = path.join(Preferences.results_dir, "Category production fit sensorimotor")
    elif model_type == ModelType.linguistic:
        base_dir = path.join(Preferences.results_dir, f"Category production fit linguistic")
    elif model_type == ModelType.combined_set_union:
        base_dir = path.join(Preferences.results_dir, "Category production fit combined set union")
    elif model_type == ModelType.naïve_sensorimotor:
        base_dir = path.join(Preferences.results_dir, "Category production fit naïve sensorimotor")
    elif model_type == ModelType.naïve_linguistic:
        base_dir = path.join(Preferences.results_dir, "Category production fit naïve linguistic")
    else:
        raise NotImplementedError()

    if conscious_access_threshold is not None:
        file_suffix = f"({model_identifier_string}) CAT={conscious_access_threshold}"
    else:
        file_suffix = f"({model_identifier_string})"

    hitrates_per_rfop.to_csv(path.join(base_dir,
                                       f"Production proportion per rank frequency of production {file_suffix}.csv"),
                             index=False)
    hitrates_per_rmr.to_csv(path.join(base_dir,
                                      f"Production proportion per rounded mean rank {file_suffix}.csv"),
                            index=False)

    # region Graph tables

    # rfop sd region
    save_hitrate_summary_figure(summary_table=hitrates_per_rfop,
                                x_selector=RANK_FREQUENCY_OF_PRODUCTION,
                                fig_title="Hitrate per RFOP",
                                fig_name=f"hitrate per RFOP {file_suffix}",
                                model_type=model_type,
                                summarise_participants_by=ParticipantSummaryType.production_proportion_mean_sd)
    # rfop traces
    save_hitrate_summary_figure(summary_table=hitrates_per_rfop,
                                x_selector=RANK_FREQUENCY_OF_PRODUCTION,
                                fig_title="Hitrate per RFOP (producible)",
                                fig_name=f"hitrate per RFOP {file_suffix}",
                                model_type=model_type,
                                summarise_participants_by=ParticipantSummaryType.individual_hitrates_producible)
    save_hitrate_summary_figure(summary_table=hitrates_per_rfop,
                                x_selector=RANK_FREQUENCY_OF_PRODUCTION,
                                fig_title="Hitrate per RFOP (all)",
                                fig_name=f"hitrate per RFOP {file_suffix}",
                                model_type=model_type,
                                summarise_participants_by=ParticipantSummaryType.individual_hitrates_all)
    # rmr sd region
    save_hitrate_summary_figure(summary_table=hitrates_per_rmr,
                                x_selector=ROUNDED_MEAN_RANK,
                                fig_title="Hitrate per RMR",
                                fig_name=f"hitrate per RMR {file_suffix}",
                                model_type=model_type,
                                summarise_participants_by=ParticipantSummaryType.production_proportion_mean_sd)
    save_hitrate_summary_figure(summary_table=hitrates_per_rmr,
                                x_selector=ROUNDED_MEAN_RANK,
                                fig_title="Hitrate per RMR (producible)",
                                fig_name=f"hitrate per RMR {file_suffix}",
                                model_type=model_type,
                                summarise_participants_by=ParticipantSummaryType.individual_hitrates_producible)
    save_hitrate_summary_figure(summary_table=hitrates_per_rmr,
                                x_selector=ROUNDED_MEAN_RANK,
                                fig_title="Hitrate per RMR (all)",
                                fig_name=f"hitrate per RMR {file_suffix}",
                                model_type=model_type,
                                summarise_participants_by=ParticipantSummaryType.individual_hitrates_all)

    # endregion

    return hitrate_fit_rfop, hitrate_fit_rmr


def process_one_model_output(main_data: DataFrame,
                             model_type: ModelType,
                             input_results_dir: str,
                             min_first_rank_freq: Optional[int],
                             conscious_access_threshold: Optional[float],
                             ):
    assert model_type in [ModelType.linguistic, ModelType.sensorimotor]
    input_results_path = Path(input_results_dir)
    model_identifier = f"{input_results_path.parent.name} {input_results_path.name}"
    output_dir = f"Category production fit {model_type.name}"
    save_item_level_data(main_data, path.join(Preferences.results_dir,
                                              output_dir,
                                              f"item-level data ({model_identifier})"
                                              + (f" CAT={conscious_access_threshold}" if conscious_access_threshold is not None else "") +
                                              ".csv"))

    hitrate_fit_rfop, hitrate_fit_rmr = save_hitrate_summary_tables(model_identifier, main_data, model_type,
                                                                    conscious_access_threshold)

    drop_missing_data_to_add_types(main_data, {TTFA: int})

    save_model_performance_stats(
        main_data,
        model_identifier=model_identifier,
        results_dir=input_results_dir,
        min_first_rank_freq=min_first_rank_freq,
        hitrate_fit_rfop=hitrate_fit_rfop,
        hitrate_fit_rmr=hitrate_fit_rmr,
        model_type=model_type,
        conscious_access_threshold=conscious_access_threshold,
    )


def process_one_model_output_naïve(main_data: DataFrame,
                                   model_type: ModelType,
                                   input_results_dir: str,
                                   min_first_rank_freq: Optional[int],
                                   ):
    assert model_type in [ModelType.naïve_linguistic, ModelType.naïve_sensorimotor]
    input_results_path = Path(input_results_dir)
    model_identifier = f"{input_results_path.parent.name} {input_results_path.name}"
    output_dir = f"Category production fit {model_type.name}"
    save_item_level_data(main_data, path.join(Preferences.results_dir,
                                              output_dir,
                                              f"item-level data ({model_identifier}).csv"))

    hitrate_fit_rfop, hitrate_fit_rmr = save_hitrate_summary_tables(model_identifier, main_data, model_type, None)

    save_model_performance_stats(
        main_data,
        model_identifier=model_identifier,
        results_dir=input_results_dir,
        min_first_rank_freq=min_first_rank_freq,
        hitrate_fit_rfop=hitrate_fit_rfop,
        hitrate_fit_rmr=hitrate_fit_rmr,
        model_type=model_type,
        conscious_access_threshold=None,
    )


def save_model_performance_stats(main_dataframe,
                                 results_dir,
                                 model_identifier: str,
                                 min_first_rank_freq: Optional[int],
                                 hitrate_fit_rfop,
                                 hitrate_fit_rmr,
                                 model_type: ModelType,
                                 conscious_access_threshold: Optional[float],
                                 ):

    # Build output dir
    if conscious_access_threshold is not None:
        filename_suffix = f" CAT={conscious_access_threshold}"
    else:
        filename_suffix = ""
    if model_type == ModelType.sensorimotor:
        specific_output_dir = "Category production fit sensorimotor"
    elif model_type == ModelType.linguistic:
        specific_output_dir = "Category production fit linguistic"
    elif model_type == ModelType.naïve_sensorimotor:
        specific_output_dir = "Category production fit naïve sensorimotor"
    elif model_type == ModelType.naïve_linguistic:
        specific_output_dir = "Category production fit naïve linguistic"
    else:
        raise NotImplementedError()
    overall_stats_output_path = path.join(
        Preferences.results_dir,
        specific_output_dir,
        f"model_effectiveness_overall ({model_identifier}){filename_suffix}.csv")

    df_dict = dict()

    if model_type in [ModelType.linguistic, ModelType.sensorimotor]:
        # Only spreading-activation models have specs, and produce TTFAs from which correlation stats are generated
        df_dict.update(GraphPropagation.load_model_spec(results_dir))
        df_dict.update(get_correlation_stats(main_dataframe, min_first_rank_freq, model_type=model_type))

    df_dict.update({
        "Hitrate within SD of mean (RFoP)": hitrate_fit_rfop,
        "Hitrate within SD of mean (RMR)": hitrate_fit_rmr,
    })

    if conscious_access_threshold is not None:
        df_dict[CAT] = conscious_access_threshold

    model_performance_data: DataFrame = DataFrame.from_records([df_dict])
    model_performance_data.to_csv(overall_stats_output_path,
                                  # As of Python 3.6, dictionary keys are ordered by insertion,
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


def hitrate_within_sd_of_mean_frac(df: DataFrame) -> DataFrame:
    # When the model hitrate is within one SD of the production proportion mean
    within = Series(
        (df[MODEL_HITRATE_ALL] > df[PRODUCTION_PROPORTION + " Mean"] - df[PRODUCTION_PROPORTION + " SD"])
        & (df[MODEL_HITRATE_ALL] < df[PRODUCTION_PROPORTION + " Mean"] + df[PRODUCTION_PROPORTION + " SD"]))
    # The fraction of times this happens
    return within.aggregate('mean')


def get_summary_table(main_dataframe, groupby_column):
    """
    Summarise main dataframe by aggregating production proportion by the stated `groupby_column` column.
    """
    df = DataFrame()

    # Individual participant columns
    for participant in CATEGORY_PRODUCTION.participants:
        df[PARTICIPANT_HITRATE_PRODUCIBLE_f.format(participant)] = (
            main_dataframe
            [main_dataframe[PARTICIPANT_SAW_CATEGORY_f.format(participant)] == True]
            [[groupby_column, PARTICIPANT_RESPONSE_HIT_f.format(participant)]].astype(float)
            .groupby(groupby_column)
            .mean()[PARTICIPANT_RESPONSE_HIT_f.format(participant)])
        df[PARTICIPANT_HITRATE_All_f.format(participant)] = (
            main_dataframe
            [main_dataframe[PARTICIPANT_SAW_CATEGORY_f.format(participant)] == True]
            [[groupby_column, PARTICIPANT_RESPONSE_HIT_f.format(participant)]].astype(float)
            .groupby(groupby_column)
            .sum()[PARTICIPANT_RESPONSE_HIT_f.format(participant)]
            / CATEGORIES_PER_PARTICIPANT)

    # Participant summary columns
    df[PRODUCTION_PROPORTION + ' Mean'] = (
        main_dataframe
        .groupby(groupby_column)
        .mean()[PRODUCTION_PROPORTION])
    df[PRODUCTION_PROPORTION + ' SD'] = (
        main_dataframe
        .groupby(groupby_column)
        .std()[PRODUCTION_PROPORTION])
    df[PRODUCTION_PROPORTION + ' Count'] = (
        main_dataframe
        .groupby(groupby_column)
        .count()[PRODUCTION_PROPORTION])
    df[PRODUCTION_PROPORTION + ' CI95'] = df.apply(
        lambda row: t_confidence_interval(row[PRODUCTION_PROPORTION + ' SD'],
                                          row[PRODUCTION_PROPORTION + ' Count'],
                                          0.95), axis=1)

    # Model columns
    df[MODEL_HITRATE_PRODUCIBLE] = (
        main_dataframe[[groupby_column, MODEL_HIT]].astype(float)
        .groupby(groupby_column)
        .mean()[MODEL_HIT]
    )
    df[MODEL_HITRATE_ALL] = (
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
    main_data: DataFrame = CATEGORY_PRODUCTION.data.copy()

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
    add_rfop_column(main_data, model_type=model_type)
    add_rmr_column(main_data)

    return main_data
