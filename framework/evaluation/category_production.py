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
from collections import defaultdict
from enum import Enum, auto
from glob import glob
from math import floor
from os import path, listdir
from pathlib import Path
from typing import DefaultDict, Dict, Set, List, Optional, Tuple

import yaml
from matplotlib import pyplot, ticker
from pandas import DataFrame, read_csv, isna, Series, NA
from pandas.errors import ParserError as PandasParserError

from .column_names import *
from ..category_production.category_production import CategoryProduction, ColNames as CPColNames
from ..cognitive_model.ldm.corpus.tokenising import modified_word_tokenize
from ..cognitive_model.utils.logging import logger
from ..cognitive_model.basic_types import ActivationValue, Component
from ..cognitive_model.utils.exceptions import ParseError
from ..cognitive_model.utils.maths import cm_to_inches
from ..cognitive_model.preferences.preferences import Preferences


CP_INSTANCE = CategoryProduction()

# Each participant saw 39 categories
CATEGORIES_PER_PARTICIPANT = 39
# Each category was seen by 20 participants
PARTICIPANTS_PER_CATEGORY = 20
# Participants were separated into 3 equipollent sets, each of which saw disjoint equipollent sets of categories.
PARTICIPANT_SETS = 3
TOTAL_PARTICIPANTS = PARTICIPANTS_PER_CATEGORY * PARTICIPANT_SETS  # 60
TOTAL_CATEGORIES = CATEGORIES_PER_PARTICIPANT * PARTICIPANT_SETS  # 117
assert (TOTAL_CATEGORIES == len(CP_INSTANCE.category_labels))


class ModelType(Enum):
    """Represents the type of model being used in the comparison."""
    # Linguistic models
    linguistic                 = auto()
    linguistic_one_hop         = auto()
    # Sensorimotor models
    sensorimotor               = auto()
    sensorimotor_one_hop       = auto()
    # Combined models
    combined_noninteractive    = auto()
    combined_interactive       = auto()

    @property
    def name(self) -> str:
        if self == ModelType.linguistic_one_hop:
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


def get_n_words_from_path_linguistic(results_dir_path) -> int:
    """
    Gets the number of words from a path storing results.
    :param results_dir_path:
    :return: n_words: int
    """
    dir_name = path.basename(results_dir_path)
    words_match = re.search(re.compile(r".* (?P<n_words>[0-9,]+) words,.*"), dir_name)
    if words_match:
        # remove the comma and parse as int
        n_words = int(words_match.group("n_words").replace(",", ""))
        return n_words
    else:
        raise ParseError(f"Could not parse number of words from {dir_name}.")


def get_firing_threshold_from_path_linguistic(results_dir_path) -> ActivationValue:
    """
    Gets the firing threshold from a path storing the results.
    :param results_dir_path:
    :return: firing_threshold: ActivationValue
    """
    dir_name = path.basename(results_dir_path)
    ft_match = re.search(re.compile(r"firing-θ (?P<firing_threshold>[0-9.]+);"), dir_name)
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


def get_model_ttfas_and_components_for_category_combined_interactive(category: str, results_dir,
                                                                     require_buffer_entry: bool = True,
                                                                     require_activations_in_component: Optional[Component] = None,
                                                                     ) -> Tuple[DefaultDict[str, int], DefaultDict[str, Component]]:
    """
    Dictionary of
        response -> time to first activation
    for the specified category.

    DefaultDict gives nans where response not found

    :param category:
    :param results_dir:
    :return:
        tuple(
            dictionary response -> time,
            dictionaryh response -> component
        )
    """

    # Try to load model response
    model_responses_path = path.join(results_dir, f"responses_{category}.csv")
    try:
        with open(model_responses_path, mode="r", encoding="utf-8") as model_responses_file:
            model_responses: DataFrame = read_csv(model_responses_file, header=0, comment="#", index_col=False,
                                                  dtype={TICK_ON_WHICH_ACTIVATED: int})

    # If the category wasn't found, there are no TTFAs
    except FileNotFoundError:
        logger.warning(f"Could not find model output file for {category} [{model_responses_path}]")
        return (defaultdict(lambda: NA), defaultdict(lambda: NA))
    except PandasParserError as er:
        logger.error(f"Corrupt file at {model_responses_path}")
        raise er

    relevant_data = model_responses.sort_values(by=TICK_ON_WHICH_ACTIVATED)

    if require_buffer_entry:
        relevant_data = relevant_data[relevant_data[ITEM_ENTERED_BUFFER] == True]
    if require_activations_in_component is not None:
        relevant_data = relevant_data[relevant_data[COMPONENT] == require_activations_in_component.name]

    # In case some of the category words are present in the response, they would count as "free" hits on tick 0.
    # Thus we exclude any such responses here.
    # Use > 1 here rather than > 0 to keep compatibility with old versions where initial activations weren't
    # processed until tick 1. This assumes that the shortest edge is > 1, which it is in everything we have or will
    # test.
    relevant_data = relevant_data[relevant_data[TICK_ON_WHICH_ACTIVATED] > 1]

    ttfas = relevant_data \
        .groupby(RESPONSE) \
        .first()[[TICK_ON_WHICH_ACTIVATED]] \
        .to_dict('dict')[TICK_ON_WHICH_ACTIVATED]

    components = relevant_data \
        .groupby(RESPONSE) \
        .first()[[COMPONENT]] \
        .to_dict('dict')[COMPONENT]

    # Return canonical values
    components = {
        response: Component[component]
        for response, component in components.items()
    }

    return (
        defaultdict(lambda: NA, ttfas),
        defaultdict(lambda: NA, components)
    )


def get_model_ttfas_for_category_linguistic(category: str, results_dir, n_words: int,
                                            firing_threshold: float) -> DefaultDict[str, int]:
    """
    Dictionary of
        response -> time to first activation
    for the specified category.

    DefaultDict gives nans where response not found

    :param category:
    :param results_dir:
    :param n_words:
    :param firing_threshold:
    :return:
    """

    # Try to load model response
    try:
        model_responses_path = path.join(results_dir, f"responses_{category}_{n_words:,}.csv")
        with open(model_responses_path, mode="r", encoding="utf-8") as model_responses_file:
            model_responses: DataFrame = read_csv(model_responses_file, header=0, comment="#", index_col=False)

        activations_which_fired = model_responses[model_responses[ACTIVATION] >= firing_threshold]\
            .sort_values(by=TICK_ON_WHICH_ACTIVATED)

        # In case some of the category words are present in the response, they would count as "free" hits on tick 0.
        # Thus we exclude any such responses here.
        # Use > 1 here rather than > 0 to keep compatibility with old versions where initial activations weren't
        # processed until tick 1. This assumes that the shortest edge is > 1, which it is in everything we have or will
        # test.
        activations_which_fired = activations_which_fired[activations_which_fired[TICK_ON_WHICH_ACTIVATED] > 1]

        ttfas = activations_which_fired\
            .groupby(RESPONSE)\
            .first()[[TICK_ON_WHICH_ACTIVATED]]\
            .to_dict('dict')[TICK_ON_WHICH_ACTIVATED]

        return defaultdict(lambda: NA, ttfas)

    # If the category wasn't found, there are no TTFAs
    except FileNotFoundError:
        return defaultdict(lambda: NA)


def get_model_ttfas_for_category_sensorimotor(category: str, results_dir) -> Dict[str, int]:
    """
    Dictionary of
        response -> time to first activation
    for the specified category.

    DefaultDict gives nans where response not found

    :param category:
    :param results_dir:
    :return:
    """

    # Check for erroneous paths
    if not path.isdir(results_dir):
        raise NotADirectoryError(results_dir)

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

        # In case some of the category words are present in the response, they would count as "free" hits on tick 0.
        # Thus we exclude any such responses here.
        # Use > 1 here rather than > 0 to keep compatibility with old versions where initial activations weren't
        # processed until tick 1. This assumes that the shortest edge is > 1, which it is in everything we have or will
        # test.
        in_buffer_data = in_buffer_data[in_buffer_data[TICK_ON_WHICH_ACTIVATED] > 1]

        ttfas: Dict = in_buffer_data \
            .groupby(RESPONSE) \
            .first()[[TICK_ON_WHICH_ACTIVATED]] \
            .to_dict('dict')[TICK_ON_WHICH_ACTIVATED]

        return defaultdict(lambda: NA, ttfas)

    # If the category wasn't found, there are no TTFAs
    except FileNotFoundError:
        return defaultdict(lambda: NA)


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
    # divided by the number of participants who saw the category in which it is a response.
    main_data[PRODUCTION_PROPORTION] = main_data.apply(lambda row: row[CPColNames.ProductionFrequency] / PARTICIPANTS_PER_CATEGORY, axis=1)


def category_response_col_names_for_model_type(model_type):
    if model_type in [ModelType.linguistic, ModelType.linguistic_one_hop]:
        c, r = CPColNames.Category, CPColNames.Response
    elif model_type in [ModelType.sensorimotor, ModelType.sensorimotor_one_hop]:
        c, r = CPColNames.CategorySensorimotor, CPColNames.ResponseSensorimotor
    elif model_type == ModelType.combined_noninteractive:
        # We could use either here
        c, r = CPColNames.Category, CPColNames.Response
    elif model_type == ModelType.combined_interactive:
        # We're using BrEng words for each of the words, so we'll use the standard columns, which are BrEng anyway
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
            ttfas_for_category: Dict[str, int] = ttfas[c]
        except KeyError:
            return NA

        # If the response was directly found, we can return it
        if r in ttfas_for_category:
            return ttfas_for_category[r]

        # Otherwise, try to break the response into components and find any one of them.
        # Component words which aren't found are just ignored.
        else:
            ttfas_for_response = [ttfas_for_category[w]
                                  for w in modified_word_tokenize(r)
                                  if (w not in CP_INSTANCE.ignored_words)
                                  and (w in ttfas_for_category)]

            # The multi-word response is said to be activated the first time any one of its constituent words are
            if len(ttfas_for_response) > 1:
                return min(ttfas_for_response)
            # If none of the constituent words have a ttfa, we have no ttfa for the multiword term
            else:
                return NA

    main_data[TTFA] = main_data.apply(get_min_ttfa_for_multiword_responses, axis=1)


def add_component_column(main_data, components: Dict[str, Dict[str, Component]], model_type: ModelType):
    """Mutates `main_data`."""
    logger.info("Adding components column")

    def get_component(row) -> str:
        category_column, response_column = category_response_col_names_for_model_type(model_type)
        c, r = row[category_column], row[response_column]
        try:
            components_for_category = components[c]
        except KeyError:
            return NA

        # Get the component if we can directly look it up, otherwise nothing
        if r in components_for_category:
            return components_for_category[r].name
        else:
            return NA

    main_data[COMPONENT] = main_data.apply(get_component, axis=1)


def add_model_hit_column(main_data, ttfa_column: str = TTFA):
    """Mutates `main_data`."""
    logger.info("Adding model hit column")
    main_data[MODEL_HIT] = main_data.apply(lambda row: not isna(row[ttfa_column]), axis=1)


def save_item_level_data(main_data: DataFrame, save_path):
    """Saves the item-level data to a csv."""
    main_data.to_csv(save_path, index=False)


def save_hitrate_summary_figure(summary_table, x_selector, fig_name, figures_dir):
    """Save a summary table as a figure."""
    x_values = summary_table.reset_index()[x_selector]
    # Participant traces
    for participant in CP_INSTANCE.participants:
        pyplot.plot(x_values,
                    summary_table[PARTICIPANT_HITRATE_All_f.format(participant)],
                    linewidth=0.2, linestyle="-", color="k", alpha=0.3,
                    zorder=0)
    # SD region
    pyplot.fill_between(x=x_values,
                        y1=summary_table['Hitrate Mean'] - summary_table[
                            'Hitrate SD'],
                        y2=summary_table['Hitrate Mean'] + summary_table[
                            'Hitrate SD'],
                        color="#AFF7F3", alpha=0.6,
                        zorder=10)
    # Mean
    pyplot.plot(x_values,
                summary_table['Hitrate Mean'],
                linewidth=0.5, linestyle="-",
                color="#000000", alpha=0.5,
                zorder=20)

    # add model performance
    pyplot.plot(x_values,
                summary_table[MODEL_HITRATE],
                linewidth=2.0, linestyle='-',
                color="#FA5300",
                zorder=30)

    pyplot.xlim((1, summary_table.reset_index()[x_selector].max()))
    pyplot.ylim((0, None))

    pyplot.gca().xaxis.set_minor_locator(ticker.MultipleLocator(1))

    pyplot.ylabel("Hit rate")
    pyplot.xlabel(x_selector)

    pyplot.gcf().set_size_inches(cm_to_inches(7), cm_to_inches(7))
    pyplot.subplots_adjust(bottom=0.15, top=0.96, left=0.19, right=0.99)
    pyplot.savefig(path.join(figures_dir, f"{fig_name}.png"), dpi=600)

    pyplot.clf()
    pyplot.cla()
    pyplot.close()


def get_hitrate_summary_tables(main_data: DataFrame):

    hitrates_per_rpf = get_summary_table(main_data, RANKED_PRODUCTION_FREQUENCY)

    # For RPF we will truncate the table at mean + 2SD (over categories) items
    # We use the same column selectors (i.e. linguistic) in all cases
    category_column, response_column = category_response_col_names_for_model_type(ModelType.linguistic)
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


def save_hitrate_summary_tables(hitrates_per_rmr, hitrates_per_rpf, model_type, file_suffix, output_dir = None):
    # Save summary tables
    if output_dir is None:
        output_dir = path.join(Preferences.evaluation_dir, model_type.model_output_dirname)
    hitrates_per_rpf.to_csv(path.join(output_dir,
                                      f"Production proportion per rank frequency of production {file_suffix}.csv"),
                            index=False)
    hitrates_per_rmr.to_csv(path.join(output_dir,
                                      f"Production proportion per rounded mean rank {file_suffix}.csv"),
                            index=False)


def save_hitrate_graphs(hitrates_per_rpf, hitrates_per_rmr, file_suffix, figures_dir):

    # rpf sd region
    save_hitrate_summary_figure(summary_table=hitrates_per_rpf,
                                x_selector=RANKED_PRODUCTION_FREQUENCY,
                                fig_name=f"hitrate per RPF {file_suffix}",
                                figures_dir=figures_dir)
    # rmr sd region
    save_hitrate_summary_figure(summary_table=hitrates_per_rmr,
                                x_selector=ROUNDED_MEAN_RANK,
                                fig_name=f"hitrate per RMR {file_suffix}",
                                figures_dir=figures_dir)


def apply_ttfa_cutoff(data, ttfa_column, ttfa_cutoff):
    """Adds a cut-off `MODEL_HIT` column to a copy of `data`."""
    cut_data = data.copy()
    cut_data[MODEL_HIT] = cut_data[ttfa_column] < ttfa_cutoff
    cut_data.fillna(value={MODEL_HIT: False}, inplace=True)
    return cut_data


def process_one_model_output(main_data: DataFrame,
                             model_type: ModelType,
                             input_results_dir,
                             min_first_rank_freq: Optional[int],
                             manual_ttfa_cutoff: Optional[int] = None,
                             stats_save_path: Optional = None,
                             figures_save_path: Optional = None,
                             ):
    """
    Computes stats and figures for the output of one run of one model.
    :param main_data:
    :param model_type:
    :param input_results_dir:
    :param min_first_rank_freq:
    :param manual_ttfa_cutoff:
        If specified, applies a TTFA cut-off to the data before processing
    :param stats_save_path:
        Supply to override default save location
    :param figures_save_path:
        Supply to override default save location
    :return:
    """
    assert model_type in [ModelType.linguistic, ModelType.sensorimotor, ModelType.linguistic_one_hop, ModelType.sensorimotor_one_hop]
    input_results_path = Path(input_results_dir)

    if stats_save_path is None:
        stats_save_path = path.join(Preferences.evaluation_dir, model_type.model_output_dirname)
    if figures_save_path is None:
        figures_save_path = path.join(Preferences.figures_dir, model_type.figures_dirname)

    model_identifier = f"{input_results_path.parent.name} {input_results_path.name}"

    file_suffix = f"({model_identifier})"

    if manual_ttfa_cutoff is not None:
        file_suffix += f" cutoff {manual_ttfa_cutoff}"
        main_data = apply_ttfa_cutoff(main_data, TTFA, manual_ttfa_cutoff)

    item_level_path = path.join(stats_save_path, f"item-level data {file_suffix}.csv")
    save_item_level_data(main_data, item_level_path)

    hitrates_per_rpf, hitrates_per_rmr = get_hitrate_summary_tables(main_data)
    save_hitrate_summary_tables(hitrates_per_rmr, hitrates_per_rpf, model_type, file_suffix, output_dir=stats_save_path)

    # Compute hitrate fits
    hitrate_fit_rpf = frac_within_sd_of_hitrate_mean(hitrates_per_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)
    hitrate_fit_rmr = frac_within_sd_of_hitrate_mean(hitrates_per_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)
    hitrate_fit_rpf_head = frac_within_sd_of_hitrate_mean(hitrates_per_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)
    hitrate_fit_rmr_head = frac_within_sd_of_hitrate_mean(hitrates_per_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)

    drop_missing_data_to_add_types(main_data, {TTFA: int})

    save_model_performance_stats(
        main_data,
        model_identifier=model_identifier,
        results_dir=input_results_dir,
        min_first_rank_freq=min_first_rank_freq,
        hitrate_fit_rpf_hr=hitrate_fit_rpf,
        hitrate_fit_rmr_hr=hitrate_fit_rmr,
        hitrate_fit_rpf_hr_head=hitrate_fit_rpf_head,
        hitrate_fit_rmr_hr_head=hitrate_fit_rmr_head,
        model_type=model_type,
        output_dir=stats_save_path,
    )

    save_hitrate_graphs(hitrates_per_rpf, hitrates_per_rmr, file_suffix=file_suffix, figures_dir=figures_save_path)

    logger.info(f"Hitrate fits for {model_type.name} model")
    logger.info(f"rmr fit: {frac_within_sd_of_hitrate_mean(hitrates_per_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hitrates_per_rmr, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")
    logger.info(f"rpf fit: {frac_within_sd_of_hitrate_mean(hitrates_per_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=True)} head only ({frac_within_sd_of_hitrate_mean(hitrates_per_rpf, test_column=MODEL_HITRATE, only_before_sd_includes_0=False)} whole graph)")


def save_model_performance_stats(main_dataframe,
                                 results_dir,
                                 model_identifier: str,
                                 min_first_rank_freq: Optional[int],
                                 hitrate_fit_rpf_hr,
                                 hitrate_fit_rmr_hr,
                                 hitrate_fit_rpf_hr_head,
                                 hitrate_fit_rmr_hr_head,
                                 model_type: ModelType,
                                 output_dir: str = None
                                 ):
    if output_dir is None:
        output_dir = path.join(
            Preferences.evaluation_dir,
            model_type.model_output_dirname)
    overall_stats_output_path = path.join(
        output_dir,
        f"model_effectiveness_overall ({model_identifier}).csv")

    df_dict = dict()

    if model_type in [ModelType.linguistic, ModelType.sensorimotor]:
        # Only spreading-activation models have specs, and produce TTFAs from which correlation stats are generated
        # TODO: this is getting a bit messy
        with open(Path(results_dir, " model_spec.yaml"), mode="r") as spec_file:
            df_dict.update(yaml.load(spec_file, yaml.SafeLoader))
        df_dict.update(get_correlation_stats(main_dataframe, min_first_rank_freq, model_type=model_type))

    df_dict.update({
        "Hitrate within SD of HR mean (RPF)": hitrate_fit_rpf_hr,
        "Hitrate within SD of HR mean (RMR)": hitrate_fit_rmr_hr,
        "Hitrate within SD of HR mean (RPF) head only": hitrate_fit_rpf_hr_head,
        "Hitrate within SD of HR mean (RMR) head only": hitrate_fit_rmr_hr_head,
    })

    model_performance_data: DataFrame = DataFrame.from_records([df_dict])
    model_performance_data.to_csv(overall_stats_output_path,
                                  # As of Python 3.7, dictionary keys are ordered by insertion,
                                  # so we should automatically get a consistent order for stacking,
                                  columns=list(df_dict.keys()),
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


def frac_within_sd_of_hitrate_mean(df: DataFrame, test_column: str, only_before_sd_includes_0: bool) -> float:
    """
    test_column: the column containing the hitrates to test
    only_before_sd_includes_0: if true, only compute fraction in region before SD region first touches 0
    """
    if only_before_sd_includes_0:
        # [PyCharm thinks the first comparison gives a bool but it's actually a Series, so ignore this warning]
        # noinspection PyUnresolvedReferences
        df = df[
            # True whenever SD region includes or is below 0
            (df["Hitrate Mean"] <= df["Hitrate SD"])
            # .cumsum() starts at 0 and increments when above is true, so will be > 0 after the first time it's true
            .cumsum() <= 0]
        logger.info(f"fraction calculated in region [1, {df.shape[0]}]")
    # When the test hitrate is within one SD of the hitrate mean
    within = Series(
        (df[test_column] > df["Hitrate Mean"] - df["Hitrate SD"])
        & (df[test_column] < df["Hitrate Mean"] + df["Hitrate SD"]))
    # The fraction of times this happens
    return within.aggregate('mean')


def get_summary_table(main_dataframe, groupby_column):
    """Summarise main dataframe by aggregating hitrate proportion by the stated `groupby_column` column."""
    df = DataFrame()

    # Individual participant columns
    for participant in CP_INSTANCE.participants:
        df[PARTICIPANT_HITRATE_All_f.format(participant)] = (
            main_dataframe
            [main_dataframe[PARTICIPANT_SAW_CATEGORY_f.format(participant)] == True]
            [[groupby_column, PARTICIPANT_RESPONSE_HIT_f.format(participant)]].astype(float)
            .groupby(groupby_column)
            .sum()[PARTICIPANT_RESPONSE_HIT_f.format(participant)]
            / CATEGORIES_PER_PARTICIPANT)

    # Participant summary columns: hitrate
    df['Hitrate Mean'] = df[[PARTICIPANT_HITRATE_All_f.format(p) for p in CP_INSTANCE.participants]].mean(axis=1, skipna=True)
    df['Hitrate SD'] = df[[PARTICIPANT_HITRATE_All_f.format(p) for p in CP_INSTANCE.participants]].std(axis=1, skipna=True)

    # Model columns
    df[MODEL_HITRATE] = (
        main_dataframe[[groupby_column, MODEL_HIT]].astype(float)
        .groupby(groupby_column)
        .sum()[MODEL_HIT]
        / TOTAL_CATEGORIES
    )

    df = df.reset_index()

    return df


def get_hitrate_variance(main_dataframe: DataFrame) -> DataFrame:
    df = DataFrame()
    df[MODEL_HITRATE_PER_CATEGORY] = (
            main_dataframe
            .groupby(CPColNames.Category)
            .mean()[MODEL_HIT])
    for participant in CP_INSTANCE.participants:
        participant_df = DataFrame()
        participant_df[PARTICIPANT_HITRATE_PER_CATEGORY_f.format(participant)] = (
            main_dataframe[main_dataframe[PARTICIPANT_SAW_CATEGORY_f.format(participant)] == True]
            .groupby(CPColNames.Category)
            .mean()[PARTICIPANT_RESPONSE_HIT_f.format(participant)])
        df = df.join(participant_df, how="left")
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
    """
    Renames existing precomputed distance columns.
    Excludes idiosyncratic responses.
    Adds computed columns:
        production proportion,
        ranked production frequency,
        rounded mean rank,
    """

    # Main dataframe holds category production data and model response data
    main_data: DataFrame = CP_INSTANCE.data.copy()

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
