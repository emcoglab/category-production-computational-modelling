import re
import logging
from collections import defaultdict
from math import floor
from os import path, listdir
from typing import DefaultDict, Dict, Set, List, Optional

from matplotlib import pyplot
from numpy import nan
from pandas import DataFrame, read_csv, isna

from category_production.category_production import CategoryProduction, ColNames as CPColNames
from evaluation.comparison import get_summary_table, hitrate_within_sd_of_mean_frac
from ldm.corpus.tokenising import modified_word_tokenize
from model.graph_propagation import GraphPropagation
from model.utils.exceptions import ParseError
from evaluation.column_names import ACTIVATION, TICK_ON_WHICH_ACTIVATED, ITEM_ENTERED_BUFFER, RESPONSE, MODEL_HIT, \
    TTFA, PRODUCTION_PROPORTION, RANK_FREQUENCY_OF_PRODUCTION, ROUNDED_MEAN_RANK, MODEL_HITRATE, CATEGORY_AVAILABLE
from preferences import Preferences

logger = logging.getLogger(__name__)

N_PARTICIPANTS = 20

CATEGORY_PRODUCTION = CategoryProduction(use_cache=True)


def get_n_words_from_path_linguistic(results_dir_path: str) -> int:
    """
    Gets the number of words from a path storing results.
    :param results_dir_path:
    :return: n_words: int
    """
    dir_name = path.basename(results_dir_path)
    words_match = re.match(re.compile(r"[^0-9,]*(?P<n_words>[0-9,]+) words;"), dir_name)
    if words_match:
        # remove the comma and parse as int
        n_words = int(words_match.group("n_words").replace(",", ""))
        return n_words
    else:
        raise ParseError(f"Could not parse number of words from {dir_name}")


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
            model_responses_df: DataFrame = read_csv(model_responses_file, header=0, comment="#", index_col=False)

        ttfas = defaultdict(lambda: nan)
        for row_i, row in model_responses_df.sort_values(by=TICK_ON_WHICH_ACTIVATED).iterrows():

            # Only consider items whose activation exceeded the CAT
            if row[ACTIVATION] < conscious_access_threshold:
                continue

            item_label = row[RESPONSE]

            # We've sorted by activation time, so we only need to consider the first entry for each item
            if item_label not in ttfas.keys():
                ttfas[item_label] = row[TICK_ON_WHICH_ACTIVATED]
        return ttfas

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
            model_responses_df: DataFrame = read_csv(model_responses_file, header=0, comment="#", index_col=False)

        ttfas = defaultdict(lambda: nan)
        for row_i, row in model_responses_df[model_responses_df[ITEM_ENTERED_BUFFER] == True].sort_values(by=TICK_ON_WHICH_ACTIVATED).iterrows():

            item_label = row[RESPONSE]

            # We've sorted by activation time, so we only need to consider the first entry for each item
            if item_label not in ttfas:
                ttfas[item_label] = row[TICK_ON_WHICH_ACTIVATED]
        return ttfas

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


def add_predictor_column_model_hit(main_data):
    """Mutates `main_data`."""
    logger.info("Adding model hit column")
    main_data[MODEL_HIT] = main_data.apply(lambda row: not isna(row[TTFA]), axis=1)


def add_predictor_column_production_proportion(main_data):
    """Mutates `main_data`."""
    logger.info("Adding production proportion column")
    # Production proportion is the number of times a response was given,
    # divided by the number of participants who gave it
    main_data[PRODUCTION_PROPORTION] = main_data.apply(lambda row: row[CPColNames.ProductionFrequency] / N_PARTICIPANTS, axis=1)


def add_rfop_column(main_data):
    """Mutates `main_data`."""
    logger.info("Adding RFoP column")
    main_data[RANK_FREQUENCY_OF_PRODUCTION] = (
        main_data
        # Within each category
        .groupby(CPColNames.CategorySensorimotor)
        # Rank the responses according to production frequency
        [CPColNames.ProductionFrequency]
        .rank(ascending=False,
              # For ties, order alphabetically (i.e. pseudorandomly (?))
              method='first'))


def add_rmr_column(main_data):
    """Mutates `main_data`."""
    logger.info("Adding RMR column")
    main_data[ROUNDED_MEAN_RANK] = main_data.apply(lambda row: floor(row[CPColNames.MeanRank]), axis=1)


def add_predictor_column_ttfa(main_data, ttfas: Dict[str, Dict[str, int]], sensorimotor: bool):
    """Mutates `main_data`."""
    logger.info("Adding TTFA column")

    def get_min_ttfa_for_multiword_responses(row) -> int:
        """
        Helper function to convert a row in the output into a ttfa when the response is formed either of a single or
        multiple norms terms.
        """

        if sensorimotor:
            c = row[CPColNames.CategorySensorimotor]
            r = row[CPColNames.ResponseSensorimotor]
        else:
            c = row[CPColNames.Category]
            r = row[CPColNames.Response]

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


def save_item_level_data(main_data: DataFrame, save_path):
    main_data.to_csv(save_path, index=False)


def save_figure(summary_table, x_selector, fig_title, fig_name, sensorimotor: bool):
    """Save a summary table as a figure."""
    # add human bounds
    pyplot.fill_between(x=summary_table.reset_index()[x_selector],
                        y1=summary_table[PRODUCTION_PROPORTION + ' Mean'] - summary_table[
                            PRODUCTION_PROPORTION + ' SD'],
                        y2=summary_table[PRODUCTION_PROPORTION + ' Mean'] + summary_table[
                            PRODUCTION_PROPORTION + ' SD'])
    pyplot.scatter(x=summary_table.reset_index()[x_selector],
                   y=summary_table[PRODUCTION_PROPORTION + ' Mean'])
    # add model performance
    pyplot.scatter(x=summary_table.reset_index()[x_selector],
                   y=summary_table[MODEL_HITRATE])

    pyplot.ylim((0, None))

    pyplot.title(fig_title)
    pyplot.xlabel(x_selector)
    pyplot.ylabel("Production proportion / hitrate")

    pyplot.savefig(
        path.join(Preferences.figures_dir,
                  f"hitrates{' sensorimotor' if sensorimotor else ''}",
                  f"{fig_name}.png"))
    pyplot.clf()
    pyplot.cla()
    pyplot.close()


def save_hitrate_summary_tables(input_results_dir: str, main_data: DataFrame, sensorimotor: bool):
    production_proportion_per_rfop = get_summary_table(main_data, RANK_FREQUENCY_OF_PRODUCTION)
    production_proportion_per_rfop_restricted = get_summary_table(
        main_data[main_data[CATEGORY_AVAILABLE]],
        RANK_FREQUENCY_OF_PRODUCTION)
    # Production proportion per rounded mean rank
    production_proportion_per_rmr = get_summary_table(main_data, ROUNDED_MEAN_RANK)
    production_proportion_per_rmr_restricted = get_summary_table(main_data[main_data[CATEGORY_AVAILABLE]],
                                                                 ROUNDED_MEAN_RANK)

    # Compute hitrate fits
    hitrate_fit_rfop = hitrate_within_sd_of_mean_frac(production_proportion_per_rfop)
    hitrate_fit_rfop_restricted = hitrate_within_sd_of_mean_frac(production_proportion_per_rfop_restricted)
    hitrate_fit_rmr = hitrate_within_sd_of_mean_frac(production_proportion_per_rmr)
    hitrate_fit_rmr_restricted = hitrate_within_sd_of_mean_frac(production_proportion_per_rmr_restricted)

    # Save summary tables
    base_dir = path.join(Preferences.results_dir, f"Category production fit{' sensorimotor' if sensorimotor else ''}")
    production_proportion_per_rfop.to_csv(path.join(base_dir,
                                                    f"Production proportion per rank frequency of production"
                                                    f" ({path.basename(input_results_dir)}).csv"),
                                          index=False)
    production_proportion_per_rmr.to_csv(path.join(base_dir,
                                                   f"Production proportion per rounded mean rank"
                                                   f" ({path.basename(input_results_dir)}).csv"),
                                         index=False)
    production_proportion_per_rfop_restricted.to_csv(path.join(base_dir,
                                                               f"Production proportion per rank frequency of production"
                                                               f" ({path.basename(input_results_dir)}) restricted.csv"),
                                                     index=False)
    production_proportion_per_rmr_restricted.to_csv(path.join(base_dir,
                                                              f"Production proportion per rounded mean rank"
                                                              f" ({path.basename(input_results_dir)}) restricted.csv"),
                                                    index=False)

    # region Graph tables

    save_figure(summary_table=production_proportion_per_rfop,
                x_selector=RANK_FREQUENCY_OF_PRODUCTION,
                fig_title="Hitrate per RFOP",
                fig_name=f"hitrate per RFOP {path.basename(input_results_dir)}",
                sensorimotor=sensorimotor)
    save_figure(summary_table=production_proportion_per_rfop_restricted,
                x_selector=RANK_FREQUENCY_OF_PRODUCTION,
                fig_title="Hitrate per RFOP (only available categories)",
                fig_name=f"restricted hitrate per RFOP {path.basename(input_results_dir)}",
                sensorimotor=sensorimotor)
    save_figure(summary_table=production_proportion_per_rmr,
                x_selector=ROUNDED_MEAN_RANK,
                fig_title="Hitrate per RMR",
                fig_name=f"hitrate per RMR {path.basename(input_results_dir)}",
                sensorimotor=sensorimotor)
    save_figure(summary_table=production_proportion_per_rmr_restricted,
                x_selector=ROUNDED_MEAN_RANK,
                fig_title="Hitrate per RMR (only available categories)",
                fig_name=f"restricted hitrate per RMR {path.basename(input_results_dir)}",
                sensorimotor=sensorimotor)

    # endregion

    hitrate_stats = {
        "hitrate_fit_rfop": hitrate_fit_rfop,
        "hitrate_fit_rfop_restricted": hitrate_fit_rfop_restricted,
        "hitrate_fit_rmr": hitrate_fit_rmr,
        "hitrate_fit_rmr_restricted": hitrate_fit_rmr_restricted
    }

    return hitrate_stats


def save_model_performance_stats(main_dataframe,
                                 results_dir,
                                 min_first_rank_freq,
                                 hitrate_fit_rfop,
                                 hitrate_fit_rfop_restricted,
                                 hitrate_fit_rmr,
                                 hitrate_fit_rmr_restricted,
                                 sensorimotor: bool):

    overall_stats_output_path = path.join(Preferences.results_dir,
                                          f"Category production fit{' sensorimotor' if sensorimotor else ''}",
                                          f"model_effectiveness_overall "
                                          f"({path.basename(results_dir)}).csv")
    model_spec = GraphPropagation.load_model_spec(results_dir)
    stats = {
        **get_correlation_stats(main_dataframe, min_first_rank_freq, sensorimotor=sensorimotor),
        # hitrate stats
        "Hitrate within SD of mean (RFoP)": hitrate_fit_rfop,
        "Hitrate within SD of mean (RFoP; available categories only)": hitrate_fit_rfop_restricted,
        "Hitrate within SD of mean (RMR)": hitrate_fit_rmr,
        "Hitrate within SD of mean (RMR; available categories only)": hitrate_fit_rmr_restricted,
    }
    model_performance_data: DataFrame = DataFrame.from_records([{
        **model_spec,
        **stats,
    }])

    model_performance_data.to_csv(overall_stats_output_path,
                                  # Make sure columns are in consistent order for stacking,
                                  # and make sure the model spec columns come first.
                                  columns=sorted(model_spec.keys()) + sorted(stats.keys()),
                                  index=False)


def get_correlation_stats(correlation_dataframe, min_first_rank_freq, sensorimotor: bool):
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
    if sensorimotor:
        available_pairs = set(correlation_dataframe[[CPColNames.Category, CPColNames.Response]]
                              .groupby([CPColNames.Category, CPColNames.Response])
                              .groups.keys())
    else:
        available_pairs = set(correlation_dataframe[[CPColNames.Category, CPColNames.Response]]
                              .groupby([CPColNames.Category, CPColNames.Response])
                              .groups.keys())

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


def drop_missing_data(main_data: DataFrame, distance_column: Optional[str]):
    """
    Mutates `main_data`.

    Set `distance_column` to None to skip it.
    :param main_data:
    :param distance_column:
    :return:
    """
    if distance_column is not None:
        main_data.dropna(inplace=True, how='any', subset=[TTFA, distance_column])
    else:
        main_data.dropna(inplace=True, how='any', subset=[TTFA])
    # Now we can convert TTFAs to ints and distances to floats as there won't be null values
    main_data[TTFA] = main_data[TTFA].astype(int)
    if distance_column is not None:
        main_data[distance_column] = main_data[distance_column].astype(float)
