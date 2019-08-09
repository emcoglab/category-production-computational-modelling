import re
from collections import defaultdict
from os import path, listdir
from typing import DefaultDict, Dict, Set, List

from numpy import nan
from pandas import DataFrame, read_csv

from model.graph_propagation import GraphPropagation
from model.utils.exceptions import ParseError
from evaluation.column_names import ACTIVATION, TICK_ON_WHICH_ACTIVATED, ITEM_ENTERED_BUFFER, RESPONSE
from preferences import Preferences


N_PARTICIPANTS = 20


def interpret_path_linguistic(results_dir_path: str) -> int:
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


def get_model_ttfas_for_category_sensorimotor(category: str, results_dir: str) -> Dict[str, int]:
    """
    Dictionary of
        response -> time to first activation
    for the specified category.
    :param category:
    :param results_dir:
    """

    # Try to load model response
    try:
        model_responses_path = path.join(
            results_dir,
            f"responses_{category}.csv")
        with open(model_responses_path, mode="r", encoding="utf-8") as model_responses_file:
            model_responses_df: DataFrame = read_csv(model_responses_file, header=0, comment="#", index_col=False)

        # We're not using the nan values here, so we just use a straight dictionary
        ttfas = dict()
        for row_i, row in model_responses_df[model_responses_df[ITEM_ENTERED_BUFFER] == True].sort_values(by=TICK_ON_WHICH_ACTIVATED).iterrows():

            item_label = row[RESPONSE]

            # We've sorted by activation time, so we only need to consider the first entry for each item
            if item_label not in ttfas:
                ttfas[item_label] = row[TICK_ON_WHICH_ACTIVATED]
        return ttfas

    # If the category wasn't found, there are no TTFAs
    except FileNotFoundError:
        return defaultdict(lambda: nan)


def save_stats(available_items,
               corr_frf_vs_ttfa,
               corr_meanrank_vs_ttfa,
               corr_prodfreq_vs_ttfa,
               first_rank_frequent_corr_rt_vs_ttfa,
               n_first_rank_frequent,
               results_dir,
               min_first_rank_freq,
               hitrate_fit_rfop,
               hitrate_fit_rfop_available_cats_only,
               hitrate_fit_rmr,
               hitrate_fit_rmr_available_cats_only,
               # restrict to TODO
               restricted=False,
               conscious_access_threshold=nan,
               ):
    overall_stats_output_path = path.join(Preferences.results_dir,
                                          "Category production fit",
                                          f"model_effectiveness_overall {'(restricted) ' if restricted else ''}"
                                          f"({path.basename(results_dir)}) CAT={conscious_access_threshold}.csv")
    model_spec = GraphPropagation.load_model_spec(results_dir)
    stats = {
        "CAT":                                      conscious_access_threshold,
        "FRF corr (-)":                             corr_frf_vs_ttfa,
        "FRF N":                                    n_first_rank_frequent,
        f"zRT corr (+; FRF≥{min_first_rank_freq})": first_rank_frequent_corr_rt_vs_ttfa,
        "zRT N":                                    n_first_rank_frequent,
        "ProdFreq corr (-)":                        corr_prodfreq_vs_ttfa,
        "ProdFreq N":                               len(available_items),
        "MeanRank corr (+)":                        corr_meanrank_vs_ttfa,
        "Mean Rank N":                              len(available_items),
        # hitrate stats
        "Hitrate within SD of mean (RFoP)":         hitrate_fit_rfop,
        "Hitrate within SD of mean (RFoP;"
        " available categories only)":              hitrate_fit_rfop_available_cats_only,
        "Hitrate within SD of mean (RMR)":          hitrate_fit_rmr,
        "Hitrate within SD of mean (RMR;"
        " available categories only)":              hitrate_fit_rmr_available_cats_only,
    }
    data: DataFrame = DataFrame.from_records([{
        **model_spec,
        **stats,
    }])

    with open(overall_stats_output_path, mode="w", encoding="utf-8") as data_file:
        data.to_csv(data_file, index=False,
                    # Make sure columns are in consistent order for stacking,
                    # and make sure the model spec columns come first.
                    columns=sorted(model_spec.keys()) + sorted(stats.keys()))
