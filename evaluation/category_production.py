import re
from collections import defaultdict
from os import path
from typing import DefaultDict

from numpy import nan
from pandas import DataFrame, read_csv

from model.graph_propagation import GraphPropagation
from model.utils.exceptions import ParseError
from preferences import Preferences

# Results DataFrame column names
RESPONSE = "Response"
NODE_ID = "Node ID"
ACTIVATION = "Activation"
TICK_ON_WHICH_ACTIVATED = "Tick on which activated"
ITEM_ENTERED_BUFFER = "Item entered WM buffer"
TTFA = "TTFA"
REACHED_CAT = "Reached conc.acc. θ"


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
        model_responses_path = path.join(
            results_dir,
            f"responses_{category}_{n_words:,}.csv")
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


def get_model_ttfas_for_category_sensorimotor(category: str, results_dir: str) -> DefaultDict[str, int]:
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
        model_responses_path = path.join(
            results_dir,
            f"responses_{category}.csv")
        with open(model_responses_path, mode="r", encoding="utf-8") as model_responses_file:
            model_responses_df: DataFrame = read_csv(model_responses_file, header=0, comment="#", index_col=False)

        ttfas = defaultdict(lambda: nan)
        for row_i, row in model_responses_df.sort_values(by=TICK_ON_WHICH_ACTIVATED).iterrows():

            item_label = row[RESPONSE]

            # Only interested if item entered the buffer
            if not row[ITEM_ENTERED_BUFFER]:
                continue

            # We've sorted by activation time, so we only need to consider the first entry for each item
            if item_label not in ttfas.keys():
                ttfas[item_label] = row[TICK_ON_WHICH_ACTIVATED]
        return ttfas

    # If the category wasn't found, there are no TTFAs
    except FileNotFoundError:
        return defaultdict(lambda: nan)


# TODO: these two functions are baseically identical. Can they be merged into one?
def save_stats_linguistic(available_items, corr_frf_vs_ttfa, corr_meanrank_vs_ttfa, corr_prodfreq_vs_ttfa,
                          first_rank_frequent_corr_rt_vs_ttfa, n_first_rank_frequent, results_dir, restricted,
                          min_first_rank_freq, conscious_access_threshold):
    overall_stats_output_path = path.join(Preferences.results_dir,
                                          "Category production fit",
                                          f"model_effectiveness_overall {'(restricted) ' if restricted else ''}"
                                          f"({path.basename(results_dir)}) CAT={conscious_access_threshold}.csv")

    data: DataFrame = DataFrame.from_records([{
        # Include model spec
        **GraphPropagation.load_model_spec(results_dir),
        "CAT":                                      conscious_access_threshold,
        "FRF corr (-)":                             corr_frf_vs_ttfa,
        "FRF N":                                    n_first_rank_frequent,
        f"zRT corr (+; FRF≥{min_first_rank_freq})": first_rank_frequent_corr_rt_vs_ttfa,
        "zRT N":                                    n_first_rank_frequent,
        "ProdFreq corr (-)":                        corr_prodfreq_vs_ttfa,
        "ProdFreq N":                               len(available_items),
        "MeanRank corr (+)":                        corr_meanrank_vs_ttfa,
        "Mean Rank N":                              len(available_items),
    }])

    with open(overall_stats_output_path, mode="w", encoding="utf-8") as data_file:
        data.to_csv(data_file, index=False)


def save_stats_sensorimotor(available_items, corr_frf_vs_ttfa, corr_meanrank_vs_ttfa, corr_prodfreq_vs_ttfa,
                            first_rank_frequent_corr_rt_vs_ttfa, n_first_rank_frequent, results_dir,
                            restricted, min_first_rank_freq):

    overall_stats_output_path = path.join(Preferences.results_dir,
                                          "Category production fit",
                                          f"model_effectiveness_overall {'(restricted) ' if restricted else ''}"
                                          f"({path.basename(results_dir)}).csv")

    data: DataFrame = DataFrame.from_records([{
        # Include model spec
        **GraphPropagation.load_model_spec(results_dir),
        "FRF corr (-)":                             corr_frf_vs_ttfa,
        "FRF N":                                    n_first_rank_frequent,
        f"zRT corr (+; FRF≥{min_first_rank_freq})": first_rank_frequent_corr_rt_vs_ttfa,
        "zRT N":                                    n_first_rank_frequent,
        "ProdFreq corr (-)":                        corr_prodfreq_vs_ttfa,
        "ProdFreq N":                               len(available_items),
        "MeanRank corr (+)":                        corr_meanrank_vs_ttfa,
        "Mean Rank N":                              len(available_items),
    }])

    with open(overall_stats_output_path, mode="w", encoding="utf-8") as data_file:
        data.to_csv(data_file, index=False)
