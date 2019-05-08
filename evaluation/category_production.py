import re
from collections import defaultdict
from os import path
from typing import DefaultDict

from numpy import nan
from pandas import DataFrame, read_csv

from model.component import ItemActivatedEvent, load_model_spec
from model.utils.exceptions import ParseError
from preferences import Preferences

# Results DataFrame column names
RESPONSE = "Response"
NODE_ID = "Node ID"
ACTIVATION = "Activation"
TICK_ON_WHICH_ACTIVATED = "Tick on which activated"
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

            activation_event = ItemActivatedEvent(label=row[RESPONSE],
                                                  activation=row[ACTIVATION],
                                                  time_activated=row[TICK_ON_WHICH_ACTIVATED])
            # We've sorted by activation time, so we only need to consider the first entry for each item
            if activation_event.label not in ttfas.keys():
                ttfas[activation_event.label] = activation_event.time_activated
        return ttfas

    # If the category wasn't found, there are no TTFAs
    except FileNotFoundError:
        return defaultdict(lambda: nan)


def get_model_ttfas_for_category_sensorimotor(category: str,
                                              results_dir: str) -> DefaultDict[str, int]:
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

            activation_event = ItemActivatedEvent(label=row[RESPONSE],
                                                  activation=row[ACTIVATION],
                                                  time_activated=row[TICK_ON_WHICH_ACTIVATED])
            # We've sorted by activation time, so we only need to consider the first entry for each item
            if activation_event.label not in ttfas.keys():
                ttfas[activation_event.label] = activation_event.time_activated
        return ttfas

    # If the category wasn't found, there are no TTFAs
    except FileNotFoundError:
        return defaultdict(lambda: nan)


def save_stats_linguistic(available_items, corr_frf_vs_ttfa, corr_meanrank_vs_ttfa, corr_prodfreq_vs_ttfa,
                          first_rank_frequent_corr_rt_vs_ttfa, n_first_rank_frequent, results_dir, restricted, min_first_rank_freq, conscious_access_threshold):
    overall_stats_output_path = path.join(Preferences.results_dir,
                                          "Category production fit",
                                          f"model_effectiveness_overall {'(restricted) ' if restricted else ''}"
                                          f"({path.basename(results_dir)}) CAT={conscious_access_threshold}.csv")

    model_spec = load_model_spec(results_dir)

    data_records = {
        f"Model":                                   model_spec["Model name"],
        f"Length factor":                           model_spec["Length factor"],
        f"SD factor":                               model_spec["SD factor"],
        f"Firing threshold":                        model_spec["Firing threshold"],
        f"Words":                                   model_spec["Words"],
        f"CAT":                                     conscious_access_threshold,
        f"FRF corr (-)":                            corr_frf_vs_ttfa,
        f"FRF N":                                   n_first_rank_frequent,
        f"zRT corr (+; FRF≥{min_first_rank_freq})": first_rank_frequent_corr_rt_vs_ttfa,
        f"zRT N":                                   n_first_rank_frequent,
        f"ProdFreq corr (-)":                       corr_prodfreq_vs_ttfa,
        f"ProdFreq N":                              len(available_items),
        f"MeanRank corr (+)":                       corr_meanrank_vs_ttfa,
        f"Mean Rank N":                             len(available_items),
    }
    data: DataFrame = DataFrame.from_records([data_records])

    with open(overall_stats_output_path, mode="w", encoding="utf-8") as data_file:
        data.to_csv(data_file, index=False, columns=[
            f"Model",
            f"Length factor",
            f"SD factor",
            f"Firing threshold",
            f"Words",
            f"CAT",
            f"FRF corr (-)",
            f"FRF N",
            f"zRT corr (+; FRF≥{min_first_rank_freq})",
            f"zRT N",
            f"ProdFreq corr (-)",
            f"ProdFreq N",
            f"MeanRank corr (+)",
            f"Mean Rank N",
        ])


def save_stats_sensorimotor(available_items, corr_frf_vs_ttfa, corr_meanrank_vs_ttfa, corr_prodfreq_vs_ttfa,
                            first_rank_frequent_corr_rt_vs_ttfa, n_first_rank_frequent, results_dir,
                            restricted, min_first_rank_freq):
    overall_stats_output_path = path.join(Preferences.results_dir,
                                          "Category production fit",
                                          f"model_effectiveness_overall {'(restricted) ' if restricted else ''}"
                                          f"({path.basename(results_dir)}).csv")

    model_spec = load_model_spec(results_dir)

    data_records = {
        f"Length factor":                           model_spec["Length factor"],
        f"Max sphere radius":                       model_spec["Max sphere radius"],
        f"Run for ticks":                           model_spec["Run for ticks"],
        f"Sigma":                                   model_spec["Log-normal sigma"],
        f"Bailout":                                 model_spec["Bailout"],
        f"FRF corr (-)":                            corr_frf_vs_ttfa,
        f"FRF N":                                   n_first_rank_frequent,
        f"zRT corr (+; FRF≥{min_first_rank_freq})": first_rank_frequent_corr_rt_vs_ttfa,
        f"zRT N":                                   n_first_rank_frequent,
        f"ProdFreq corr (-)":                       corr_prodfreq_vs_ttfa,
        f"ProdFreq N":                              len(available_items),
        f"MeanRank corr (+)":                       corr_meanrank_vs_ttfa,
        f"Mean Rank N":                             len(available_items),
    }
    data: DataFrame = DataFrame.from_records([data_records])

    with open(overall_stats_output_path, mode="w", encoding="utf-8") as data_file:
        data.to_csv(data_file, index=False, columns=[
            f"Length factor",
            f"Max sphere radius",
            f"Run for ticks",
            f"Sigma",
            f"Bailout",
            f"FRF corr (-)",
            f"FRF N",
            f"zRT corr (+; FRF≥{min_first_rank_freq})",
            f"zRT N",
            f"ProdFreq corr (-)",
            f"ProdFreq N",
            f"MeanRank corr (+)",
            f"Mean Rank N",
        ])
