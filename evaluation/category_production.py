import re
from logging import getLogger
from collections import defaultdict
from os import path
from typing import DefaultDict

from numpy import nan
from pandas import DataFrame, read_csv

from model.component import ItemActivatedEvent
from model.utils.exceptions import ParseError

logger = getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


# Results DataFrame column names
RESPONSE = "Response"
NODE_ID = "Node ID"
ACTIVATION = "Activation"
TICK_ON_WHICH_ACTIVATED = "Tick on which activated"
TTFA = "TTFA"
REACHED_CAT = "Reached conc.acc. θ"


def interpret_path(results_dir_path: str) -> int:
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
        logger.info(f"Results are from graph with {n_words:,} words")
        return n_words
    else:
        raise ParseError(f"Could not parse number of words from {dir_name}")


def get_model_ttfas_for_category(category: str, results_dir: str, n_words: int) -> DefaultDict[str, int]:
    """
    Dictionary of
        response -> time to first activation
    for the specified category.

    DefaultDict gives nans where response not found

    :param category:
    :param results_dir:
    :param n_words:
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
            if not row[REACHED_CAT]:
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
