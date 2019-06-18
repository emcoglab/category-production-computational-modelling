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
import logging
import sys
from os import path
from typing import Dict

from numpy import nan, array
from pandas import DataFrame

from category_production.category_production import CategoryProduction
from category_production.category_production import ColNames as CPColNames
from evaluation.category_production import TTFA, get_model_ttfas_for_category_sensorimotor, save_stats_sensorimotor
from ldm.corpus.tokenising import modified_word_tokenize
from ldm.utils.maths import DistanceType, distance
from preferences import Preferences
from sensorimotor_norms.exceptions import WordNotInNormsError
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


category_production = CategoryProduction(use_cache=True)
sensorimotor_norms = SensorimotorNorms()


def get_sensorimotor_distance_minkowski3(row):
    c = row[CPColNames.CategorySensorimotor]
    r = row[CPColNames.ResponseSensorimotor]

    try:
        category_vector = array(sensorimotor_norms.vector_for_word(c))
    except WordNotInNormsError:
        return nan

    try:
        response_vector = array(sensorimotor_norms.vector_for_word(r))
    except WordNotInNormsError:
        return nan

    return distance(category_vector, response_vector, DistanceType.Minkowski3)


def main(input_results_dir: str, min_first_rank_freq: int = None):

    # Set defaults
    if min_first_rank_freq is None:
        min_first_rank_freq = 1

    logger.info(f"Looking at output from model.")

    # region Build main dataframe and save item-level data

    # Main dataframe holds category production data and model response data
    main_dataframe: DataFrame = category_production.data.copy()

    main_dataframe = add_predictor_columns_ttfa_distance(main_dataframe, input_results_dir, distance_column=f"{DistanceType.Minkowski3.name} distance")

    # Save item-level data
    per_category_stats_output_path = path.join(Preferences.results_dir,
                                               "Category production fit",
                                               f"item-level data ({path.basename(input_results_dir)}).csv")
    main_dataframe.to_csv(per_category_stats_output_path, index=False)

    # endregion

    # region Compute correlations with DVs

    # frf vs ttfa
    corr_frf_vs_ttfa = main_dataframe[CPColNames.FirstRankFrequency].corr(main_dataframe[TTFA], method='pearson')

    # The Pearson's correlation over all categories between average first-response RT (for responses with
    # first-rank frequency ≥4) and the time to the first activation (TTFA) within the model.
    first_rank_frequent_data = main_dataframe[main_dataframe[CPColNames.FirstRankFrequency] >= min_first_rank_freq]
    n_first_rank_frequent = first_rank_frequent_data.shape[0]
    first_rank_frequent_corr_rt_vs_ttfa = first_rank_frequent_data[CPColNames.MeanZRT].corr(
        first_rank_frequent_data[TTFA], method='pearson')

    # Pearson's correlation between production frequency and ttfa
    corr_prodfreq_vs_ttfa = main_dataframe[CPColNames.ProductionFrequency].corr(main_dataframe[TTFA], method='pearson')

    corr_meanrank_vs_ttfa = main_dataframe[CPColNames.MeanRank].corr(main_dataframe[TTFA], method='pearson')

    # endregion

    # Save model-performance stats
    available_pairs = set(main_dataframe[[CPColNames.CategorySensorimotor, CPColNames.ResponseSensorimotor]]
                          .groupby([CPColNames.CategorySensorimotor, CPColNames.ResponseSensorimotor])
                          .groups.keys())
    save_stats_sensorimotor(available_pairs, corr_frf_vs_ttfa, corr_meanrank_vs_ttfa, corr_prodfreq_vs_ttfa,
                            first_rank_frequent_corr_rt_vs_ttfa, n_first_rank_frequent, input_results_dir,
                            False, min_first_rank_freq)


def add_predictor_columns_ttfa_distance(main_dataframe, input_results_dir, distance_column):
    """Add columns for TTFA and distance, dropping appropriate rows."""

    # Drop precomputed distance measures
    main_dataframe.drop(['Sensorimotor.distance.cosine.additive', 'Linguistic.PPMI'], axis=1, inplace=True)

    # Get TTFAs and distances
    model_ttfas: Dict[str, Dict[str, int]] = dict()  # category -> response -> TTFA
    for category in category_production.category_labels_sensorimotor:
        model_ttfas[category] = get_model_ttfas_for_category_sensorimotor(category, input_results_dir)

    def get_min_ttfa_for_multiword_responses(row) -> int:
        """
        Helper function to convert a row in the output into a ttfa when the response is formed either of a single or
        multiple norms terms.
        """
        c = row[CPColNames.CategorySensorimotor]
        r = row[CPColNames.ResponseSensorimotor]

        # If the category is not found in the dictionary, it was not accessed by the model so no TTFA will be present
        # for any response.
        try:
            # response -> TTFA
            c_ttfas: Dict[str, int] = model_ttfas[c]
        except KeyError:
            return nan

        # If the response was directly found, we can return it
        if r in c_ttfas:
            return c_ttfas[r]

        # Otherwise, try to break the response into components and find any one of them
        else:
            r_ttfas = [c_ttfas[w]
                       for w in modified_word_tokenize(r)
                       if (w not in category_production.ignored_words)
                       and (w in c_ttfas)]

            # The multi-word response is said to be activated the first time any one of its constituent words are
            if len(r_ttfas) > 1:
                return min(r_ttfas)
            # If none of the constituent words have a ttfa, we have no ttfa for the multiword term
            else:
                return nan

    main_dataframe[TTFA] = main_dataframe.apply(get_min_ttfa_for_multiword_responses, axis=1)
    main_dataframe[distance_column] = main_dataframe.apply(get_sensorimotor_distance_minkowski3, axis=1)

    # Drop rows corresponding to responses which weren't produced by the model
    main_dataframe = main_dataframe[main_dataframe[TTFA].notnull()]
    main_dataframe = main_dataframe[main_dataframe[distance_column].notnull()]

    # Now we can convert TTFAs to ints and distances to floats as there won't be null values
    main_dataframe[TTFA] = main_dataframe[TTFA].astype(int)
    main_dataframe[distance_column] = main_dataframe[distance_column].astype(float)
    return main_dataframe


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Compare spreading activation results with Category Production data.")
    parser.add_argument("path", type=str, help="The path in which to find the results.")
    parser.add_argument("min_frf", type=int, nargs="?", default=None, help="The minimum FRF required for zRT and FRF correlations.")
    args = parser.parse_args()

    main(args.path, args.min_frf)

    logger.info("Done!")
