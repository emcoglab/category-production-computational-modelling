"""
===========================
Compare model to Briony's category production actual responses.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2018
---------------------------
"""

import logging
import sys
from os import path

from numpy import nan, mean
from pandas import read_csv, DataFrame
from scipy.stats import spearmanr, pearsonr

from category_production.category_production import CategoryProduction
from ldm.core.corpus.indexing import FreqDist
from ldm.preferences.preferences import Preferences as CorpusPreferences
from model.temporal_spreading_activation import ActivatedNodeEvent
from preferences import Preferences

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


# Results DataFrame column names
# TODO: repeated values 😕
RESPONSE = "Response"
NODE_ID = "Node ID"
ACTIVATION = "Activation"
TICK_ON_WHICH_ACTIVATED = "Tick on which activated"


def comment_line_from_str(message: str) -> str:
    return f"# {message}\n"


def main():

    n_words: int = 3_000
    min_first_rank_freq: int = 4

    corpus = CorpusPreferences.source_corpus_metas.bbc
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    filtered_words = set(freq_dist.most_common_tokens(n_words))

    cp = CategoryProduction()

    category_comparisons = []

    # Average RTs for all category–first-rank-member pairs (with FRF ≥ `min_first_rank_freq`)
    first_rank_mean_rts = []
    # Corresponding time-to-activation of member nodes from category seed
    first_rank_tsa_times = []

    for category_label in cp.category_labels:

        # Skip the check if the category won't be in the network
        if category_label not in filtered_words:
            continue

        # Dictionary of differently-ordered lists of words
        actual_response_words = [r
                                 for r in cp.responses_for_category(category_label, single_word_only=True)
                                 if r in filtered_words]
        n_actual_responses_in_corpus = len(actual_response_words)
        response_corpus_coverage_percent = 100 * n_actual_responses_in_corpus / len(cp.responses_for_category(category_label, single_word_only=True))

        # Load model responses
        try:
            model_responses_path = path.join(Preferences.output_dir, f"Category production traces ({n_words:,} words)", f"responses_{category_label}_{n_words:,}.csv")
            with open(model_responses_path, mode="r", encoding="utf-8") as model_responses_file:
                model_responses_df = read_csv(model_responses_file, header=0, comment="#", index_col=False)
        except FileNotFoundError as e:
            # Skip any we don't have yet
            logger.warning(f"File not found: {e.filename}")
            continue

        model_response_entries = []
        for row_i, row in model_responses_df.sort_values(by=TICK_ON_WHICH_ACTIVATED).iterrows():
            model_response_entries.append(ActivatedNodeEvent(
                node=row[RESPONSE], activation=row[ACTIVATION], tick_activated=row[TICK_ON_WHICH_ACTIVATED]))

        # Get overlap
        model_response_overlap_entries = []
        for mr in model_response_entries:
            # Only interested in overlap
            if mr.node not in actual_response_words:
                continue
            # Only interested in unique entries
            if mr.node in [existing_mr.node for existing_mr in model_response_overlap_entries]:
                continue
            model_response_overlap_entries.append(mr)

        overlap_size = len(model_response_overlap_entries)
        if n_actual_responses_in_corpus > 0:
            overlap_percent = 100 * overlap_size / n_actual_responses_in_corpus
        else:
            overlap_percent = nan

        # Comparison vectors

        # model response vector will contain ticks on which the entry was (first) activated
        model_time_to_first_activation = []
        # production frequency vector will contain the production frequency
        production_frequencies = []
        # mean rank vector will contain mean ranks
        mean_ranks = []
        for common_entry in model_response_overlap_entries:
            model_time_to_first_activation.append(common_entry.tick_activated)
            mean_ranks.append(cp.data_for_category_response_pair(category_label, common_entry.node, CategoryProduction.ColNames.MeanRank))
            production_frequencies.append(cp.data_for_category_response_pair(category_label, common_entry.node, CategoryProduction.ColNames.ProductionFrequency))

        # noinspection PyTypeChecker
        mean_rank_corr, _ = spearmanr(model_time_to_first_activation, mean_ranks)
        # noinspection PyTypeChecker
        production_frequency_corr, _ = spearmanr(model_time_to_first_activation, production_frequencies)

        # First rank frequency RTs

        # Get the first-rank responses
        first_rank_entries_this_cat = [r
                                       # which were also found by the model
                                       for r in model_response_overlap_entries
                                       # but only those above threshold
                                       if cp.data_for_category_response_pair(category_label, r.node, CategoryProduction.ColNames.FirstRankFrequency) >= min_first_rank_freq]
        for response in first_rank_entries_this_cat:
            first_rank_mean_rts.append(mean(list(cp.rts_for_category_response_pair(category_label, response.node))))
            first_rank_tsa_times.append(response.tick_activated)

        category_comparisons.append((
            category_label,
            n_actual_responses_in_corpus,
            response_corpus_coverage_percent,
            overlap_size,
            overlap_percent,
            str([e.node for e in model_response_overlap_entries]),
            str(model_time_to_first_activation),
            str(mean_ranks),
            mean_rank_corr,
            str(production_frequencies),
            production_frequency_corr,
        ))

    first_rank_rt_corr, _ = pearsonr(first_rank_mean_rts, first_rank_tsa_times)
    logger.info(
        f"First response RT correlation (Pearson's; positive is better fit; FRF≥{min_first_rank_freq}) = {first_rank_rt_corr} (N = {len(first_rank_mean_rts)})")

    model_effectiveness_path = path.join(Preferences.output_dir, f"Category production traces ({n_words:,} words)", f"model_effectiveness_{n_words:,}.csv")

    category_comparisons_df = DataFrame(category_comparisons, columns=[
        f"Category",
        f"Actual responses in corpus",
        f"Corpus coverage of actual responses (%)",
        f"Model response overlap size",
        f"Model response overlap %",
        f"Model response overlap words",
        f"Model: time to first activation",
        f"Mean ranks",
        f"Production frequencies",
        f"Mean rank correlation (Spearman's; positive is better fit)",
        f"Production frequency correlation (Spearman's; negative is better fit)",
    ])
    with open(model_effectiveness_path, mode="w", encoding="utf-8") as output_file:
        category_comparisons_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
