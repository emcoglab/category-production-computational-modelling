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

from pandas import read_csv
from scipy.stats import spearmanr

from category_production.category_production import CategoryProduction
from ldm.core.corpus.indexing import FreqDist
from ldm.preferences.preferences import Preferences as CorpusPreferences
from model.temporal_spreading_activation import ActivatedNodeEvent
from model.utils.email import Emailer
from preferences import Preferences

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


# Results DataFrame column names
RESPONSE = "Response"
NODE_ID = "Node ID"
ACTIVATION = "Activation"
TICK_ON_WHICH_ACTIVATED = "Tick on which activated"


def comment_line_from_str(message: str) -> str:
    return f"# {message}\n"


def main():

    n_words = 10_000

    corpus = CorpusPreferences.source_corpus_metas.bbc
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    filtered_words = set(freq_dist.most_common_tokens(n_words))

    cp = CategoryProduction()

    for category_label in cp.category_labels:

        # Skip the check if the category won't be in the network
        if category_label not in filtered_words:
            continue

        # Dictionary of differently-ordered lists of words
        actual_response_words_by_mean_rank = [r
                                              for r in cp.responses_for_category(category_label, single_word_only=True, sort_by=CategoryProduction.ColNames.MeanRank)
                                              if r in filtered_words]
        n_actual_response_words = len(actual_response_words_by_mean_rank)

        # Load model responses
        model_responses_path = path.join(Preferences.output_dir, f"Category production traces ({n_words:,} words)", f"responses_{category_label}_{n_words:,}.csv")
        model_responses_df = read_csv(model_responses_path, header=0, comments="#", index=False)
        model_response_entries = []
        for row in model_responses_df.sort_values(by=TICK_ON_WHICH_ACTIVATED).iterrows():
            model_response_entries.append(ActivatedNodeEvent(
                node=row[RESPONSE], activation=row[ACTIVATION], tick_activated=row[TICK_ON_WHICH_ACTIVATED]))

        # Get overlap
        model_response_overlap_entries = []
        for mr in model_response_entries:
            # Only interested in overlap
            if mr.node not in actual_response_words_by_mean_rank:
                continue
            # Only interested in unique entries
            if mr.node in [existing_mr.node for existing_mr in model_response_overlap_entries]:
                continue

        coverage_percent = 100 * len(set(model_response_overlap_entries)) / n_actual_response_words

        # Comparison vectors

        # model response vector will contain ticks on which the entry was (first) activated
        model_response_vector_for_overlap = []
        # production frequency vector will contain the production frequency
        production_frequency_vector_for_overlap = []
        # mean rank vector will contain mean ranks
        mean_rank_vector_for_overlap = []
        for common_entry in model_response_overlap_entries:
            model_response_vector_for_overlap.append(common_entry.tick_activated)
            mean_rank_vector_for_overlap.append(cp.data_for_category_response_pair(category_label, common_entry.node, CategoryProduction.ColNames.MeanRank))
            production_frequency_vector_for_overlap.append(cp.data_for_category_response_pair(category_label, common_entry.node, CategoryProduction.ColNames.ProductionFrequency))

        mean_rank_corr, _ = spearmanr(model_response_vector_for_overlap, model_response_vector_for_overlap)
        production_frequency_corr, _ = spearmanr(model_response_vector_for_overlap, production_frequency_vector_for_overlap)

        # endregion

        # region save comparison output file

        model_effectiveness_path = path.join(Preferences.output_dir, f"Category production traces ({n_words:,} words)", f"model_effectiveness_{category_label}_{n_words:,}.csv")

        with open(model_effectiveness_path, mode="w", encoding="utf-8") as model_efficacy_file:

            model_efficacy_file.write("Response overlap:\n")
            model_efficacy_file.write("\t" + "\n\t".join([mr.node for mr in model_response_overlap_entries]) + "\n")
            model_efficacy_file.write("\n")

            model_efficacy_file.write("Coverage: what percentage of human-listed responses are found by the model in the first 1000 ticks or before bailout?\n")
            model_efficacy_file.write("{percent:.2f}%\n".format(percent=coverage_percent))
            model_efficacy_file.write("\n")

            model_efficacy_file.write("Mean rank correlation:\n")
            model_efficacy_file.write(f"\t{mean_rank_corr}")
            model_efficacy_file.write("\n")

            model_efficacy_file.write("Production frequency correlation:\n")
            model_efficacy_file.write(f"\t{production_frequency_corr}")
            model_efficacy_file.write("\n")

            model_efficacy_file.write("Actual responses (those present in corpus, ordered by Mean Rank):\n")
            model_efficacy_file.write("\t" + "\n\t".join(actual_response_words_by_mean_rank) + "\n")
            model_efficacy_file.write("\n")

    emailer = Emailer(Preferences.email_connection_details_path)
    emailer.send_email(f"Done running {path.basename(__file__)} with {n_words} words.", Preferences.target_email_address)


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
