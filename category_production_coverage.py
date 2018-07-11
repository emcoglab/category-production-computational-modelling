"""
===========================
Corpus coverage of category production data.
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

from numpy import nan as nnan
from pandas import DataFrame

from category_production.category_production import CategoryProduction
from ldm.core.corpus.indexing import FreqDist
from ldm.preferences.preferences import Preferences as CorpusPreferences
from preferences import Preferences

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def briony_vocab_overlap(top_n_words):
    """Coverage of vocabulary in Briony's Category Production dataset."""
    # Category production words
    category_production = CategoryProduction()
    category_production_words = category_production.vocabulary_single_word

    # Frequent words in corpus
    corpus = CorpusPreferences.source_corpus_metas.bbc
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    corpus_words = freq_dist.most_common_tokens(top_n_words)

    # Useful numbers to report
    n_cat_prod_words = len(category_production_words)
    n_intersection = len(set(category_production_words).intersection(set(corpus_words)))
    n_missing = len(set(category_production_words) - set(corpus_words))
    n_unused = len(set(corpus_words) - set(category_production_words))

    logger.info(f"Category production dataset contains {n_cat_prod_words} words.")
    logger.info(f"This includes {n_intersection} of the top {top_n_words} words in the {corpus.name} corpus.")
    logger.info(f"{n_missing} category production words are not present.")
    logger.info(f"{n_unused} graph words are not used in the category production set.")


def briony_categories_overlap(top_n_words):
    """Coverage of individual category responses in Briony's Category Production dataset."""
    # Category production words
    category_production = CategoryProduction()
    categories = category_production.category_labels

    # Frequent words in corpus
    corpus = CorpusPreferences.source_corpus_metas.bbc
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    corpus_words = set(freq_dist.most_common_tokens(top_n_words))

    logger.info(f"Top {top_n_words:,} words:")

    results = []
    for category in categories:
        # Skip multi-word categories
        if " " in category:
            continue
        if category in corpus_words:
            responses = category_production.responses_for_category(category, single_word_only=True)
            n_present_responses = len(set(responses).intersection(set(corpus_words)))
            percent_present_responses = 100 * n_present_responses / len(responses)
            logger.info(f"{category}:\t{percent_present_responses:0.2f}%")
        else:
            logger.info(f"{category}:\t(not present)")
            n_present_responses = nnan
            percent_present_responses = nnan
        results.append({
            "Number of words": top_n_words,
            "Category": category,
            "Number of single-word responses covered": n_present_responses,
            "% of single-word responses covered": percent_present_responses
        })
    return results


def main():
    word_counts = [
        1_000,
        3_000,
        5_000,
        10_000,
        20_000,
        30_000,
        50_000,
        100_000,
        200_000,
        300_000,
    ]
    results = DataFrame()
    for word_count in word_counts:
        results_this_word_count = DataFrame(briony_categories_overlap(word_count))
        results = results.append(results_this_word_count, ignore_index=True)

    # Save the overall data
    filename = f"category production coverage words.csv"
    results.to_csv(path.join(Preferences.output_dir, filename),
                   columns=["Number of words",
                            "Category",
                            "Number of single-word responses covered",
                            "% of single-word responses covered"],
                   index=False)


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")