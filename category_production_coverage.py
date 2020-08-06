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

import sys
from typing import Set

from pandas import DataFrame

from category_production.category_production import CategoryProduction, ColNames as CPColNames
from ldm.corpus.indexing import FreqDist
from ldm.preferences.preferences import Preferences as CorpusPreferences

from model.utils.logging import logger
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms

CP = CategoryProduction()


def is_single_word(word: str) -> bool:
    return " " not in word


def log_coverage(test_vocabulary: Set[str], test_name: str, use_sensorimotor: bool):

    category_labels = CP.category_labels_sensorimotor if use_sensorimotor else CP.category_labels


    results = []
    n_categories_overall = 0
    n_categories_covered = 0
    n_responses_overall = 0
    n_responses_covered = 0
    for category in category_labels:
        if not is_single_word(category):
            continue
        n_categories_overall += 1
        single_word_responses = CP.responses_for_category(
            category=category,
            single_word_only=True,
            sort_by=CPColNames.Response,
            use_sensorimotor=use_sensorimotor)
        single_word_responses_within_word_limit = [
            word
            for word in single_word_responses
            if word in test_vocabulary
        ]

        n_single_word_responses = len(single_word_responses)
        n_responses_overall += n_single_word_responses
        if category in test_vocabulary:
            n_categories_covered += 1
            n_single_word_responses_in_graph = len(single_word_responses_within_word_limit)
        else:
            n_single_word_responses_in_graph = 0
        n_responses_covered += n_single_word_responses_in_graph

        results.append({
            "Single-word category": category,
            "Single-word responses": n_single_word_responses,
            "Single-word responses within graph": n_single_word_responses_in_graph,
            "% coverage": 100 * n_single_word_responses_in_graph / n_single_word_responses,
        })
    DataFrame.from_records(results).to_csv(f"/Users/caiwingfield/Desktop/{test_name} words overlap.csv", columns=[
        "Single-word category",
        "Single-word responses",
        "Single-word responses within graph",
        "% coverage",
    ], index=False)
    logger.info(f"{test_name} words: "
                f"categories {n_categories_covered}/{n_categories_overall}"
                f" ({100 * n_categories_covered / n_categories_overall:.1f}%); "
                f"responses {n_responses_covered}/{n_responses_overall}"
                f" ({100 * n_responses_covered / n_responses_overall:.1f}%)")


def main():
    word_counts = [
        1_000,
        3_000,
        10_000,
        30_000,
        40_000,
        60_000,
        100_000,
        300_000,
    ]

    # Linguistic
    corpus = CorpusPreferences.source_corpus_metas.bbc
    freq_dist = FreqDist.load(corpus.freq_dist_path)

    for word_count in word_counts:
        corpus_words = set(freq_dist.most_common_tokens(word_count))
        log_coverage(test_vocabulary=corpus_words,
                     test_name=f"{word_count:,}",
                     use_sensorimotor=False)

    # Sensorimotor
    sm = SensorimotorNorms()
    log_coverage(test_vocabulary=set(sm.iter_words()),
                 test_name="Sensorimotor",
                 use_sensorimotor=True)


if __name__ == '__main__':
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
