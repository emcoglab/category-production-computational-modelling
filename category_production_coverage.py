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

from pandas import DataFrame

from category_production.category_production import CategoryProduction, ColNames as CPColNames
from ldm.corpus.indexing import FreqDist
from ldm.preferences.preferences import Preferences as CorpusPreferences

from model.utils.logging import logger


def is_single_word(word: str) -> bool:
    return " " not in word


def main():
    word_counts = [
        1_000,
        3_000,
        10_000,
        30_000,
        40_000,
        100_000,
        300_000,
    ]

    cp = CategoryProduction()
    corpus = CorpusPreferences.source_corpus_metas.bbc
    freq_dist = FreqDist.load(corpus.freq_dist_path)

    for word_count in word_counts:
        corpus_words = set(freq_dist.most_common_tokens(word_count))
        results = []
        n_categories_overall = 0
        n_categories_covered = 0
        n_responses_overall = 0
        n_responses_covered = 0
        for category in cp.category_labels:
            if not is_single_word(category):
                continue
            n_categories_overall += 1
            single_word_responses = cp.responses_for_category(category, single_word_only=True, sort_by=CPColNames.Response)
            single_word_responses_within_word_limit = [
                word
                for word in single_word_responses
                if word in corpus_words
            ]

            n_single_word_responses = len(single_word_responses)
            n_responses_overall += n_single_word_responses
            if category in corpus_words:
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
        DataFrame.from_records(results).to_csv(f"/Users/caiwingfield/Desktop/{word_count:,} words overlap.csv", columns=[
            "Single-word category",
            "Single-word responses",
            "Single-word responses within graph",
            "% coverage",
        ], index=False)

        logger.info(f"{word_count:,} words: "
                    f"categories {n_categories_covered}/{n_categories_overall}"
                    f" ({100*n_categories_covered/n_categories_overall:.1f}%); "
                    f"responses {n_responses_covered}/{n_responses_overall}"
                    f" ({100*n_responses_covered/n_responses_overall:.1f}%)")


if __name__ == '__main__':
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
