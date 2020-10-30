"""
===========================
Model coverage of category production data.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2018, 2020
---------------------------
"""
from __future__ import annotations

import sys
from typing import Set, Tuple
from dataclasses import dataclass, field

from category_production.category_production import CategoryProduction
from model.ldm.corpus.indexing import FreqDist
from model.ldm.corpus.tokenising import modified_word_tokenize
from model.ldm.preferences.preferences import Preferences as CorpusPreferences
from model.utils.logging import logger
from model.sensorimotor_norms.sensorimotor_norms import SensorimotorNorms


@dataclass
class Coverage:
    categories_single: Set[str] = field(default_factory=set)
    categories_multi:  Set[str] = field(default_factory=set)
    # Store responses tagged by their categories to prevent collisions
    responses_single:  Set[Tuple[str, str]] = field(default_factory=set)
    responses_multi:   Set[Tuple[str, str]] = field(default_factory=set)

    @property
    def categories_both(self) -> Set[str]:
        return self.categories_single | self.categories_multi

    @property
    def responses_both(self) -> Set[Tuple[str, str]]:
        return self.responses_single | self.responses_multi

    def __or__(self, other: Coverage) -> Coverage:
        return Coverage(
            categories_single=self.categories_single | other.categories_single,
            categories_multi =self.categories_multi  | other.categories_multi,
            responses_single =self.responses_single  | other.responses_single,
            responses_multi  =self.responses_multi   | other.responses_multi,
        )


CP = CategoryProduction()
SN = SensorimotorNorms()


def is_single_word(word: str) -> bool:
    return " " not in word


def term_in_sn(t: str) -> bool:
    if SN.has_word(t):
        return True
    for w in tokenise(t):
        if SN.has_word(w):
            return True
    return False


def tokenise(w: str) -> Set[str]:
    return {
        t
        for t in modified_word_tokenize(w)
        if t not in CP.ignored_words
    }


def main(word_count: int, freq_dist: FreqDist):

    corpus_words = set(freq_dist.most_common_tokens(word_count))

    sensorimotor_coverage = Coverage()
    linguistic_coverage = Coverage()
    cp_size = Coverage()

    # Everything stored by the original label, not the sensorimotor label. This includes single/multi designations.
    for category in CP.category_labels:
        category_s = CP.translate_linguistic2sensorimotor[category]
        # Remember if the category was in the model
        had_category_linguistic   = False
        had_category_sensorimotor = False

        if is_single_word(category):
            # Single-word category

            cp_size.categories_single.add(category)
            # Check single-word category in (frequency-filtered) corpus
            if category in corpus_words:
                linguistic_coverage.categories_single.add(category)
                had_category_linguistic = True
            # Check if in sensorimotor norms
            if term_in_sn(category_s):
                sensorimotor_coverage.categories_single.add(category)
                had_category_sensorimotor = True

        else:
            # Multi-word category

            cp_size.categories_multi.add(category)
            # Check multi-word category in (frequency-filtered) corpus
            # Use the same strategy as the model:
            # 1. Tokenise the category
            # 2. If any token is there, we have a hit
            if tokenise(category) & corpus_words:
                linguistic_coverage.categories_multi.add(category)
                had_category_linguistic = True
            # Check if in sensorimotor norms
            if term_in_sn(category_s):
                sensorimotor_coverage.categories_multi.add(category)
                had_category_sensorimotor = True

        for response in CP.responses_for_category(category):
            response_s = CP.translate_linguistic2sensorimotor[response]
            if is_single_word(response):
                # Single-word response
                cp_size.responses_single.add((category, response))
                if had_category_linguistic and (response in corpus_words):
                    linguistic_coverage.responses_single.add((category, response))
                if had_category_sensorimotor and (term_in_sn(response_s)):
                    sensorimotor_coverage.responses_single.add((category, response))

            else:
                # Multi-word response
                cp_size.responses_multi.add((category, response))
                # Check multi-word response in corpus using the same strategy as above
                if had_category_linguistic:
                    if tokenise(response) & corpus_words:
                        linguistic_coverage.responses_multi.add((category, response))
                if had_category_sensorimotor and term_in_sn(response_s):
                    sensorimotor_coverage.responses_multi.add((category, response))

    combined_coverage = linguistic_coverage | sensorimotor_coverage
    
    # Categories

    logger.info(f"--- Categories: Single words ---")
    logger.info(f"Linguistic categories:   {len(linguistic_coverage.categories_single)}/{len(cp_size.categories_single)} ({100*len(linguistic_coverage.categories_single)/len(cp_size.categories_single):.2f}%)")
    logger.info(f"Sensorimotor categories: {len(sensorimotor_coverage.categories_single)}/{len(cp_size.categories_single)} ({100*len(sensorimotor_coverage.categories_single)/len(cp_size.categories_single):.2f}%)")
    logger.info(f"Combined categories:     {len(combined_coverage.categories_single)}/{len(cp_size.categories_single)} ({100*len(combined_coverage.categories_single)/len(cp_size.categories_single):.2f}%)")

    logger.info(f"--- Categories: Multi-words ---")
    logger.info(f"Linguistic categories:   {len(linguistic_coverage.categories_multi)}/{len(cp_size.categories_multi)} ({100*len(linguistic_coverage.categories_multi)/len(cp_size.categories_multi):.2f}%)")
    logger.info(f"Sensorimotor categories: {len(sensorimotor_coverage.categories_multi)}/{len(cp_size.categories_multi)} ({100*len(sensorimotor_coverage.categories_multi)/len(cp_size.categories_multi):.2f}%)")
    logger.info(f"Combined categories:     {len(combined_coverage.categories_multi)}/{len(cp_size.categories_multi)} ({100*len(combined_coverage.categories_multi)/len(cp_size.categories_multi):.2f}%)")

    logger.info(f"--- Categories: All ---")
    logger.info(f"Linguistic categories:   {len(linguistic_coverage.categories_both)}/{len(cp_size.categories_both)} ({100*len(linguistic_coverage.categories_both)/len(cp_size.categories_both):.2f}%)")
    logger.info(f"Sensorimotor categories: {len(sensorimotor_coverage.categories_both)}/{len(cp_size.categories_both)} ({100*len(sensorimotor_coverage.categories_both)/len(cp_size.categories_both):.2f}%)")
    logger.info(f"Combined categories:     {len(combined_coverage.categories_both)}/{len(cp_size.categories_both)} ({100*len(combined_coverage.categories_both)/len(cp_size.categories_both):.2f}%)")

    logger.info("")

    # Responses

    logger.info(f"--- Responses Single words ---")
    logger.info(f"Linguistic responses:   {len(linguistic_coverage.responses_single)}/{len(cp_size.responses_single)} ({100*len(linguistic_coverage.responses_single)/len(cp_size.responses_single):.2f}%)")
    logger.info(f"Sensorimotor responses: {len(sensorimotor_coverage.responses_single)}/{len(cp_size.responses_single)} ({100*len(sensorimotor_coverage.responses_single)/len(cp_size.responses_single):.2f}%)")
    logger.info(f"Combined responses:     {len(combined_coverage.responses_single)}/{len(cp_size.responses_single)} ({100*len(combined_coverage.responses_single)/len(cp_size.responses_single):.2f}%)")

    logger.info(f"--- Responses Multi-words ---")
    logger.info(f"Linguistic responses:   {len(linguistic_coverage.responses_multi)}/{len(cp_size.responses_multi)} ({100*len(linguistic_coverage.responses_multi)/len(cp_size.responses_multi):.2f}%)")
    logger.info(f"Sensorimotor responses: {len(sensorimotor_coverage.responses_multi)}/{len(cp_size.responses_multi)} ({100*len(sensorimotor_coverage.responses_multi)/len(cp_size.responses_multi):.2f}%)")
    logger.info(f"Combined responses:     {len(combined_coverage.responses_multi)}/{len(cp_size.responses_multi)} ({100*len(combined_coverage.responses_multi)/len(cp_size.responses_multi):.2f}%)")

    logger.info(f"--- Responses All ---")
    logger.info(f"Linguistic responses:   {len(linguistic_coverage.responses_both)}/{len(cp_size.responses_both)} ({100*len(linguistic_coverage.responses_both)/len(cp_size.responses_both):.2f}%)")
    logger.info(f"Sensorimotor responses: {len(sensorimotor_coverage.responses_both)}/{len(cp_size.responses_both)} ({100*len(sensorimotor_coverage.responses_both)/len(cp_size.responses_both):.2f}%)")
    logger.info(f"Combined responses:     {len(combined_coverage.responses_both)}/{len(cp_size.responses_both)} ({100*len(combined_coverage.responses_both)/len(cp_size.responses_both):.2f}%)")


if __name__ == '__main__':
    logger.info("Running %s" % " ".join(sys.argv))
    main(word_count=40_000, freq_dist=FreqDist.load(CorpusPreferences.source_corpus_metas.bbc.freq_dist_path))
    logger.info("Done!")
