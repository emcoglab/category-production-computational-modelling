"""
===========================
Overlap between linguistic and sensorimotor components.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2020
---------------------------
"""
from pathlib import Path
from typing import Set

from ldm.corpus.indexing import FreqDist
from ldm.preferences.preferences import Preferences as CorpusPreferences
from ldm.utils.logging import print_progress

from model.utils.logging import logger
from preferences import Preferences
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms


def main():

    top_n = 60_000

    output_dir = Path(Preferences.ancillary_dir)
    output_dir.mkdir(exist_ok=True)

    corpus = CorpusPreferences.source_corpus_metas.bbc
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    linguistic_words: Set[str] = set(freq_dist.most_common_tokens(top_n=top_n))

    sensorimotor_norms = SensorimotorNorms()
    sensorimotor_words = set(sensorimotor_norms.iter_words())

    logger.info("Finding linguistic-unique terms")
    linguistic_unique_words = linguistic_words - sensorimotor_words
    linguistic_unique_words = [
        w
        for w in freq_dist.most_common_tokens(top_n=top_n)
        if w in linguistic_unique_words
    ]
    logger.info("Finding sensorimotor-unique terms")
    sensorimotor_unique_words = list(sensorimotor_words - linguistic_words)

    logger.info("Writing output lists")
    linguistic_unique_word_list_path   = Path(output_dir, "linguistic-unique word list.txt")
    sensorimotor_unique_word_list_path = Path(output_dir, "sensorimotor-unique word list.txt")
    with open(linguistic_unique_word_list_path, mode="w", encoding="utf-8") as linguistic_list_file:
        for i, word in enumerate(linguistic_unique_words):
            linguistic_list_file.write(f"{word}\n")
            print_progress(i+1, len(linguistic_unique_words))
    with open(sensorimotor_unique_word_list_path, mode="w", encoding="utf-8") as sensorimotor_list_file:
        for i, word in enumerate(sensorimotor_unique_words):
            sensorimotor_list_file.write(f"{word}\n")
            print_progress(i+1, len(sensorimotor_unique_words))


if __name__ == '__main__':
    from sys import argv
    logger.info("Running %s" % " ".join(argv))
    main()
    logger.info("Done!")
