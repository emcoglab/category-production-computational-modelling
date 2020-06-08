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

from model.utils.logging import logger
from preferences import Preferences
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms


def main():

    output_dir = Path(Preferences.ancillary_dir)

    corpus = CorpusPreferences.source_corpus_metas.bbc
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    linguistic_words: Set[str] = set(freq_dist.most_common_tokens())

    sensorimotor_norms = SensorimotorNorms()
    sensorimotor_words = set(sensorimotor_norms.iter_words())

    linguistic_unique_words = [
        w
        for w in freq_dist.most_common_tokens()
        if w in linguistic_words - sensorimotor_words
    ]
    sensorimotor_unique_words = list(sensorimotor_words - linguistic_words)

    linguistic_unique_word_list_path   = Path(output_dir, "linguistic-unique word list.txt")
    sensorimotor_unique_word_list_path = Path(output_dir, "sensorimotor-unique word list.txt")
    with open(linguistic_unique_word_list_path, mode="w", encoding="utf-8") as linguistic_list_file:
        for word in linguistic_unique_words:
            linguistic_list_file.write(f"{word}\n")
    with open(sensorimotor_unique_word_list_path, mode="w", encoding="utf-8") as sensorimotor_list_file:
        for word in sensorimotor_unique_words:
            sensorimotor_list_file.write(f"{word}\n")


if __name__ == '__main__':
    from sys import argv
    logger.info("Running %s" % " ".join(argv))
    main()
    logger.info("Done!")
