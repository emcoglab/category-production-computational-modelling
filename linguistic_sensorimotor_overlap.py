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
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Set, Dict

import yaml

from ldm.corpus.indexing import FreqDist
from ldm.preferences.preferences import Preferences as CorpusPreferences
from ldm.utils.logging import print_progress

from model.utils.logging import logger
from preferences import Preferences
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms

sensorimotor_norms = SensorimotorNorms()
corpus = CorpusPreferences.source_corpus_metas.bbc
freq_dist = FreqDist.load(corpus.freq_dist_path)


def list_candidate_translations_breng_ameng(breng_fragment: str,
                                            ameng_fragmet: str,
                                            strict_breng: bool,
                                            strict_ameng: bool,
                                            ):
    """
    Translating BrEng to AmEng.
    Heuristic for candidate translation:
        breng word doesn't exist in ameng list
        ameng'd word does
    """

    breng_words = set(freq_dist.most_common_tokens())
    ameng_words = set(sensorimotor_norms.iter_words())

    translation_dictionary = dict()

    logger.info(f"Starting with {len(breng_words):,} BrEng words")
    for breng_word in sorted(breng_words):
        if breng_word in ameng_words:
            # The word is already AmEng
            continue
        if any(c in breng_word for c in "' ./-"):
            continue
        if breng_fragment in breng_word:
            # Can be translated
            ameng_word = breng_word.replace(breng_fragment, ameng_fragmet)
            if (ameng_word in ameng_words) or not strict_ameng:
                # Translated version is now AmEng
                translation_dictionary[breng_word] = ameng_word

    logger.info("Mopping up any extras which are not found in the breng list")
    for ameng_word in sorted(ameng_words):
        if ameng_word in breng_words:
            continue
        if any(c in ameng_word for c in "' ./-"):
            continue
        if ameng_fragmet in ameng_word:
            breng_word = ameng_word.replace(ameng_fragmet, breng_fragment)
            if breng_word in translation_dictionary:
                continue
            if (breng_word in breng_words) or not strict_breng:
                translation_dictionary[breng_word] = ameng_word

    for breng_word, ameng_word in translation_dictionary.items():
        print(f"{breng_word}: {ameng_word}")
    print(f"{len(translation_dictionary):,}")


def compare_with_tysto(only_in_tysto: bool):

    with open("/Users/caiwingfield/Resilio Sync/Lancaster/dictionaries/tysto.yaml", mode="r") as t:
        tysto_dict = yaml.load(t, Loader=yaml.SafeLoader)

    my_dict = dict()
    for my_file in glob("/Users/caiwingfield/Resilio Sync/Lancaster/dictionaries/breng_ameng/*.yaml"):
        with open(my_file, mode="r") as f:
            d = yaml.load(f, Loader=yaml.SafeLoader)
            my_dict.update(d)

    i = 0
    if only_in_tysto:
        for b, a in tysto_dict.items():
            if b not in my_dict:
                i += 1
                print(f"{b}: {a}")
    else:
        for b, a in my_dict.items():
            if b not in tysto_dict:
                i += 1
                print(f"{b}: {a}")
    print(i)


def distribute_tysto():

    rules: Dict[str, str] = {
        "l": "ll",
        "ys": "yz",
        "ae": "e",
        "oe": "e",
        "ce": "se",
        "re": "er",
        "is": "iz",
        "ll": "l",
        "ou": "o",
        "uo": "o",
    }

    distribution = defaultdict(dict)

    with open("/Users/caiwingfield/Resilio Sync/Lancaster/dictionaries/tysto_new.yaml", mode="r") as t:
        tysto_dict: Dict[str, str] = yaml.load(t, Loader=yaml.SafeLoader)

    for br, am in tysto_dict.items():
        this_item_done = False
        for br_fragment, am_fragment in rules.items():
            if br.replace(br_fragment, am_fragment) == am:
                distribution[br_fragment][br] = am
                this_item_done = True
                break
        if not this_item_done:
            distribution["misc"][br] = [am]

    for rule in distribution.keys():
        with open(f"/Users/caiwingfield/Desktop/{rule}.txt", mode="w") as outfile:
            yaml.dump(distribution[rule], outfile, yaml.SafeDumper)


def list_unique_words():

    top_n = 60_000

    output_dir = Path(Preferences.ancillary_dir)

    linguistic_words: Set[str] = set(freq_dist.most_common_tokens(top_n=top_n))

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
    # list_candidate_translations_breng_ameng(
    #     breng_fragment="ae",
    #     ameng_fragmet="e",
    #     strict_breng=True,
    #     strict_ameng=True,
    # )
    distribute_tysto()
    logger.info("Done!")
