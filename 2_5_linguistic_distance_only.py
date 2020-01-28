"""
===========================
Model responses to Briony's category production categories using a linguistic distance-only model.
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
from os import path, makedirs
from typing import Optional

from pandas import DataFrame

from category_production.category_production import CategoryProduction, ColNames as CPColNames
from cli.lookups import get_corpus_from_name, get_model_from_params
from evaluation.column_names import MODEL_HIT
from ldm.corpus.indexing import FreqDist
from ldm.corpus.tokenising import modified_word_tokenize
from ldm.model.base import DistributionalSemanticModel, VectorSemanticModel
from ldm.model.ngram import NgramModel
from ldm.utils.maths import DistanceType
from model.utils.file import comment_line_from_str
from model.version import VERSION
from model.naïve_linguistic import LinguisticVectorDistanceOnlyModel, LinguisticNgramDistanceOnlyModel
from preferences import Preferences

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def main(quantile: float,
         n_words: int,
         corpus_name: str,
         model_name: str,
         radius: int,
         distance_type: Optional[DistanceType]):

    corpus = get_corpus_from_name(corpus_name)
    freq_dist = FreqDist.load(corpus.freq_dist_path)
    distributional_model: DistributionalSemanticModel = get_model_from_params(corpus, freq_dist, model_name, radius)

    filtered_words = set(freq_dist.most_common_tokens(n_words))

    if distributional_model.model_type.metatype == DistributionalSemanticModel.MetaType.ngram:
        assert isinstance(distributional_model, NgramModel)
        lm = LinguisticNgramDistanceOnlyModel(quantile=quantile,
                                              distributional_model=distributional_model,
                                              n_words=n_words)
        model_dirname = f"{distributional_model.name} {n_words:,} words quantile {lm.quantile}"
    else:
        assert isinstance(distributional_model, VectorSemanticModel)
        lm = LinguisticVectorDistanceOnlyModel(quantile=quantile,
                                               distance_type=distance_type,
                                               distributional_model=distributional_model,
                                               n_words=n_words)
        model_dirname = f"{distributional_model.name} {n_words:,} words {distance_type.name} "

    response_dir = path.join(Preferences.output_dir,
                             "Category production",
                             f"Linguistic distance-only {VERSION}",
                             model_dirname)

    if not path.isdir(response_dir):
        logger.warning(f"{response_dir} directory does not exist; making it.")
        makedirs(response_dir)
    model_responses_path = path.join(response_dir, f"hits.csv")

    cp = CategoryProduction()

    # Record model details
    csv_comments = [
        f"Linguistic distance-only model:",
        f"\t        model = {distributional_model.name}",
        f"\t        words = {n_words:_}",
        f"\t     quantile = {quantile}",
    ]
    if distance_type is not None:
        csv_comments.append(f"\tdistance type = {distance_type.name}")

    hits = []
    for category in cp.category_labels:
        logger.info(f"Checking hits for category \"{category}\"")
        if category in filtered_words:
            category_words = [category]
        else:
            category_words = [w
                              for w in modified_word_tokenize(category)
                              if w not in cp.ignored_words
                              # Ignore words which aren't available: activate all words we can
                              and w in filtered_words]
        for response in cp.responses_for_category(category):
            try:
                # Hit if hit for any category word
                hit = any(lm.is_hit(c, response) for c in category_words)
            except LookupError as er:
                logger.warning(er)
                hit = False
            hits.append((
                category, response, hit
            ))

    hits_df = DataFrame(hits, columns=[CPColNames.Category, CPColNames.Response, MODEL_HIT])

    with open(model_responses_path, mode="w", encoding="utf-8") as output_file:
        # Write comments
        for comment in csv_comments:
            output_file.write(comment_line_from_str(comment))
        # Write data
        hits_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Run temporal spreading activation on a graph.")

    parser.add_argument("-q", "--quantile", type=float, required=True)
    parser.add_argument("-w", "--words", type=int, required=True,
                        help="The number of words to use from the corpus. (Top n words.)")
    parser.add_argument("-c", "--corpus_name", required=True, type=str)
    parser.add_argument("-m", "--model_name", required=True, type=str)
    parser.add_argument("-r", "--radius", required=True, type=int)
    parser.add_argument('-d', "--distance_type", type=str, default=None)

    args = parser.parse_args()

    main(quantile=args.quantile,
         n_words=args.words,
         corpus_name=args.corpus_name,
         model_name=args.model_name,
         radius=args.radius,
         distance_type=args.distance_type)
    logger.info("Done!")
