"""
===========================
Model responses to Briony's category production categories using a sensorimotor distance-only model.
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
from evaluation.column_names import MODEL_HIT
from ldm.corpus.tokenising import modified_word_tokenize
from ldm.utils.maths import DistanceType
from model.naïve_sensorimotor import SensorimotorDistanceOnlyModelComponent
from model.utils.file import comment_line_from_str
from model.version import VERSION
from preferences import Preferences

logger = logging.getLogger(__name__)
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


def main(quantile: float, distance_type: Optional[DistanceType]):

    sm = SensorimotorDistanceOnlyModelComponent(quantile=quantile, distance_type=distance_type)
    model_dirname = f"{distance_type.name} quantile {sm.quantile}"
    response_dir = path.join(Preferences.output_dir,
                             "Category production",
                             f"Sensorimotor distance-only {VERSION}",
                             model_dirname)

    if not path.isdir(response_dir):
        logger.warning(f"{response_dir} directory does not exist; making it.")
        makedirs(response_dir)

    cp = CategoryProduction()

    # Record model details
    csv_comments = [
        f"Sensorimotor distance-only model:",
        f"\t        model = sensorimotor",
        f"\tdistance type = {distance_type.name}",
    ]

    model_responses_path = path.join(response_dir, "hits.csv")
    hits = []
    for i, category in enumerate(cp.category_labels_sensorimotor, start=1):
        logger.info(f"Checking hits for category \"{category}\" ({i}/{len(cp.category_labels_sensorimotor)})")
        if category in sm.words:
            category_words = [category]
        else:
            category_words = [w for w in modified_word_tokenize(category) if w not in cp.ignored_words]
        for response in cp.responses_for_category(category, use_sensorimotor=True, force_unique=True):
            try:
                # Hit if hit for any category word
                hit = any(sm.is_hit(c, response) for c in category_words)
            except LookupError as er:
                logger.warning(er)
                hit = False
            hits.append((
                category, response, hit
            ))

    hits_df = DataFrame(hits, columns=[CPColNames.CategorySensorimotor, CPColNames.ResponseSensorimotor, MODEL_HIT])

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
    parser.add_argument('-d', "--distance_type", type=str, default=None)

    args = parser.parse_args()

    main(quantile=args.quantile, distance_type=DistanceType.from_name(args.distance_type))
    logger.info("Done!")
