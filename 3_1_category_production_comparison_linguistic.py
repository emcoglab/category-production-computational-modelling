"""
===========================
Compare model to Briony's category production actual responses.
Also works for the one-hop linguistic model.

Pass the parent location to a bunch of results.
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

import argparse
import sys
from os import path

from framework.category_production.category_production import CategoryProduction

from framework.evaluation.category_production import get_n_words_from_path_linguistic, get_model_ttfas_for_category_linguistic, \
    add_ttfa_column, get_firing_threshold_from_path_linguistic, ModelType, find_output_dirs, \
    prepare_category_production_data, process_one_model_output, add_model_hit_column
from framework.cognitive_model.utils.logging import logger

CP = CategoryProduction()


def main(input_results_dir: str,
         min_first_rank_freq: int,
         variant: str,
         ):

    if variant == "full":
        model_type = ModelType.linguistic
    elif variant == "one-hop":
        model_type = ModelType.linguistic_one_hop
    else:
        raise NotImplementedError()

    model_output_dirs = find_output_dirs(root_dir=input_results_dir)

    for model_output_dir in model_output_dirs:
        logger.info(path.basename(model_output_dir))
        main_data = prepare_category_production_data(model_type)

        n_words = get_n_words_from_path_linguistic(model_output_dir)

        ttfas = {
            category: get_model_ttfas_for_category_linguistic(category, model_output_dir, n_words, get_firing_threshold_from_path_linguistic(model_output_dir))
            for category in CP.category_labels
        }
        add_ttfa_column(main_data, model_type=model_type, ttfas=ttfas)
        add_model_hit_column(main_data)

        process_one_model_output(main_data, model_type, model_output_dir, min_first_rank_freq)


if __name__ == '__main__':
    logger.info("Running %s" % " ".join(sys.argv))

    parser = argparse.ArgumentParser(description="Compare spreading activation results with Category Production data.")
    parser.add_argument("path", type=str, help="The path in which to find the results.")
    parser.add_argument("min_frf", type=int, nargs="?", default=1, help="The minimum FRF required for zRT and FRF correlations. Omit to use 1.")
    parser.add_argument("-v-", "--variant", type=str, choices=["one-hop", "full"])
    args = parser.parse_args()

    main(args.path, args.min_frf, args.variant)

    logger.info("Done!")
