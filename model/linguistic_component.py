"""
===========================
The linguistic component of the model.
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

from enum import Enum, auto
from os import path
from typing import Set
import logging

import yaml
from pandas import DataFrame

from ldm.corpus.corpus import CorpusMetadata
from ldm.corpus.indexing import FreqDist
from ldm.model.base import DistributionalSemanticModel
from ldm.utils.maths import DistanceType
from model.common import ActivationValue, ItemLabel, _load_labels
from model.graph import Graph
from model.temporal_spreading_activation import TemporalSpreadingActivation
from model.utils.maths import make_decay_function_exponential_with_decay_factor, make_decay_function_gaussian_with_sd
from preferences import Preferences

logger = logging.getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


class EdgePruningType(Enum):
    Length     = auto()
    Percent    = auto()
    Importance = auto()


class LinguisticComponent(TemporalSpreadingActivation):
    """
    The linguistic component of the model.
    """

    def __init__(self,
                 n_words: int,
                 distributional_model: DistributionalSemanticModel,
                 freq_dist: FreqDist,
                 length_factor: int,
                 node_decay_factor: float,
                 edge_decay_sd_factor: float,
                 impulse_pruning_threshold: ActivationValue,
                 firing_threshold: ActivationValue,
                 distance_type: DistanceType = None,
                 edge_pruning=None,
                 edge_pruning_type: EdgePruningType = None,
                 ):

        node_labelling_dictionary = load_labels_from_corpus(distributional_model.corpus_meta, n_words)

        super(LinguisticComponent, self).__init__(
            graph=_load_graph(n_words, length_factor, distributional_model,
                              distance_type, edge_pruning_type, edge_pruning),
            idx2label=node_labelling_dictionary,
            impulse_pruning_threshold=impulse_pruning_threshold,
            firing_threshold=firing_threshold,
            node_decay_function=make_decay_function_exponential_with_decay_factor(
                decay_factor=node_decay_factor),
            edge_decay_function=make_decay_function_gaussian_with_sd(
                sd=edge_decay_sd_factor * length_factor)
        )

        self.available_words: Set[ItemLabel] = set(freq_dist.most_common_tokens(n_words))


def _load_graph(n_words, length_factor, distributional_model, distance_type, edge_pruning_type, edge_pruning) -> Graph:

    # Check if distance_type is needed and get filename
    if distributional_model.model_type.metatype is DistributionalSemanticModel.MetaType.ngram:
        assert distance_type is None
        graph_file_name = f"{distributional_model.name} {n_words} words length {length_factor}.edgelist"
    else:
        assert distance_type is not None
        graph_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words length {length_factor}.edgelist"

    # Load graph
    if edge_pruning is None:
        logger.info(f"Loading graph from {graph_file_name}")
        graph = Graph.load_from_edgelist(file_path=path.join(Preferences.graphs_dir, graph_file_name))

    elif edge_pruning_type is EdgePruningType.Length:
        logger.info(f"Loading graph from {graph_file_name}, pruning any edge longer than {edge_pruning}")
        graph = Graph.load_from_edgelist(file_path=path.join(Preferences.graphs_dir, graph_file_name),
                                         ignore_edges_longer_than=edge_pruning,
                                         keep_at_least_n_edges=Preferences.min_edges_per_node)

    elif edge_pruning_type is EdgePruningType.Percent:
        quantile_file_name = f"{distributional_model.name} {distance_type.name} {n_words} words length {length_factor} edge length quantiles.csv"
        quantile_data = DataFrame.from_csv(path.join(Preferences.graphs_dir, quantile_file_name), header=0,
                                           index_col=None)
        pruning_length = quantile_data[
            # Use 1 - so that smallest top quantiles get converted to longest edges
            quantile_data["Top quantile"] == 1 - (edge_pruning / 100)
            ]["Pruning length"].iloc[0]
        logger.info(f"Loading graph from {graph_file_name}, pruning longest {edge_pruning}% of edges (anything over {pruning_length})")
        graph = Graph.load_from_edgelist(file_path=path.join(Preferences.graphs_dir, graph_file_name),
                                         ignore_edges_longer_than=edge_pruning,
                                         keep_at_least_n_edges=Preferences.min_edges_per_node)

    elif edge_pruning_type is EdgePruningType.Importance:
        logger.info(
            f"Loading graph from {graph_file_name}, pruning longest {edge_pruning}% of edges")
        graph = Graph.load_from_edgelist_with_importance_pruning(
            file_path=path.join(Preferences.graphs_dir, graph_file_name),
            ignore_edges_with_importance_greater_than=edge_pruning,
            keep_at_least_n_edges=Preferences.min_edges_per_node)

    else:
        raise NotImplementedError()

    return graph


def save_model_spec_linguistic(edge_decay_sd_factor, firing_threshold, length_factor, model_name, n_words,
                               response_dir):
    spec = {
        "Model name": model_name,
        "Length factor": length_factor,
        "SD factor": edge_decay_sd_factor,
        "Firing threshold": firing_threshold,
        "Words": n_words,
    }
    with open(path.join(response_dir, " model_spec.yaml"), mode="w", encoding="utf-8") as spec_file:
        yaml.dump(spec, spec_file, yaml.SafeDumper)


def load_labels_from_corpus(corpus: CorpusMetadata, n_words: int):
    return _load_labels(path.join(Preferences.graphs_dir, f"{corpus.name} {n_words} words.nodelabels"))
