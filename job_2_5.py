from typing import Dict

from model.utils.job import LinguisticPropagationJob, LinguisticOneHopJobSpec

logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


class Job_2_5(LinguisticPropagationJob):

    # model_name -> n_words -> RAM/G
    RAM: Dict[str, Dict[int, int]] = {
        "pmi_ngram": {
            1_000: 2,
            3_000: 3,
            10_000: 7,
            30_000: 11,
            40_000: 15,
        },
        "ppmi_ngram": {
            1_000: 2,
            3_000: 3,
            10_000: 5,
            30_000: 7,
            40_000: 9,
        }
    }

    def __init__(self, spec: LinguisticOneHopJobSpec):
        super().__init__(
            script_number="2_5",
            script_name="2_5_linguistic_one_hop.py",
            spec=spec)

    @property
    def _ram_requirement_g(self):
        assert isinstance(self.spec, LinguisticOneHopJobSpec)
        return self.RAM[self.spec.model_name][self.spec.n_words]


if __name__ == '__main__':

    n_words = 40_000
    impulse_pruning_threshold = 0.05
    node_decay_factor = 0.99
    model_radius = 5
    corpus_name = "bbc"

    specs = [
        LinguisticOneHopJobSpec(model_name="ppmi_ngram", firing_threshold=0.5, edge_decay_sd=10, impulse_pruning_threshold=impulse_pruning_threshold, node_decay_factor=node_decay_factor, model_radius=model_radius, corpus_name=corpus_name, pruning=None, pruning_type=None, n_words=n_words, length_factor=10, bailout=None, run_for_ticks=None),
        LinguisticOneHopJobSpec(model_name="ppmi_ngram", firing_threshold=0.6, edge_decay_sd=10, impulse_pruning_threshold=impulse_pruning_threshold, node_decay_factor=node_decay_factor, model_radius=model_radius, corpus_name=corpus_name, pruning=None, pruning_type=None, n_words=n_words, length_factor=10, bailout=None, run_for_ticks=None),
        LinguisticOneHopJobSpec(model_name="ppmi_ngram", firing_threshold=0.7, edge_decay_sd=10, impulse_pruning_threshold=impulse_pruning_threshold, node_decay_factor=node_decay_factor, model_radius=model_radius, corpus_name=corpus_name, pruning=None, pruning_type=None, n_words=n_words, length_factor=10, bailout=None, run_for_ticks=None),
        LinguisticOneHopJobSpec(model_name="ppmi_ngram", firing_threshold=0.8, edge_decay_sd=10, impulse_pruning_threshold=impulse_pruning_threshold, node_decay_factor=node_decay_factor, model_radius=model_radius, corpus_name=corpus_name, pruning=None, pruning_type=None, n_words=n_words, length_factor=10, bailout=None, run_for_ticks=None),
        LinguisticOneHopJobSpec(model_name="ppmi_ngram", firing_threshold=0.9, edge_decay_sd=10, impulse_pruning_threshold=impulse_pruning_threshold, node_decay_factor=node_decay_factor, model_radius=model_radius, corpus_name=corpus_name, pruning=None, pruning_type=None, n_words=n_words, length_factor=10, bailout=None, run_for_ticks=None),
    ]

    for job in [Job_2_5(spec) for spec in specs]:
        job.submit()
