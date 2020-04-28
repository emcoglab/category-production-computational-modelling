from typing import Dict

from model.utils.job import LinguisticPropagationJob, LinguisticPropagationJobSpec


class Job_2_3(LinguisticPropagationJob):

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

    def __init__(self, spec: LinguisticPropagationJobSpec):
        super().__init__(
            script_number="2_3",
            script_name="2_3_category_production_ngram_tsa.py",
            spec=spec)

    @property
    def _ram_requirement_g(self):
        assert isinstance(self.spec, LinguisticPropagationJobSpec)
        return self.RAM[self.spec.model_name][self.spec.n_words]


if __name__ == '__main__':

    n_words = 40_000
    bailout = int(n_words / 2)
    impulse_pruning_threshold = 0.05
    node_decay_factor = 0.99
    model_radius = 5
    corpus_name = "bbc"

    specs = [
        LinguisticPropagationJobSpec(model_name="ppmi_ngram", firing_threshold=0.9, edge_decay_sd=15, impulse_pruning_threshold=impulse_pruning_threshold, node_decay_factor=node_decay_factor, model_radius=model_radius, corpus_name=corpus_name, pruning=None, pruning_type=None, n_words=n_words, length_factor=10, run_for_ticks=3_000, bailout=bailout),
    ]

    for job in [Job_2_3(spec) for spec in specs]:
        job.submit()
