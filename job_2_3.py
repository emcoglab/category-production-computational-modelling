import logging

from model.utils.job import LinguisticSAJob, LinguisticSASpec


logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


class Job_2_3(LinguisticSAJob):

    # graph_size -> RAM/G
    RAM = {
        1_000: 2,
        3_000: 3,
        10_000: 7,
        30_000: 11,
        40_000: 15,
    }

    def __init__(self, spec: LinguisticSASpec, run_for_ticks: int, bailout: int):
        super().__init__(
            script_number="2_3",
            script_name="2_3_category_production_ngram_tsa.py",
            spec=spec,
            run_for_ticks=run_for_ticks,
            bailout=bailout)

    @property
    def command(self) -> str:
        cmd = self.script_name
        # script args
        cmd += f" --bailout {self.bailout}"
        cmd += f" --corpus_name {self.spec.corpus_name}"
        cmd += f" --firing_threshold {self.spec.firing_threshold}"
        cmd += f" --impulse_pruning_threshold {self.spec.impulse_pruning_threshold}"
        cmd += f" --length_factor {self.spec.length_factor}"
        cmd += f" --model_name {self.spec.model_name}"
        cmd += f" --node_decay_factor {self.spec.node_decay_factor}"
        cmd += f" --radius {self.spec.model_radius}"
        cmd += f" --edge_decay_sd_factor {self.spec.edge_decay_sd}"
        cmd += f" --run_for_ticks {self.run_for_ticks}"
        cmd += f" --words {int(self.spec.graph_size)}"
        return cmd

    @property
    def _ram_requirement_g(self):
        assert isinstance(self.spec, LinguisticSASpec)
        return self.RAM[self.spec.graph_size]

if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)

    graph_size = 40_000
    bailout = int(graph_size / 2)
    impulse_pruning_threshold = 0.05
    node_decay_factor = 0.99
    model_radius = 5
    corpus_name = "bbc"

    specs = [
        LinguisticSASpec(graph_size=graph_size, model_name="pmi_ngram", length_factor=10, firing_threshold=0.7, edge_decay_sd=35, impulse_pruning_threshold=impulse_pruning_threshold, node_decay_factor=node_decay_factor, model_radius=model_radius, corpus_name=corpus_name, pruning=None),
        LinguisticSASpec(graph_size=graph_size, model_name="pmi_ngram", length_factor=10, firing_threshold=0.6, edge_decay_sd=25, impulse_pruning_threshold=impulse_pruning_threshold, node_decay_factor=node_decay_factor, model_radius=model_radius, corpus_name=corpus_name, pruning=None),
        LinguisticSASpec(graph_size=graph_size, model_name="pmi_ngram", length_factor=10, firing_threshold=0.7, edge_decay_sd=30, impulse_pruning_threshold=impulse_pruning_threshold, node_decay_factor=node_decay_factor, model_radius=model_radius, corpus_name=corpus_name, pruning=None),
        LinguisticSASpec(graph_size=graph_size, model_name="pmi_ngram", length_factor=10, firing_threshold=0.6, edge_decay_sd=30, impulse_pruning_threshold=impulse_pruning_threshold, node_decay_factor=node_decay_factor, model_radius=model_radius, corpus_name=corpus_name, pruning=None),
        LinguisticSASpec(graph_size=graph_size, model_name="ppmi_ngram", length_factor=10, firing_threshold=0.7, edge_decay_sd=35, impulse_pruning_threshold=impulse_pruning_threshold, node_decay_factor=node_decay_factor, model_radius=model_radius, corpus_name=corpus_name, pruning=None),
        LinguisticSASpec(graph_size=graph_size, model_name="ppmi_ngram", length_factor=10, firing_threshold=0.5, edge_decay_sd=15, impulse_pruning_threshold=impulse_pruning_threshold, node_decay_factor=node_decay_factor, model_radius=model_radius, corpus_name=corpus_name, pruning=None),
        LinguisticSASpec(graph_size=graph_size, model_name="ppmi_ngram", length_factor=10, firing_threshold=0.7, edge_decay_sd=20, impulse_pruning_threshold=impulse_pruning_threshold, node_decay_factor=node_decay_factor, model_radius=model_radius, corpus_name=corpus_name, pruning=None),
        LinguisticSASpec(graph_size=graph_size, model_name="ppmi_ngram", length_factor=10, firing_threshold=0.5, edge_decay_sd=20, impulse_pruning_threshold=impulse_pruning_threshold, node_decay_factor=node_decay_factor, model_radius=model_radius, corpus_name=corpus_name, pruning=None),
        LinguisticSASpec(graph_size=graph_size, model_name="ppmi_ngram", length_factor=10, firing_threshold=0.5, edge_decay_sd=30, impulse_pruning_threshold=impulse_pruning_threshold, node_decay_factor=node_decay_factor, model_radius=model_radius, corpus_name=corpus_name, pruning=None),
    ]

    for job in [Job_2_3(spec, run_for_ticks=3_000, bailout=bailout) for spec in specs]:
        job.submit()
