from jobs.job import LinguisticSAJob, LinguisticSASpec


class Job_2_3(LinguisticSAJob):

    # graph_size -> RAM/G
    RAM = {
        1_000: 2,
        3_000: 3,
        10_000: 7,
        30_000: 11,
        40_000: 15,
    }

    def __init__(self, spec: LinguisticSASpec, run_for_ticks: int, bailout: int = None):
        super().__init__(
            script_number="2_3",
            script_name="2_3_category_production_ngram_tsa.py",
            spec=spec,
            run_for_ticks=run_for_ticks,
            bailout=bailout)

    @property
    def qsub_command(self) -> str:
        cmd = f"qsub"
        # qsub args
        cmd += f" -S {self._python_location}"
        cmd += f" -q {self._queue}"
        cmd += f" -N {self.name}"
        cmd += f" -m e -M c.wingfield@lancaster.ac.uk"
        cmd += f" -l h_vmem={self.RAM[self.spec.max_radius]}G"
        # script
        cmd += f" {self.script_name}"
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