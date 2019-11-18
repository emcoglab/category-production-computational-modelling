from jobs.job import LinguisticSAJob, LinguisticSASpec


class Job_2_2(LinguisticSAJob):

    # graph_size -> pruning -> RAM/G
    RAM = {
        3_000:  {100: 5,   90: 5,   80: 5,   70: 5,   60: 5,   50: 5 , 40: 5 , 30: 5 , 20: 5 , 10: 5 , 0: 5  },
        10_000: {100: 30,  90: 26,  80: 24,  70: 22,  60: 20,  50: 15, 40: 13, 30: 12, 20: 12, 10: 12, 0: 12 },
        20_000: {100: 70,  90: 65,  80: 60,  70: 55,  60: 50,  50: 45, 40: 35, 30: 30, 20: 30, 10: 25, 0: 25 },
        30_000: {100: 160, 90: 140, 80: 120, 70: 110, 60: 100, 50: 90, 40: 80, 30: 70, 20: 70, 10: 60, 0: 60 },
        40_000: {                                              50: 160,                        10: 80,      },
    }

    def __init__(self, spec: LinguisticSASpec, run_for_ticks: int, bailout: int = None):
        super().__init__(
            script_number="2_2",
            script_name="2_2_category_production_importance_pruned_tsa.py",
            spec=spec,
            run_for_ticks=run_for_ticks,
            bailout=bailout)

    @property
    def name(self) -> str:
        # importance pruned
        return super().name + "_im"

    @property
    def qsub_command(self) -> str:
        cmd = f"qsub"
        # qsub args
        cmd += f" -S {self._python_location}"
        cmd += f" -q {self._queue}"
        cmd += f" -N {self.name}"
        cmd += f" -m e -M c.wingfield@lancaster.ac.uk"
        cmd += f" -l h_vmem={self.RAM[self.spec.max_radius][self.spec.pruning]}G"
        # script
        cmd += f" {self.script_name}"
        # script args
        cmd += f" --prune_importance {self.spec.pruning}"
        cmd += f" --bailout {self.bailout}"
        cmd += f" --corpus_name {self.spec.corpus_name}"
        cmd += f" --firing_threshold {self.spec.firing_threshold}"
        cmd += f" --impulse_pruning_threshold {self.spec.impulse_pruning_threshold}"
        cmd += f" --distance_type {self.spec.distance_type.name}"
        cmd += f" --length_factor {self.spec.length_factor}"
        cmd += f" --model_name {self.spec.model_name}"
        cmd += f" --node_decay_factor {self.spec.node_decay_factor}"
        cmd += f" --radius {self.spec.model_radius}"
        cmd += f" --edge_decay_sd_factor {self.spec.edge_decay_sd}"
        cmd += f" --run_for_ticks {self.run_for_ticks}"
        cmd += f" --words {int(self.spec.graph_size)}"
        return cmd
