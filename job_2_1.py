from model.utils.job import LinguisticPropagationJob, LinguisticPropagationSpec


class Job_2_1(LinguisticPropagationJob):

    # graph_size -> pruning -> RAM/G
    RAM = {
        3_000:  {0:  3,  10:  3,  20:  3,  30:  3,  40:  2,  50:  2},
        10_000: {0: 20,  10: 20,  20: 17,  30: 14,  40: 13,  50: 12},
        20_000: {0: 65,  10: 60,  20: 55,  30: 50,  40: 40,  50: 35},
        30_000: {0: 120, 10: 120, 20: 90,  30: 90,  40: 80,  50: 80},
    }

    def __init__(self, spec: LinguisticPropagationSpec, run_for_ticks: int, bailout: int = None):
        super().__init__(
            script_number="2_1",
            script_name="2_1_category_production_pruned_tsa.py",
            spec=spec,
            run_for_ticks=run_for_ticks,
            bailout=bailout)

    @property
    def name(self) -> str:
        # percentage pruned
        return super().name + "_pc"

    @property
    def qsub_command(self) -> str:
        cmd = self.script_name
        # script args
        cmd += f" --prune_percent {self.spec.pruning}"
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

    @property
    def _ram_requirement_g(self):
        assert isinstance(self.spec, LinguisticPropagationSpec)
        return self.RAM[self.spec.graph_size][self.spec.pruning]
