from model.utils.job import LinguisticPropagationJob, LinguisticPropagationJobSpec


class Job_2_1(LinguisticPropagationJob):

    # n_words -> pruning -> RAM/G
    RAM = {
        3_000:  {0:  3,  10:  3,  20:  3,  30:  3,  40:  2,  50:  2},
        10_000: {0: 20,  10: 20,  20: 17,  30: 14,  40: 13,  50: 12},
        20_000: {0: 65,  10: 60,  20: 55,  30: 50,  40: 40,  50: 35},
        30_000: {0: 120, 10: 120, 20: 90,  30: 90,  40: 80,  50: 80},
    }

    def __init__(self, spec: LinguisticPropagationJobSpec):
        super().__init__(
            script_number="2_1",
            script_name="2_1_category_production_pruned_tsa.py",
            spec=spec)

    @property
    def name(self) -> str:
        # percentage pruned
        return super().name + "_pc"

    @property
    def _ram_requirement_g(self):
        assert isinstance(self.spec, LinguisticPropagationJobSpec)
        return self.RAM[self.spec.n_words][self.spec.pruning]
