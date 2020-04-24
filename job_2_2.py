from model.utils.job import LinguisticPropagationJob, LinguisticPropagationJobSpec


class Job_2_2(LinguisticPropagationJob):

    # n_words -> pruning -> RAM/G
    RAM = {
        3_000:  {100: 5,   90: 5,   80: 5,   70: 5,   60: 5,   50: 5 , 40: 5 , 30: 5 , 20: 5 , 10: 5 , 0: 5  },
        10_000: {100: 30,  90: 26,  80: 24,  70: 22,  60: 20,  50: 15, 40: 13, 30: 12, 20: 12, 10: 12, 0: 12 },
        20_000: {100: 70,  90: 65,  80: 60,  70: 55,  60: 50,  50: 45, 40: 35, 30: 30, 20: 30, 10: 25, 0: 25 },
        30_000: {100: 160, 90: 140, 80: 120, 70: 110, 60: 100, 50: 90, 40: 80, 30: 70, 20: 70, 10: 60, 0: 60 },
        40_000: {                                              50: 160,                        10: 80,      },
    }

    def __init__(self, spec: LinguisticPropagationJobSpec):
        super().__init__(
            script_number="2_2",
            script_name="2_2_category_production_importance_pruned_tsa.py",
            spec=spec)

    @property
    def name(self) -> str:
        # importance pruned
        return super().name + "_im"

    @property
    def _ram_requirement_g(self):
        assert isinstance(self.spec, LinguisticPropagationJobSpec)
        return self.RAM[self.spec.n_words][self.spec.pruning]
