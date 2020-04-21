from dataclasses import dataclass
from typing import List

from model.utils.job import Job, Spec
from ldm.utils.maths import DistanceType


@dataclass
class Spec_1_6(Spec):
    length_factor: int
    distance_type: DistanceType
    pruning_length: int

    @property
    def cli_args(self) -> List[str]:
        return super().cli_args + [
            f" --length_factor {self.length_factor}",
            f" --distance_type {self.distance_type.name}",
            f" --pruning_length {self.pruning_length}",
        ]

    @property
    def shorthand(self) -> str:
        return f"sm_" \
               f"{self.pruning_length}"


class Job_1_6(Job):

    # pruning -> RAM/G
    RAM = {
        20:  2,
        40:  2,
        60:  2,
        80:  2,
        100: 5,
        120: 10,
        140: 20,
        160: 30,
        180: 40,
        200: 60,
        220: 70,
        240: 100,
        260: 120,
    }

    def __init__(self, spec: Spec_1_6):
        super().__init__(
            script_number="1_6",
            script_name="1_6_sensorimotor_neighbourhood_densities.py",
            spec=spec)

    @property
    def name(self) -> str:
        # percentage pruned
        return super().name + "_nbhd"

    @property
    def _ram_requirement_g(self):
        assert isinstance(self.spec, Spec_1_6)
        return self.RAM[self.spec.pruning_length]


if __name__ == '__main__':
    for pruning_length in range(20, 261, 20):
        Job_1_6(Spec_1_6(pruning_length=pruning_length,
                         distance_type=DistanceType.Minkowski3,
                         length_factor=100)
                ).submit()
