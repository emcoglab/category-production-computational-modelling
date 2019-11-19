from dataclasses import dataclass

from jobs.job import Job, Spec
from ldm.utils.maths import DistanceType


@dataclass
class Spec_1_6(Spec):
    distance_type: DistanceType
    pruning_length: int

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
    def qsub_command(self) -> str:
        cmd = f"qsub"
        # qsub args
        cmd += f" -S {self._python_location}"
        cmd += f" -q {self._queue}"
        cmd += f" -N {self.name}"
        cmd += f" -m e -M c.wingfield@lancaster.ac.uk"
        cmd += f" -l h_vmem={self.RAM[self.spec.pruning_length]}G"
        # script
        cmd += f" {self.script_name}"
        # script args
        cmd += f" --length_factor {self.spec.length_factor}"
        cmd += f" --distance_type {self.spec.distance_type.name}"
        cmd += f" --pruning_length {self.spec.pruning_length}"
        return cmd
