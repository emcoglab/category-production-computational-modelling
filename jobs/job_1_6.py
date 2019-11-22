import logging
from dataclasses import dataclass

from jobs.job import Job, SASpec
from ldm.utils.maths import DistanceType


@dataclass
class Spec_1_6(SASpec):
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


logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


if __name__ == '__main__':
    logging.basicConfig(format=logger_format, datefmt=logger_dateformat, level=logging.INFO)

    for pruning_length in range(20, 261, 20):
        Job_1_6(Spec_1_6(pruning_length=pruning_length,
                         distance_type=DistanceType.Minkowski3,
                         length_factor=100)
                ).submit()
