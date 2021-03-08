from dataclasses import dataclass
from typing import List

from framework.cli.job import Job, JobSpec, _SerialisableDict
from framework.cognitive_model.ldm.utils.maths import DistanceType


@dataclass
class JobSpec_0_4(JobSpec):
    length_factor: int
    distance_type: DistanceType
    pruning_distance: float

    def _to_dict(self) -> _SerialisableDict:
        return {
            **super()._to_dict(),
            "Length factor": str(self.length_factor),
            "Distance type": self.distance_type.name,
            "Pruning distance": str(self.pruning_distance),
        }

    @classmethod
    def _from_dict(cls, dictionary: _SerialisableDict):
        return cls(length_factor=int(dictionary["Length factor"]),
                   distance_type=DistanceType.from_name(dictionary["Distance type"]),
                   pruning_distance=float(dictionary["Pruning distance"]))

    @property
    def cli_args(self) -> List[str]:
        args = super().cli_args + [
            f" --length_factor {self.length_factor}",
            f" --distance_type {self.distance_type.name}",
            f" --pruning_distance {self.pruning_distance}"
        ]
        return args

    @property
    def shorthand(self) -> str:
        return f"04_" \
               f"{self.length_factor}_" \
               f"{self.pruning_distance}"
               
    # This isn't used
    def output_location_relative(self):
        pass


class Job_0_4(Job):
    RAM = 8

    def __init__(self, spec: JobSpec_0_4):
        super().__init__(
            script_number="0_4",
            script_name="0_4_preprune_sensorimotor_graphs.py",
            spec=spec)

    @property
    def _ram_requirement_g(self):
        assert isinstance(self.spec, JobSpec_0_4)
        return self.RAM


if __name__ == '__main__':
    Job_0_4(
        JobSpec_0_4(distance_type=DistanceType.Minkowski3, length_factor=162, pruning_distance=1.5)
    ).submit()
