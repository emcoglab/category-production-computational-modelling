from dataclasses import dataclass
from typing import List

from framework.cli.job import Job, JobSpec, _SerialisableDict
from framework.cognitive_model.ldm.utils.maths import DistanceType


@dataclass
class JobSpec_0_3(JobSpec):
    length_factor: int
    distance_type: DistanceType
    use_breng_translation: bool

    def _to_dict(self) -> _SerialisableDict:
        return {
            **super()._to_dict(),
            "Length factor": str(self.length_factor),
            "Distance type": self.distance_type.name,
            "Use BrEng translation": str(self.use_breng_translation)
        }

    @classmethod
    def _from_dict(cls, dictionary: _SerialisableDict):
        return cls(length_factor=int(dictionary["Length factor"]),
                   distance_type=DistanceType.from_name(dictionary["Distance type"]),
                   use_breng_translation=bool(dictionary["Use BrEng translation"]))

    @property
    def cli_args(self) -> List[str]:
        return super().cli_args + [
            f" --length_factor {self.length_factor}",
        ]

    @property
    def shorthand(self) -> str:
        return f"03_" \
               f"{self.length_factor}"


class Job_0_3(Job):
    RAM = 35

    def __init__(self, spec: JobSpec_0_3):
        super().__init__(
            script_number="0_3",
            script_name="0_3_save_sensorimotor_graph.py",
            spec=spec)

    @property
    def _ram_requirement_g(self):
        assert isinstance(self.spec, JobSpec_0_3)
        return self.RAM


if __name__ == '__main__':
    Job_0_3(
        JobSpec_0_3(distance_type=DistanceType.Minkowski3, use_breng_translation=True, length_factor=159)
    ).submit()
