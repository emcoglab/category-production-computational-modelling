from dataclasses import dataclass
from typing import List

from framework.cli.job import Job, JobSpec, _SerialisableDict


@dataclass
class JobSpec_0_3(JobSpec):
    def _to_dict(self) -> _SerialisableDict:
        return {
            **super()._to_dict(),
            "Length factor": str(self.length_factor),
        }

    @classmethod
    def _from_dict(cls, dictionary: _SerialisableDict):
        return cls(length_factor=int(dictionary["Length factor"]))

    length_factor: int

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
    Job_0_3(spec=JobSpec_0_3(length_factor=100)).submit()
