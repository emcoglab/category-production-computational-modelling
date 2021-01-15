from pathlib import Path
from typing import Dict

from framework.cli.job import InteractiveCombinedJob, InteractiveCombinedJobSpec

logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


class Job_2_7(InteractiveCombinedJob):

    # max_sphere_radius -> RAM/G
    SM_RAM: Dict[int, int] = {
        100: 5,
        150: 30,
        198: 55,  # 198 is the largest min edge length, so the threshold below which the graph becomes disconnected
        200: 60,
        250: 120,
    }
    LING_RAM: Dict[str, Dict[int, int]] = {
        "pmi_ngram": {
            1_000: 2,
            3_000: 3,
            10_000: 7,
            30_000: 11,
            40_000: 15,
            60_000: 20,
        },
        "ppmi_ngram": {
            1_000: 2,
            3_000: 3,
            10_000: 5,
            30_000: 7,
            40_000: 9,
            60_000: 11,
        }
    }

    def __init__(self, spec: InteractiveCombinedJobSpec):
        super().__init__(
            script_number="2_7",
            script_name="2_7_interactive_combined.py",
            spec=spec)

    @property
    def _ram_requirement_g(self):
        assert isinstance(self.spec, InteractiveCombinedJobSpec)
        return self.SM_RAM[int(self.spec.sensorimotor_spec.max_radius * self.spec.sensorimotor_spec.length_factor)] \
               + self.LING_RAM[self.spec.linguistic_spec.model_name][self.spec.linguistic_spec.n_words]


if __name__ == '__main__':

    jobs = [
        Job_2_7(s)
        for s in InteractiveCombinedJobSpec.load_multiple(
            Path(Path(__file__).parent, "job_specifications/job_interactive_testing.yaml"))
    ]
    for job in jobs:
        job.run_locally()
