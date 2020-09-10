from pathlib import Path
from typing import Dict

from model.utils.job import LinguisticPropagationJob, LinguisticOneHopJobSpec

logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


class Job_2_5(LinguisticPropagationJob):

    # model_name -> n_words -> RAM/G
    RAM: Dict[str, Dict[int, int]] = {
        "pmi_ngram": {
            1_000: 2,
            3_000: 3,
            10_000: 7,
            30_000: 11,
            40_000: 15,
        },
        "ppmi_ngram": {
            1_000: 2,
            3_000: 3,
            10_000: 5,
            30_000: 7,
            40_000: 9,
        }
    }

    def __init__(self, spec: LinguisticOneHopJobSpec):
        super().__init__(
            script_number="2_5",
            script_name="2_5_linguistic_one_hop.py",
            spec=spec)

    @property
    def _ram_requirement_g(self):
        assert isinstance(self.spec, LinguisticOneHopJobSpec)
        return self.RAM[self.spec.model_name][self.spec.n_words]


if __name__ == '__main__':
    jobs = [
        Job_2_5(s)
        for s in LinguisticOneHopJobSpec.load_multiple(
            Path(Path(__file__).parent, "job_specifications/job_cognition_paper_linguistic_search.yaml"))
    ]
    for job in jobs:
        job.run_locally(extra_arguments=[])
