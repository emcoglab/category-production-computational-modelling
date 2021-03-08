from pathlib import Path
from typing import Dict

from framework.cli.job import LinguisticPropagationJob, LinguisticPropagationJobSpec


class Job_2_3(LinguisticPropagationJob):

    # model_name -> n_words -> RAM/G
    RAM: Dict[str, Dict[int, int]] = {
        "pmi_ngram": {
            1_000: 3,
            3_000: 4,
            10_000: 10,
            30_000: 16,
            40_000: 22,
        },
        "ppmi_ngram": {
            1_000: 3,
            3_000: 4,
            10_000: 8,
            30_000: 10,
            40_000: 14,
        }
    }

    def __init__(self, spec: LinguisticPropagationJobSpec):
        super().__init__(
            script_number="2_3",
            script_name="2_3_category_production_ngram_tsa.py",
            spec=spec)

    @property
    def _ram_requirement_g(self):
        assert isinstance(self.spec, LinguisticPropagationJobSpec)
        return self.RAM[self.spec.model_name][self.spec.n_words]


if __name__ == '__main__':
    jobs = [
        Job_2_3(s)
        for s in LinguisticPropagationJobSpec.load_multiple(
            Path(Path(__file__).parent, "job_specifications/job_journal_paper_linguistic.yaml"))
    ]
    for job in jobs:
        job.run_locally(extra_arguments=[])
