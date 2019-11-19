"""
Submits a job batch
"""
from jobs.job_1_6 import Job_1_6, Spec_1_6
from ldm.utils.maths import DistanceType

if __name__ == '__main__':

    for pruning_length in range(20, 261, 20):
        Job_1_6(Spec_1_6(pruning_length=pruning_length,
                         distance_type=DistanceType.Minkowski3,
                         length_factor=100)
                ).submit()
