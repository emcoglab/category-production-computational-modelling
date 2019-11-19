"""
===========================
Code for submission of jobs on SGE Wayland.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2019
---------------------------
"""
import subprocess
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ldm.utils.maths import DistanceType


@dataclass
class Spec(metaclass=ABCMeta):
    length_factor: int

    @abstractmethod
    @property
    def shorthand(self) -> str:
        raise NotImplementedError


@dataclass
class SensorimotorSASpec(Spec):
    max_radius: int
    sigma: float
    median: int
    buffer_threshold: float
    accessible_set_threshold: float
    distance_type: DistanceType
    buffer_capacity: int
    accessible_set_capacity: int

    @property
    def shorthand(self) -> str:
        return f"sm_" \
               f"r{self.max_radius}_" \
               f"m{self.median}_" \
               f"s{self.sigma}_" \
               f"a{self.accessible_set_threshold}_" \
               f"ac{self.accessible_set_capacity}_" \
               f"b{self.buffer_threshold}"


@dataclass
class LinguisticSASpec(Spec):
    graph_size: int
    firing_threshold: float
    model_name: str
    model_radius: int
    corpus_name: str
    edge_decay_sd: float
    impulse_pruning_threshold: float
    node_decay_factor: float
    pruning: int
    distance_type: Optional[DistanceType]

    @property
    def shorthand(self):
        return f"{int(self.graph_size / 1000)}k_" \
               f"f{self.firing_threshold}_" \
               f"s{self.edge_decay_sd}_" \
               f"{self.model_name}_" \
               f"pr{self.pruning}"


class Job(metaclass=ABCMeta):
    _python_location = "/Users/cai/Applications/miniconda3/bin/python"
    _queue = "serial"

    def __init__(self,
                 script_number: str,
                 script_name: str,
                 spec: Spec,
                 ):
        self._number: str = script_number
        self.short_name: str = "j" + script_number.replace("_", "")
        self.script_name: str = "../" + script_name
        self.spec = spec

    @property
    def name(self) -> str:
        return f"{self.short_name}{self.spec.shorthand}"

    @abstractmethod
    @property
    def qsub_command(self) -> str:
        raise NotImplementedError()

    def submit(self):
        subprocess.run(self.qsub_command)


class SAJob(Job, metaclass=ABCMeta):
    def __init__(self,
                 script_number: str,
                 script_name: str,
                 spec: Spec,
                 run_for_ticks: int,
                 bailout: int = None):
        super().__init__(
            script_number=script_number,
            script_name=script_name,
            spec=spec)
        self.run_for_ticks: int = run_for_ticks
        self.bailout: int = bailout


class SensorimotorSAJob(SAJob, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        self.spec: SensorimotorSASpec
        super().__init__(*args, **kwargs)


class LinguisticSAJob(SAJob, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        self.spec: LinguisticSASpec
        super().__init__(*args, **kwargs)
