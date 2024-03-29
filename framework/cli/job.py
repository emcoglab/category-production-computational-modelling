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
from __future__ import annotations

from pathlib import Path
from subprocess import run
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Union

import yaml

from framework.cli.lookups import get_model_from_params, get_corpus_from_name
from framework.cognitive_model.components import ModelComponent, FULL_ACTIVATION
from framework.cognitive_model.ldm.corpus.indexing import FreqDist
from framework.cognitive_model.ldm.model.base import LinguisticDistributionalModel
from framework.cognitive_model.linguistic_components import LinguisticComponent
from framework.cognitive_model.linguistic_propagator import LinguisticPropagator
from framework.cognitive_model.sensorimotor_components import SensorimotorComponent, BufferedSensorimotorComponent
from framework.cognitive_model.sensorimotor_propagator import SensorimotorPropagator
from framework.cognitive_model.ldm.utils.maths import DistanceType
from framework.cognitive_model.basic_types import ActivationValue
from framework.cognitive_model.graph import EdgePruningType
from framework.cognitive_model.attenuation_statistic import AttenuationStatistic
from framework.cognitive_model.version import VERSION, GIT_HASH

_SerialisableDict = Dict[str, str]


# region Job Specs

@dataclass
class JobSpec(ABC):

    def __post_init__(self):
        self._validate()

    def _validate(self) -> None:
        """
        Returns silently if class is valid
        """
        return

    @property
    @abstractmethod
    def shorthand(self) -> str:
        """
        A short name which may not uniquely define the spec, but can be used to
        disambiguate job names, etc.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def cli_args(self) -> List[str]:
        """
        List of key-value pairs in `--arg_name val` format.
        """
        # By default, no arguments. This way we can confidently supply *super().cli_args().
        return []

    @abstractmethod
    def _to_dict(self) -> _SerialisableDict:
        """Serialise."""
        return {
            "Version": VERSION,
            "Commit": GIT_HASH,
        }

    @classmethod
    @abstractmethod
    def _from_dict(cls, dictionary: _SerialisableDict):
        """Deserialise.  Does not preserve Version or Commit."""
        raise NotImplementedError()

    @abstractmethod
    def output_location_relative(self) -> Path:
        """
        Path for a job's output to be saved, relative to the parent output directory.
        """
        raise NotImplementedError()

    def save(self, in_location: Path) -> None:
        """
        Save the model spec in a common format.
        Creates the output location if it doesn't already exist.
        """
        if not in_location.is_dir():
            in_location.mkdir(parents=True)
        with open(Path(in_location, " model_spec.yaml"), mode="w", encoding="utf-8") as spec_file:
            yaml.dump(self._to_dict(), spec_file, yaml.SafeDumper,
                      # Always serialise in block style
                      default_flow_style=False)

    @classmethod
    def load(cls, filename: Path):
        with open(filename, mode="r", encoding="utf-8") as file:
            return cls._from_dict(yaml.load(file, yaml.SafeLoader))

    @classmethod
    def load_multiple(cls, filename: Path) -> List:
        with open(filename, mode="r", encoding="utf-8") as file:
            return [
                cls._from_dict(d)
                # Unclear why we need the [0] here, but we seem to. Must just be the behaviour of yaml.load_all
                for d in list(yaml.load_all(file, yaml.SafeLoader))[0]
            ]

    def csv_comments(self) -> List[str]:
        """List of commented"""
        return [
            f"# \t{k} = {v}"
            for k, v in self._to_dict().items()
        ]


@dataclass
class PropagationJobSpec(JobSpec, ABC):
    length_factor: int
    run_for_ticks: Optional[int]
    bailout: Optional[int]

    @property
    @abstractmethod
    def cli_args(self) -> List[str]:
        args = [
            f"--length_factor {self.length_factor}",
        ]
        if self.run_for_ticks is not None:
            args.append(f"--run_for_ticks {self.run_for_ticks}")
        if self.bailout is not None:
            args.append(f"--bailout {self.bailout}")
        return args

    def _to_dict(self) -> _SerialisableDict:
        d = super()._to_dict()
        d.update({
            "Length factor": str(self.length_factor),
        })
        if self.run_for_ticks is not None:
            d.update({
                "Run for ticks": str(self.run_for_ticks),
            })
        if self.bailout is not None:
            d.update({
                "Bailout": str(self.bailout),
            })
        return d

    @abstractmethod
    def to_component(self, component_class) -> ModelComponent:
        raise NotImplementedError()


@dataclass
class SensorimotorPropagationJobSpec(PropagationJobSpec):
    max_radius: float
    node_decay_sigma: float
    node_decay_median: float
    distance_type: DistanceType
    accessible_set_threshold: float
    accessible_set_capacity: Optional[int]
    attenuation_statistic: AttenuationStatistic
    use_activation_cap: bool
    use_breng_translation: bool

    @property
    def cli_args(self) -> List[str]:
        args = super().cli_args + [
            f"--distance_type {self.distance_type.name}",
            f"--max_sphere_radius {self.max_radius}",
            f"--accessible_set_threshold {self.accessible_set_threshold}",
            f"--length_factor {self.length_factor}",
            f"--node_decay_median {self.node_decay_median}",
            f"--node_decay_sigma {self.node_decay_sigma}",
            f"--attenuation {self.attenuation_statistic.name}",
        ]
        if self.accessible_set_capacity is not None:
            args.append(f"--accessible_set_capacity {self.accessible_set_capacity}")
        if self.use_breng_translation:
            args.append(f"--use_breng_translation")
        if self.use_activation_cap:
            args.append(f"--use_activation_cap")
        return args

    @property
    def shorthand(self) -> str:
        shorthand = "sm_"
        if self.use_breng_translation:
            shorthand += "breng_"
        shorthand += f"r{self.max_radius}_" \
                     f"m{self.node_decay_median}_" \
                     f"s{self.node_decay_sigma}_" \
                     f"a{self.accessible_set_threshold}_"
        if self.accessible_set_capacity is not None:
            shorthand += f"ac{self.accessible_set_capacity}"
        else:
            shorthand += f"ac-"
        if self.use_activation_cap:
            shorthand += f"cap_"
        return shorthand

    def output_location_relative(self) -> Path:
        breng_string = " BrEng" if self.use_breng_translation else ""
        cap_string   = " capped;" if self.use_activation_cap else ""
        return Path(
            f"Sensorimotor {VERSION}{breng_string}",
            f"{self.distance_type.name} length {self.length_factor} att {self.attenuation_statistic.name};"
            f" max-r {self.max_radius};"
            f" n-decay-m {self.node_decay_median};"
            f" n-decay-σ {self.node_decay_sigma};"
            f" as-θ {self.accessible_set_threshold};"
            f" as-cap {self.accessible_set_capacity:,};"
            f"{cap_string}"
            f" run-for {self.run_for_ticks};"
            f" bail {self.bailout}"
        )

    def _to_dict(self) -> _SerialisableDict:
        d = super()._to_dict()
        d.update({
            "Distance type": self.distance_type.name,
            "Length factor": str(self.length_factor),
            "Max sphere radius": str(self.max_radius),
            "Log-normal median": str(self.node_decay_median),
            "Log-normal sigma": str(self.node_decay_sigma),
            "Accessible set threshold": str(self.accessible_set_threshold),
            "Accessible set capacity": str(self.accessible_set_capacity),
            "Attenuation statistic": self.attenuation_statistic.name,
            "Use BrEng translation": str(self.use_breng_translation),
            "Use activation cap": str(self.use_activation_cap),
        })
        return d

    @classmethod
    def _from_dict(cls, dictionary: _SerialisableDict) -> SensorimotorPropagationJobSpec:
        return cls(
            length_factor           =int(dictionary["Length factor"]),
            run_for_ticks           =int(dictionary["Run for ticks"]),
            bailout                 =dictionary["Bailout"] if "Bailout" in dictionary else None,
            max_radius              =float(dictionary["Max radius"]),
            node_decay_sigma        =float(dictionary["Log-normal sigma"]),
            node_decay_median       =float(dictionary["Log-normal median"]),
            accessible_set_capacity =int(dictionary["Accessible set capacity"]),
            accessible_set_threshold=ActivationValue(dictionary["Accessible set threshold"]),
            distance_type           =DistanceType.from_name(dictionary["Distance type"]),
            attenuation_statistic   =AttenuationStatistic.from_slug(dictionary["Attenuation statistic"]),
            use_breng_translation   =bool(dictionary["Use BrEng translation"]) if "Use BrEng translation" in dictionary else False,
            use_activation_cap      =bool(dictionary["Use activation cap"]) if "Use activation cap" in dictionary else False,
        )

    def to_component(self, component_class) -> SensorimotorComponent:
        return component_class(
            propagator=SensorimotorPropagator(
                distance_type=self.distance_type,
                length_factor=self.length_factor,
                max_sphere_radius=self.max_radius,
                node_decay_lognormal_median=self.node_decay_median,
                node_decay_lognormal_sigma=self.node_decay_sigma,
                use_breng_translation=self.use_breng_translation,
                shelf_life=self.run_for_ticks,
            ),
            accessible_set_threshold=self.accessible_set_threshold,
            accessible_set_capacity=self.accessible_set_capacity,
            attenuation_statistic=self.attenuation_statistic,
            activation_cap=FULL_ACTIVATION if self.use_activation_cap else None,
            use_breng_translation=self.use_breng_translation,
        )


@dataclass
class BufferedSensorimotorPropagationJobSpec(SensorimotorPropagationJobSpec):
    buffer_threshold: float
    buffer_capacity: Optional[int]

    @property
    def cli_args(self) -> List[str]:
        args = super().cli_args + [
            f"--buffer_capacity {self.buffer_capacity}",
            f"--buffer_threshold {self.buffer_threshold}",
        ]
        return args

    @property
    def shorthand(self) -> str:
        return super().shorthand + "_" \
               f"b{self.buffer_threshold}"

    def output_location_relative(self) -> Path:
        breng_string = " BrEng" if self.use_breng_translation else ""
        cap_string   = " capped;" if self.use_activation_cap else ""
        return Path(
            f"Sensorimotor {VERSION}{breng_string}",
            f"{self.distance_type.name} length {self.length_factor} attenuate {self.attenuation_statistic.name};"
            f" max-r {self.max_radius};"
            f" n-decay-m {self.node_decay_median};"
            f" n-decay-σ {self.node_decay_sigma};"
            f" as-θ {self.accessible_set_threshold};"
            f" as-cap {self.accessible_set_capacity:,};"
            f" buff-θ {self.buffer_threshold};"
            f" buff-cap {self.buffer_capacity};"
            f"{cap_string}"
            f" run-for {self.run_for_ticks};"
            f" bail {self.bailout}",
        )

    def _to_dict(self) -> _SerialisableDict:
        d = super()._to_dict()
        d.update({
            "Buffer capacity": str(self.buffer_capacity),
            "Buffer threshold": str(self.buffer_threshold),
        })
        return d

    @classmethod
    def _from_dict(cls, dictionary: _SerialisableDict) -> BufferedSensorimotorPropagationJobSpec:
        return cls(
            length_factor           =int(dictionary["Length factor"]),
            run_for_ticks           =dictionary["Run for ticks"] if "Run for ticks" in dictionary else None,
            bailout                 =dictionary["Bailout"] if "Bailout" in dictionary else None,
            max_radius              =float(dictionary["Max sphere radius"]),
            node_decay_sigma        =float(dictionary["Log-normal sigma"]),
            node_decay_median       =float(dictionary["Log-normal median"]),
            buffer_capacity         =int(dictionary["Buffer capacity"]),
            buffer_threshold        =ActivationValue(dictionary["Buffer threshold"]),
            accessible_set_capacity =int(dictionary["Accessible set capacity"]),
            accessible_set_threshold=ActivationValue(dictionary["Accessible set threshold"]),
            distance_type           =DistanceType.from_name(dictionary["Distance type"]),
            attenuation_statistic   =AttenuationStatistic.from_slug(dictionary["Attenuation statistic"]),
            use_breng_translation   =bool(dictionary["Use BrEng translation"]) if "Use BrEng translation" in dictionary else False,
            use_activation_cap      =bool(dictionary["Use activation cap"]) if "Use activation cap" in dictionary else False,
        )

    def to_component(self, component_class) -> BufferedSensorimotorComponent:
        return component_class(
            propagator=SensorimotorPropagator(
                distance_type=self.distance_type,
                length_factor=self.length_factor,
                max_sphere_radius=self.max_radius,
                node_decay_lognormal_median=self.node_decay_median,
                node_decay_lognormal_sigma=self.node_decay_sigma,
                use_breng_translation=self.use_breng_translation,
            ),
            accessible_set_threshold=self.accessible_set_threshold,
            accessible_set_capacity=self.accessible_set_capacity,
            attenuation_statistic=self.attenuation_statistic,
            activation_cap=FULL_ACTIVATION if self.use_activation_cap else None,
            buffer_capacity=self.buffer_capacity,
            buffer_threshold=self.buffer_threshold,
            use_breng_translation=self.use_breng_translation,
        )


@dataclass
class LinguisticPropagationJobSpec(PropagationJobSpec):
    n_words: int
    firing_threshold: ActivationValue
    model_name: str
    model_radius: int
    corpus_name: str
    edge_decay_sd: float
    node_decay_factor: float
    accessible_set_threshold: float
    accessible_set_capacity: Optional[int]
    use_activation_cap: bool
    impulse_pruning_threshold: ActivationValue
    pruning_type: Optional[EdgePruningType]
    pruning: Optional[int]
    distance_type: Optional[DistanceType] = None

    @property
    def cli_args(self) -> List[str]:
        args = super().cli_args + [
            f"--words {self.n_words}",
            f"--firing_threshold {self.firing_threshold}",
            f"--model_name {self.model_name}",
            f"--radius {self.model_radius}",
            f"--corpus_name {self.corpus_name}",
            f"--edge_decay_sd {self.edge_decay_sd}",
            f"--node_decay_factor {self.node_decay_factor}",
            f"--accessible_set_threshold {self.accessible_set_threshold}",
            f"--impulse_pruning_threshold {self.impulse_pruning_threshold}",
        ]
        if self.accessible_set_capacity is not None:
            args.append(f"--accessible_set_capacity {self.accessible_set_capacity}")
        if self.use_activation_cap:
            args.append(f"--use_activation_cap")
        if self.pruning is not None:
            if self.pruning_type == EdgePruningType.Importance:
                args.append(f"--prune_importance {self.pruning}")
            elif self.pruning_type == EdgePruningType.Percent:
                args.append(f"--prune_percent {self.pruning}")
            elif self.pruning_type == EdgePruningType.Length:
                args.append(f"--prune_length {self.pruning}")
            else:
                raise NotImplementedError()
        if self.distance_type is not None:
            args.append(f"--distance_type {self.distance_type.name}")
        return args

    @property
    def shorthand(self):
        shorthand = f"{int(self.n_words / 1000)}k_" \
               f"f{self.firing_threshold}_" \
               f"s{self.edge_decay_sd}_" \
               f"a{self.accessible_set_threshold}_"
        if self.accessible_set_capacity is not None:
            shorthand += f"ac{self.accessible_set_capacity}_"
        else:
            shorthand += f"ac-_"
        if self.use_activation_cap:
            shorthand += f"cap_"
        shorthand += f"{self.model_name}_" \
                     f"pr{self.pruning}"
        return shorthand

    def output_location_relative(self) -> Path:
        if self.pruning_type is None:
            pruning_suffix = ""
        elif self.pruning_type == EdgePruningType.Percent:
            pruning_suffix = f", longest {self.pruning}% edges removed"
        elif self.pruning_type == EdgePruningType.Importance:
            pruning_suffix = f", importance pruning {self.pruning}"
        else:
            raise NotImplementedError()

        if self.distance_type is not None:
            model_name = (f"{self.model_name}"
                              f" {self.distance_type.name}"
                              f" {self.n_words:,} words, length {self.length_factor}{pruning_suffix}")
        else:
            model_name = (f"{self.model_name}"
                              f" {self.n_words:,} words, length {self.length_factor}{pruning_suffix}")

        cap_string = " capped;" if self.use_activation_cap else ""

        return Path(
            f"Linguistic {VERSION}",
            f"{model_name};"
            f" firing-θ {self.firing_threshold};"
            f" n-decay-f {self.node_decay_factor};"
            f" e-decay-sd {self.edge_decay_sd};"
            f" as-θ {self.accessible_set_threshold};"
            f" as-cap {self.accessible_set_capacity};"
            f"{cap_string}"
            f" imp-prune-θ {self.impulse_pruning_threshold};"
            f" run-for {self.run_for_ticks};"
            f" bail {self.bailout}",
        )

    def _to_dict(self) -> _SerialisableDict:
        d = super()._to_dict()
        d.update({
            "Words": str(self.n_words),
            "Model name": self.model_name,
            "Model radius": str(self.model_radius),
            "Corpus name": self.corpus_name,
            "Length factor": str(self.length_factor),
            "Edge decay SD": str(self.edge_decay_sd),
            "Node decay": str(self.node_decay_factor),
            "Accessible set threshold": str(self.accessible_set_threshold),
            "Accessible set capacity": str(self.accessible_set_capacity),
            "Firing threshold": str(self.firing_threshold),
            "Impulse pruning threshold": str(self.impulse_pruning_threshold),
            "Use activation cap": str(self.use_activation_cap)
        })
        if self.distance_type is not None:
            d.update({
                "Distance type": self.distance_type.name,
            })
        if self.pruning_type is not None:
            d.update({
                "Pruning type": self.pruning_type.name,
                "Pruning": str(self.pruning),
            })
        return d

    @classmethod
    def _from_dict(cls, dictionary: _SerialisableDict):
        return cls(
            length_factor            =int(dictionary["Length factor"]),
            run_for_ticks            =int(dictionary["Run for ticks"]),
            bailout                  =dictionary["Bailout"] if "Bailout" in dictionary else None,
            distance_type            =DistanceType.from_name(dictionary["Distance type"]) if "Distance type" in dictionary else None,
            n_words                  =int(dictionary["Words"]),
            firing_threshold         =ActivationValue(dictionary["Firing threshold"]),
            model_name               =str(dictionary["Model name"]),
            model_radius             =int(dictionary["Model radius"]),
            corpus_name              =str(dictionary["Corpus name"]),
            edge_decay_sd            =float(dictionary["Edge decay SD"]),
            node_decay_factor        =float(dictionary["Node decay"]),
            accessible_set_capacity  =int(dictionary["Accessible set capacity"]) if dictionary["Accessible set capacity"] != 'None' else None,
            accessible_set_threshold =ActivationValue(dictionary["Accessible set threshold"]),
            impulse_pruning_threshold=ActivationValue(dictionary["Impulse pruning threshold"]),
            pruning_type             =EdgePruningType.from_name(dictionary["Pruning type"]) if "Pruning type" in dictionary else None,
            pruning                  =int(dictionary["Pruning"]) if "Pruning" in dictionary else None,
            use_activation_cap       =bool(dictionary["Use activation cap"]) if "Use activation cap" in dictionary else False,
        )

    def to_component(self, component_class) -> LinguisticComponent:
        corpus = get_corpus_from_name(self.corpus_name)
        freq_dist = FreqDist.load(corpus.freq_dist_path)
        distributional_model: LinguisticDistributionalModel = get_model_from_params(
            corpus, freq_dist, self.model_name, self.model_radius)
        return component_class(
            propagator=LinguisticPropagator(
                distance_type=self.distance_type,
                length_factor=self.length_factor,
                n_words=self.n_words,
                distributional_model=distributional_model,
                node_decay_factor=self.node_decay_factor,
                edge_decay_sd=self.edge_decay_sd,
                edge_pruning_type=self.pruning_type,
                edge_pruning=self.pruning,
            ),
            accessible_set_threshold=self.accessible_set_threshold,
            accessible_set_capacity=self.accessible_set_capacity,
            firing_threshold=self.firing_threshold,
            activation_cap=FULL_ACTIVATION if self.use_activation_cap else None,
        )


class LinguisticOneHopJobSpec(LinguisticPropagationJobSpec):
    def output_location_relative(self) -> Path:
        return Path(
            f"Linguistic one-hop {VERSION}",
            *super().output_location_relative().parts[1:]
        )


class BufferedSensorimotorOneHopJobSpec(BufferedSensorimotorPropagationJobSpec):
    def output_location_relative(self) -> Path:
        breng_string = " BrEng" if self.use_breng_translation else ""
        return Path(
            f"Sensorimotor one-hop {VERSION}{breng_string}",
            *super().output_location_relative().parts[1:]
        )


@dataclass
class CombinedJobSpec(JobSpec, ABC):
    linguistic_spec: LinguisticPropagationJobSpec
    sensorimotor_spec: SensorimotorPropagationJobSpec

    def _validate(self) -> None:
        super()._validate()
        assert self.linguistic_spec.run_for_ticks == self.sensorimotor_spec.run_for_ticks

    @property
    def cli_args(self) -> List[str]:
        linguistic_args = [
            f"--linguistic_{a.lstrip('-')}"
            for a in self.linguistic_spec.cli_args
            # ignore args which would be shared between component jobs; move to container model
            if all(ignored_arg not in a for ignored_arg in ["bailout", "run_for_ticks"])
        ]
        sensorimotor_args = [
            f"--sensorimotor_{a.lstrip('-')}"
            for a in self.sensorimotor_spec.cli_args
            if all(ignored_arg not in a for ignored_arg in ["bailout", "run_for_ticks"])
        ]
        return linguistic_args + sensorimotor_args

    def _to_dict(self) -> _SerialisableDict:
        return {
            **super()._to_dict(),
            **{
                self._linguistic_prefix() + key: value
                for key, value in self.linguistic_spec._to_dict().items()
            },
            **{
                self._sensorimotor_prefix() + key: value
                for key, value in self.sensorimotor_spec._to_dict().items()
            }
        }

    @classmethod
    def _linguistic_prefix(cls):
        return "(Linguistic) "

    @classmethod
    def _sensorimotor_prefix(cls):
        return "(Sensorimotor) "

    @classmethod
    def _trim_and_filter_keys(cls, d: _SerialisableDict, prefix: str):
        return {
            key[len(prefix):]: value
            for key, value in d.items()
            if key.startswith(prefix)
        }


@dataclass
class NoninteractiveCombinedJobSpec(CombinedJobSpec):
    def output_location_relative(self) -> Path:
        pass

    @property
    def shorthand(self) -> str:
        return f"ni_{self.linguistic_spec.shorthand}_{self.sensorimotor_spec.shorthand}"


@dataclass
class InteractiveCombinedJobSpec(CombinedJobSpec):
    lc_to_smc_delay: int
    smc_to_lc_delay: int
    lc_to_smc_threshold: ActivationValue
    smc_to_lc_threshold: ActivationValue
    buffer_threshold: ActivationValue
    cross_component_attenuation: float
    buffer_capacity_linguistic_items: Optional[int]
    buffer_capacity_sensorimotor_items: Optional[int]
    run_for_ticks: int
    bailout: Optional[int]

    def output_location_relative(self) -> Path:
        return Path(
            f"Interactive combined {VERSION}",
            *self.sensorimotor_spec.output_location_relative().parts[1:],  # Skip the type name and version number
            *self.linguistic_spec.output_location_relative().parts[1:],  # Skip the type name and version number
            f"delay-ls {self.lc_to_smc_delay};"
            f" delay-sl {self.smc_to_lc_delay};"
            f" θ-ls {ActivationValue(self.lc_to_smc_threshold)};"
            f" θ-sl {ActivationValue(self.smc_to_lc_threshold)};"
            f" cca {float(self.cross_component_attenuation)};"
            f" buff-θ {ActivationValue(self.buffer_threshold)};"
            f" buff-cap-l {self.buffer_capacity_linguistic_items};"
            f" buff-cap-s {self.buffer_capacity_sensorimotor_items};"
            f" run-for {self.run_for_ticks};"
            f" bail {self.bailout}"
        )

    @property
    def shorthand(self) -> str:
        return (f"ic_"
                f"{self.linguistic_spec.shorthand}_"
                f"{self.sensorimotor_spec.shorthand}_"
                f"ls{self.lc_to_smc_delay}-{self.lc_to_smc_threshold}_"
                f"sl{self.smc_to_lc_delay}-{self.smc_to_lc_threshold}_"
                f"cca{self.cross_component_attenuation}_"
                f"b{self.buffer_threshold}_"
                f"bcl{self.buffer_capacity_linguistic_items}_"
                f"bcs{self.buffer_capacity_sensorimotor_items}")

    @property
    def cli_args(self) -> List[str]:
        args = super().cli_args + [
            f"--buffer_threshold {self.buffer_threshold}",
            f"--buffer_capacity_linguistic_items {self.buffer_capacity_linguistic_items}",
            f"--buffer_capacity_sensorimotor_items {self.buffer_capacity_sensorimotor_items}",
            f"--lc_to_smc_delay {self.lc_to_smc_delay}",
            f"--smc_to_lc_delay {self.smc_to_lc_delay}",
            f"--lc_to_smc_threshold {self.lc_to_smc_threshold}",
            f"--smc_to_lc_threshold {self.smc_to_lc_threshold}",
            f"--cross_component_attenuation {self.cross_component_attenuation}",
            f"--run_for_ticks {self.run_for_ticks}",
        ]
        if self.bailout is not None:
            args.append(f"--bailout {self.bailout}")
        return args

    def _to_dict(self) -> _SerialisableDict:
        d = {
            **super()._to_dict(),
            "Linguistic to sensorimotor delay": str(self.lc_to_smc_delay),
            "Sensorimotor to linguistic delay": str(self.smc_to_lc_delay),
            "Linguistic to sensorimotor threshold": str(self.lc_to_smc_threshold),
            "Sensorimotor to linguistic threshold": str(self.smc_to_lc_threshold),
            "Cross-component attenuation": str(self.cross_component_attenuation),
            "Buffer threshold": str(self.buffer_threshold),
            "Buffer capacity (linguistic items)": str(self.buffer_capacity_linguistic_items),
            "Buffer capacity (sensorimotor items)": str(self.buffer_capacity_sensorimotor_items),
            "Run for ticks": str(self.run_for_ticks),
        }
        if self.bailout is not None:
            d["Bailout"] = str(self.bailout)
        return d

    @classmethod
    def _from_dict(cls, dictionary: _SerialisableDict) -> InteractiveCombinedJobSpec:
        return cls(
            linguistic_spec=LinguisticPropagationJobSpec._from_dict({
                **cls._trim_and_filter_keys(dictionary, cls._linguistic_prefix()),
                # Push the shared values into the separate dictionaries
                **{
                    "Bailout": dictionary["Bailout"],
                    "Run for ticks": dictionary["Run for ticks"],
                }
            }),
            sensorimotor_spec=SensorimotorPropagationJobSpec._from_dict({
                **cls._trim_and_filter_keys(dictionary, cls._sensorimotor_prefix()),
                **{
                    "Bailout": dictionary["Bailout"],
                    "Run for ticks": dictionary["Run for ticks"],
                    "Use BrEng translation": True,
                }
            }),
            lc_to_smc_delay=int(dictionary["Linguistic to sensorimotor delay"]),
            smc_to_lc_delay=int(dictionary["Sensorimotor to linguistic delay"]),
            lc_to_smc_threshold=ActivationValue(dictionary["Linguistic to sensorimotor threshold"]),
            smc_to_lc_threshold=ActivationValue(dictionary["Sensorimotor to linguistic threshold"]),
            cross_component_attenuation=float(dictionary["Cross-component attenuation"]),
            buffer_threshold=ActivationValue(dictionary["Buffer threshold"]),
            buffer_capacity_linguistic_items=int(dictionary["Buffer capacity (linguistic items)"]),
            buffer_capacity_sensorimotor_items=int(dictionary["Buffer capacity (sensorimotor items)"]),
            # These replace items which would be shared by individual components
            run_for_ticks=int(dictionary["Run for ticks"]),
            bailout=int(dictionary["Bailout"]) if "Bailout" in dictionary else None,
        )

# endregion


# region Jobs

class Job(ABC):
    _shim = "framework/cognitive_model/utils/shim.sh"

    def __init__(self,
                 script_number: str,
                 script_name: str,
                 spec: JobSpec,
                 ):
        self._number: str = script_number
        self.short_name: str = "j" + script_number.replace("_", "")
        self.script_name: str = script_name  # "../" + script_name
        self.module_name: str = Job._without_py(script_name)
        self.spec = spec

    @property
    def name(self) -> str:
        return f"{self.short_name}_{self.spec.shorthand}"

    @property
    @abstractmethod
    def _ram_requirement_g(self) -> float:
        raise NotImplementedError

    # @final <- TODO: wait for 3.8
    @property
    def qsub_command(self) -> str:
        """The qsub command to run, complete with arguments, to execute this job."""
        cmd = f"qsub"
        # qsub args
        cmd += f" -N {self.name}"
        cmd += f" -l h_vmem={self._ram_requirement_g}G"
        # script
        cmd += f" {self._shim} "
        cmd += self.command
        return cmd

    @property
    def command(self) -> str:
        """The CLI command to run, complete with arguments, to execute this job."""
        cmd = self.script_name
        cmd += " "  # separates args from script name
        cmd += " ".join(self.spec.cli_args)
        return cmd

    def run_locally(self, extra_arguments: Optional[Union[List[str], str]] = None):
        """
        Run the job on the local machine.
        :param extra_arguments:
            Either an extra CLI argument to include which isn't specified in the job spec, or a list of such arguments.
            None, "", or [] implies no extra arguments.
        :return:
        """
        if extra_arguments is None:
            extra_arguments = []
        elif isinstance(extra_arguments, str):
            extra_arguments = [extra_arguments]
        command = self.command + " " + " ".join(extra_arguments)
        print(command)
        run(f"python {command}", shell=True)

    def submit(self, extra_arguments: Optional[Union[List[str], str]] = None):
        if extra_arguments is None:
            extra_arguments = []
        elif isinstance(extra_arguments, str):
            extra_arguments = [extra_arguments]
        command = self.qsub_command + " " + " ".join(extra_arguments)
        print(command)
        run(command, shell=True)

    @classmethod
    def _without_py(cls, script_name: str) -> str:
        if script_name.endswith(".py"):
            return script_name[:-3]
        else:
            return script_name


class PropagationJob(Job, ABC):
    def __init__(self,
                 script_number: str,
                 script_name: str,
                 spec: JobSpec):
        super().__init__(
            script_number=script_number,
            script_name=script_name,
            spec=spec)


class SensorimotorPropagationJob(PropagationJob, ABC):
    def __init__(self, *args, **kwargs):
        self.spec: BufferedSensorimotorPropagationJobSpec
        super().__init__(*args, **kwargs)


class LinguisticPropagationJob(PropagationJob, ABC):
    def __init__(self, *args, **kwargs):
        self.spec: LinguisticPropagationJobSpec
        super().__init__(*args, **kwargs)


class InteractiveCombinedJob(PropagationJob, ABC):
    def __init__(self, *args, **kwargs):
        self.spec: InteractiveCombinedJobSpec
        super().__init__(*args, **kwargs)

# endregion
