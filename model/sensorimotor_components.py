from __future__ import annotations

from enum import Enum, auto
from logging import getLogger
from typing import Optional, List, Dict, Set

from model.basic_types import ActivationValue, ItemIdx, ItemLabel
from model.buffer import AccessibleSet, WorkingMemoryBuffer
from model.events import ModelEvent, ItemActivatedEvent
from model.sensorimotor_propagator import SensorimotorPropagator
from model.utils.iterable import partition
from model.utils.maths import prevalence_from_fraction_known, scale_prevalence_01
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms

logger = getLogger()
logger_format = '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
logger_dateformat = "%Y-%m-%d %H:%M:%S"


class SensorimotorComponent:

    def __init__(self,
                 propagator: SensorimotorPropagator,
                 activation_cap: ActivationValue,
                 norm_attenuation_statistic: NormAttenuationStatistic,
                 accessible_set_threshold: ActivationValue,
                 accessible_set_capacity: Optional[int],
                 ):
        """
        :param activation_cap:
            If None is supplied, no cap is used.
        :param accessible_set_threshold:
            Used to determine what counts as "activated" and in the accessible set.
        """

        assert (activation_cap
                # If activation_cap == accessible_set_threshold, items will only enter the accessible set when fully
                # activated.
                >= accessible_set_threshold
                # accessible_set_threshold must be strictly positive, else no item can ever be reactivated (since
                # membership to the accessible set is a guard to reactivation).
                > 0)

        # Data

        # Use >= and < to test for above/below
        # Cap on a node's total activation after receiving incoming.
        self.activation_cap: ActivationValue = activation_cap

        sensorimotor_norms = SensorimotorNorms()

        def get_statistic_for_item(idx: ItemIdx):
            """Gets the correct statistic for an item."""
            if norm_attenuation_statistic is NormAttenuationStatistic.FractionKnown:
                # Fraction known will all be in the range [0, 1], so we can use it as a scaling factor directly
                return sensorimotor_norms.fraction_known(self.propagator.idx2label[idx])
            elif norm_attenuation_statistic is NormAttenuationStatistic.Prevalence:
                # Brysbaert et al.'s (2019) prevalence has a defined range, so we can affine-scale it into [0, 1] for the
                # purposes of attenuating the activation
                return scale_prevalence_01(prevalence_from_fraction_known(sensorimotor_norms.fraction_known(self.propagator.idx2label[idx])))
            else:
                raise NotImplementedError()

        self._attenuation_statistic: Dict[ItemIdx, float] = {
            idx: get_statistic_for_item(idx)
            for idx in self.propagator.graph.nodes
        }

        # region Resettable
        # These fields are reinitialised in .reset()

        # The set of items which are "accessible to conscious awareness" even if they are not in the working memory
        # buffer
        self.accessible_set: AccessibleSet = AccessibleSet(accessible_set_threshold, accessible_set_capacity)

        # TODO: really, this should be hidden, and the present class should provide the external interface
        self.propagator: SensorimotorPropagator = propagator

        # region modulations and guards

        # No presynaptic guards
        self.propagator.presynaptic_guards.extend([])
        self.propagator.presynaptic_modulations.extend([
            self._apply_activation_cap,
            self._attenuate_by_statistic,
            self._apply_memory_pressure,
        ])
        self.propagator.postsynaptic_modulations.extend([
            self._apply_activation_cap
        ])
        self.propagator.postsynaptic_guards.extend([
            self._not_in_accessible_set
        ])

        # endregion

    # TODO: not sure this is the best place for these to live
    def _apply_activation_cap(self, idx: ItemIdx, activation: ActivationValue) -> ActivationValue:
        """If accumulated activation is over the cap, apply the cap."""
        # The activation cap, if used, MUST be greater than the firing threshold (this is checked in __init__, so
        # applying the cap does not effect whether the node will fire or not)
        return min(activation, self.activation_cap)

    def _attenuate_by_statistic(self, idx: ItemIdx, activation: ActivationValue) -> ActivationValue:
        # Attenuate the incoming activations to a concept based on a statistic of the concept
        return activation * self._attenuation_statistic[idx]

    def _apply_memory_pressure(self, idx: ItemIdx, activation: ActivationValue) -> ActivationValue:
        # When AS is full, MP is 1, and activation is killed.
        # When AS is empty, MP is 0, and activation is unaffected.
        return activation * 1 - self.accessible_set.pressure

    def _not_in_accessible_set(self, idx: ItemIdx, activation: ActivationValue) -> bool:
        # Node will only fire if it's not in the accessible set
        return idx not in self.accessible_set

    def reset(self):
        self.propagator.reset()
        self.accessible_set.clear()

    def tick(self) -> List[ModelEvent]:
        # Decay events before activating anything new
        # (in case accessible set membership is used to modulate or guard anything)
        self.accessible_set.prune_decayed_items(activation_lookup=self.propagator.activation_of_item_with_idx,
                                                time=self.propagator.clock)

        logger.info(
            f"\tAS: {len(self.accessible_set)}/{self.accessible_set.capacity if self.accessible_set.capacity is not None else '∞'} "
            f"(MP: {self.accessible_set.pressure})")

        # Proceed with ._evolve_model() and record what became activated
        # Activation and firing may be affected by the size of or membership to the accessible set and the buffer, but
        # nothing will ENTER it until later, and everything that will LEAVE this tick already has done so.
        tick_events = self.propagator.tick()
        activation_events, other_events = partition(tick_events, lambda e: isinstance(e, ItemActivatedEvent))

        # Update accessible set
        self.accessible_set.present_items(activation_events=activation_events,
                                          activation_lookup=self.propagator.activation_of_item_with_idx,
                                          time=self.propagator.clock)

        return activation_events + other_events

    @property
    def concept_labels(self) -> Set[ItemLabel]:
        """Labels of concepts"""
        return set(w for i, w in self.propagator.idx2label.items())


class BufferedSensorimotorComponent(SensorimotorComponent):

    def __init__(self,
                 propagator: SensorimotorPropagator,
                 activation_cap: ActivationValue,
                 norm_attenuation_statistic: NormAttenuationStatistic,
                 accessible_set_threshold: ActivationValue,
                 accessible_set_capacity: Optional[int],
                 buffer_threshold: ActivationValue,
                 buffer_capacity: Optional[int],
                 ):
        """
        :param buffer_capacity:
            The maximum size of the buffer. After this, qualifying items will displace existing items rather than just
            being added.
        :param buffer_threshold:
            The minimum activation required for a concept to enter the working_memory_buffer.
        """

        super().__init__(
            propagator=propagator,
            activation_cap=activation_cap,
            norm_attenuation_statistic=norm_attenuation_statistic,
            accessible_set_threshold=accessible_set_threshold,
            accessible_set_capacity=accessible_set_capacity,
        )

        assert (activation_cap
                # If activation_cap == buffer_threshold, items will only enter the buffer when fully activated.
                >= buffer_threshold
                # If buffer_pruning_threshold == accessible_set_threshold then the only things in the accessible set
                # will be those items which were displaced from the buffer before being pruned. We probably won't use
                # this but it's not invalid or degenerate.
                >= accessible_set_threshold)

        # region Resettable
        # These fields are reinitialised in .reset()

        # The set of items which are currently being consciously considered.
        #
        # A fixed size (self.buffer_capacity).  Items may enter the buffer when they are activated and leave when they
        # decay sufficiently (self.buffer_pruning_threshold) or are displaced.
        #
        # This is updated each .tick() based on items which fired (a prerequisite for entering the buffer)
        self.working_memory_buffer: WorkingMemoryBuffer = WorkingMemoryBuffer(buffer_threshold, buffer_capacity)

        # endregion

    def tick(self) -> List[ModelEvent]:
        # Decay events before activating anything new
        # (in case buffer membership is used to modulate or guard anything)
        decay_events = self.working_memory_buffer.prune_decayed_items(
            activation_lookup=self.propagator.activation_of_item_with_idx,
            time=self.propagator.clock)

        tick_events = super().tick()
        activation_events, other_events = partition(tick_events, lambda e: isinstance(e, ItemActivatedEvent))

        # Update buffer
        # Some events will get updated commensurately.
        # `activation_events` may now contain some non-activation events.
        activation_events = self.working_memory_buffer.present_items(
            activation_events,
            activation_lookup=self.propagator.activation_of_item_with_idx,
            time=self.propagator.clock)

        return decay_events + activation_events + other_events

    def reset(self):
        super().reset()
        self.working_memory_buffer.clear()


class NormAttenuationStatistic(Enum):
    """The statistic to use for attenuating activation of norms labels."""
    FractionKnown = auto()
    Prevalence = auto()

    @property
    def name(self) -> str:
        """The name of the NormAttenuationStatistic"""
        if self is NormAttenuationStatistic.FractionKnown:
            return "Fraction known"
        if self is NormAttenuationStatistic.Prevalence:
            return "Prevalence"
        else:
            raise NotImplementedError()
