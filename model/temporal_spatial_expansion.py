"""
===========================
Temporal spatial expansion.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2018
---------------------------
"""
from typing import Dict, Set

from ldm.utils.maths import DistanceType
from model.component import ModelComponent, ActivationValue, ItemLabel, ActivationRecord, ItemActivatedEvent
from model.points_in_space import PointsInSpace, PointIdx
from model.utils.math import decay_function_lognormal_median


class TemporalSpatialExpansion(ModelComponent):
    def __init__(self,
                 points_in_space: PointsInSpace,
                 item_labelling_dictionary: Dict[PointIdx, ItemLabel],
                 expansion_rate: float,
                 max_radius: float,
                 distance_type: DistanceType,
                 decay_median: float,
                 decay_shape: float,
                 decay_threshold: ActivationValue):

        super().__init__(item_labelling_dictionary=item_labelling_dictionary)

        self.points: PointsInSpace = points_in_space
        self.expansion_rate: float = expansion_rate
        self.max_radius: float = max_radius
        self.distance_type: DistanceType = distance_type

        # Use >= and < to test for above/below
        self.decay_threshold: ActivationValue = decay_threshold

        self._decay_function = decay_function_lognormal_median(decay_median, decay_shape)

        # Dictionary point_idx -> radius
        self.spheres = dict()

    # TODO: repeated code
    def n_suprathreshold_items(self) -> int:
        """
        The number of nodes which are above the firing threshold.
        May take a long time to compute.
        :return:
        """
        return len([
            p
            for p in self.points.iter_points()
            # TODO: this is the wrong threshold to use
            if self.activation_of_item_with_idx(p.idx) >= self.decay_threshold
        ])

    def activation_of_item_with_idx(self, idx) -> ActivationValue:
        initial_activation, time_activated = self._activation_records[idx]
        age = self.clock - time_activated
        return self._decay_function(age, initial_activation)

    def tick(self) -> Set[ItemActivatedEvent]:

        self.clock += 1

        self._decay_activations()

        self._grow_spheres()

        points_which_became_active = self._apply_activations()

        return set(ItemActivatedEvent(label=self.idx2label[point_idx], activation=self._activation_records[point_idx], time_activated=self.clock)
                   for point_idx in points_which_became_active)

    def _grow_spheres(self):
        """
        Radiates spheres.

        MUST be called before _apply_activations in a tick.
        """

        # TODO: This could be made more clever and efficient by working out which points will be reached when at the
        # TODO: time the sphere is created, and just looking that up here.

        for centre_idx, old_radius in self.spheres.items():
            new_radius = old_radius + self.expansion_rate

            if new_radius > self.max_radius:
                self.spheres.pop(centre_idx)
            else:
                self.spheres[centre_idx] = new_radius

            # get ready to activate points within sphere
            for reached_point_idx in self.points.points_between_spheres(centre_idx,
                                                                        outer_radius=new_radius,
                                                                        inner_radius=old_radius,
                                                                        distance_type=self.distance_type):
                # Pass on full activation when reached by scheduling it NOW
                self.schedule_activation_of_item_with_idx(
                    idx=reached_point_idx,
                    activation=self._activation_records[centre_idx].activation,
                    arrival_time=self.clock)

    def _decay_activations(self):

        # Since _activation_records will be computed according to need using the decay function, decaying them just involves
        # incrementing the clock.

        # However we need to remove concepts from the buffer if they have decayed too much
        for activated_point_idx, activation_record in self._activation_records.items():
            age = self.clock - activation_record.time_activated
            decayed_activation = self._decay_function(age, activation_record.activation)
            if decayed_activation < self.decay_threshold:
                self._activation_records.pop(activated_point_idx)

    def activate_item_with_idx(self, point_idx: int, incoming_activation: ActivationValue) -> bool:
        """
        Activate a point.
        :param point_idx:
        :param incoming_activation:
        :return:
            True if the item fired, else false
        """

        # Create sphere if not already activated
        if point_idx not in self._activation_records.keys():
            new_activation = incoming_activation
            self.spheres[point_idx] = 0
            did_activate = True

        # Otherwise accumulate incoming_activation, but reset clock
        else:
            current_activation = self._activation_records[point_idx].activation
            new_activation = current_activation + incoming_activation
            did_activate = False

        self._activation_records[point_idx] = ActivationRecord(new_activation, self.clock)

        return did_activate
