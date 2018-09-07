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
from typing import Dict, Set, Tuple

from ldm.core.utils.maths import DistanceType
from model.component import ModelComponent, ActivationValue, Label, ActivationRecord, ItemActivatedEvent
from model.points_in_space import PointsInSpace, Idx
from model.utils.math import decay_function_lognormal_median


class TemporalSpatialExpansion(ModelComponent):
    def __init__(self,
                 points_in_space: PointsInSpace,
                 item_labelling_dictionary: Dict[Idx, Label],
                 expansion_rate: float,
                 max_radius: float,
                 distance_type: DistanceType,
                 decay_median: float,
                 decay_shape: float,
                 decay_threshold: ActivationValue,
                 conscious_access_threshold: ActivationValue):

        super().__init__(item_labelling_dictionary=item_labelling_dictionary)

        self.points: PointsInSpace = points_in_space
        self.expansion_rate: float = expansion_rate
        self.max_radius: float = max_radius
        self.distance_type: DistanceType = distance_type

        # Use >= and < to test for above/below
        self.decay_threshold: ActivationValue = decay_threshold
        self.conscious_access_threshold: ActivationValue = conscious_access_threshold

        self._decay_function = decay_function_lognormal_median(decay_median, decay_shape)

        # Dictionary point_idx -> activation
        self.activations: Dict[int, ActivationRecord] = dict()

        # Dictionary point_idx -> radius
        self.spheres = dict()

    def tick(self) -> Set[ItemActivatedEvent]:

        self.clock += 1

        self._decay_activations()

        points_which_became_consciously_active = self._grow_spheres()

        return set(ItemActivatedEvent(self.idx2label[point_idx], self.activations[point_idx], self.clock)
                   for point_idx in points_which_became_consciously_active)

    def _grow_spheres(self) -> Set[int]:
        """
        Radiates spheres
        :return:
            Set of point indices which became consciously active.
        """

        # TODO: This could be made more clever and efficient by working out which points will be reached when at the
        # TODO: time the sphere is created, and just looking that up here.

        points_which_activated: Set[int] = set()
        points_which_crossed_c_a_t: Set[int] = set()

        for centre_idx, old_radius in self.spheres.items():
            new_radius = old_radius + self.expansion_rate

            if new_radius > self.max_radius:
                self.spheres.pop(centre_idx)
            else:
                self.spheres[centre_idx] = new_radius

            # activate points within sphere
            for reached_point_idx in self.points.points_between_spheres(centre_idx,
                                                                        outer_radius=new_radius,
                                                                        inner_radius=old_radius,
                                                                        distance_type=self.distance_type):
                # Pass on full activation when reached
                did_activate, did_cross_c_a_t = self.activate_item_with_idx(reached_point_idx, self.activations[centre_idx].activation)
                if did_activate:
                    points_which_activated.add(reached_point_idx)
                if did_cross_c_a_t:
                    points_which_crossed_c_a_t.add(reached_point_idx)

        return points_which_crossed_c_a_t

    def _decay_activations(self):

        # Since activations will be computed according to need using the decay function, decaying them just involves
        # incrementing the clock.

        # However we need to remove concepts from the buffer if they have decayed too much
        for activated_point_idx, activation_record in self.activations.items():
            age = self.clock - activation_record.time_activated
            decayed_activation = self._decay_function(age, activation_record.activation)
            if decayed_activation < self.decay_threshold:
                self.activations.pop(activated_point_idx)

    def activate_item_with_idx(self, point_idx: int, incoming_activation: ActivationValue) -> Tuple[bool, bool]:
        """
        Activate a point.
        :param point_idx:
        :param incoming_activation:
        :return:
            Tuple of bools:
            (
                Point became newly activated (True) or just absorbed and accumulated (False),
                Point crossed conscious access threshold (True) or not (False)
            )
        """

        # Create sphere if not already activated
        if point_idx not in self.activations.keys():
            new_activation = incoming_activation
            self.spheres[point_idx] = 0
            did_activate = True
            # Since the point started at zero activation, we crossed the c_a_t iff the new activation is greater than it
            did_cross_conscious_access_threshold = incoming_activation > self.conscious_access_threshold

        # Otherwise accumulate incoming_activation, but reset clock
        else:
            current_activation = self.activations[point_idx].activation
            currently_below_c_a_t = current_activation > self.conscious_access_threshold
            new_activation = current_activation + incoming_activation
            did_activate = False
            did_cross_conscious_access_threshold = currently_below_c_a_t and (new_activation > self.conscious_access_threshold)

        self.activations[point_idx] = ActivationRecord(new_activation, self.clock)

        return did_activate, did_cross_conscious_access_threshold
