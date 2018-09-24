"""
===========================
Sensorimotor space.
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
from typing import Iterable

from pandas import DataFrame, read_csv

from ldm.core.utils.exceptions import WordNotFoundError
from model.points_in_space import PointsInSpace
from preferences import Preferences


class ColNames(object):
    """Column names used in sensorimotor data."""

    WORD = "word"

    # Sensory
    TOUCH = "haptic.mean"
    HEARING = "auditory.mean"
    SEEING = "visual.mean"
    SMELLING = "olfactory.mean"
    TASTING = "gustatory.mean"
    INTEROCEPTION = "interoception.mean"

    # Motor
    HEAD = "head.mean"
    MOUTH = "mouth.mean"
    HAND = "hand.mean"
    FOOT = "foot.mean"
    TORSO = "torso.mean"

    sensorimotor_cols = [
        TOUCH,
        HEARING,
        SEEING,
        SMELLING,
        TASTING,
        INTEROCEPTION,
        HEAD,
        MOUTH,
        HAND,
        FOOT,
        TORSO,
    ]


class SensorimotorNorms(object):

    def __init__(self):
        self.data: DataFrame = read_csv(Preferences.sensorimotor_norms_path, index_col=None, header=0,
                                        usecols=[ColNames.WORD] + ColNames.sensorimotor_cols)

        # TODO: verify which
        self.n_dims: int = self._sensorimotor_data.shape[1]

        # Trim whitespace and convert words to lower case
        self.data[ColNames.WORD] = self.data[ColNames.WORD].str.strip()
        self.data[ColNames.WORD] = self.data[ColNames.WORD].str.lower()

    @property
    def _sensorimotor_data(self):
        """A view on the data which just contains the sensorimotor values."""
        return self.data[ColNames.sensorimotor_cols]

    def iter_words(self) -> Iterable[str]:
        for word in self.data[ColNames.WORD]:
            yield word

    def has_word(self, word: str) -> bool:
        return word in self.iter_words()

    def vector_for_word(self, word):
        row = self.data[self.data[ColNames.WORD] == word]

        # Make sure we only got one row
        n_rows = row.shape[0]
        if n_rows is 0:
            # No rows: word wasn't found
            raise WordNotFoundError(word)
        elif n_rows > 1:
            # More than one row: word wasn't a unique row identifier
            # Something has gone wrong!
            raise Exception()

        return [
            row.iloc[0][col_name]
            for col_name in ColNames.sensorimotor_cols
        ]

    def as_points_in_space(self) -> PointsInSpace:
        return PointsInSpace(
            data_matrix=self._sensorimotor_data.values,
            labelling_dictionary={
                i: word
                for i, word in enumerate(self.iter_words())
            })
