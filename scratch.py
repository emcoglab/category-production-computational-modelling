from numpy import array

from model.points_in_space import PointsInSpace
from sensorimotor_norms.sensorimotor_norms import SensorimotorNorms


def main():
    smn = SensorimotorNorms()
    words = ["ant", "bee"]

    sm_data = smn.matrix_for_words(words)

    pis = PointsInSpace(
        data_matrix=array(sm_data)
    )

    labelling_dictionary = {i: word for i, word in enumerate(words)}

    print(pis.point_with_idx(0))
    print(pis.point_with_idx(1))


if __name__ == '__main__':
    main()
