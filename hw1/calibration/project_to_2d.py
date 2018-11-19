import numpy as np
import cv2



def project_to_2d(coordinates_3d):
    """

    :param coordinates_3d: 3d coordinates of point. tuple (x,y,z)
    :return: 2d coordinates in 2d camera projection.
    """
    calibration_matrix = [[2560.5518418869233, 0.0, 1533.530354490828],
                          [0.0, 2560.2351813290616, 1206.2615781781851],
                          [0.0, 0.0, 1.0]]

    calibration_matrix = np.matrix(calibration_matrix)


    coordinates_3d = np.array(coordinates_3d)
    coordinates_3d.shape = (3,1)

    #multiply calibration matrix by 3d coordinate vector
    p_2d = calibration_matrix * coordinates_3d
    #divide resulting vector by z
    p_2d = p_2d / coordinates_3d[2]

    return p_2d

if __name__ == '__main__':
    coord_1 = (1,1,1)
    coord_2 = (10,1,1)

    print('2d projection of coordinate 1 %:\n',coord_1)
    coord_2d = project_to_2d(coord_1)
    print coord_2d

    print('2d projection of coordinate 2 %:\n', coord_2)
    coord_2d = project_to_2d(coord_2)
    print coord_2d
