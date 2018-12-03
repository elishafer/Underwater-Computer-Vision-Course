import numpy as np
import cv2
import math



def project_to_2d(coordinates_3d, distortion=True):
    """

    :param coordinates_3d: 3d coordinates of point. tuple (x,y,z)
    :return: 2d coordinates in 2d camera projection.
    """
    coordinates_3d = np.array(coordinates_3d,dtype='float')

    calibration_matrix =   [[ 2614.6799607,     0         , 1632.33532693],
                            [    0        ,  2626.31303303,  1228.99718842],
                            [    0        ,     0         ,    1        ]]

    calibration_matrix = np.matrix(calibration_matrix)

    distortion_coefficients = [ 0.31995069, -1.5584381 ,  0.00068326,  0.00059707,  2.33263373]
    distortion_coefficients = np.array(distortion_coefficients)


    # distort the resulting image:
    k_1 = distortion_coefficients[0]
    k_2 = distortion_coefficients[1]
    k_3 = distortion_coefficients[4]
    p_1 = distortion_coefficients[2]
    p_2 = distortion_coefficients[3]

    x = coordinates_3d[0]/coordinates_3d[2]
    y = coordinates_3d[1]/coordinates_3d[2]
    if distortion == True:
        r = math.sqrt(x ** 2 + y ** 2)
        distortion_calculation = (1 + k_1 * r ** 2 + k_2 * r ** 4 + k_3 * r**6)
        x_distorted = x * distortion_calculation + 2 * p_1 * x * y + p_2 * (r ** 2 + 2 * x ** 2)
        y_distorted = y * distortion_calculation + p_1 * (r ** 2 + 2 * y ** 2) + 2 * p_2 * x * y
    else:
        x_distorted = x
        y_distorted = y

    projected_coordinates = np.matrix([x_distorted,y_distorted,1])
    projected_coordinates.shape = (3,1)

    #multiply calibration matrix by 3d coordinate vector
    p_2d = calibration_matrix * projected_coordinates

    return (p_2d[:2])

if __name__ == '__main__':
    coord_1 = (1,1,1)
    coord_2 = (10,1,1)
    coord_3 = (1,1,10)

    print('2d projection of coordinate 1 %:\n',coord_1)
    coord_2d = project_to_2d(coord_1)
    print coord_2d

    print('2d projection of coordinate 2 %:\n', coord_2)
    coord_2d = project_to_2d(coord_2)
    print coord_2d

    print('2d projection of coordinate 3 %:\n', coord_3)
    coord_2d = project_to_2d(coord_3)
    print coord_2d
