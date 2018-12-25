
"""UW strobe simulator for HW2"""

__author__      = "Elisei Shafer"

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
import cv2
from PIL import Image
from mpl_toolkits import mplot3d
import sys
sys.path.append('../../hw1/calibration')
from project_to_2d import project_to_2d

def depthmap_loader(path_to_depthmap):
    """
    Load depthmap that's made in OpenSfm, clean it up and interpolate
    :param path_to_depthmap:
    :return:
    """
    image_original_size = (3264,2248)

    depthmap = np.load(path_to_depthmap)
    plt.imshow(depthmap['depth'])
    plt.show()
    result, odm = depthmap_preprocessor(depthmap)

    im = Image.fromarray(result)
    im = im.resize(image_original_size)

    result = np.array(im)

    return odm, result

def depthmap_preprocessor(depthmap, dfilter='interpolation'):
    dm = depthmap['depth']  # np.where( depthmap['score'] > 0.97, depthmap['depth'], 0)

    #Median Blur LPF to remove speckle noise from the distance map.
    #Image is mapped to uint8 since medianBlur accepts only uint for window
    #size 7x7.
    max_value = dm.max()
    dm_norm = dm*255/max_value
    dm_norm = cv2.medianBlur(dm_norm.astype(np.uint8), 7)
    dm = dm_norm.astype(np.float32)*max_value/255

    #interpolate the image to remove all zeros.
    if dfilter=='interpolation':
        mask = ~(dm == 0)
        xx, yy = np.meshgrid(np.arange(dm.shape[1]), np.arange(dm.shape[0]))
        xym = np.vstack((np.ravel(xx[mask]), np.ravel(yy[mask]))).T
        data0 = np.ravel(dm[mask])
        interp0 = interpolate.NearestNDInterpolator(xym, data0)
        result0 = interp0(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)

    return result0, dm

def build_3d_map(depthmap):

    calibration_matrix = [[2614.6799607, 0, 1632.33532693],
                          [0, 2626.31303303, 1228.99718842],
                          [0, 0, 1]]
    image_size = (3264, 2248)
    world_coordinates = np.zeros([image_size[1], image_size[0], 3])
    c_x = 1632.33532693
    c_y = 1228.99718842
    f_x = 2614.6799607
    f_y = 2626.31303303

    it = np.nditer(depthmap, flags=['multi_index'])
    while not it.finished:
        u = it.multi_index[0]
        v = it.multi_index[1]
        z = depthmap[u][v]
        world_coordinates[u][v] = np.transpose(np.linalg.inv(calibration_matrix) * np.transpose(np.matrix([u,v,1]))* z)
        it.iternext()

    return world_coordinates

def get_scene_center_coordinates(map_3d):

    return map_3d[map_3d.shape[0]/2][map_3d.shape[1]/2]

# def get_strobe_rotation(scene_center_coordinates):
#
#     strobe_rotation =



if __name__ == '__main__':
    # path_to_depthmap = '../../hw1/sfm/depthmaps/20181125_105644.jpg.clean.npz'
    # (fdm, rdm)  = depthmap_loader(path_to_depthmap)
    image_size = (3264, 2248)
    #
    #
    # plt.imshow(fdm)
    # plt.show()
    # plt.imshow(rdm)
    # plt.show()

    # map_3d = build_3d_map(rdm)
    map_3d = np.load('map_3d_7mp.npy')
    dm2 = np.zeros(image_size)


    # for e, row in enumerate(map_3d):
    #     for f, i in enumerate(row):
    #         x,y = np.array(project_to_2d(i, distortion=False))
    #         dm2[f-1][e-1] = [x,y,map_3d[f-1][e-1][2]]