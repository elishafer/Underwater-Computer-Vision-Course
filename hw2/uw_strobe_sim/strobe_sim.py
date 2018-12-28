
"""UW strobe simulator for HW2"""

__author__      = "Elisei Shafer"

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
import cv2
from PIL import Image
from mpl_toolkits import mplot3d
from scipy import integrate
import sys
sys.path.append('../../hw1/calibration')
from project_to_2d import project_to_2d

def depthmap_loader(path_to_depthmap):
    """
    Load depthmap that's made in OpenSfm, clean it up and interpolate
    :param path_to_depthmap:
    :return:
    """
    image_original_size = (3264,2448)

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
    image_size = (3264, 2448)
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

def load_image(path_to_img):

    img = Image.open(path_to_img)

    return img

def normalize_image_values(img, global_max=0.2):

    img = np.array(img,dtype='float64')
    img = img*0.2/255

    return img


def compute_I_d(map_3d, strobe_t, L_0, c, I_s=1 ):
    scene_center = get_scene_center_coordinates(map_3d)

    vector_strobe_center = scene_center - strobe_t
    vector_strobe_point = map_3d[:,:] - strobe_t
    norm_strobe_center = np.linalg.norm(vector_strobe_center)
    norm_strobe_point = np.linalg.norm(vector_strobe_point, axis=2)
    norm_camera_point = np.linalg.norm(map_3d,axis=2)

    theta = np.arccos(
                        (np.dot(vector_strobe_point, vector_strobe_center))/
                        (norm_strobe_point * norm_strobe_center)
    )
    Q = np.cos(0.8*theta)
    vector_normal = np.array([0,0,1])
    cos_phi = np.dot(vector_strobe_point, vector_normal)/ \
              (norm_strobe_point)

    I_d = (I_s * Q[:, :, None] * cos_phi[:, :, None] * L_0 * np.exp(
        -1 * c * (norm_strobe_point[:, :, None] + norm_camera_point[:, :, None]))) / \
          (norm_strobe_point[:, :, None] ** 2)

    return I_d

def compute_backscatter(map_3d, i_s, c, beta_hg, g=0.85):
    scene_center = get_scene_center_coordinates(map_3d)

    vector_strobe_center = scene_center - strobe_t
    vector_strobe_point = map_3d[:, :] - strobe_t
    norm_strobe_center = np.linalg.norm(vector_strobe_center)
    norm_strobe_point = np.linalg.norm(vector_strobe_point, axis=2)
    norm_camera_point = np.linalg.norm(map_3d, axis=2)
    x_opn = map_3d / map_3d[:, :, 2]
    B = integrate.quad(B_point, 0, map_3d[:,:,2], args=(x_opn,vector_strobe_center,norm_strobe_center,
                                                        i_s, c, beta_hg, g ))
    return B

def B_point(z, x_opn, vector_strobe_center, norm_strobe_center, i_s, c, beta_hg, g):
    vector_camera_point = x_opn*z
    vector_strobe_point = vector_camera_point - strobe_t
    norm_strobe_point = np.linalg.norm(vector_strobe_point)
    norm_camera_point = np.linalg.norm(vector_camera_point)

    theta = np.arccos(
        (np.dot(vector_strobe_point, vector_strobe_center)) /
        (norm_strobe_point * norm_strobe_center)
    )
    Q = np.cos(0.8 * theta)
    cos_psi = -1 * np.dot(vector_strobe_point, vector_camera_point)/(norm_strobe_point* norm_camera_point)
    beta = beta_hg * ((1 - g) ** 2) / (1 + g ** 2 - 2 * g * cos_psi)
    B_point = i_s * Q * np.exp(-1 * c * (norm_strobe_point + norm_camera_point))/(norm_strobe_point ** 2) * beta

    return B_point




if __name__ == '__main__':
    path_to_depthmap = '../../hw1/sfm/depthmaps/20181125_105644.jpg.clean.npz'
    (fdm, rdm)  = depthmap_loader(path_to_depthmap)
    image_size = (3264, 2448)
       #
    #
    plt.imshow(fdm)
    plt.show()
    plt.imshow(rdm)
    plt.show()

    # map_3d = build_3d_map(rdm)
    # np.save('map_3d_7mp', map_3d)
    map_3d = np.load('map_3d_7mp.npy')
    dm2 = np.zeros(image_size)

    # strobe_t = np.array([0.5, 0.5, 0])
    # map_3d = map_3d/10
    # scene_center = get_scene_center_coordinates(map_3d)
    # vector_strobe_center = scene_center - strobe_t
    # vector_strobe_point = map_3d[:, :] - strobe_t
    # theta = np.arccos((np.dot(vector_strobe_point, vector_strobe_center)) /
    #                   (np.linalg.norm(vector_strobe_point, axis=2)* np.linalg.norm(vector_strobe_center)))

    map_3d = map_3d/10
    L_0 = normalize_image_values(Image.open('../../hw1/sfm/images/20181125_105644.jpg'))
    # Strobe without backscatter
    c = np.array([0.228, 0.046, 0.019])
    strobe_t = 1 * np.array([-0.1, 0.1, 0])
    I_d = compute_I_d(map_3d,strobe_t,L_0,c, I_s=3)
    plt.imshow(I_d)
    plt.show()
    strobe_t = 5 * np.array([-0.1, 0.1, 0])
    I_d = compute_I_d(map_3d, strobe_t, L_0, c, I_s=4)
    plt.imshow(I_d)
    plt.show()

    c = np.array([0.236, 0.068, 0.077])
    strobe_t = 1 * np.array([-0.1, 0.1, 0])
    I_d = compute_I_d(map_3d, strobe_t, L_0, c, I_s=3)
    plt.imshow(I_d)
    plt.show()
    strobe_t = 5 * np.array([-0.1, 0.1, 0])
    I_d = compute_I_d(map_3d, strobe_t, L_0, c, I_s=5)
    plt.imshow(I_d)
    plt.show()