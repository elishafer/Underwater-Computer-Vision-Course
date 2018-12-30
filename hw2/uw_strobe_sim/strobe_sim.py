
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

def depthmap_loader(path_to_depthmap, image_resize=True, resize_size=(3264,2448)):
    """
    Load depthmap that's made in OpenSfm, clean it up and interpolate
    :param path_to_depthmap:
    :return:
    """

    depthmap = np.load(path_to_depthmap)
    plt.imshow(depthmap['depth'])
    plt.colorbar()
    plt.show()
    result, odm = depthmap_preprocessor(depthmap)

    if image_resize == True:
        im = Image.fromarray(result)
        im = im.resize(resize_size)
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

def build_3d_map(depthmap, image_size, calibration_matrix):

    world_coordinates = np.zeros([image_size[1], image_size[0], 3])

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

def load_image(path_to_img):

    img = Image.open(path_to_img)

    return img

def normalize_image_values(img, global_max=0.2):

    img = np.array(img,dtype='float64')
    img = img*global_max/255

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

def compute_backscatter(map_3d, strobe_t, i_s, c, beta_hg, g=0.85):
    scene_center = get_scene_center_coordinates(map_3d)

    vector_strobe_center = scene_center - strobe_t
    vector_strobe_point = map_3d[:, :] - strobe_t
    norm_strobe_center = np.linalg.norm(vector_strobe_center)
    norm_strobe_point = np.linalg.norm(vector_strobe_point, axis=2)
    norm_camera_point = np.linalg.norm(map_3d, axis=2)
    x_opn = map_3d / map_3d[:, :, 2,None]
    B = np.zeros(map_3d[:,:,0].shape)
    for i in xrange(map_3d.shape[0]):
        print("integrating on row %d", i)
        for j in xrange(map_3d.shape[1]):

            B[i][j], err = integrate.quad(B_point, 0.001, map_3d[i,j,2], args=(x_opn[i,j],vector_strobe_center,norm_strobe_center,
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
    (fdm, rdm)  = depthmap_loader(path_to_depthmap, image_resize=False)
    image_size = (640, 480)
    # image_size = (3264, 2448)

    plt.imshow(fdm)
    plt.colorbar()
    plt.show()
    plt.imshow(rdm)
    plt.colorbar()
    plt.show()

    # calibration_matrix = [[2614.6799607/5.1, 0, 1632.33532693/5.1],
    #                       [0, 2626.31303303/5.1, 1228.99718842/5.1],
    #                       [0, 0, 1]]
    #
    # map_3d = build_3d_map(rdm, image_size, calibration_matrix)
    # map_3d.astype('half')
    # np.savez_compressed('map_3d_1mp', map_3d=map_3d)
    map_3d = np.load('map_3d_1mp.npz')
    map_3d = map_3d['map_3d']

    # strobe_t = np.array([0.5, 0.5, 0])
    # map_3d = map_3d/10
    # scene_center = get_scene_center_coordinates(map_3d)
    # vector_strobe_center = scene_center - strobe_t
    # vector_strobe_point = map_3d[:, :] - strobe_t
    # theta = np.arccos((np.dot(vector_strobe_point, vector_strobe_center)) /
    #                   (np.linalg.norm(vector_strobe_point, axis=2)* np.linalg.norm(vector_strobe_center)))

    map_3d = map_3d/5
    # L_0 = normalize_image_values(Image.open('../../hw1/sfm/images/20181125_105644.jpg'))
    L_0 = normalize_image_values(Image.open('../20181125_105644_scaled.png'))
    # Strobe without backscatter
    c = np.array([0.228, 0.046, 0.019])
    beta_hg = np.array([1.22, 2.05, 3.06]) * 10 ** (-3)

    # strobe_t = 1 * np.array([-0.1, 0.1, 0])
    # I_d = compute_I_d(map_3d,strobe_t,L_0,c, I_s=12)
    # plt.imshow(I_d)
    # plt.show()

    # strobe_t = 5 * np.array([-0.1, 0.1, 0])
    # I_d = compute_I_d(map_3d, strobe_t, L_0, c, I_s=16)
    # plt.imshow(I_d)
    # plt.show()
    # #
    c = np.array([0.236, 0.068, 0.077])
    beta_hg = np.array([0.314, 0.395, 0.469])

    # strobe_t = 1 * np.array([-0.1, 0.1, 0])
    # I_d = compute_I_d(map_3d, strobe_t, L_0, c, I_s=12)
    # plt.imshow(I_d)
    # plt.show()

    strobe_t = 5 * np.array([-0.1, 0.1, 0])
    I_d = compute_I_d(map_3d, strobe_t, L_0, c, I_s=20)
    plt.imshow(I_d)
    plt.show()

    # bs = np.zeros(map_3d.shape)
    # for i in xrange(3):
    #     bs[:,:,i] = compute_backscatter(map_3d, strobe_t, 1, c[i],beta_hg[i])