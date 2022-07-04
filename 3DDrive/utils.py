import os
import sys
import struct
import yaml
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw


def show_pcd(file_no, data_type):
    """pcd 출력"""

    if data_type == 'lidar':
        base = os.path.join(os.getcwd(), 'input\lidar')
    else:
        base = os.path.join(os.getcwd(), 'input\radar')

    path = os.path.join(base, f'{file_no:06d}.pcd')
    pcd = o3d.io.read_point_cloud(path)
    o3d.visualization.draw_geometries(
        [pcd],
        zoom=0.3412,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[2.6172, 2.0475, 1.532],
        up=[-0.0694, -0.9768, 0.2024]
    )


def pcd2bin(read_path, save_name):
    """pcd -> array -> bin"""
    # 함수 수정 필요
    pcd = o3d.io.read_point_cloud(read_path)
    pcd_array = np.array(pcd.points, dtype=np.float32)
    result = pcd_array.tobytes()

    with open('test.bin') as f:
        f.write(result)


def show_velo2cam(
    img=r'C:\workspace\codes\calib\radar-lidar-camera-calibration\input\kitti-velo2cam\000007.png',
    binary=r'C:\workspace\codes\calib\radar-lidar-camera-calibration\input\kitti-velo2cam\000007.bin',
    cal=r'C:\workspace\codes\calib\radar-lidar-camera-calibration\input\kitti-velo2cam\000007.txt'
):
    """kitti-velo2cam"""
    # https://github.com/azureology/kitti-velo2cam

    with open(cal, 'r') as f:
        calib = f.readlines()

    # P2 (3 x 4) for left eye
    P2 = np.matrix([float(x)
                   for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3, 4)
    R0_rect = np.matrix(
        [float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3, 3)
    # Add a 1 in bottom-right, reshape to 4 x 4
    R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0], axis=0)
    R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0, 1], axis=1)
    Tr_velo_to_cam = np.matrix(
        [float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3, 4)
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0, 0, 0, 1], axis=0)

    # read raw data from binary
    scan = np.fromfile(binary, dtype=np.float32).reshape((-1, 4))
    points = scan[:, 0:3]  # lidar xyz (front, left, up)
    # TODO: use fov filter?
    velo = np.insert(points, 3, 1, axis=1).T
    velo = np.delete(velo, np.where(velo[0, :] < 0), axis=1)
    cam = P2 * R0_rect * Tr_velo_to_cam * velo
    cam = np.delete(cam, np.where(cam[2, :] < 0)[1], axis=1)
    # get u,v,z
    cam[:2] /= cam[2, :]
    # do projection staff
    plt.figure(figsize=(12, 5), dpi=96, tight_layout=True)
    png = mpimg.imread(img)
    IMG_H, IMG_W, _ = png.shape
    # restrict canvas in range
    plt.axis([0, IMG_W, IMG_H, 0])
    plt.imshow(png)
    # filter point out of canvas
    u, v, z = cam
    u_out = np.logical_or(u < 0, u > IMG_W)
    v_out = np.logical_or(v < 0, v > IMG_H)
    outlier = np.logical_or(u_out, v_out)
    cam = np.delete(cam, np.where(outlier), axis=1)
    # generate color map from depth
    u, v, z = cam
    plt.scatter([u], [v], c=[z], cmap='rainbow_r', alpha=0.5, s=2)
    # plt.title(name)
    # plt.savefig(f'./data_object_image_2/testing/projection/{name}.png',bbox_inches='tight')
    plt.show()


def quaternion_rotation_matrix(Q):
    # ./calib.txt의 Quaternion 값을 rotation matrix로 역산

    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix


def get_overlay_lidar(config_path, img_path, lidar_path):

    with open(config_path, 'r') as f:
        f.readline()
        config = yaml.load(f, Loader=yaml.FullLoader)
        lens = config['lens']
        fx = float(config['fx'])
        fy = float(config['fy'])
        cx = float(config['cx'])
        cy = float(config['cy'])
        k1 = float(config['k1'])
        k2 = float(config['k2'])
        p1 = float(config['p1/k3'])
        p2 = float(config['p2/k4'])
        Q = np.array(config['Q'], dtype=np.float32)
        tvec = np.array(config['tvec'], dtype=np.float32).reshape(3, 1)

    image = Image.open(img_path)
    image_compare = image.copy()
    background = image.copy()

    px = background.load()
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            px[i, j] = (255, 255, 255)

    pcd_load = o3d.io.read_point_cloud(lidar_path)
    pc_array = np.asarray(pcd_load.points)

    K = np.matrix([[fx, 0.0, cx],
                   [0.0, fy, cy],
                   [0.0, 0.0, 1.0]])
    D = np.array([k1, k2, p1, p2])  # config.yaml

    rmat = quaternion_rotation_matrix(Q)
    # rotatino matrix를 rotation vector로 역산
    rvec, jac = cv2.Rodrigues(rmat)
    # pcd 3차원 좌표를 2차원에 투사
    res_loc, _ = cv2.projectPoints(pc_array, rvec, tvec, K, D)

    draw_overlay = ImageDraw.Draw(image)
    draw_background = ImageDraw.Draw(background)
    radius = 2.5

    for pt in res_loc:
        xp = pt[0, 0]
        yp = pt[0, 1]
        draw_overlay.ellipse((xp - radius, yp - radius, xp + radius, yp + radius),
                             fill=(0, 0, 255))
        draw_background.ellipse((xp - radius, yp - radius, xp + radius, yp + radius),
                                fill=(0, 0, 255))

    return image_compare, image, background
