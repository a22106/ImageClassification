import os
import struct
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import sys
import yaml
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
import utils
import time


def save_calib_outputs(START, END):
    total_start = time.time()

    for view_no in range(START, END):
        loop_start = time.time()

        lidar_path = f'C:/workspace/pythonProject/radar-lidar-camera-calibration/input/lidar/{view_no:06d}.pcd'

        for i in range(1, 6):
            config_path = f'C:/workspace/pythonProject/radar-lidar-camera-calibration/input/calib/config0{i}.yaml'
            img_path = f'C:/workspace/pythonProject/radar-lidar-camera-calibration/input/camera/0{i}/{view_no:06d}.jpg'
            out_path = f'C:/workspace/pythonProject/radar-lidar-camera-calibration/output/0{i}/{view_no:06d}.jpg'
            origin, overlay, lidar = utils.get_overlay_lidar(
                config_path, img_path, lidar_path)

            fig = plt.figure(dpi=600)
            plt.subplot(1, 3, 1)
            plt.imshow(origin)
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(overlay)
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(lidar)
            plt.axis('off')

            plt.subplots_adjust(wspace=0.02)

            plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        print(f'{view_no:06d}.pcd / Loop time: {time.time() - loop_start:.5f} seconds')

    print(f'Total time: {time.time() - total_start:.5f} seconds')


if __name__ == '__main__':
    START, END = 212, 600
    save_calib_outputs(START, END)
