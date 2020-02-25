#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: jartichoker
Created: 2020-01-29


"""

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from libs.Image import Image


def create_corner_heatmap(cal_dir):
    image_helper = Image(cal_dir)
    setupInfo = image_helper.setup_info()
    num_cam = image_helper.num_cam()
    downsize = 4

    for cam_idx in range(num_cam):
        detected_arr = image_helper.load_np_file(f'intrinsic_pts{cam_idx}.npy')
        coverage_map = np.zeros((int(setupInfo.SensorInfo.height/downsize), int(setupInfo.SensorInfo.width/downsize)))

        for detected_im in detected_arr:
            for detected_pt in detected_im:
                coverage_map[int(detected_pt[0][1]/downsize), int(detected_pt[0][0]/downsize)] += 1

        plt.imsave(os.path.join(image_helper.directory(), f'{image_helper.get_cam_name(cam_idx)}_checkerboard_heatmap.png'),
                   coverage_map, cmap='hot')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Create Heatmap")
    parser.add_argument('--cal_dir')

    args, unknown = parser.parse_known_args()
    if unknown:
        print("Unknown options: {}".format(unknown))

    create_corner_heatmap(args.cal_dir)
