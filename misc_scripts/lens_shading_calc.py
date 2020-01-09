#  Copyright (c) 2019, The LightCo
#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without
#  modification, are strictly prohibited without prior permission of
#  The LightCo.
#
#  @author  cstanski
#  @version V1.0.0
#  @date    Sep 2019
#  @brief
#  Stereo calibration test script

import cv2
import numpy as np
import os
import importlib
import argparse
from libs.Image import *
from scipy import ndimage
import matplotlib as matplot
matplot.use('TkAgg')
import matplotlib.pyplot as plt


# Open a figure to avoid cv2.imshow crash
plt.figure(1)
####################
# Input Parameters
####################

parser = argparse.ArgumentParser(description="Stereo Calibrate 2")
parser.add_argument('--image_dir', default='Calibration_Aug23')
parser.add_argument('--cal_dir', default='Calibration_Aug23')
parser.add_argument('--filter_size', type=int, default=65, help="Filters raw vignetting data using a 2D filter of filter_size by filter size")

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

common_gain_flag = True  # compute the lens shading gain using gray image if true (if false compute gain separately for each channel)
####################

image_helper = Image(args.image_dir)
cal_file_helper = Image(args.cal_dir)
display_size = image_helper.display_size(1024)
setupInfo = image_helper.setup_info()

# Initialize smoothing filter
k = 0
if args.filter_size > 0:
    k = np.ones((args.filter_size, args.filter_size)) / (args.filter_size * args.filter_size)

orientation = 0
all_files_read = False
while not all_files_read:
    for cam_idx in range(image_helper.num_cam()):
        img, gray = image_helper.read_image_file(cam_idx, orientation, scale_to_8bit=False)
        if img is None:
            all_files_read = True
            break

        if common_gain_flag is True:
            max_val = np.max(gray)
            gain = (1/gray) * max_val
            if args.filter_size > 0:
                gain = ndimage.convolve(gain, k, mode='nearest')
            img_new = np.empty(img.shape)
            img_new[:,:,0] = gain
            img_new[:,:,1] = gain
            img_new[:,:,2] = gain
        else:
            max_val_vec = np.max(img,axis=(0,1))
            img_new = (1/img) * max_val_vec
            _, _, c = img_new.shape
            if args.filter_size > 0:
                for i in range(c):
                    img_new[:, :, i] = ndimage.convolve(img_new[:, :, i], k, 'nearest')

        cal_file_helper.save_np_file("lens_shading_{}".format(setupInfo.RigInfo.module_name[cam_idx]), img_new)

        img2 = cv2.resize(img, None, fx=display_size, fy=display_size)
        cv2.imshow("{}".format(setupInfo.RigInfo.module_name[cam_idx]), img2)
        all_files_read = True

    if not all_files_read:
        key = cv2.waitKey(0)
        if key == 27:
            break
    orientation += 1


