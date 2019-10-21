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

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

####################

image_helper = Image(args.image_dir)
cal_file_helper = Image(args.cal_dir)
display_size = image_helper.display_size(1024)
setupInfo = image_helper.setup_info()

orientation = 0
all_files_read = False
while not all_files_read:
    for cam_idx in range(image_helper.num_cam()):
        img, gray = image_helper.read_image_file(cam_idx, orientation, scale_to_8bit=False)
        if img is None:
            all_files_read = True
            break
        max_val = np.max(gray)
        img_new = (1/img) * max_val
        cal_file_helper.save_np_file("lens_shading_{}".format(cam_idx), img_new)

        img2 = cv2.resize(img, None, fx=display_size, fy=display_size)
        cv2.imshow("{}".format(setupInfo.RigInfo.module_name[cam_idx]), img2)
        all_files_read = True

    if not all_files_read:
        key = cv2.waitKey(0)
        if key == 27:
            break
    orientation += 1


