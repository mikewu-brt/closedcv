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
from libs.LensDistortion import *
import matplotlib as matplot
matplot.use('TkAgg')
import matplotlib.pyplot as plt


####################
# Input Parameters
####################

parser = argparse.ArgumentParser(description="Stereo Calibrate 2")
parser.add_argument('--image_dir', default='Calibration_Aug23')
parser.add_argument('--cal_dir', default='Calibration_Aug23')
parser.add_argument('--start_idx', type=int, default=0)

# Open a figure to avoid cv2.imshow crash
plt.figure(1)
args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))


use_lens_shading = False

####################

image_helper = Image(args.image_dir)
cal_file_helper = Image(args.cal_dir)
display_size = image_helper.display_size(1024)
setupInfo = image_helper.setup_info()

if use_lens_shading is True:
    lens_distortion = []
    for cam_idx in range(image_helper.num_cam()):
        lens_distortion.append(LensDistortion(cam_idx, None, args.cal_dir))

orientation = args.start_idx
all_files_read = False
while not all_files_read:
    for cam_idx in range(image_helper.num_cam()):

        img_tmp, _ = image_helper.read_image_file(cam_idx, orientation, scale_to_8bit=False)
        if img_tmp is None:
            all_files_read = True
            break

        if use_lens_shading:
            img = lens_distortion[cam_idx].correct_vignetting(img_tmp, None,apply_flag=use_lens_shading,
                                                              alpha=0.7)
        else:
            img = img_tmp

        img2 = cv2.resize(img, None, fx=display_size, fy=display_size)
        cv2.imshow("{}".format(setupInfo.RigInfo.module_name[cam_idx]), img2)

    if not all_files_read:
        key = cv2.waitKey(0)
        if key == 27:
            all_files_read = True
    orientation += 1


