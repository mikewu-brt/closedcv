#  Copyright (c) 2019, The LightCo
#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without
#  modification, are strictly prohibited without prior permission of
#  The LightCo.
#
#  @author  cstanski
#  @version V1.0.0
#  @date    Dec 2019
#  @brief
#

import cv2
import argparse
from libs.Image import *
from libs.CalibrationInfo import *
from libs.LensDistortion import *
import matplotlib as matplot
matplot.use('TkAgg')
import matplotlib.pyplot as plt

plt.figure(0).clear()

parser = argparse.ArgumentParser(description="Compute Distortion Map (ASIC)")
parser.add_argument('--cal_dir', default='calibration_nov21_25mm')
#parser.add_argument('--cal_dir', default='calibration_nov21_25mm_distortion_map_r1_16')
parser.add_argument('--decimate', type=int, default=16)

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

# Read image file
image_file = "../Depth Captures/Lenna.png"
img = cv2.imread(image_file)
cv2.imshow("Orig", img)
ny, nx, _ = img.shape

# Add distortion
lens_distortion = LensDistortion(0, args.cal_dir, args.cal_dir)
if lens_distortion.distortion_map() is not None:
    print("Using Distortion Map")
    lens_distortion.plot_distortion_map(0)
else:
    print("Creating distortion map from K and D")
    cal_info = CalibrationInfo(args.cal_dir)
    cv_dist_map, _ = cv2.initUndistortRectifyMap(cameraMatrix=cal_info.K(0), distCoeffs=cal_info.D(0), R=np.identity(3),
                                               newCameraMatrix=cal_info.K(0), size=(nx, ny), m1type=cv2.CV_32FC2)
    lens_distortion.set_opencv_distortion_map(cv_dist_map)

"""
Decimate the distortion map
"""
asic_dist_map = lens_distortion.asic_distortion_map(decimate=16)

image_helper = Image(args.cal_dir)
image_helper.save_text_file("asic_mapx", asic_dist_map[:, :, 0])
image_helper.save_text_file("asic_mapy", asic_dist_map[:, :, 1])
image_helper.save_np_file("asic_map", asic_dist_map)

# Extrapolate map back
lens_distortion2 = LensDistortion(0)
lens_distortion2.set_asic_distortion_map(asic_dist_map, (nx, ny))
dist_img = lens_distortion2.correct_distortion(img)


cv2.imshow("Distorted", dist_img)
cv2.imwrite("../Depth Captures/Lenna_distorted.png", dist_img)


