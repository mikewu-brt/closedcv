#  Copyright (c) 2019, The LightCo
#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without
#  modification, are strictly prohibited without prior permission of
#  The LightCo.
#
#  This module works on a single capture for the camera rig. It uses the known distance to the object and its image to
#  compute the focal length of all the cameras on the rig
#
#  @author  yhussain
#  @version V1.0.0
#  @date    Oct 2019
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


####################
# Input Parameters
####################

parser = argparse.ArgumentParser(description="Compute Magnification")
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
sensorInfo = setupInfo.SensorInfo

center = np.array([sensorInfo.width/2, sensorInfo.height/2])

# Open a figure to avoid cv2.imshow crash
plt.figure(1)
# Checkerboard info
nx = setupInfo.ChartInfo.nx
ny = setupInfo.ChartInfo.ny
checker_size_mm = setupInfo.ChartInfo.size_mm
pixel_size = sensorInfo.pixel_size_um/1000
obj_dist = setupInfo.CalibMagInfo.obj_dist

mag_x = np.zeros(image_helper.num_cam(), np.float32)
mag_y = np.zeros(image_helper.num_cam(), np.float32)
fx = np.zeros(image_helper.num_cam(), np.float32)
fy = np.zeros(image_helper.num_cam(), np.float32)
avg_focal_length = np.zeros(image_helper.num_cam(), np.float32)
max_focal_length = np.zeros(image_helper.num_cam(), np.float32)

delta = setupInfo.CalibMagInfo.delta

all_files_read = False
while not all_files_read:
    for cam_idx in range(image_helper.num_cam()):
        filename = setupInfo.CalibMagInfo.input_image_filename[cam_idx]
        img, gray = image_helper.read_image_file(cam_idx, 0, scale_to_8bit=True, file_name = filename)
        print("Searching")
        ret, corners = cv2.findChessboardCornersSB(img, (nx, ny), None)
        if ret:
            print("Chessboard Found")
            corners = corners[::-1] # reverse the array
            imagePoints = np.reshape(np.squeeze(corners), [ny,nx,-1]) - center
            imagePoints_norm = np.linalg.norm(imagePoints, axis=2)
            min_val = np.min(imagePoints_norm)
            [y_loc, x_loc] = np.where(imagePoints_norm == min_val)
            start_x = max(x_loc[0] - delta,0)
            start_y = max(y_loc[0] - delta,0)
            end_x = min(x_loc[0]+delta, nx-1)
            end_y = min(y_loc[0]+delta, ny-1)
            x_mean = np.mean(imagePoints[start_y:end_y, start_x + 1:end_x, 0] -imagePoints[start_y: end_y, start_x: end_x - 1, 0]);
            y_mean = np.mean(imagePoints[start_y + 1:end_y, start_x: end_x, 1]- imagePoints[ start_y: end_y - 1, start_x: end_x, 1]);
            mag_x[cam_idx] = x_mean * pixel_size / checker_size_mm;
            mag_y[cam_idx] = y_mean * pixel_size / checker_size_mm;
            fx[cam_idx] = (mag_x[cam_idx] / (1 - mag_x[cam_idx])) * obj_dist[cam_idx];
            fy[cam_idx] = (mag_y[cam_idx] / (1 - mag_y[cam_idx])) * obj_dist[cam_idx];
            avg_focal_length[cam_idx] = (fx[cam_idx] / pixel_size + fy[cam_idx] / pixel_size) / 2
            max_focal_length[cam_idx] = max(fx[cam_idx] / pixel_size, fy[cam_idx] / pixel_size)
            print("avg_focal_length:", cam_idx)
            print(avg_focal_length[cam_idx])
        else:
            print("Chessboard not found")

        if img is None:
            all_files_read = True
            break

        # save the average and maximum focal lengths
        cal_file_helper.save_np_file("computed_avg_focal_lengths", avg_focal_length)
        cal_file_helper.save_np_file("computed_max_focal_lengths",  max_focal_length)
        img2 = cv2.resize(img, None, fx=display_size, fy=display_size)
        cv2.imshow("{}".format(setupInfo.RigInfo.module_name[cam_idx]), img2)
        all_files_read = True



