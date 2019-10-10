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

def compute_entrance_to_pupil_adjustment_val(mag_x=None, mag_y=None, obj_dist=None):
    if mag_x is None:
        mag_x = np.array([[0.007385166740417, 0.007243190574646, 0.007378269195557],
                          [0.005509874916077, 0.005497940254211, 0.005500683593750]])
        mag_y = np.array([[0.007386892986298, 0.007249995517731, 0.007380636310577],
                          [0.005514597797394, 0.005503431606293, 0.005507277774811] ])
        obj_dist = np.array([[3.373324500000000*1e3, 3.436324500000000*1e3, 3.373324500000000*1e3],
                             [4.529324500000000*1e3, 4.532324500000000*1e3, 4.531324499999999*1e3]])

    # take average of amgnifications in x and y dimensions
    mag = (mag_x + mag_y)/2

    factor = mag/(1-mag)

    distance_to_pupil_adj = (factor[1,:] * obj_dist[1,:] - factor[0,:] * obj_dist[0,:]) / (factor[0,:] - factor[1,:])
    #print(distance_to_pupil_adj)
    return(distance_to_pupil_adj);

def compute_magnification(args=None, delta_shift=0):

    if args is None:
        print("need to provide arguments")
        return;

    image_helper = Image(args.image_dir)
    cal_file_helper = Image(args.cal_dir)
    display_size = image_helper.display_size(1024)
    setupInfo = image_helper.setup_info()
    sensorInfo = setupInfo.SensorInfo

    center = np.array([sensorInfo.width / 2, sensorInfo.height / 2])

    # Open a figure to avoid cv2.imshow crash
    plt.figure(1)
    # Checkerboard info
    nx = setupInfo.ChartInfo.nx
    ny = setupInfo.ChartInfo.ny
    checker_size_mm = setupInfo.ChartInfo.size_mm
    pixel_size = sensorInfo.pixel_size_um / 1000
    obj_dist = setupInfo.CalibMagInfo.obj_dist + delta_shift

    num_data = 2
    mag_x = np.zeros([num_data, image_helper.num_cam()], np.float32)
    mag_y = np.zeros([num_data, image_helper.num_cam()], np.float32)
    fx = np.zeros([num_data, image_helper.num_cam()], np.float32)
    fy = np.zeros([num_data, image_helper.num_cam()], np.float32)
    avg_focal_length = np.zeros([num_data, image_helper.num_cam()], np.float32)
    max_focal_length = np.zeros([num_data, image_helper.num_cam()], np.float32)

    delta = setupInfo.CalibMagInfo.delta

    all_files_read = False
    while not all_files_read:
        for k in range(num_data):
            for cam_idx in range(image_helper.num_cam()):
                filename = setupInfo.CalibMagInfo.input_image_filename[k, cam_idx]
                img, gray = image_helper.read_image_file(cam_idx, 0, scale_to_8bit=True, file_name=filename)
                print("Searching")
                ret, corners = cv2.findChessboardCornersSB(img, (nx, ny), None)
                if ret:
                    print("Chessboard Found")
                    corners = corners[::-1]  # reverse the array
                    imagePoints = np.reshape(np.squeeze(corners), [ny, nx, -1]) - center
                    imagePoints_norm = np.linalg.norm(imagePoints, axis=2)
                    min_val = np.min(imagePoints_norm)
                    [y_loc, x_loc] = np.where(imagePoints_norm == min_val)
                    start_x = max(x_loc[0] - delta, 0)
                    start_y = max(y_loc[0] - delta, 0)
                    end_x = min(x_loc[0] + delta, nx - 1)
                    end_y = min(y_loc[0] + delta, ny - 1)
                    x_mean = np.mean(
                        imagePoints[start_y:end_y, start_x + 1:end_x, 0] - imagePoints[start_y: end_y, start_x: end_x - 1,
                                                                       0]);
                    y_mean = np.mean(
                        imagePoints[start_y + 1:end_y, start_x: end_x, 1] - imagePoints[start_y: end_y - 1, start_x: end_x,
                                                                        1]);
                    mag_x[k, cam_idx] = x_mean * pixel_size / checker_size_mm;
                    mag_y[k, cam_idx] = y_mean * pixel_size / checker_size_mm;
                    fx[k, cam_idx] = (mag_x[k, cam_idx] / (1 - mag_x[k, cam_idx])) * obj_dist[k, cam_idx];
                    fy[k, cam_idx] = (mag_y[k, cam_idx] / (1 - mag_y[k, cam_idx])) * obj_dist[k, cam_idx];
                    avg_focal_length[k, cam_idx] = (fx[k, cam_idx] / pixel_size + fy[k, cam_idx] / pixel_size) / 2
                    max_focal_length[k, cam_idx] = max(fx[k, cam_idx] / pixel_size, fy[k, cam_idx] / pixel_size)
                    #print("avg_focal_length:", cam_idx)
                    #print(avg_focal_length[k, cam_idx])
            else:
                print("Chessboard not found")

            if img is None:
                all_files_read = True
                break

            # save the average and maximum focal lengths
            cal_file_helper.save_np_file("computed_avg_focal_lengths", avg_focal_length)
            cal_file_helper.save_np_file("computed_max_focal_lengths", max_focal_length)
            img2 = cv2.resize(img, None, fx=display_size, fy=display_size)
            img2 = cv2.circle(img2, (img2.shape[1] >> 1, img2.shape[0] >> 1), 5, (255, 0, 0), 3)
            cv2.imshow("{}".format(setupInfo.RigInfo.module_name[cam_idx]), img2)
        all_files_read = True
    #print(mag_x)
    #print(mag_y)
    #print(obj_dist)
    print("avg_focal_length:")
    print(avg_focal_length)
    distance_to_pupil = compute_entrance_to_pupil_adjustment_val(mag_x, mag_y, obj_dist)
    return (distance_to_pupil);

####################
# Input Parameters
####################

parser = argparse.ArgumentParser(description="Compute Magnification")
parser.add_argument('--image_dir', default='Calibration_Aug23')
parser.add_argument('--cal_dir', default='Calibration_Aug23')

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

delta = compute_magnification(args, 0)
print("delta:")
print(delta)

#two methiods: (1) take the mean of the three values for the camera or (2) take the highest magnitude
#delta_new = compute_magnification(args, np.mean(delta))
delta_new = compute_magnification(args, np.max(np.abs(delta)) * np.sign(delta))

print("delta_new:")
print(delta_new)


####################
