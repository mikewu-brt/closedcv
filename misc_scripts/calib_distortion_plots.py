"""
 * Copyright (c) 2018, The LightCo
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are strictly prohibited without prior permission of
 * The LightCo.
 *
 * @author  Yunus Hussain
 * @version V1.0.0
 * @date    July 2019
 * @brief
 *  Stereo calibration test script
 *
 ******************************************************************************/
"""
import cv2
import numpy as np
import os
import importlib
import argparse
import math
import sys
from libs.Image import *
from libs.CalibrationInfo import *
import matplotlib as matplot
matplot.use('TkAgg')
import matplotlib.pyplot as plt

####################
# Input Parameters
####################

parser = argparse.ArgumentParser(description="Stereo Calibrate")
parser.add_argument('--image_dir', default='Oct2_cal')
parser.add_argument('--cal_dir', default='Oct2_cal')
parser.add_argument('--json_filename', default='', help='calibration json file to read intrinsics from')

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

####################


image_helper = Image(args.image_dir)
setupInfo = image_helper.setup_info()
display_size = image_helper.display_size(1024)

# Optical parameters
fl_mm = setupInfo.LensInfo.fl_mm
pixel_size_um = setupInfo.SensorInfo.pixel_size_um

# Relative camera positions in meters (Initial guess)
cam_position_m = setupInfo.RigInfo.cam_position_m

# Checkerboard info
nx = setupInfo.ChartInfo.nx
ny = setupInfo.ChartInfo.ny
checker_size_mm = setupInfo.ChartInfo.size_mm
num_cam = cam_position_m.shape[0]

# Checkerboard
objp = np.zeros((nx*ny, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * (checker_size_mm * 1.0e-3)

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


intrinsic_pts = []
rvecs = []
tvecs = []
chessboard_detect = []
view_error = []
for cam_idx in range(num_cam):
    intrinsic_pts.append(image_helper.load_np_file("intrinsic_pts{}.npy".format(cam_idx)))
    rvecs.append(image_helper.load_np_file("rvecs{}.npy".format(cam_idx)))
    tvecs.append(image_helper.load_np_file("tvecs{}.npy".format(cam_idx)))
    chessboard_detect.append(image_helper.load_np_file("chessboard_detect{}.npy".format(cam_idx)))
    view_error.append(image_helper.load_np_file("view_error{}.npy".format(cam_idx)))

imgshape = (setupInfo.SensorInfo.width, setupInfo.SensorInfo.width)
cal_info = CalibrationInfo(args.cal_dir, args.json_filename)
K = cal_info.K()
D = cal_info.D()

# Misc control

show_reproject_error = True
reverse_y_axis = False

######################

num_cam = image_helper.num_cam()

# Open a figure to avoid cv2.imshow crash
plt.figure(1)

corners2 = np.empty((num_cam, 1, nx*ny, 1, 2))

imgpointsproj = []
imgpointsproj_nodistcorrect = []
imgpts_ref = []
for cam_idx in range(num_cam):
    print("Compute intrisics for cam {}".format(cam_idx))
    K1 = K[cam_idx]
    D1 = D[cam_idx]
    rvecs1 = rvecs[cam_idx]
    tvecs1 = tvecs[cam_idx]

    imgpointsproj.append([])
    imgpointsproj_nodistcorrect.append([])

    mean_error = 0
    mean_error_nodist = 0
    for i in range(len(rvecs1)):
        imgpoints2, _ = cv2.projectPoints(objp, rvecs1[i], tvecs1[i], K1, D1)
        rvecs1_mat = cv2.Rodrigues(rvecs1[i])

        # project world points to image plane without any distortion correction
        D2 = np.zeros_like(D1)
        imgpoints2_nodistcorrect, _ = cv2.projectPoints(objp, rvecs1[i], tvecs1[i], K1, D2)

        imgpointsproj_nodistcorrect[cam_idx].append(imgpoints2_nodistcorrect)
        imgpointsproj[cam_idx].append(imgpoints2)

        error = cv2.norm(intrinsic_pts[cam_idx][i].astype(np.float32), imgpoints2, cv2.NORM_L2) / math.sqrt(len(imgpoints2))

        error_nodist = cv2.norm(intrinsic_pts[cam_idx][i].astype(np.float32), imgpoints2_nodistcorrect, cv2.NORM_L2) / math.sqrt(len(imgpoints2))

        mean_error += error
        mean_error_nodist += error_nodist
        print("error = ", error)
        print("error_nodist = ", error_nodist)

    print("total error: {}".format(mean_error / len(rvecs1)))
    print("total error nodist: {}".format(mean_error_nodist / len(rvecs1)))
    print("")

exit_app = False
for cam_idx in range(num_cam):
    if exit_app:
        break
    print("cam_idx = {}".format(cam_idx))
    print("i range = {}".format(len(rvecs[cam_idx])))
    for i in range(len(rvecs[cam_idx])):
        print("i = {}".format(i))
        print(" Showing Camera {} - {} of {}".format(cam_idx, i, len(rvecs[cam_idx])-1))

        delta_x = imgpointsproj[cam_idx][i][:, 0, 0] - intrinsic_pts[cam_idx][i][:, 0, 0]
        delta_y = imgpointsproj[cam_idx][i][:, 0, 1] - intrinsic_pts[cam_idx][i][:, 0, 1]
        error_x = cv2.norm(delta_x, cv2.NORM_L2) / math.sqrt(len(delta_x))
        error_y = cv2.norm(delta_y, cv2.NORM_L2) / math.sqrt(len(delta_y))
        print("error_x: {:.4f}".format(error_x))
        print("error_y: {:.4f}".format(error_y))

        delta_x_nodistcorrect = imgpointsproj_nodistcorrect[cam_idx][i][:, 0, 0] - intrinsic_pts[cam_idx][i][:, 0, 0]
        delta_y_nodistcorrect = imgpointsproj_nodistcorrect[cam_idx][i][:, 0, 1] - intrinsic_pts[cam_idx][i][:, 0, 1]
        error_x_nodistcorrect = cv2.norm(delta_x_nodistcorrect, cv2.NORM_L2) / math.sqrt(len(delta_x_nodistcorrect))
        error_y_nodistcorrect = cv2.norm(delta_y_nodistcorrect, cv2.NORM_L2) / math.sqrt(len(delta_y_nodistcorrect))
        print("error_x_nodist: {:.4f}".format(error_x_nodistcorrect))
        print("error_y_nodist: {:.4f}".format(error_y_nodistcorrect))

        print("View error (calibrate_camera): {:.4f}".format(view_error[cam_idx][i]))

        imgpoints_x = intrinsic_pts[cam_idx][i][:, 0, 0]
        imgpoints_y = intrinsic_pts[cam_idx][i][:, 0, 1]
        image_index = np.where(chessboard_detect[cam_idx])

        img2, gray = image_helper.read_image_file(cam_idx, image_index[0][i])
        img2 = cv2.resize(img2, None, fx=display_size, fy=display_size)

        # ##########undistort and check #############
        gray = gray.astype(np.float32)
        # undistort the image
        undistorted_img = cv2.undistort(gray, K[cam_idx], D[cam_idx], None, K[cam_idx])

        # find corners and refine
        undistorted_img = undistorted_img.astype(np.uint8)
        ret, corners = cv2.findChessboardCornersSB(undistorted_img, (nx, ny))
        if ret:
            print("Search Done")

            imgpoints_nodist = corners
            print("imgpoints_nodist shape =", imgpoints_nodist.shape)
            print("Image Index = ", image_index[0][i])
            img2 = cv2.resize(undistorted_img, None, fx=display_size, fy=display_size)
            # ##########undistort and check #############

            cv2.imshow("Image {}".format(cam_idx), img2)

            plotx = plt.figure(1)
            plotx.clear()
            plt.title("Distortion Corrected")
            plotx.gca().invert_yaxis()
            plt.xlabel("error")
            x = imgpoints_x
            y = imgpoints_y.copy()
            u = delta_x
            v = delta_y.copy()
            if reverse_y_axis:
                y = y[::-1]
                v = v[::-1]
            plt.quiver(x, y, u, v, angles='xy', units='xy', scale_units='inches', scale=2.0, pivot='tail')
            plotx.text(.7, .9, "error_x in pixels = {:.3f}\nerror_y in pixels = {:.3f}".format(error_x, error_y))
            plt.draw()

            plotx = plt.figure(2)
            plotx.clear()
            plt.title("No Distortion Correction")
            plt.gca().invert_yaxis()
            x = imgpoints_x
            y = imgpoints_y.copy()
            u = delta_x_nodistcorrect
            v = delta_y_nodistcorrect.copy()
            if reverse_y_axis:
                y = y[::-1]
                v = v[::-1]
            plt.quiver(x, y, u, v, angles='xy', units='xy', scale_units='inches', scale=2.0, pivot='tail')
            plotx.text(.7, .9, "error_x in pixels = {:.3f}\nerror_y in pixels = {:.3f}".format(
                error_x_nodistcorrect, error_y_nodistcorrect))
            plt.draw()

            plotx = plt.figure(3)
            plotx.clear()
            plt.gca().invert_yaxis()
            if reverse_y_axis:
                plt.scatter(imgpoints_x, imgpoints_y[::-1], color='k', s=1)
                plt.scatter(imgpoints_nodist[:, 0, 0], imgpoints_nodist[::-1, 0, 1], color='g', s=1)
            else:
                plt.scatter(imgpoints_x, imgpoints_y, color='k', s=1)
                plt.scatter(imgpoints_nodist[:, 0, 0], imgpoints_nodist[:, 0, 1], color='g', s=1)
            plt.draw()

            print("")

            plt.figure(4)
            plt.title("Click mouse here to advance, 'q' to exit")

            if plt.waitforbuttonpress(-1.0):
                print("Exiting")
                exit_app = True
                break

        else:
            print("Chessboard not detected.  Aborting!!!")
            break
