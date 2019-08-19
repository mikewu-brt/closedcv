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
import matplotlib as matplot
matplot.use('TkAgg')
import importlib
import argparse
import matplotlib.pyplot as plt
import math

####################
# Input Parameters
####################

parser = argparse.ArgumentParser(description="Stereo Calibrate")
parser.add_argument('--image_dir', default='cal_dfk33ux265_08072019')

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

sys.path.append(args.image_dir)
####################
max_pixel_value = 1024.0
display_size = 1/2

setupInfo = importlib.import_module("setup")
#import setup as setupInfo
#setupInfo = importlib.import_module("{}setup.py".format(args.image_dir))

# Optical parameters
fl_mm = setupInfo.LensInfo.fl_mm
pixel_size_um = setupInfo.SensorInfo.pixel_size_um

# Relative camera positions in meters (Initial guess)
cam_position_m = setupInfo.RigInfo.cam_position_m

# Checkerboard info
nx = setupInfo.ChartInfo.nx
ny = setupInfo.ChartInfo.ny
checker_size_mm = setupInfo.ChartInfo.size_mm

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

image_dir = args.image_dir
objpoints = np.load(os.path.join(image_dir, "objpoints.npy"))
imgpoints = np.load(os.path.join(image_dir, "imgpoints.npy"))
imgshape = tuple(np.load(os.path.join(image_dir, "img_shape.npy")))
tvecs = np.load(os.path.join(image_dir, "tvecs.npy"))
rvecs = np.load(os.path.join(image_dir, "rvecs.npy"))
K = np.load(os.path.join(image_dir, "K.npy"))
D = np.load(os.path.join(image_dir, "D.npy"))
chessboard_detect = np.load(os.path.join(image_dir, "chessboard_detect.npy"))

# set up the image index array by skipping over images where chessboard pattern was not found
image_index = np.where(chessboard_detect == True)


# Misc control

show_reproject_error = True
reverse_y_axis = False

######################

num_cam = cam_position_m.shape[0]
if num_cam != setupInfo.RigInfo.image_filename.shape[0]:
    print("Number of camera positions and number of filenames must match")
    exit(1)


# Open a figure to avoid cv2.imshow crash
plt.figure(1)

corners2 = np.empty((num_cam, 1, nx*ny, 1, 2))

imgpointsproj = np.empty_like(imgpoints)
imgpointsproj_nodistcorrect = np.empty_like(imgpoints)
imgpts_ref = []
for cam_idx in range(num_cam):
    print("Compute intrisics for cam {}".format(cam_idx))
    #ret, K1, D1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpts, imgshape, K_guess.copy(), None, flags=i_flags)
    K1 = K[cam_idx]
    D1 = D[cam_idx]
    rvecs1 = rvecs[cam_idx]
    tvecs1 = tvecs[cam_idx]

    mean_error = 0
    mean_error_nodist = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs1[i], tvecs1[i], K1, D1)
        rvecs1_mat = cv2.Rodrigues(rvecs1[i])

        # project world points to image plane without any didtortion correction
        D2 = np.zeros_like(D1)
        imgpoints2_nodistcorrect, _ = cv2.projectPoints(objpoints[i], rvecs1[i], tvecs1[i], K1, D2)

        imgpointsproj_nodistcorrect[cam_idx,i] = imgpoints2_nodistcorrect
        imgpointsproj[cam_idx,i] = imgpoints2

        error = cv2.norm(imgpoints[cam_idx][i].astype(np.float32), imgpoints2, cv2.NORM_L2) / math.sqrt(len(imgpoints2))

        error_nodist = cv2.norm(imgpoints[cam_idx][i].astype(np.float32), imgpoints2_nodistcorrect, cv2.NORM_L2) / math.sqrt(len(imgpoints2))

        mean_error += error
        mean_error_nodist += error_nodist
        print("error = ", error)
        print("error_nodist = ", error_nodist)


    print("total error: {}".format(mean_error / len(objpoints)))
    print("total error nodist: {}".format(mean_error_nodist / len(objpoints)))


matplot.interactive(True)
offset = len(objpoints)
print("lenobjpoints=", len(objpoints))
print("shape_imgpointsro", imgpoints.shape)
for i in range(len(objpoints)):
    for cam_idx in range(num_cam):
        #cam_idx = 0
        print(" Showing plots #", i, i + len(objpoints))
        delta_x = imgpointsproj[cam_idx,i,:,0,0] - imgpoints[cam_idx,i,:,0,0]
        delta_y = imgpointsproj[cam_idx,i,:,0,1] - imgpoints[cam_idx,i,:,0,1]
        error_x = cv2.norm(delta_x, cv2.NORM_L2)/math.sqrt(len(delta_x))
        error_y = cv2.norm(delta_y, cv2.NORM_L2)/math.sqrt(len(delta_y))
        print("error_x =", error_x)
        print("error_y =", error_y)
        delta_x_nodistcorrect = imgpointsproj_nodistcorrect[cam_idx,i,:,0,0] - imgpoints[cam_idx,i,:,0,0]
        delta_y_nodistcorrect = imgpointsproj_nodistcorrect[cam_idx,i,:,0,1] - imgpoints[cam_idx,i,:,0,1]
        error_x_nodistcorrect = cv2.norm(delta_x_nodistcorrect, cv2.NORM_L2)/math.sqrt(len(delta_x_nodistcorrect))
        error_y_nodistcorrect = cv2.norm(delta_y_nodistcorrect, cv2.NORM_L2)/math.sqrt(len(delta_y_nodistcorrect))
        print("error_x_nodist in pixels =", error_x_nodistcorrect)
        print("error_y_nodist in pixels =", error_y_nodistcorrect)
        imgpoints_x = imgpoints[cam_idx, i,:,0,0]
        imgpoints_y = imgpoints[cam_idx,i,:,0,1]
        fname = os.path.join(image_dir, setupInfo.RigInfo.image_filename[cam_idx].format(image_index[0][i]))
        raw = np.load(fname)
        raw = raw.astype(np.float32) * 256.0 / max_pixel_value
        raw = raw.astype(np.uint8)
        print(fname)
        img2 = cv2.cvtColor(raw, cv2.COLOR_BayerBG2BGR)
        img2 = cv2.resize(img2, None, fx=display_size, fy=display_size)

        ###########undistort and check #############
        gray = cv2.cvtColor(raw, cv2.COLOR_BayerBG2GRAY)
        gray = gray.astype(np.float32)
        #undistort the image
        undistorted_img = cv2.undistort(gray, K[cam_idx], D[cam_idx], None, K[cam_idx])

        # find corners and refine
        undistorted_img = undistorted_img.astype(np.uint8)
        ret, corners = cv2.findChessboardCorners(undistorted_img, (nx, ny), None)
        if ret:
            print("Search Done")
            corners2[cam_idx, 0, :, :, :] = cv2.cornerSubPix(undistorted_img, corners, (11, 11), (-1, -1), criteria)
        else:
            print("failed to find chessboard patterns, Aborting!!!!")
            break
        imgpoints_nodist = corners2[cam_idx,0,:,:,:]
        print("imgpoints_nodist shape =", imgpoints_nodist.shape)
        print("Image Index = ", image_index[0][i])
        img2 = cv2.resize(undistorted_img, None, fx=display_size, fy=display_size)
        ###########undistort and check #############

        cv2.imshow("Image {}".format(cam_idx), img2)
        #cv2.waitKey()

        plotx = plt.figure(cam_idx*3 + 1)
        plotx.gca().invert_yaxis()
        plt.ion()
        #plt.title("interactive test")
        plt.xlabel("error")
        if reverse_y_axis == True:
            plt.quiver(imgpoints_x, imgpoints_y[::-1],delta_x,delta_y[::-1], angles='xy', units='xy', scale_units='inches', scale=2.0)
        else:
            plt.quiver(imgpoints_x, imgpoints_y,delta_x,delta_y, angles='xy',units='xy', scale_units='inches', scale = 2.0 )
        plotx.text(.5,.9,'error_x in pixels = %s\nerror_y in pixels =%s' %(error_x, error_y))

        plt.draw()

        plotx = plt.figure(cam_idx*3 + 2)
        plt.gca().invert_yaxis()
        if reverse_y_axis == True:
            plt.quiver(imgpoints_x, imgpoints_y[::-1], delta_x_nodistcorrect,delta_y_nodistcorrect[::-1], angles='xy', units='xy', scale_units='inches', scale = 2.0) # reverse y co-ord
        else:
            plt.quiver(imgpoints_x, imgpoints_y, delta_x_nodistcorrect,delta_y_nodistcorrect, angles='xy', units='xy', scale_units='inches', scale=2.0)
        plotx.text(.5,.9,'error_x in pixels = %s\nerror_y in pixels =%s' %(error_x_nodistcorrect, error_y_nodistcorrect))

        plotx = plt.figure(cam_idx*3 + 3)
        plt.gca().invert_yaxis()
        if reverse_y_axis == True:
            plt.scatter(imgpoints_x, imgpoints_y[::-1], color='k')
            plt.scatter(imgpoints_nodist[:,0,0], imgpoints_nodist[::-1,0,1], color = 'g')
        else:
            plt.scatter(imgpoints_x, imgpoints_y, color='k')
            plt.scatter(imgpoints_nodist[:,0,0], imgpoints_nodist[:,0,1], color = 'g')

        #plt.draw_all
    plt.show(block=True)

        #cv2.waitKey()

