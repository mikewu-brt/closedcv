"""
 * Copyright (c) 2018, The LightCo
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are strictly prohibited without prior permission of
 * The LightCo.
 *
 * @author  Chuck Stanski
 * @version V1.0.0
 * @date    July 2019
 * @brief
 *  Stereo calibration test script
 *
"""
import cv2
import numpy as np
import os
import importlib
import argparse
import matplotlib as matplot
matplot.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import math


####################
# Input Parameters
####################

parser = argparse.ArgumentParser(description="Stereo Calibrate")
parser.add_argument('--image_dir', default='Calibration_Aug15_large_board')

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

####################

setupInfo = importlib.import_module("{}.setup".format(args.image_dir))

# check if env variable PATH_TO_IMAGE_DIR is set, if not use relative path
path_to_image_dir = os.getenv("PATH_TO_IMAGE_DIR")
if path_to_image_dir == None:
    path_to_image_dir = '.'

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

# Misc control
show_images = True
process_image_files = True
force_fx_eq_fy = False
estimate_distortion = True

display_size = 1/2

######################

num_cam = cam_position_m.shape[0]
if num_cam != setupInfo.RigInfo.image_filename.shape[0]:
    print("Number of camera positions and number of filenames must match")
    exit(1)


# Open a figure to avoid cv2.imshow crash
plt.figure(1)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nx*ny, 3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) * (checker_size_mm * 1.0e-3)

# Arrays to store object points and image points from all the images.
#objpoints = np.empty((1, nx*ny, 3))                 # 3D points in World space (relative)
chessboard_detect = []
objpoints = []                  # 3D points in World space (relative)
imgpoints = np.empty((num_cam, 1, nx*ny, 1, 2))     # Image points

corners2 = np.empty((num_cam, 1, nx*ny, 1, 2), dtype=np.float32)
all_files_read = False
orientation = 0
checkerboard_reversal = 0
if process_image_files:
    while True:
        chessboard_found = True
        # Load images
        for cam_idx in range(num_cam):

            fname = os.path.join(path_to_image_dir,args.image_dir, setupInfo.RigInfo.image_filename[cam_idx].format(orientation))
            print(fname)
            try:
                raw = np.load(fname)
            except:
                all_files_read = True
                break

            raw = raw.astype(np.float32) * 256.0 / 1024.0
            raw = raw.astype(np.uint8)
            gray = cv2.cvtColor(raw, cv2.COLOR_BayerBG2GRAY)
            img = cv2.cvtColor(raw, cv2.COLOR_BayerBG2BGR)

            print("Searching {}".format(fname))
            #ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            ret, corners = cv2.findChessboardCornersSB(gray, (nx, ny))
            if ret:
                print("Search Done")
                #corners2[cam_idx, 0, :, :, :] = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                corners2[cam_idx, 0, :, :, :] = corners
                if show_images:
                    img2 = cv2.drawChessboardCorners(img, (nx, ny), corners2[cam_idx, 0], ret)
                    img2 = cv2.resize(img2, None, fx=display_size, fy=display_size)
                    cv2.imshow("Image {}".format(cam_idx), img2)
                    cv2.waitKey(500)
            else:
                chessboard_found = False
                break

        if all_files_read:
            break
        elif chessboard_found:
            print("corners2 for ref cam = ", 0, corners2[ 0,0,0,0,:])
            print("corners2 for aux cam = ", 1, corners2[ 1,0,0,0,:])
            point0 = corners2[0,0,0,0,:]
            point1 = corners2[1,0,0,0,:]
            delta_y = math.fabs(point1[1]-point0[1])/min(point0[1], point1[1])
            delta_x = math.fabs(point1[0]-point0[0])/min(point0[0], point1[0])
            if (delta_y > 5.0) or (delta_x > 5.0):
                print("POSSIBLE CHECKERBOARD REVERSAL in x or  y")
                checkerboard_reversal += 1
            else:
                # Add points to arrays
                objpoints.append(objp)
                if orientation == 0:
                    imgpoints = corners2.copy()
                else:
                    imgpoints = np.concatenate((imgpoints, corners2.copy()), axis=1)
        else:
            print("Chessboard not found in {}".format(fname))
            if show_images:
                img2 = cv2.resize(img, None, fx=display_size, fy=display_size)
                cv2.imshow("Bad Image {}".format(cam_idx), img2)
                cv2.waitKey(500)
        chessboard_detect.append(chessboard_found)
        orientation += 1

    np.save(os.path.join(path_to_image_dir, args.image_dir, "objpoints"), objpoints)
    np.save(os.path.join(path_to_image_dir,args.image_dir, "imgpoints"), imgpoints)
    np.save(os.path.join(path_to_image_dir,args.image_dir, "chessboard_detect"), chessboard_detect)
else:
    objpoints = np.load(os.path.join(path_to_image_dir,args.image_dir, "objpoints.npy"))
    imgpoints = np.load(os.path.join(path_to_image_dir,args.image_dir, "imgpoints.npy"))
    chessboard_detect = np.load(os.path.join(path_to_image_dir,args.image_dir, "chessboard_detect.npy"))


print("")
print("Calibrating Cameras")
f = fl_mm * 1.0e-3 / (pixel_size_um * 1.0e-6)
K_guess = np.array([[f, 0, setupInfo.SensorInfo.width*0.5], [0, f, setupInfo.SensorInfo.height*0.5], [0, 0, 1]])

R_guess = np.identity(3)

i_flags = cv2.CALIB_USE_INTRINSIC_GUESS
i_flags |= cv2.CALIB_ZERO_TANGENT_DIST
i_flags |= cv2.CALIB_FIX_K3
if not estimate_distortion:
    i_flags |= cv2.CALIB_FIX_K1
    i_flags |= cv2.CALIB_FIX_K2
    i_flags |= cv2.CALIB_FIX_K4
    i_flags |= cv2.CALIB_FIX_K5
    i_flags |= cv2.CALIB_FIX_K6
if force_fx_eq_fy:
    i_flags |= cv2.CALIB_FIX_ASPECT_RATIO

e_flags = cv2.CALIB_FIX_INTRINSIC
e_flags |= cv2.CALIB_FIX_TANGENT_DIST
e_flags |= cv2.CALIB_FIX_K1
e_flags |= cv2.CALIB_FIX_K2
e_flags |= cv2.CALIB_FIX_K3
e_flags |= cv2.CALIB_FIX_K4
e_flags |= cv2.CALIB_FIX_K5
e_flags |= cv2.CALIB_FIX_K6
e_flags |= cv2.CALIB_USE_EXTRINSIC_GUESS

K = []
D = []
R = []
T = []
rvecs = []
tvecs = []
imgpts_ref = []
view_error = []
#img_size = (setupInfo.SensorInfo.width, setupInfo.SensorInfo.height)
img_size = (setupInfo.SensorInfo.height, setupInfo.SensorInfo.width)
for cam_idx in range(num_cam):
    print("Compute intrisics for cam {}".format(cam_idx))
    print("   Sensor Type: {}".format(setupInfo.SensorInfo.type))
    print("   Sensor Pixel Size (um): {}".format(pixel_size_um))
    print("   Focal Length (mm): {}".format(fl_mm))
    print("   Chart: {}".format(setupInfo.ChartInfo.name))
    imgpts = []
    if cam_idx == 0:
        for i in range(imgpoints.shape[1]):
            imgpts_ref.append(imgpoints[cam_idx, i, :, :, :].astype(np.float32))
            imgpts.append(imgpts_ref[i])
    else:
        for i in range(imgpoints.shape[1]):
            imgpts.append(imgpoints[cam_idx, i, :, :, :].astype(np.float32))

    ret, K1, D1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpts, img_size, K_guess.copy(), None, flags=i_flags)

    K.append(K1)
    D.append(D1)
    rvecs.append(rvecs1)
    tvecs.append(tvecs1)

    print("")
    print("")
    print("Camera {} - {}".format(cam_idx, setupInfo.SensorInfo.type))
    print("")
    print("Principle point: ({:.2f}, {:.2f}) - Difference from ideal: ({:.2f}, {:.2f})".format(
        K[cam_idx][0, 2], K[cam_idx][1, 2],
        setupInfo.SensorInfo.width * 0.5 - K[cam_idx][0, 2],
        setupInfo.SensorInfo.height * 0.5 - K[cam_idx][1, 2]))
    print("")
    print("Focal length (mm): ({:.2f}, {:.2f}) - Expected {}mm".format(
        K[cam_idx][0,0] * pixel_size_um * 1.0e-3, K[cam_idx][1,1] * pixel_size_um * 1.0e-3, fl_mm))

    print("")
    print("Camera Matrix:")
    print(K1)

    print("")
    print("Distortion Vector:")
    print(D1)

    if cam_idx != 0:
        T_guess = np.matmul(R_guess, -cam_position_m[cam_idx].reshape(3,1))
        ret, K1, D1, K2, D2, R1, T1, E, F, viewErr = cv2.stereoCalibrateExtended(objpoints, imgpts_ref, imgpts,
                                  K[0], D[0], K[cam_idx], D[cam_idx], img_size, R_guess.copy(), T_guess, flags=e_flags)
        R.append(R1)
        T.append(T1)
        view_error.append(viewErr)

        print("")
        print("Rotation Matrix:")
        print(R1)

        print("")
        print("Translation Matrix:")
        print(T1)

        c = np.matmul(-np.linalg.inv(R1), T1)
        print("")
        print("World coordinate of 2nd camera (m)")
        print(c)
    else:
        R.append(np.identity(3))
        T.append(np.zeros((3, 1)))


# Save results
np.save(os.path.join(path_to_image_dir,args.image_dir, "D"), D)
np.save(os.path.join(path_to_image_dir,args.image_dir, "K"), K)
np.save(os.path.join(path_to_image_dir,args.image_dir, "R"), R)
np.save(os.path.join(path_to_image_dir,args.image_dir, "T"), T)
np.save(os.path.join(path_to_image_dir,args.image_dir, "tvecs"), tvecs)
np.save(os.path.join(path_to_image_dir,args.image_dir, "rvecs"), rvecs)
