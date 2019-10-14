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
from libs.CalibrationInfo import *
import matplotlib as matplot
matplot.use('TkAgg')
import matplotlib.pyplot as plt


####################
# Input Parameters
####################

parser = argparse.ArgumentParser(description="Stereo Calibrate 2")
parser.add_argument('--image_dir', default='Oct2_cal')
parser.add_argument('--cal_dir', default='Oct2_cal')
parser.add_argument('--fix_focal_length', action="store_true", default=False, help='use this option if the focal length is to be determined explicitly and then set to that value')

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

####################

image_helper = Image(args.image_dir)
cal_file_helper = Image(args.cal_dir)
setupInfo = image_helper.setup_info()
img_size = (setupInfo.SensorInfo.width, setupInfo.SensorInfo.height)

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
use_rgb_image = True
force_fx_eq_fy = True
estimate_distortion = True
use_2step_findchessboard = False

display_size = image_helper.display_size(1024)

######################

num_cam = image_helper.num_cam()

# Open a figure to avoid cv2.imshow crash
plt.figure(1)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nx*ny, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * (checker_size_mm * 1.0e-3)

# Arrays to store object points and image points from all the images.
objpoints = []                  # 3D points in World space (relative)
intrinsic_pts = []
chessboard_detect = []
for cam_idx in range(num_cam):
    intrinsic_pts.append([])
    chessboard_detect.append([])

corners2 = np.empty((num_cam, 1, nx*ny, 1, 2), dtype=np.float32)
all_files_read = False
orientation = 0
num_files_missing = 0
if process_image_files:
    while not all_files_read:
        chessboard_found = True
        # Load images
        for cam_idx in range(num_cam):
            img, gray = image_helper.read_image_file(cam_idx, orientation)
            if img is None:
                all_files_read = True
                break

            print("Searching")
            if use_2step_findchessboard:
                if use_rgb_image:
                    ret, corners = cv2.findChessboardCorners(img, (nx, ny), None)
                else:
                    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            else:
                if use_rgb_image:
                    ret, corners = cv2.findChessboardCornersSB(img, (nx, ny), None)
                else:
                    ret, corners = cv2.findChessboardCornersSB(gray, (nx, ny), None)

            if ret:
                print("Chessboard Found")
                chessboard_detect[cam_idx].append(True)
                if use_2step_findchessboard:
                    corners2[cam_idx, 0, :, :, :] = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                else:
                    corners2[cam_idx, 0, :, :, :] = corners[::-1]
                intrinsic_pts[cam_idx].append(corners2[cam_idx, 0].copy())
                if show_images:
                    img2 = cv2.drawChessboardCorners(img, (nx, ny), corners2[cam_idx, 0], True)
                    img2 = cv2.resize(img2, None, fx=display_size, fy=display_size)
                    cv2.imshow("{}".format(setupInfo.RigInfo.module_name[cam_idx]), img2)
                    cv2.waitKey(500)
            else:
                print("Chessboard not found")
                chessboard_found = False
                chessboard_detect[cam_idx].append(False)
                if show_images:
                    img2 = cv2.resize(img, None, fx=display_size, fy=display_size)
                    cv2.imshow("Bad Image {}".format(setupInfo.RigInfo.module_name[cam_idx]), img2)
                    cv2.waitKey(500)
            print("")

        if all_files_read:
            pass

        orientation += 1

    cal_file_helper.save_np_file("objpoints", objpoints)
    for cam_idx in range(num_cam):
        cal_file_helper.save_np_file("intrinsic_pts{}".format(cam_idx), intrinsic_pts[cam_idx])
        cal_file_helper.save_np_file("chessboard_detect{}".format(cam_idx), chessboard_detect[cam_idx])

else:
    objpoints = cal_file_helper.load_np_file("objpoints.npy")
    intrinsic_pts = []
    rvecs = []
    tvecs = []
    chessboard_detect = []
    for cam_idx in range(num_cam):
        intrinsic_pts.append(cal_file_helper.load_np_file("intrinsic_pts{}.npy".format(cam_idx)))
        rvecs.append(cal_file_helper.load_np_file("rvecs{}.npy".format(cam_idx)))
        tvecs.append(cal_file_helper.load_np_file("tvecs{}.npy".format(cam_idx)))
        chessboard_detect.append(cal_file_helper.load_np_file("chessboard_detect{}.npy".format(cam_idx)))


print("")
print("Calibrating Cameras")
f = fl_mm * 1.0e-3 / (pixel_size_um * 1.0e-6)
#K_guess = np.array([[f, 0, setupInfo.SensorInfo.width*0.5], [0, f, setupInfo.SensorInfo.height*0.5], [0, 0, 1]])

R_guess = np.identity(3)

i_flags = cv2.CALIB_USE_INTRINSIC_GUESS
if args.fix_focal_length is True:
    i_flags |= cv2.CALIB_FIX_FOCAL_LENGTH
i_flags |= cv2.CALIB_ZERO_TANGENT_DIST
i_flags |= cv2.CALIB_FIX_K3
# i_flags |= cv2.CALIB_RATIONAL_MODEL
# i_flags |= cv2.CALIB_THIN_PRISM_MODEL

if not estimate_distortion:
    i_flags |= cv2.CALIB_ZERO_TANGENT_DIST
    i_flags |= cv2.CALIB_FIX_K1
    i_flags |= cv2.CALIB_FIX_K2
    i_flags |= cv2.CALIB_FIX_K3
    i_flags |= cv2.CALIB_FIX_K4
    i_flags |= cv2.CALIB_FIX_K5
    i_flags |= cv2.CALIB_FIX_K6
    i_flags |= cv2.CALIB_FIX_S1_S2_S3_S4

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
e_flags |= cv2.CALIB_FIX_S1_S2_S3_S4
e_flags |= cv2.CALIB_USE_EXTRINSIC_GUESS

print("")
print("i_flags = 0x{}".format(hex(i_flags)))
print("e_flags = 0x{}".format(hex(e_flags)))

K = []
D = []
R = []
T = []
view_error = []
stereo_view_error = []
rvecs = []
tvecs = []
for cam_idx in range(num_cam):
    print("")
    print("Compute intrinsics for cam {}".format(setupInfo.RigInfo.module_name[cam_idx]))
    print("   Manufacturer: {}".format(setupInfo.RigInfo.camera_module_manufacturer))
    print("   Module Model: {}".format(setupInfo.RigInfo.camera_module_model))
    print("   Sensor Type: {}".format(setupInfo.SensorInfo.type))
    print("   Sensor Pixel Size (um): {}".format(pixel_size_um))
    print("   Lens Type: {}".format(setupInfo.LensInfo.type))
    print("   Focal Length (mm): {}".format(fl_mm))
    print("   Chart: {}, ({}, {}) x {} mm".format(setupInfo.ChartInfo.name, setupInfo.ChartInfo.nx, setupInfo.ChartInfo.ny, setupInfo.ChartInfo.size_mm))

    if args.fix_focal_length is True:
        K_guess = np.array([[setupInfo.CalibMagInfo.fixed_focal_length[cam_idx], 0, setupInfo.SensorInfo.width*0.5],
                            [0, setupInfo.CalibMagInfo.fixed_focal_length[cam_idx], setupInfo.SensorInfo.height*0.5],
                            [0, 0, 1]])
    else:
        K_guess = np.array([[f, 0, setupInfo.SensorInfo.width*0.5],
                            [0, f, setupInfo.SensorInfo.height*0.5],
                            [0, 0, 1]])
    # Intrinsic data
    obj_pts = []
    img_pts = []
    for capture_idx in range(len(intrinsic_pts[cam_idx])):
        obj_pts.append(objp)
        img_pts.append(intrinsic_pts[cam_idx][capture_idx])

    reproj_error, K1, D1, rvecs1, tvecs1, I, E, viewErr = cv2.calibrateCameraExtended(obj_pts, img_pts, img_size, K_guess.copy(), None, flags=i_flags)
    K.append(K1)
    D.append(D1)
    view_error.append(viewErr)
    rvecs.append(rvecs1)
    tvecs.append(tvecs1)

    print("")
    print("")
    print("Camera {} - {}".format(setupInfo.RigInfo.module_name[cam_idx], setupInfo.SensorInfo.type))
    print("")
    print("Num intrinsic poses: {}".format(len(obj_pts)))
    print("Principle point: ({:.2f}, {:.2f}) - Difference from ideal: ({:.2f}, {:.2f})".format(
        K[cam_idx][0, 2], K[cam_idx][1, 2],
        setupInfo.SensorInfo.width / 2 - K[cam_idx][0, 2],
        setupInfo.SensorInfo.height / 2 - K[cam_idx][1, 2]))
    print("")
    print("Focal length (mm): ({:.2f}, {:.2f}) - Expected {}mm".format(
        K[cam_idx][0, 0] * pixel_size_um * 1.0e-3, K[cam_idx][1, 1] * pixel_size_um * 1.0e-3, fl_mm))

    print("")
    print("Camera Matrix round 1:")
    print(K1)

    print("")
    print("Distortion Vector round 1:")
    print(D1)

    print("")
    print("Reprojection Error {}".format(reproj_error))

    if cam_idx != 0:
        # Extrinsic data - pair up  ref cam with all the other cams independently
        imgpts = []
        imgpts_ref = []
        objpoints = []  # 3D points in World space (relative)
        count_ref = 0
        count_aux = 0
        if cam_idx != 0:
            for i in range(len(chessboard_detect[0])):
                if chessboard_detect[0][i] and chessboard_detect[cam_idx][i]:
                    imgpts_ref.append(intrinsic_pts[0][count_ref])
                    imgpts.append(intrinsic_pts[cam_idx][count_aux])
                    objpoints.append(objp)
                if chessboard_detect[0][i]:
                    count_ref += 1
                if chessboard_detect[cam_idx][i]:
                    count_aux += 1

        T_guess = np.matmul(R_guess, -cam_position_m[cam_idx].reshape(3, 1))
        ret, K1, D1, K2, D2, R1, T1, E, F, viewErr = cv2.stereoCalibrateExtended(objpoints, imgpts_ref, imgpts,
                                  K[0], D[0], K[cam_idx], D[cam_idx], img_size, R_guess.copy(), T_guess, flags=e_flags)

        print("")
        print("Num extrinsic poses: {}".format(len(objpoints)))
        print("")
        print("Camera Matrix Camera {} round 2:".format(setupInfo.RigInfo.module_name[0]))
        print(K[0])
        print("")
        print("Camera Matrix Camera {} round 2:".format(setupInfo.RigInfo.module_name[cam_idx]))
        print(K[cam_idx])

        print("")
        print("Distortion Vector Camera {} round 2:".format(setupInfo.RigInfo.module_name[0]))
        print(D[0])
        print("")
        print("Distortion Vector Camera {} round 2:".format(setupInfo.RigInfo.module_name[cam_idx]))
        print(D[cam_idx])

        R.append(R1)
        T.append(T1)
        stereo_view_error.append(viewErr)

        print("")
        print("Rotation Matrix:")
        print(R1)

        A, J = cv2.Rodrigues(R1)
        A *= 180.0 / 3.14159
        print("")
        print("Rotation Vector (deg):")
        print(A)

        print("")
        print("Translation Matrix:")
        print(T1)

        c = np.matmul(-np.linalg.inv(R1), T1)
        print("")
        print("World coordinate of camera {} (m)".format(setupInfo.RigInfo.module_name[cam_idx]))
        print(c)
    else:
        R.append(np.identity(3))
        T.append(np.zeros((3, 1)))


# Save results
cal_info = CalibrationInfo(args.cal_dir, K=np.array(K), D=np.array(D), R=np.array(R), T=np.array(T))
cal_info.write_json("calibration.json")

for cam_idx in range(num_cam):
    cal_file_helper.save_np_file("rvecs{}".format(cam_idx), rvecs[cam_idx])
    cal_file_helper.save_np_file("tvecs{}".format(cam_idx), tvecs[cam_idx])
    cal_file_helper.save_np_file("view_error{}".format(cam_idx), np.squeeze(view_error[cam_idx]))
    if cam_idx > 0:
        cal_file_helper.save_np_file("stereo_view_error{}".format(cam_idx), np.squeeze(stereo_view_error[cam_idx-1]))

D_np = np.asarray(D)
K_np = np.asarray(K)
K_np = np.reshape(K_np, [-1, num_cam])
R_np = np.asarray(R)
T_np = np.asarray(T)
cal_file_helper.save_text_file("D.txt", np.squeeze(D_np))
cal_file_helper.save_text_file("K.txt", K_np)
cal_file_helper.save_text_file("R.txt", np.reshape(R_np, [-1, num_cam]))
cal_file_helper.save_text_file("T.txt", np.squeeze(T_np))
