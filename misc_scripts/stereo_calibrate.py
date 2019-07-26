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
 ******************************************************************************/
"""
import cv2
import numpy as np
import os
import matplotlib as matplot
matplot.use('TkAgg')
import matplotlib.pyplot as plt

plt.figure(1)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
nx = 17
ny = 11
checker_size_mm = 280 / 13
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) * checker_size_mm

# Libra devices
fl_mm = 3.95
pixel_size_um = 1.25

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints_l = [] # 2d points in image plane.
imgpoints_r = [] # 2d points in image plane.
image_dir = "cal_072519_1"
show_images = True

orientation = 0
capture = 0
if True:
    while True:
        fname = os.path.join(image_dir, "left{}_{}.npy".format(orientation, capture))
        try:
            raw = np.load(fname)
        except:
            break
        raw = raw.astype(np.float32) * 256.0 / 1024.0
        raw = raw.astype(np.uint8)
        gray_l = cv2.cvtColor(raw, cv2.COLOR_BayerBG2GRAY)
        img_l = cv2.cvtColor(raw, cv2.COLOR_BayerBG2BGR)

        fname = os.path.join(image_dir, "right{}_{}.npy".format(orientation, capture))
        raw = np.load(fname)
        raw = raw.astype(np.float32) * 256.0 / 1024.0
        raw = raw.astype(np.uint8)
        gray_r = cv2.cvtColor(raw, cv2.COLOR_BayerBG2GRAY)
        img_r = cv2.cvtColor(raw, cv2.COLOR_BayerBG2BGR)

        # Find the chess board corners
        print("Searching orientation {}, capture {}".format(orientation, capture))
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, (nx,ny), None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, (nx,ny), None)
        print("Search Done")

        if (ret_l == True) and (ret_r == True):
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), criteria)
            imgpoints_l.append(corners2)
            if show_images:
                img2 = cv2.drawChessboardCorners(img_l, (nx,ny), corners2, ret_l)
                img2 = cv2.resize(img2, None, fx=1/4, fy=1/4)
                cv2.imshow('Image Left', img2)
                cv2.waitKey(500)

            corners2 = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), criteria)
            imgpoints_r.append(corners2)
            if show_images:
                img2 = cv2.drawChessboardCorners(img_r, (nx,ny), corners2, ret_r)
                img2 = cv2.resize(img2, None, fx=1/4, fy=1/4)
                cv2.imshow('Image Right', img2)
                cv2.waitKey(500)
        else:
            print("No points added for orientation {}, capture {}".format(orientation, capture))
            img2 = cv2.resize(img_l, None, fx=1/4, fy=1/4)
            cv2.imshow('Bad Image Left', img2)
            img2 = cv2.resize(img_r, None, fx=1/4, fy=1/4)
            cv2.imshow('Bad Image Right', img2)
            cv2.waitKey(500)

        orientation += 1

    imgshape = gray_l.shape[::-1]
    np.save("objpoints", objpoints)
    np.save("imgpoints_l", imgpoints_l)
    np.save("imgpoints_r", imgpoints_r)
    np.save("img_shape", imgshape)
else:
    objpoints = np.load("objpoints.npy")
    imgpoints_l = np.load("imgpoints_l.npy")
    imgpoints_r = np.load("imgpoints_r.npy")
    imgshape = tuple(np.load("img_shape.npy"))


print("")
print("Calibrating Cameras")
f = fl_mm / pixel_size_um * 1000
K = np.array([[f, 0, imgshape[0]/2], [0, f, imgshape[1]/2], [0, 0, 1]])
flags = cv2.CALIB_RATIONAL_MODEL
flags |= cv2.CALIB_USE_INTRINSIC_GUESS
flags |= cv2.CALIB_FIX_K1
flags |= cv2.CALIB_FIX_K2
flags |= cv2.CALIB_FIX_K3
flags |= cv2.CALIB_FIX_K4
flags |= cv2.CALIB_FIX_K5
flags |= cv2.CALIB_FIX_K6
K1 = K.copy()
ret, K1, D1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints_l, imgshape, K1, None, flags=flags)
K2 = K.copy()
ret, K2, D2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints_r, imgshape, K2, None, flags=flags)
del rvecs1, rvecs2, tvecs1, tvecs2
ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_l, imgpoints_r,
      K1, D1, K2, D2, imgshape, flags=cv2.CALIB_FIX_INTRINSIC)

P1 = np.matmul(K1, np.hstack((np.identity(3), np.zeros((3,1)))))
P2 = np.matmul(K2, np.hstack((R, T)))

print("")
print("Left Camera")
print("Principle point: ({:.2f}, {:.2f}) - Difference from ideal: ({:.2f}, {:.2f})".format(K1[0,2], K1[1,2],
      4032/2-K1[0,2], 3016/2-K1[1,2]))
print("Focal length (mm): ({:.2f}, {:.2f}) - Expected 3.95mm".format(
    K1[0,0] * pixel_size * 1000.0, K1[1,1] * pixel_size_um * 1.0e-3))

print("")
print("Right Camera")
print("Principle point: ({:.2f}, {:.2f}) - Difference from ideal: ({:.2f}, {:.2f})".format(K2[0,2], K2[1,2],
   4032/2-K1[0,2], 3016/2-K1[1,2]))
print("Focal length (mm): ({:.2f}, {:.2f}) - Expected 3.95mm".format(
   K2[0,0] * pixel_size * 1000.0, K2[1,1] * pixel_size_um * 1.0e-3))

print("")
print("Rotation Matrix:")
print(R)

print("")
print("Translation Matrix:")
print(T)

c2 = np.matmul(-np.linalg.inv(R), T)
print("")
print("World coordinate of 2nd camera (mm)")
print(c2)
