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
import matplotlib as matplot
matplot.use('TkAgg')
import matplotlib.pyplot as plt

def mag_calib(image_dir):

    setupInfo = importlib.import_module("{}.setup".format(image_dir))

    # Optical parameters
    fl_mm = setupInfo.LensInfo.fl_mm
    pixel_size_um = setupInfo.SensorInfo.pixel_size_um

    # Checkerboard info
    nx = setupInfo.ChartInfo.nx
    ny = setupInfo.ChartInfo.ny
    checker_size_mm = setupInfo.ChartInfo.size_mm

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Misc control
    show_images = True
    process_image_files = True
    force_fx_eq_fy = True

    num_cam = setupInfo.RigInfo.image_filename.shape[0]

    # Open a figure to avoid cv2.imshow crash
    plt.figure(1)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) * (checker_size_mm * 1.0e-3)

    # Arrays to store object points and image points from all the images.
    #objpoints = np.empty((1, nx*ny, 3))                 # 3D points in World space (relative)
    objpoints = []                  # 3D points in World space (relative)
    imgpoints = np.empty((num_cam, 1, nx*ny, 1, 2))     # Image points

    corners2 = np.empty((num_cam, 1, nx*ny, 1, 2))
    all_files_read = False
    orientation = 0
    if process_image_files:
        while True:
            chessboard_found = True
            # Load images
            for cam_idx in range(num_cam):
                fname = os.path.join(image_dir, setupInfo.RigInfo.image_filename[cam_idx].format(orientation))
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
                ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
                if ret:
                    print("Search Done")
                    corners2[cam_idx, 0, :, :, :] = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    if show_images:
                        img2 = cv2.drawChessboardCorners(img, (nx, ny), corners2[cam_idx,:-1], ret)
                        img2 = cv2.resize(img2, None, fx=1/4, fy=1/4)
                        cv2.imshow("Image {}".format(cam_idx), img2)
                        cv2.waitKey(500)
                else:
                    chessboard_found = False
                    break

            if all_files_read:
                break
            elif chessboard_found:
                # Add points to arrays
                objpoints.append(objp)
                if orientation == 0:
                    imgpoints = corners2.copy()
                else:
                    imgpoints = np.concatenate((imgpoints, corners2), axis=1)
            else:
                print("Chessboard not found in {}".format(fname))
                if show_images:
                    img2 = cv2.resize(img, None, fx=1/4, fy=1/4)
                    cv2.imshow("Bad Image {}".format(cam_idx), img2)
                    cv2.waitKey(500)

            orientation += 1

        imgshape = gray.shape[::-1]
        np.save(os.path.join(image_dir, "objpoints"), objpoints)
        np.save(os.path.join(image_dir, "imgpoints"), imgpoints)
        np.save(os.path.join(image_dir, "img_shape"), imgshape)
    else:
        objpoints = np.load(os.path.join(image_dir, "objpoints.npy"))
        imgpoints = np.load(os.path.join(image_dir, "imgpoints.npy"))
        imgshape = tuple(np.load(os.path.join(image_dir, "img_shape.npy")))


    print("")
    print("Calibrating Camera Magnitude")
    f = fl_mm * 1.0e-3 / (pixel_size_um * 1.0e-6)
    K_guess = np.array([[f, 0, imgshape[0]/2], [0, f, imgshape[1]/2], [0, 0, 1]])

    i_flags = cv2.CALIB_USE_INTRINSIC_GUESS
    i_flags = cv2.CALIB_FIX_PRINCIPAL_POINT
    i_flags |= cv2.CALIB_ZERO_TANGENT_DIST
    i_flags |= cv2.CALIB_FIX_K1
    i_flags |= cv2.CALIB_FIX_K2
    i_flags |= cv2.CALIB_FIX_K3
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
    imgpts_ref = []
    for cam_idx in range(num_cam):
        print("Compute magnitude for cam {}".format(cam_idx))
        imgpts = []
        if cam_idx == 0:
            for i in range(imgpoints.shape[1]):
                imgpts_ref.append(imgpoints[cam_idx, i, :, :, :].astype(np.float32))
                imgpts.append(imgpts_ref[i])
        else:
            for i in range(imgpoints.shape[1]):
                imgpts.append(imgpoints[cam_idx, i, :, :, :].astype(np.float32))

        ret, K1, D1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpts, imgshape, K_guess.copy(), None, flags=i_flags)

        K.append(K1)

    # Save results
    np.save(os.path.join(image_dir, "K"), K)

    return K