#  Copyright (c) 2019, The LightCo
#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without
#  modification, are strictly prohibited without prior permission of
#  The LightCo.
#
#  @author  cstanski
#  @version V1.0.0
#  @date    Oct 2019
#  @brief
#  Stereo calibration test script
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
import matplotlib as matplot
matplot.use('TkAgg')
import matplotlib.pyplot as plt


####################
# Input Parameters
####################

parser = argparse.ArgumentParser(description="Stereo Calibrate 2")
parser.add_argument('--image_dir', default='Calibration_Aug23')
parser.add_argument('--cal_dir', default='Calibration_Aug23')

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

####################

image_helper = Image(args.image_dir)
display_size = image_helper.display_size(1024)
setupInfo = image_helper.setup_info()

nx = setupInfo.ChartInfo.nx
ny = setupInfo.ChartInfo.ny
center = np.array([setupInfo.SensorInfo.width * 0.5, setupInfo.SensorInfo.height * 0.5])
max_dist = setupInfo.SensorInfo.width * 1.0 / 16.0

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nx*ny, 3), np.float32)
checker_size_mm = setupInfo.ChartInfo.size_mm
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * (checker_size_mm * 1.0e-3)
xi = cv2.convertPointsToHomogeneous(objp[:, 0:2])


orientation = 0
all_files_read = False
while not all_files_read:
    for cam_idx in range(image_helper.num_cam()):
        img, _ = image_helper.read_image_file(cam_idx, orientation, scale_to_8bit=True)
        if img is None:
            all_files_read = True
            break

        ret, corners = cv2.findChessboardCornersSB(img, (nx, ny), None)
        img2 = cv2.drawChessboardCorners(img.copy(), (nx, ny), corners, True)

        # Compute the distance to the center of the sensor
        dist = []
        if ret:
            corners = corners[::-1]
            dist = np.linalg.norm(np.squeeze(corners - center), axis=1)

            if dist[dist <= max_dist].shape[0] >= 4:
                obj_pts = objp[dist <= max_dist]
                img_pts = corners[dist <= max_dist]
                H, mask = cv2.findHomography(obj_pts, img_pts)
                print("H {} - num points: {}".format(setupInfo.RigInfo.module_name[cam_idx], len(obj_pts)))
                print("{}".format(H))

                # Compute the ideal corners using homography
                xip = cv2.convertPointsFromHomogeneous(np.matmul(H, np.squeeze(xi.copy()).T).T)
                error = xip - corners
                p = plt.figure(cam_idx)
                p.clear()
                x = corners[:, 0, 0]
                y = corners[:, 0, 1]
                u = error[:, 0, 0]
                v = error[:, 0, 1]
                plt.gca().invert_yaxis()
                plt.quiver(x, y, u, v, angles='xy', units='xy', scale_units='inches', scale=2.0, pivot='tail')
                plt.draw()

                p = plt.figure(10+cam_idx)
                p.clear()
                plt.gca().invert_yaxis()
                plt.scatter(xip[:, 0, 0], xip[::-1, 0, 1], color='k', s=1)
                plt.scatter(corners[:, 0, 0], corners[::-1, 0, 1], color='g', s=1)
                plt.draw()

        else:
            print("Not enough points to compute homography")
            sys.exit(-1)

        img2 = cv2.resize(img2, None, fx=display_size, fy=display_size)
        img2 = cv2.circle(img2, (img2.shape[1] >> 1, img2.shape[0] >> 1), 5, (255, 0, 0), 3)
        cv2.imshow("{}".format(setupInfo.RigInfo.module_name[cam_idx]), img2)

    if not all_files_read:
        plt.waitforbuttonpress(0.1)
        key = cv2.waitKey(0)
        if key == 27:
            break
    orientation += 1


