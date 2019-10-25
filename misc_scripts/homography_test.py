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
import argparse
from libs.Image import *
from libs.LensDistortion import *
import matplotlib as matplot
matplot.use('TkAgg')
import matplotlib.pyplot as plt


####################
# Input Parameters
####################

parser = argparse.ArgumentParser(description="Homography Test")
parser.add_argument('--image_dir', default='tv_86in_circlegrid_oct18')

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

####################

estimate_from_center_only = True
use_saved_results = False

image_helper = Image(args.image_dir)
display_size = image_helper.display_size(1024)
setupInfo = image_helper.setup_info()

nx = setupInfo.ChartInfo.nx
ny = setupInfo.ChartInfo.ny
center = np.array([setupInfo.SensorInfo.width * 0.5, setupInfo.SensorInfo.height * 0.5])
max_dist = setupInfo.SensorInfo.width * 1.0 / 8.0

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nx*ny, 3), np.float32)
checker_size_mm = setupInfo.ChartInfo.size_mm
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * (checker_size_mm * 1.0e-3)
xi = cv2.convertPointsToHomogeneous(objp[:, 0:2])

plt.figure(0)

find_ret = []
find_corners = []
if use_saved_results:
    # Load previous results
    find_ret = image_helper.load_np_file("distortion_find_ret.npy")
    find_corners = image_helper.load_np_file("distortion_find_corners.npy")

orientation = 0
lens_shade_filter = []
for cam_idx in range(image_helper.num_cam()):
    lens_shade_filter.append(image_helper.load_np_file("lens_shading_{}.npy".format(cam_idx)))
    if not use_saved_results:
        find_ret.append([])
        find_corners.append([])

distortion_error = []
all_files_read = False
while not all_files_read:
    for cam_idx in range(image_helper.num_cam()):
        if orientation == 0:
            distortion_error.append([])

        img_tmp, _ = image_helper.read_image_file(cam_idx, orientation, scale_to_8bit=False)
        if img_tmp is None:
            all_files_read = True
            break

        image_normalized = (img_tmp - 64.0 * 16.0) * lens_shade_filter[cam_idx] + 64.0 * 16.0
        image_normalized[image_normalized < 0.0] = 0.0
        max_val = np.max(image_normalized)
        print("max value: {}".format(max_val))
        image_normalized = (image_normalized/max_val) * ((256*256) - 1)
        print("Normalized max value: {}".format(np.max(image_normalized)))

        img = np.round(image_normalized/256).astype(np.uint8)

        if use_saved_results:
            ret = find_ret[cam_idx][orientation]
            corners = find_corners[cam_idx][orientation]
        else:
            print("Searching...")
            ret, corners = cv2.findCirclesGrid(img, (nx, ny), None)

            #ret, corners = cv2.findChessboardCornersSB(img, (nx, ny), None)
            find_ret[cam_idx].append(ret)
            if ret:
                find_corners[cam_idx].append(corners)
            else:
                # Fill with zeros
                find_corners[cam_idx].append(np.zeros((nx * ny, 1, 2)))

        # Compute the distance to the center of the sensor
        dist = []
        if ret:
            #corners = corners[::-1]
            img = cv2.drawChessboardCorners(img.copy(), (nx, ny), corners, True)

            obj_pts = []
            dist = np.linalg.norm(np.squeeze(corners - center), axis=1)
            if estimate_from_center_only and dist[dist <= max_dist].shape[0] >= 4:
                obj_pts = objp[dist <= max_dist]
                img_pts = corners[dist <= max_dist]
            elif not estimate_from_center_only and len(corners) >= 4:
                obj_pts = objp
                img_pts = corners
            else:
                print("Not enough points to compute homography")
                # sys.exit(-1)

            if len(obj_pts) > 0:
                H, mask = cv2.findHomography(obj_pts, img_pts)
                print("H {} - num points: {}".format(setupInfo.RigInfo.module_name[cam_idx], len(obj_pts)))
                print("{}".format(H))

                # Compute the ideal corners using homography
                xip = cv2.convertPointsFromHomogeneous(np.matmul(H, np.squeeze(xi.copy()).T).T)
                error = corners - xip
                distortion_error[cam_idx].append(np.array(error))
                p = plt.figure(cam_idx)
                p.clear()
                x = xip[:, 0, 0]
                y = xip[:, 0, 1]
                u = -1.0 * error[:, 0, 0]
                v = -1.0 * error[:, 0, 1]
                plt.gca().invert_yaxis()
                plt.quiver(x, y, u, v, angles='uv', units='xy', minlength=1, scale_units='xy', scale=0.01, pivot='tip')
                plt.draw()

                #p = plt.figure(10+cam_idx)
                #p.clear()
                #plt.gca().invert_yaxis()
                #plt.scatter(xip[:, 0, 0], xip[::-1, 0, 1], color='k', s=1)
                #plt.scatter(corners[:, 0, 0], corners[::-1, 0, 1], color='g', s=1)
                #plt.draw()

                max_x_idx = np.argmax(np.abs(u))
                max_y_idx = np.argmax(np.abs(v))
                print("Max error ({}, {})".format(u[max_x_idx], v[max_y_idx]))
        else:
            print("Chessboard not found")
            distortion_error[cam_idx].append(np.zeros((nx * ny, 1, 2)))

        img2 = cv2.resize(img, None, fx=display_size, fy=display_size)
        img2 = cv2.circle(img2, (img2.shape[1] >> 1, img2.shape[0] >> 1), int(max_dist * display_size + 0.5), (255, 0, 0), 3)
        cv2.imshow("{}".format(setupInfo.RigInfo.module_name[cam_idx]), img2)

        print("")

    if not all_files_read:
        plt.waitforbuttonpress(0.1)
        key = cv2.waitKey(50)
        if key == 27:
            break
    orientation += 1


if not use_saved_results:
    find_ret = np.array(find_ret)
    find_corners = np.array(find_corners)
    image_helper.save_np_file("distortion_find_ret", find_ret)
    image_helper.save_np_file("distortion_find_corners", find_corners)


distortion_error = np.array(distortion_error)
img_shape = (setupInfo.SensorInfo.height, setupInfo.SensorInfo.width)
c = np.squeeze(find_corners[0, find_ret[0]], axis=2).reshape(-1, 2)
e = np.squeeze(distortion_error[0, find_ret[0]], axis=2).reshape(-1, 2)

lens_distortion = LensDistortion()
dist_map = lens_distortion.compute_distortion_map(img_shape, c, e)

step = 1
plt.figure(10).clear()
plt.imshow(dist_map[::step, ::step, 0])
plt.colorbar()
plt.title('X Error')

plt.figure(11).clear()
plt.imshow(dist_map[::step, ::step, 1])
plt.colorbar()
plt.title('Y Error')

if False:
    use_saved_results = False

    find_ret = []
    find_corners = []
    if use_saved_results:
        # Load previous results
        find_ret = image_helper.load_np_file("distortion_find_ret.npy")
        find_corners = image_helper.load_np_file("distortion_find_corners.npy")

    orientation = 0
    lens_shade_filter = []
    for cam_idx in range(image_helper.num_cam()):
        lens_shade_filter.append(image_helper.load_np_file("lens_shading_{}.npy".format(cam_idx)))
        if not use_saved_results:
            find_ret.append([])
            find_corners.append([])

    distortion_error = []
    all_files_read = False
    while not all_files_read:
        for cam_idx in range(image_helper.num_cam()):
            if orientation == 0:
                distortion_error.append([])

            img_tmp, _ = image_helper.read_image_file(cam_idx, orientation, scale_to_8bit=False)
            if img_tmp is None:
                all_files_read = True
                break

            image_normalized = (img_tmp - 64.0 * 16.0) * lens_shade_filter[cam_idx] + 64.0 * 16.0
            image_normalized[image_normalized < 0.0] = 0.0
            max_val = np.max(image_normalized)
            print("max value: {}".format(max_val))
            image_normalized = (image_normalized/max_val) * ((256*256) - 1)
            print("Normalized max value: {}".format(np.max(image_normalized)))

            img = np.round(image_normalized/256).astype(np.uint8)

            img = lens_distortion.correct_distortion(img)

            if use_saved_results:
                ret = find_ret[cam_idx][orientation]
                corners = find_corners[cam_idx][orientation]
            else:
                print("Searching...")
                ret, corners = cv2.findCirclesGrid(img, (nx, ny), None)

                #ret, corners = cv2.findChessboardCornersSB(img, (nx, ny), None)
                find_ret[cam_idx].append(ret)
                find_corners[cam_idx].append(corners)

            # Compute the distance to the center of the sensor
            dist = []
            if ret:
                #corners = corners[::-1]
                img = cv2.drawChessboardCorners(img.copy(), (nx, ny), corners, True)

                obj_pts = []
                dist = np.linalg.norm(np.squeeze(corners - center), axis=1)
                if estimate_from_center_only and dist[dist <= max_dist].shape[0] >= 4:
                    obj_pts = objp[dist <= max_dist]
                    img_pts = corners[dist <= max_dist]
                elif not estimate_from_center_only and len(corners) >= 4:
                    obj_pts = objp
                    img_pts = corners
                else:
                    print("Not enough points to compute homography")
                    # sys.exit(-1)

                if len(obj_pts) > 0:
                    H, mask = cv2.findHomography(obj_pts, img_pts)
                    print("H {} - num points: {}".format(setupInfo.RigInfo.module_name[cam_idx], len(obj_pts)))
                    print("{}".format(H))

                    # Compute the ideal corners using homography
                    xip = cv2.convertPointsFromHomogeneous(np.matmul(H, np.squeeze(xi.copy()).T).T)
                    error = corners - xip
                    distortion_error[cam_idx].append(error)
                    p = plt.figure(cam_idx)
                    p.clear()
                    x = xip[:, 0, 0]
                    y = xip[:, 0, 1]
                    u = -1.0 * error[:, 0, 0]
                    v = -1.0 * error[:, 0, 1]
                    plt.gca().invert_yaxis()
                    plt.quiver(x, y, u, v, angles='uv', units='xy', minlength=1, scale_units='xy', scale=0.01, pivot='tip')
                    plt.draw()

                    #p = plt.figure(10+cam_idx)
                    #p.clear()
                    #plt.gca().invert_yaxis()
                    #plt.scatter(xip[:, 0, 0], xip[::-1, 0, 1], color='k', s=1)
                    #plt.scatter(corners[:, 0, 0], corners[::-1, 0, 1], color='g', s=1)
                    #plt.draw()

                    max_x_idx = np.argmax(np.abs(u))
                    max_y_idx = np.argmax(np.abs(v))
                    print("Max error ({}, {})".format(u[max_x_idx], v[max_y_idx]))
            else:
                print("Chessboard not found")

        img2 = cv2.resize(img, None, fx=display_size, fy=display_size)
        img2 = cv2.circle(img2, (img2.shape[1] >> 1, img2.shape[0] >> 1), 5, (255, 0, 0), 3)
        cv2.imshow("{}".format(setupInfo.RigInfo.module_name[cam_idx]), img2)

        print("")

    if not all_files_read:
        plt.waitforbuttonpress(0.1)
        key = cv2.waitKey(50)
        if key == 27:
            break
    orientation += 1