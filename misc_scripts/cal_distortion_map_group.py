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
#

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
parser.add_argument('--cal_dir', default=None)
parser.add_argument('--use_distortion_from', default=None)

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

####################

use_saved_results = True

image_helper = Image(args.image_dir)
display_size = image_helper.display_size(1024)
setupInfo = image_helper.setup_info()

center = np.array([setupInfo.SensorInfo.width * 0.5, setupInfo.SensorInfo.height * 0.5])
max_dist = setupInfo.SensorInfo.width * 1.0 / 1.0

plt.figure(0)

if args.use_distortion_from is not None:
    lens_distortion = LensDistortion(0, args.use_distortion_from, args.cal_dir)
    print("")
    print("Using distortion map from: {}".format(args.use_distortion_from))
    lens_distortion.plot_distortion_map(30, 64)
    plt.title("Lens Distortion Map")
    plt.draw()
else:
    lens_distortion = LensDistortion(0, args.cal_dir, args.cal_dir)

# Search for corners
find_ret = []
find_corners = []
group = 0
pose_cnt = 0
orientation = 0
while not use_saved_results and group < len(setupInfo.ChartInfo.pose_info):

    if pose_cnt == 0:
        nx = setupInfo.ChartInfo.pose_info[group]['nx']
        ny = setupInfo.ChartInfo.pose_info[group]['ny']
        find_ret.append([])
        find_corners.append([])
        num_pose_in_group = setupInfo.ChartInfo.pose_info[group]['pose_nx'] * \
                            setupInfo.ChartInfo.pose_info[group]['pose_ny']

    img_tmp, _ = image_helper.read_image_file(0, orientation, scale_to_8bit=False)
    if img_tmp is None:
        print("Error - Exiting early, no more files to read")
        break

    img = lens_distortion.correct_vignetting(img_tmp, alpha=0.7, scale_to_8bit=True)

    print("Searching...")
    ret, corners = cv2.findCirclesGrid(img, (nx, ny), None)

    # ret, corners = cv2.findChessboardCornersSB(img, (nx, ny), None)
    find_ret[group].append(ret)
    if ret:
        print("Chessboard found")
        find_corners[group].append(corners)
        img = cv2.drawChessboardCorners(img.copy(), (nx, ny), corners, True)
    else:
        print("Chessboard not found")
        # Fill with zeros
        find_corners[group].append(np.zeros((nx * ny, 1, 2)))

    img2 = cv2.resize(img, None, fx=display_size, fy=display_size)
    img2 = cv2.circle(img2, (img2.shape[1] >> 1, img2.shape[0] >> 1), int(max_dist * display_size + 0.5), (255, 0, 0), 3)
    cv2.imshow("{}".format(setupInfo.RigInfo.module_name[0]), img2)
    print("")

    key = cv2.waitKey(50)
    if key == 27:
        break

    pose_cnt += 1
    if pose_cnt >= num_pose_in_group:
        pose_cnt = 0
        group += 1

    orientation += 1

# Load saved results or store saved results
if use_saved_results:
    find_ret = []
    find_corners = []
    for group in range(len(setupInfo.ChartInfo.pose_info)):
        find_ret.append(image_helper.load_np_file("distortion_find_ret_{}.npy".format(group)))
        find_corners.append(image_helper.load_np_file("distortion_find_corners_{}.npy".format(group)).astype(np.float32))
else:
    for group in range(len(find_ret)):
        find_ret[group] = np.array(find_ret[group])
        find_corners[group] = np.array(find_corners[group], dtype=np.float32)
        image_helper.save_np_file("distortion_find_ret_{}".format(group), find_ret[group])
        image_helper.save_np_file("distortion_find_corners_{}".format(group), find_corners[group])


# Compute the homography
obj_pts = []
img_pts = []
H = []
distortion_error = []
dist_map = []
for group in range(len(find_ret)):
    # Gather data to compute the homography
    obj_pts.append([])
    img_pts.append([])
    H.append([])

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    pose_x = 0
    pose_y = 0
    nx = setupInfo.ChartInfo.pose_info[group]['nx']
    ny = setupInfo.ChartInfo.pose_info[group]['ny']
    objp = np.zeros((nx * ny, 3), np.float32)
    checker_size_mm = setupInfo.ChartInfo.size_mm
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * (checker_size_mm * 1.0e-3)

    for ret in find_ret[group]:
        if ret:
            obj_corners = objp.copy() + np.array([setupInfo.ChartInfo.pose_info[group]['pose_dx'] * pose_x / 1000.0,
                                                  setupInfo.ChartInfo.pose_info[group]['pose_dy'] * pose_y / 1000.0,
                                                  0])
            pose_idx = setupInfo.ChartInfo.pose_info[group]['pose_nx'] * pose_y + pose_x
            corners = find_corners[group][pose_idx].copy()

            # Undistort
            if args.use_distortion_from is not None:
                corners = lens_distortion.correct_distortion_points(corners)

            dist = np.linalg.norm(np.squeeze(corners - center), axis=1)
            obj_pts[group].append(obj_corners[dist <= max_dist])
            img_pts[group].append(corners[dist <= max_dist])

        pose_x += 1
        if pose_x >= setupInfo.ChartInfo.pose_info[group]['pose_nx']:
            pose_x = 0
            pose_y += 1

    # Compute the homography
    o = np.empty((0, 3))
    for o1 in obj_pts[group]:
        o = np.vstack((o, o1))

    i = np.empty((0, 1, 2))
    for i1 in img_pts[group]:
        i = np.vstack((i, i1))
    h, mask = cv2.findHomography(o, i)
    H[group].append(h)
    print("H {} - num points: {}".format(setupInfo.RigInfo.module_name[0], len(o)))
    print("{}".format(h))
    print("")


    # Compute the errors per corner
    distortion_error.append([])
    pose_x = 0
    pose_y = 0
    nx = setupInfo.ChartInfo.pose_info[group]['nx']
    ny = setupInfo.ChartInfo.pose_info[group]['ny']
    for ret in find_ret[group]:
        if ret:
            # Compute the ideal corners using homography
            obj_corners = objp.copy() + np.array([setupInfo.ChartInfo.pose_info[group]['pose_dx'] * pose_x / 1000.0,
                                                  setupInfo.ChartInfo.pose_info[group]['pose_dy'] * pose_y / 1000.0,
                                                  1.0])
            xip = cv2.convertPointsFromHomogeneous(np.matmul(H[group], obj_corners.T).T)
            idx = setupInfo.ChartInfo.pose_info[group]['pose_nx'] * pose_y + pose_x
            corners = find_corners[group][idx].copy()

            # Undistort
            if args.use_distortion_from is not None:
                corners = lens_distortion.correct_distortion_points(corners)

            error = corners - xip
            distortion_error[group].append(np.array(error))
            p = plt.figure(group)
            p.clear()
            x = xip[:, 0, 0]
            y = xip[:, 0, 1]
            u = -1.0 * error[:, 0, 0]
            v = -1.0 * error[:, 0, 1]
            plt.gca().invert_yaxis()
            plt.quiver(x, y, u, v, angles='uv', units='xy', minlength=1, scale_units='xy', scale=0.01, pivot='tip')
            plt.title("Distortion error for group {}, pose {}, {}".format(group, pose_x, pose_y))
            plt.draw()

            max_x_idx = np.argmax(np.abs(u))
            max_y_idx = np.argmax(np.abs(v))
            avg_error = np.average(np.linalg.norm(np.squeeze(error), axis=1))
            print("Group: {}, Pose: {}, {} - Max error: ({}, {}), Avg error: {}".format(group, pose_x, pose_y,
                                                                          u[max_x_idx], v[max_y_idx],
                                                                          avg_error))

            if plt.waitforbuttonpress(0.1):
                group = len(find_ret)
                break
        else:
            print("Group: {}, Pose: {}, {} - Checkerboard not found ".format(group, pose_x, pose_y))
            distortion_error[group].append(np.zeros((nx * ny, 1, 2)))

        pose_x += 1
        if pose_x >= setupInfo.ChartInfo.pose_info[group]['pose_nx']:
            pose_x = 0
            pose_y += 1

    if args.use_distortion_from is None:
        # Compute the distortion map
        dist_error = np.array(distortion_error[group])
        img_shape = (setupInfo.SensorInfo.height, setupInfo.SensorInfo.width)
        c = np.squeeze(find_corners[group][find_ret[group]], axis=2).reshape(-1, 2)
        e = np.squeeze(dist_error[find_ret[group]], axis=2).reshape(-1, 2)

        dist_map.append([])
        dist_map[group] = lens_distortion.compute_distortion_map(img_shape, c, e)

if args.use_distortion_from is None:
    step = 1
    plt.figure(10).clear()
    plt.imshow(dist_map[0][::step, ::step, 0])
    plt.colorbar()
    plt.title('X Error')
    plt.draw()

    plt.figure(20).clear()
    plt.imshow(dist_map[0][::step, ::step, 1])
    plt.colorbar()
    plt.title('Y Error')
    plt.draw()

    # FIXME(Chuck) - Need to combine all distortion maps into a single map. simply combine for not but should align
    img_shape = (setupInfo.SensorInfo.height, setupInfo.SensorInfo.width)
    c = np.squeeze(find_corners[0][find_ret[0]], axis=2).reshape(-1, 2)
    e = np.squeeze(np.array(distortion_error[0])[find_ret[0]], axis=2).reshape(-1, 2)
    for group in range(1, len(dist_map)):
        c = np.concatenate((c, np.squeeze(find_corners[group][find_ret[group]], axis=2).reshape(-1, 2)))
        e = np.concatenate((e, np.squeeze(np.array(distortion_error[group])[find_ret[group]], axis=2).reshape(-1, 2)))

        lens_distortion.set_distortion_map(dist_map[group])
        lens_distortion.plot_distortion_map(30 + group, 64)
        plt.title("Lens Distortion Map for group {}".format(group))
        plt.draw()

    dist_map.append([])
    dist_map[-1] = lens_distortion.compute_distortion_map(img_shape, c, e)
    lens_distortion.plot_distortion_map(40, 64)
    plt.title("Lens Distortion Map for overall")
    plt.draw()

    # Save map
    print("")
    print("Saving distortion map")
    print("")
    cal_helper = Image(args.cal_dir)
    cal_helper.save_np_file("distortion_map_{}".format(setupInfo.RigInfo.module_name[0]), dist_map[-1])


    step = 1
    plt.figure(10).clear()
    plt.imshow(dist_map[-1][::step, ::step, 0])
    plt.colorbar()
    plt.title('X Error')
    plt.draw()

    plt.figure(20).clear()
    plt.imshow(dist_map[-1][::step, ::step, 1])
    plt.colorbar()
    plt.title('Y Error')
    plt.draw()

    plt.waitforbuttonpress(0.1)

