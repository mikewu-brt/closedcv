"""
 * Copyright (c) 2018, The LightCo
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are strictly prohibited without prior permission of
 * The LightCo.
 *
 * @author  Chuck Stanski
 * @version V1.0.0
 * @date    August 2019
 * @brief
 *   Computes distance assuming a checkboard chart
 *
"""
import importlib
import argparse
from libs.Stereo import *
import matplotlib as matplot
matplot.use('TkAgg')
import matplotlib.pyplot as plt

####################
# Input Parameters
####################

parser = argparse.ArgumentParser(description="Stereo Calibrate")
parser.add_argument('--image_dir', default='Distance_Aug15_0')

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

####################

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

setupInfo = importlib.import_module("{}.setup".format(args.image_dir))
nx = setupInfo.ChartInfo.nx
ny = setupInfo.ChartInfo.ny
num_cam = setupInfo.RigInfo.image_filename.size
orientation = 0
all_files_read = False
stereo = Stereo(args.image_dir)
corners2 = np.empty((num_cam, 1, nx*ny, 1, 2))
while True:
    chessboard_found = True
    # Load images
    for cam_idx in range(num_cam):
        fname = os.path.join(args.image_dir, setupInfo.RigInfo.image_filename[cam_idx].format(orientation))
        try:
            raw = np.load(fname)
        except:
            all_files_read = True
            break

        raw = raw.astype(np.float32) * 256.0 / 1024.0
        raw = raw.astype(np.uint8)
        gray = cv2.cvtColor(raw, cv2.COLOR_BayerBG2GRAY)
        img = cv2.cvtColor(raw, cv2.COLOR_BayerBG2BGR)

        #print("Searching {}".format(fname))
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            #print("Search Done")
            corners2[cam_idx, 0, :, :, :] = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        else:
            #print("Chessboard not found in {}".format(fname))
            chessboard_found = False

    if all_files_read:
        break
    elif chessboard_found:
        # Compute reprojection error
        pts = np.empty((2, corners2.shape[2], 1, 2))
        pts[0] = corners2[0, 0]
        for cam_idx in range(1, num_cam):
            pts[1] = corners2[cam_idx, 0]
            distance, disparity = stereo.compute_distance_disparity(pts)
            reproj_pts = stereo.reproject_points(pts[0], distance, 1)
            reproj_err_vect = pts[1] - reproj_pts
            reproj_xy_err = np.sqrt(np.average(np.square(reproj_err_vect), axis=0))
            reproj_err = np.sqrt(np.average(np.average(np.square(reproj_err_vect), axis=2)))
            print("")
            print("Camera {}, Orientation {}".format(cam_idx, orientation))
            print("Average distance (m): {}".format(np.average(distance)))
            print("Reprojection error (avg pixels): {}".format(reproj_err))
            print("    avg X error: {}".format(reproj_xy_err[0, 0]))
            print("    avg Y error: {}".format(reproj_xy_err[0, 1]))

            plt.figure(orientation*(num_cam-1)+cam_idx).clear()
            x = pts[1].reshape(-1, 2)[:, 0]
            y = pts[1].reshape(-1, 2)[:, 1]
            u = reproj_err_vect.reshape(-1, 2)[:, 0]
            v = reproj_err_vect.reshape(-1, 2)[:, 1]
            q = plt.quiver(x, y, u, v, pivot='tip')
            plt.title("Reprojection Error - Camera {}, Orientation {}, Distance {:.2f}".format(cam_idx, orientation,
                                                                                           np.average(distance)))
            plt.xlabel("Avg pixel error: {:.2f}, Avg X {:.2f}, Avg Y {:.2f}".format(reproj_err, reproj_xy_err[0, 0],
                                                                                    reproj_xy_err[0, 1]))
    else:
        print("Chessboard not found in {}".format(fname))

    orientation += 1
