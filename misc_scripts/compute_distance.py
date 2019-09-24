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
 *   Computes distance to "matched" points in an image
 *
"""
import importlib
import argparse
from libs.Stereo import *
import importlib
import matplotlib as matplot
matplot.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

####################
# Input Parameters
####################

parser = argparse.ArgumentParser(description="Compute distance using random spacial points")
parser.add_argument('--cal_dir', default='Calibration_Aug15_large_board')
parser.add_argument('--image_dir', default='Outside_Aug15_0')

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

####################a

plt.figure(1)

# Match algo
orb = cv2.AKAZE_create()

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

disp_calc = cv2.StereoBM_create(numDisparities=1600, blockSize=5)

stereo = Stereo(args.cal_dir)

img_idx = []
ham_distance = []
pts_ref = []
pts_src = []

setup_info = importlib.import_module("{}.setup".format(args.image_dir))
num_cam =setup_info.RigInfo.image_filename.size
orientation = 0
all_files_read = False
while not all_files_read:
    for cam_idx in range(num_cam):
        fname = os.path.join(args.image_dir, setup_info.RigInfo.image_filename[cam_idx].format(orientation))
        try:
            raw = np.load(fname)
        except:
            all_files_read = True
            break

        raw = raw.astype(np.float32) * 256.0 / 1024.0
        raw = raw.astype(np.uint8)
        gray = cv2.cvtColor(raw, cv2.COLOR_BayerBG2GRAY)
        img = cv2.cvtColor(raw, cv2.COLOR_BayerBG2BGR)

        if cam_idx == 0:
            # save reference image
            gray_ref = gray
            img_ref = img
        else:
            # Rectifiy views
            rect_ref, rect_src, P1, P2, R1, R2 = stereo.rectify_views(gray_ref.copy(), gray.copy(), cam_idx)

            plt.figure(5).clear()
            rect_overlay = cv2.addWeighted(rect_ref, 0.5, rect_src, 0.5, 0.0)
            plt.imshow(rect_overlay)
            plt.title("Rectified Overlay")

            # Test
            plt.figure(cam_idx).clear()
            d = np.arange(5, 400)
            z = -P2[0, 3] / d
            plt.plot(d, z)
            plt.title("Disparity distance")
            plt.xlabel("Disparity (pixels)")
            plt.ylabel("Distance (m)")
            plt.grid()

            # find the keypoints and descriptors with SIFT
            kp_ref, des_ref = orb.detectAndCompute(rect_ref, None)
            kp_src, des_src = orb.detectAndCompute(rect_src, None)

            # Match descriptors.
            matches = bf.match(des_ref, des_src)

            # Filter matches that are not correct
            for m in matches:
                if m.distance < 12:
                    dx = kp_src[m.trainIdx].pt[0] - kp_ref[m.queryIdx].pt[0]
                    dy = kp_src[m.trainIdx].pt[1] - kp_ref[m.queryIdx].pt[1]
                    if (dx < -20) and (abs(dy) < 5):
                        img_idx.append(orientation)
                        ham_distance.append(m.distance)
                        pts_ref.append(kp_ref[m.queryIdx].pt)
                        pts_src.append(kp_src[m.trainIdx].pt)

            # Test Plot dx vs dy
            diff = np.array(pts_src) - np.array(pts_ref)
            plt.figure(10+cam_idx).clear()
            plt.scatter(diff[:, 0], diff[:, 1])
            plt.title("Scatter plot of matched points")
            plt.xlabel("Disparity (pixels)")
            plt.ylabel("Delta Y (pixels)")
            plt.grid()

            # Truncate for opencv - Better solution?
            pts_ref = np.int32(pts_ref)
            pts_src = np.int32(pts_src)

            img3, img4 = stereo.draw_epilines(rect_ref.copy(), rect_src.copy(), pts_ref, pts_src, cam_idx=cam_idx, P1=P1, P2=P2)
            img3 = cv2.resize(img3, dsize=None, fx=1/2, fy=1/2)
            img4 = cv2.resize(img4, dsize=None, fx=1/2, fy=1/2)
            cv2.imshow("Rectified Reference epilines", img3)
            cv2.imshow("Rectified Source epilines", img4)

            pts = []
            pts.append(pts_ref)
            pts.append(pts_src)
            distance, disparity = stereo.compute_distance_disparity(pts, T=P2[:, 3])

            plt.figure(15+cam_idx).clear()
            plt.scatter(disparity, distance)
            plt.grid()
            plt.title("Disparity / Distance(m)")
            plt.xlabel("Disparity")
            plt.ylabel("Distance")

            break
    all_files_read = True
