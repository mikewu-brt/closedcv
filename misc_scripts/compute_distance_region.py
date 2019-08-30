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
import argparse
from libs.Stereo import *
import importlib
import matplotlib as matplot
matplot.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

plt.figure(1).clear()
orientation = 0

ref_pts = []
src_pts = []


####################
# Input Parameters
####################

parser = argparse.ArgumentParser(description="Compute distance using random spacial points")
parser.add_argument('--cal_dir', default='Calibration_Aug23')
parser.add_argument('--image_dir', default='Outside_Aug15_0')

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

scale = 0.5
use_saved_results = False

search_y_min = -6
search_y_max = 6
max_disparity = 800

####################a

stereo = Stereo(args.cal_dir)

setup_info = importlib.import_module("{}.setup".format(args.image_dir))
num_cam = setup_info.RigInfo.image_filename.size
all_files_read = False

if not use_saved_results:
    while not all_files_read:
        for cam_idx in range(num_cam):
            fname = os.path.join(args.image_dir, setup_info.RigInfo.image_filename[cam_idx].format(orientation))
            try:
                raw = np.load(fname)
            except:
                all_files_read = True
                break

            img = cv2.cvtColor(raw.copy(), cv2.COLOR_BayerBG2BGR)

            if cam_idx == 0:
                # save reference image
                img_ref = img.copy()
            else:
                # Rectifiy views
                img_src = img.copy()
                rect_ref, rect_src, P1, P2, R1, R2 = stereo.rectify_views(img_ref.copy(), img_src.copy(), cam_idx)

                # Plot rectified views
                rect_overlay = cv2.addWeighted(rect_ref, 0.5, rect_src, 0.5, 0.0)
                img = cv2.resize(rect_overlay, dsize=None, fx=scale, fy=scale)
                cv2.imshow("Rect Overlay", img)

                # Disparity / Distance chart for this view
                plt.figure(cam_idx).clear()
                d = np.arange(5, 400)
                z = -P2[0, 3] / d
                plt.plot(d, z)
                plt.title("Disparity distance")
                plt.xlabel("Disparity (pixels)")
                plt.ylabel("Distance (m)")
                plt.grid()

                # Choose region
                print("")
                print("Orientation {}".format(orientation))
                print("Press \'s\' to select ROI from reference image, ESC to exit, \'n\' to advance to next capture")
                show_w = int(rect_ref.shape[1] * scale)
                show_h = int(rect_ref.shape[0] * scale)
                max_x = rect_ref.shape[1] - show_w - 1
                max_y = rect_ref.shape[0] - show_h - 1
                ref_img = cv2.resize(rect_ref.copy(), dsize=None, fx=scale, fy=scale)
                cv2.imshow("Reference", ref_img)
                cv2.waitKey(100)

                while True:
                    key = cv2.waitKey(200)
                    if key == 27:
                        all_files_read = True
                        break
                    elif key == ord('n'):
                        if len(ref_pts) > orientation:
                            print("Advancing to next capture")
                            break
                        else:
                            print("Cannot advance until points are chosen")
                    elif key == ord('s'):
                        roi = cv2.selectROI("Reference", ref_img)
                        if roi[2] > 0:
                            xs = int(1.0 / scale * roi[0])
                            xe = xs + int(1.0 / scale * roi[2])
                            ys = int(1.0 / scale * roi[1])
                            ye = ys + int(1.0 / scale * roi[3])
                            max_sum = 1.0 / scale * roi[2] * roi[3] * 3.0 * pow(2, 16)
                            max_xs = 0
                            max_ys = 0

                            for y_search in range(search_y_min, search_y_max+1):

                                srcxs = max(0, xs - max_disparity)
                                srcxe = min(srcxs + int(1.0 / scale * roi[2]), rect_ref.shape[1])
                                srcys = max(0, ys + y_search)
                                srcye = min(srcys + int(1.0 / scale * roi[3]), rect_src.shape[0])

                                while (srcxs < xs) and (srcxe < rect_src.shape[1]):
                                    abs_diff = np.sum(np.abs(np.subtract(rect_ref[ys:ye, xs:xe], rect_src[srcys:srcye, srcxs:srcxe], dtype=np.float)))
                                    if abs_diff < max_sum:
                                        max_sum = abs_diff
                                        max_xs = srcxs
                                        max_ys = srcys

                                    srcxs += 1
                                    srcxe += 1

                            if len(ref_pts) <= orientation:
                                ref_pts.append((xs, ys))
                                src_pts.append((max_xs, max_ys))
                            else:
                                ref_pts[orientation] = (xs, ys)
                                src_pts[orientation] = (max_xs, max_ys)

                            pts = []
                            pts.append(np.array(ref_pts[orientation]).reshape(1, 2))
                            pts.append(np.array(src_pts[orientation]).reshape(1, 2))
                            dis, dip = stereo.compute_distance_disparity(pts, T=P2[:, 3])
                            print("Reference ROI: ({}, {}, {}, {})".format(xs, ys, xe-xs, ye-ys))
                            print("Source ROI top: ({}, {})".format(max_xs, max_ys))
                            print("Disparity: {}, Distance: {}".format(dip, dis))
                            print("")

        orientation += 1

    ref_pts = np.array(ref_pts).reshape(-1, 2)
    src_pts = np.array(src_pts).reshape(-1, 2)
    i = input("Do you wish to save the results (Y/n): ")
    if i == 'Y':
        print("Saving results")
        np.save(os.path.join(args.image_dir, "ref_pts"), ref_pts)
        np.save(os.path.join(args.image_dir, "src_pts"), src_pts)


else:
    print("Using saved results")

    # Load results from files
#    disparity = np.load(os.path.join(args.image_dir, "disparity.npy"))
#    distance = np.load(os.path.join(args.image_dir, "distance.npy"))
    ref_pts = np.load(os.path.join(args.image_dir, "ref_pts.npy"))
    src_pts = np.load(os.path.join(args.image_dir, "src_pts.npy"))
    R1, R2, P1, P2, Q, roi1, roi2 = stereo.rectification_matrix(1)

# Compute results
pts = []
pts.append(np.array(ref_pts).reshape(-1, 2))
pts.append(np.array(src_pts).reshape(-1, 2))
distance, disparity = stereo.compute_distance_disparity(pts, T=P2[:, 3])

# Plot results
# Correct for angles in the ground truth measurements
gt = np.load(os.path.join(args.image_dir, "ground_truth.npy"))
delta = np.linalg.norm(ref_pts - P2[:2, 2], axis=1)
theta = np.arctan(delta / P2[0,0])
new_gt = np.cos(theta) * gt[0:len(theta)]

# Sort the Ground truth (for display purposes)
i = np.argsort(new_gt)

plt.figure(10).clear()
plt.scatter(-disparity[0][i], distance[0][i])
plt.scatter(-disparity[0][i], new_gt[i])
plt.grid()
plt.xlabel("Disparity (pixels)")
plt.ylabel("Distance (m)")
plt.title("Disparity / Distance")
plt.legend(("Computed Distance", "Ground Truth"))


dx = P2[0, 3] / new_gt
g1 = P2[0, 3] / (dx + 1)
g2 = P2[0, 3] / (dx - 1)
g3 = P2[0, 3] / (dx + 2)
g4 = P2[0, 3] / (dx - 2)

err = distance[0] - new_gt
plt.figure(11).clear()
plt.plot(new_gt[i], err[i])
plt.plot(new_gt[i], (g3-new_gt)[i])
plt.plot(new_gt[i], (g1-new_gt)[i])
plt.plot(new_gt[i], (g2-new_gt)[i])
plt.plot(new_gt[i], (g4-new_gt)[i])
plt.grid()
plt.xlabel("Distance (m)")
plt.ylabel("Error (m)")
plt.title("Distance Error")
plt.legend(("Error", "+2 pixel error", "+1 pixel error", "-1 pixel error", "-2 pixel error"))


pixel_err = disparity[0] - dx
plt.figure(12).clear()
plt.plot(new_gt[i], pixel_err[i])
plt.grid()
plt.xlabel("Distance (m)")
plt.ylabel("Error (pixels)")
plt.title("Pixel error vs distance")

plt.figure(13).clear()
diff = src_pts - ref_pts
plt.scatter(-diff[:, 0], diff[:, 1])
plt.grid()
plt.xlabel("Disparity X (pixels)")
plt.ylabel("Disparity Y (pixels)")
plt.title("Y Disparity Error - Search range ({}, {})".format(search_y_min, search_y_max))

cv2.destroyAllWindows()
