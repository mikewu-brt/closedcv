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
from libs.Image import *
from libs.LensDistortion import *
import importlib
import matplotlib as matplot
matplot.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

plt.figure(1).clear()
orientation = 0

####################
# Input Parameters
####################

parser = argparse.ArgumentParser(description="Compute distance using random spacial points")
parser.add_argument('--cal_dir', default='Calibration_Aug23')
parser.add_argument('--image_dir', default='Outside_Aug15_0')
parser.add_argument('--start_idx', type=int, default=0)

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

use_saved_results = False

search_y_min = -6
search_y_max = 6
max_disparity = 800

####################a

stereo = Stereo(args.cal_dir)
image_helper = Image(args.image_dir)
setup_info = image_helper.setup_info()
scale = image_helper.display_size(2000)

num_cam = image_helper.num_cam()
all_files_read = False

ref_pts = []
src_pts = []
lens = []
for cam_idx in range(num_cam):
    ref_pts.append([])
    src_pts.append([])
    lens.append(LensDistortion(cam_idx, args.cal_dir, args.cal_dir))

if not use_saved_results:
    while not all_files_read:
        for cam_idx in range(num_cam):
            img, _ = image_helper.read_image_file(cam_idx, orientation + args.start_idx, scale_to_8bit=False)
            if img is None:
                all_files_read = True
                break

            if lens[cam_idx].distortion_map() is not None:
                print("Correct {} image with distortion map".format(setup_info.RigInfo.module_name[cam_idx]))
                img = lens[cam_idx].correct_distortion(img)

            if cam_idx == 0:
                # save reference image
                img_ref = img.copy()
                ref_pts[0].append((0, 0))
                src_pts[0].append((0, 0))
            else:
                # Rectifiy views
                img_src = img.copy()
                rect_ref, rect_src, P1, P2, R1, R2 = stereo.rectify_views(img_ref.copy(), img_src.copy(), cam_idx)

                # Plot rectified views
                rect_overlay = cv2.addWeighted(rect_ref, 0.5, rect_src, 0.5, 0.0)
                img = cv2.resize(rect_overlay, dsize=None, fx=scale, fy=scale)
                cv2.imshow("Rect Overlay Cam {} - Cam {}".format(setup_info.RigInfo.module_name[0],
                                                                 setup_info.RigInfo.module_name[cam_idx]), img)

                # Disparity / Distance chart for this view
                plt.figure(cam_idx).clear()
                d = np.arange(5, 400)
                # Take abs for display purposes
                z = np.absolute(-P2[0, 3] / d)
                plt.plot(d, z)
                plt.title("Disparity distance")
                plt.xlabel("Disparity (pixels)")
                plt.ylabel("Distance (m)")
                plt.grid()

                # Choose region
                print("")
                print("Orientation {}, Cam {}".format(orientation, setup_info.RigInfo.module_name[cam_idx]))
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
                        for cleanup_idx in range(cam_idx):
                            ref_pts[cleanup_idx].pop()
                            src_pts[cleanup_idx].pop()
                        all_files_read = True
                        break
                    elif key == ord('n'):
                        if len(ref_pts[cam_idx]) > orientation:
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
                            min_sum = 1.0 / scale * roi[2] * roi[3] * 3.0 * pow(2, 16)
                            min_xs = 0
                            min_ys = 0

                            for y_search in range(search_y_min, search_y_max+1):

                                srcxs = max(0, xs - max_disparity)
                                srcxe = min(srcxs + int(1.0 / scale * roi[2]), rect_src.shape[1])
                                srcys = max(0, ys + y_search)
                                srcye = min(srcys + int(1.0 / scale * roi[3]), rect_src.shape[0])

                                while (srcxs <= (xs + max_disparity)) and (srcxe < rect_src.shape[1]):
                                    abs_diff = np.sum(np.abs(np.subtract(rect_ref[ys:ye, xs:xe], rect_src[srcys:srcye, srcxs:srcxe], dtype=np.float)))
                                    if abs_diff < min_sum:
                                        min_sum = abs_diff
                                        min_xs = srcxs
                                        min_ys = srcys

                                    srcxs += 1
                                    srcxe += 1

                            if len(ref_pts[cam_idx]) <= orientation:
                                ref_pts[cam_idx].append((xs, ys))
                                src_pts[cam_idx].append((min_xs, min_ys))
                            else:
                                ref_pts[cam_idx][orientation] = (xs, ys)
                                src_pts[cam_idx][orientation] = (min_xs, min_ys)

                            pts = []
                            pts.append(np.array(ref_pts[cam_idx][orientation]).reshape(1, 2))
                            pts.append(np.array(src_pts[cam_idx][orientation]).reshape(1, 2))
                            print(P2)
                            dis, dip = stereo.compute_distance_disparity(pts, T=P2[:, 3])
                            print("Reference ROI: ({}, {}, {}, {})".format(xs, ys, xe-xs, ye-ys))
                            print("Source ROI top: ({}, {})".format(min_xs, min_ys))
                            print("Disparity: {}, Distance: {}".format(dip, dis))
                            print("")

                            # Draw line on rectified view
                            min_ye = min_ys + int(1.0 / scale * roi[3])
                            min_ym = int((min_ys + min_ye) / 2.0)
                            min_xe = min_xs + int(1.0 / scale * roi[2])
                            updated_overlay = cv2.line(rect_overlay.copy(), (0, min_ym), (rect_overlay.shape[1], min_ym), (0, 255, 0), int(1.0 / scale))
                            updated_overlay = cv2.line(updated_overlay, (0, min_ys), (rect_overlay.shape[1], min_ys), (255, 0, 0), int(1.0 / scale))
                            updated_overlay = cv2.line(updated_overlay, (0, min_ye), (rect_overlay.shape[1], min_ye), (0, 0, 255), int(1.0 / scale))
                            updated_overlay = cv2.line(updated_overlay, (min_xs, 0), (min_xs, rect_overlay.shape[0]), (0, 0, 255), int(1.0 / scale))
                            updated_overlay = cv2.line(updated_overlay, (min_xe, 0), (min_xe, rect_overlay.shape[0]), (0, 0, 255), int(1.0 / scale))
                            img = cv2.resize(updated_overlay, dsize=None, fx=scale, fy=scale)
                            cv2.imshow("Rect Overlay Cam {} - Cam {}".format(setup_info.RigInfo.module_name[0],
                                                                 setup_info.RigInfo.module_name[cam_idx]), img)
                    if all_files_read:
                        break

        orientation += 1

    for i in range(len(ref_pts)):
        if i == 0:
            ref = np.expand_dims(np.array(ref_pts[i]), axis=0)
            src = np.expand_dims(np.array(src_pts[i]), axis=0)
        else:
            ref = np.concatenate((ref, np.expand_dims(np.array(ref_pts[i]), axis=0)), axis=0)
            src = np.concatenate((src, np.expand_dims(np.array(src_pts[i]), axis=0)), axis=0)
    ref_pts = ref
    src_pts = src
    i = input("Do you wish to save the results (Y/n): ")
    if i == 'Y':
        print("Saving results")
        image_helper.save_np_file("ref_pts", ref_pts)
        image_helper.save_np_file("src_pts", src_pts)


else:
    print("Using saved results")

    # Load results from files
    ref_pts = image_helper.load_np_file("ref_pts.npy")
    src_pts = image_helper.load_np_file("src_pts.npy")

gt = image_helper.load_np_file("ground_truth.npy")[args.start_idx:]
for cam_idx in range(1, num_cam):
    R1, R2, P1, P2, Q, roi1, roi2 = stereo.rectification_matrix(cam_idx)
    # Compute results
    pts = []
    pts.append(ref_pts[cam_idx])
    pts.append(src_pts[cam_idx])
    distance, disparity = stereo.compute_distance_disparity(pts, T=P2[:, 3])
    disparity = np.squeeze(disparity)
    distance = np.squeeze(distance)

    # Plot results
    # Correct for angles in the ground truth measurements
    delta_x = ref_pts[cam_idx][:, 0] - P2[0, 2]
    delta_y = ref_pts[cam_idx][:, 1] - P2[1, 2]
    theta = np.arctan(delta_x / P2[0, 0])
    alpha = np.arctan(delta_y / P2[1, 1])
    new_gt = np.cos(theta) * np.cos(alpha) * gt[0:len(theta)]

    # Sort the Ground truth (for display purposes)
    i = np.argsort(new_gt)

    plt.figure(0 + 10 * cam_idx).clear()
    plt.scatter(-disparity[i], distance[i])
    plt.scatter(-disparity[i], new_gt[i])
    plt.grid()
    plt.xlabel("Disparity (pixels)")
    plt.ylabel("Distance (m)")
    plt.title("Disparity / Distance Cam {} {} - {}".format(setup_info.RigInfo.module_name[0],
                                                           setup_info.RigInfo.module_name[cam_idx], args.cal_dir))
    plt.legend(("Computed Distance", "Ground Truth"))


    dx = P2[0, 3] / new_gt
    g1 = P2[0, 3] / (dx + 1)
    g2 = P2[0, 3] / (dx - 1)
    g3 = P2[0, 3] / (dx + 2)
    g4 = P2[0, 3] / (dx - 2)

    err = distance - new_gt
    plt.figure(1 + 10 * cam_idx).clear()
    plt.plot(new_gt[i], err[i])
    plt.plot(new_gt[i], (g3-new_gt)[i])
    plt.plot(new_gt[i], (g1-new_gt)[i])
    plt.plot(new_gt[i], (g2-new_gt)[i])
    plt.plot(new_gt[i], (g4-new_gt)[i])
    plt.grid()
    plt.xlabel("Distance (m)")
    plt.ylabel("Error (m)")
    plt.title("Distance Error Cam {} {} - {}".format(setup_info.RigInfo.module_name[0],
                                                     setup_info.RigInfo.module_name[cam_idx], args.cal_dir))
    plt.legend(("Error", "+2 pixel error", "+1 pixel error", "-1 pixel error", "-2 pixel error"))


    pixel_err = disparity - dx
    plt.figure(2 + 10 * cam_idx).clear()
    plt.plot(new_gt[i], pixel_err[i])
    plt.grid()
    plt.xlabel("Distance (m)")
    plt.ylabel("Error (pixels)")
    plt.title("Pixel error vs distance Cam {} {} - {}".format(setup_info.RigInfo.module_name[0],
                                                              setup_info.RigInfo.module_name[cam_idx], args.cal_dir))

    plt.figure(3 + 10 * cam_idx).clear()
    diff = src_pts[cam_idx] - ref_pts[cam_idx]
    plt.scatter(-diff[:, 0], diff[:, 1])
    plt.grid()
    plt.xlabel("Disparity X (pixels)")
    plt.ylabel("Disparity Y (pixels)")
    plt.title("Y Disparity Error Cam {} {} - Search range ({}, {}) - {}".format(setup_info.RigInfo.module_name[0],
                                                                                setup_info.RigInfo.module_name[cam_idx],
                                                                                search_y_min, search_y_max, args.cal_dir))

cv2.destroyAllWindows()
