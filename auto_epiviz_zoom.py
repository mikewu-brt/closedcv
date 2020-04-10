#  Copyright (c) 2019, The LightCo
#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without
#  modification, are strictly prohibited without prior permission of
#  The LightCo.
#
#  This module works on a single capture for the camera rig. It uses the known distance to the object and its image to
#  compute the focal length of all the cameras on the rig
#
#  @author  yhussain
#  @version V1.0.0
#  @date    Oct 2019
#  @brief
#  Auto Epiviz Tool

import cv2
import numpy as np
import os
import importlib
import argparse
from libs.Image import *
import json
import OpenEXR
import Imath
from libs.LensDistortion import *
import matplotlib as matplot
matplot.use('TkAgg')
import matplotlib.pyplot as plt


def plot_camera(ax, title, img):
    ax.imshow(img)
    ax.set_title(title)
    ax.set_axis_off()

#def compute_fundamental_matrix( K1, K2, R, t)

def skew_symetric(vector_in):
    """
    Converts vector_in into a skew symetric matrix.  This is useful for computing cross products
    of 2 vectors through matrix multiplication.

    A x B = [A]x B where [A]x is the skew_symetric(A).  [A]x is a 3x3 matrix and B is a 3x1 matrix.

    :param vector_in: 3x1 or 1x3 vector
    :return s: [vector_in]x
    """
    if vector_in.size != 3:
        print("skew_symetric matrix must only contain 3 elements")
        exit(-1)

    s = np.zeros((3, 3), dtype=np.float)
    s[0, 1] = -vector_in[2]
    s[1, 0] = vector_in[2]

    s[0, 2] = vector_in[1]
    s[2, 0] = -vector_in[1]

    s[1, 2] = -vector_in[0]
    s[2, 1] = vector_in[0]
    return s

def compute_projection_matrix(K, R, T):
    P = np.matmul(K, np.hstack((R, T)))
    return P

def compute_fundamental_matrix(K1, K2, R, T):
    """
    Computes the fundamental matrix from KRT.  Assumes P1 = K1 [ I | 0]

    :return F: Fundamental matrix
    """
    P1 = compute_projection_matrix(K1, np.identity(3), np.zeros((3, 1)))
    P2 = compute_projection_matrix(K2, R, T)
    Pp = np.vstack((np.linalg.inv(K1), np.zeros(3)))
    C = np.zeros((4, 1))
    C[3] = 1
    F = np.matmul(skew_symetric(np.matmul(P2, C)), np.matmul(P2, Pp))
    return F

def convert_exr_image(exrfile):
    exr = OpenEXR.InputFile(exrfile)
    DW = exr.header()['dataWindow']
    rows, cols = (DW.max.y - DW.min.y + 1, DW.max.x - DW.min.x + 1)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    nptype = (np.float32, (rows, cols))
    chans = [np.frombuffer(exr.channel(c, FLOAT), dtype=nptype, count=1).squeeze() for c in ('R', 'G', 'B')]
    return np.dstack(chans)

def parse_calib_data(calibdata, check_dist_norm = False):
    K = np.zeros((3, 3))
    R = np.zeros((3, 3))
    T = np.zeros((3, 1))
    D = np.zeros((5, 1))

    # Rotation
    R[0, 0] = calibdata["rvec"][0]
    R[0, 1] = calibdata["rvec"][1]
    R[0, 2] = calibdata["rvec"][2]
    R[1, 0] = calibdata["rvec"][3]
    R[1, 1] = calibdata["rvec"][4]
    R[1, 2] = calibdata["rvec"][5]
    R[2, 0] = calibdata["rvec"][6]
    R[2, 2] = calibdata["rvec"][7]
    R[2, 2] = calibdata["rvec"][8]

    # Translation
    T[0, 0] = calibdata["tvec"][0]
    T[1, 0] = calibdata["tvec"][1]
    T[2, 0] = calibdata["tvec"][2]

    # Kmat
    kmat = calibdata["intrinsic_kmat"]["fd_0"]["k_mat"]
    K[0, 0] = kmat[0]
    K[0, 1] = kmat[1]
    K[0, 2] = kmat[2]
    K[1, 0] = kmat[3]
    K[1, 1] = kmat[4]
    K[1, 2] = kmat[5]
    K[2, 0] = kmat[6]
    K[2, 2] = kmat[7]
    K[2, 2] = kmat[8]

    # Distortion
    indist = calibdata["distort_params"]
    if check_dist_norm:
        assert(K[0, 2] == indist["distortion_center"][0])
        assert(K[1, 2] == indist["distortion_center"][1])
        assert(K[0, 0] == indist["norm_x_y"][0])
        assert(K[1, 1] == indist["norm_x_y"][1])
    D[0, 0] = indist["distortion_coeffs"][0]
    D[1, 0] = indist["distortion_coeffs"][1]
    D[2, 0] = indist["distortion_coeffs"][2]
    D[3, 0] = indist["distortion_coeffs"][3]
    D[4, 0] = indist["distortion_coeffs"][4]

    return K, R, T, D


    # @staticmethod
def drawlines(img1, img2, lines, pts1, pts2, circles=True):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r = img1.shape[0]
    c = img1.shape[1]
    if np.ndim(img1) < 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if np.ndim(img2) < 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        pt1 = np.int32(pt1)
        pt2 = np.int32(pt2)
        if circles:
            img1 = cv2.circle(img1, (pt1[0,0],pt1[0,1]), 5, color, circle_thickness)
            #img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
            img2 = cv2.circle(img2, (pt2[0,0], pt2[0,1]), 5, color, circle_thickness)
    return img1, img2

####################
# Input Parameters
####################

#sys.argv = "auto_epiviz.py  --image_dir 2020-04-02_calibration --cal_dir  4_2_cal_f16/save_rt_cv_init_ref_fppd_04_02/cv_bypassed  --correspond_dir  4_2_cal_f16    ".split()
parser = argparse.ArgumentParser(description="Auto Epiviz")
parser.add_argument('--image_dir', default='2020-04-02_calibration')
parser.add_argument('--cal_dir', default='4_2_cal_f16/save_rt_cv_init_ref_fppd_04_02/cv_bypassed')
parser.add_argument('--correspond_dir', default='4_2_cal_f16')
parser.add_argument('--upsample', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=33)
parser.add_argument('--point_idx', type=int, default=0)
parser.add_argument('--frame_num', type=int, default=0)
parser.add_argument('--image_width', type=int, default=3208)
parser.add_argument('--image_height', type=int, default=2200)
parser.add_argument('--read_image_bin', action="store_true", default=False, help='use this option if we want to read images in  bin files, however this will require a setup.py file ')
parser.add_argument('--read_correspondence_exr', action="store_true", default=False, help='use this option if we want to read correspondences exr files otherwise npy files are assumed ')
#parser.add_argument('--fix_slant', action="store_true", default=False, help='use this option to account for any slant in the checkerboard for magnitude estimation')

# Open a figure to avoid cv2.imshow crash
plt.figure(1)

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

print("")
print("Command line options:")
print("  {}".format(args))
print("")

num_cam = 4
module_name  = ["A1", "A2", "A3", "A4"]
circle_thickness = 3

img_size = (args.image_height, args.image_width)


path_to_image_dir = os.getenv("PATH_TO_IMAGE_DIR")
#path_to_image_dir = "/Users/yhussain/project/cal_images/"

# read the calibration files
K = []
R = []
T = []
D = []
for cam_idx in range(num_cam):
    filename = path_to_image_dir + "/" + args.cal_dir + "/calib{}.json".format(cam_idx)
    print(filename)
    K1, R1, T1, D1  = parse_calib_data(json.load(open(filename, 'r')), False)
    K.append(K1)
    R.append(R1)
    T.append(T1)
    D.append(D1)
#    def parse_calib_data_fil(filename, check_dist_norm = False):
#    return parse_calib_data(json.load( open(filename, 'r')), check_dist_norm)

image_show = False
#orientation = 30
#orientation = 80
#choose a point in the corners
#point_index = 200 # choose randomly
#point_index = 800 # choose randomly
#patch_size = (33,33)
#upsample_factor = 9

upsample_factor = args.upsample
patch_size = (args.patch_size, args.patch_size)
point_index = args.point_idx
orientation = args.frame_num

img_stack = []
if args.read_image_bin:
    image_helper = Image(args.image_dir)
    display_size = image_helper.display_size(1024)
    num_cam = image_helper.num_cam()
    setupInfo = image_helper.setup_info()
    sensorInfo = setupInfo.SensorInfo
    img_size = (setupInfo.SensorInfo.width, setupInfo.SensorInfo.height)
    # read the bin files
    all_files_read = False
    while not all_files_read:
        for cam_idx in range(image_helper.num_cam()):
            img, gray = image_helper.read_image_file(cam_idx, orientation, scale_to_8bit=True)
            if img is None:
                all_files_read = True
                break
            img_stack.append(img)
            if image_show:
                img2 = cv2.resize(img, None, fx=display_size, fy=display_size)
                cv2.imshow("{}".format(setupInfo.RigInfo.module_name[cam_idx]), img2)
                cv2.waitKey(500)
        all_files_read = True  # need to read only one set

else:
    for cam_idx in range(num_cam):
        filename = path_to_image_dir + "/" + args.image_dir + "/img_{}_0.jpg".format(module_name[cam_idx])
        print(filename)
        img = cv2.imread(filename)
        if img_size != img.shape[0:2]:
            print("Image size mismatch Please Check!!")
            print(img_size)
            print(img.shape)
            exit(1);

        img_stack.append(img)



# read EXR files
if args.read_correspondence_exr:
    filename = path_to_image_dir + "/" + args.correspond_dir +  "/f{0:03d}.exr".format(orientation)
    print(filename)
    exr_np = convert_exr_image(filename)
else:
    filename = path_to_image_dir + "/" + args.correspond_dir
    print(filename)
    exr_np = np.load(filename, allow_pickle=True)
    print(exr_np)

# get new optimal camera matrix for each of the cameras and undistort image and the correspondence sets
K_new = []
K1_new = np.empty(K[0].shape)
new_img_size = img_size
undist_img_stack = []
undist_points_stack = []
corner_detected = []
for cam_idx in range(num_cam):
    K1_new, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix=K[cam_idx], distCoeffs = D[cam_idx],
                                                imageSize=img_size, alpha=0,  newImgSize=new_img_size)
    print(roi)
    K_new.append(K1_new)
    print(new_img_size)
    undist_img = cv2.undistort(img_stack[cam_idx], K[cam_idx], D[cam_idx], newCameraMatrix=K_new[cam_idx])
    undist_img_stack.append(undist_img)

    # convert to float64 else errors out
    if args.read_correspondence_exr:
        corners = np.ndarray.astype(exr_np[cam_idx, :, 0:2], dtype=np.float64)
    else:
        corners = np.ndarray.astype(np.array(exr_np[cam_idx]), dtype=np.float64)
        corners = corners.reshape(-1,2)

    print(corners.shape)
    print(corners)
    if(corners[0,0] == -1):
        corner_detected.append(False)
    else:
        corner_detected.append(True)


    undist_points = cv2.undistortPoints(src=corners, cameraMatrix=K[cam_idx], distCoeffs=D[cam_idx], P=K_new[cam_idx])

    undist_points_stack.append(undist_points)



F_stack = []
image_patches = []

# for each baseline
for cam_idx in range(1,num_cam):

    # select the ROI of size 33x33 around the center point for each image


    corner_point_ref = undist_points_stack[0][point_index]
    print(corner_point_ref)
    y_ref = np.int32(np.round(corner_point_ref[0,1]))
    x_ref = np.int32(np.round(corner_point_ref[0,0]))
    print(" x_ref, y_ref = ", x_ref, y_ref)
    roi_patch_ref = ( x_ref - patch_size[0]//2, y_ref - patch_size[1]//2, x_ref+patch_size[0]//2 + 1, y_ref+patch_size[1]//2 + 1)
    print(roi_patch_ref)

    corner_point_src = undist_points_stack[cam_idx][point_index]
    print(corner_point_src)

    if corner_detected[cam_idx]:
        y_src = np.int32(np.round(corner_point_src[0,1]))
        x_src = np.int32(np.round(corner_point_src[0,0]))
        print(" x_src, y_src = ", x_src, y_src)
        roi_patch_src = ( x_src - patch_size[0]//2, y_src - patch_size[1]//2, x_src+patch_size[0]//2 + 1, y_src+patch_size[1]//2 + 1)
        print(roi_patch_src)


        # compute fundamental matrix between each pair
        F = compute_fundamental_matrix(K_new[0], K_new[cam_idx], R[cam_idx], T[cam_idx])
        F_stack.append(F)


        # covert correspondence points to Homogeneous co-ordinates
        pts_ref = cv2.convertPointsToHomogeneous(corner_point_ref)
        pts_src = cv2.convertPointsToHomogeneous(corner_point_src)

        # comoute the epipolar lines
        line_src = cv2.computeCorrespondEpilines(pts_ref, 1, F).reshape(-1, 3)
        line_ref = cv2.computeCorrespondEpilines(pts_src, 2, F).reshape(-1, 3)


        img_src, img1 = drawlines( undist_img_stack[cam_idx].copy(), undist_img_stack[0].copy(), line_src, pts_src, pts_ref)
        img3 = cv2.resize(img_src, None, fx=1/2, fy=1/2)
        cv2.imshow("{}_full".format(module_name[cam_idx]), img3)
        cv2.waitKey(500)
        img_src_nocircles, img1_nocircles = drawlines( undist_img_stack[cam_idx].copy(), undist_img_stack[0].copy(), line_src, pts_src, pts_ref, circles=False)



        # select roi
        img_roi_src = img_src[roi_patch_src[1]:roi_patch_src[3], roi_patch_src[0]:roi_patch_src[2]]
        img_roi_src_nocircles = img_src_nocircles[roi_patch_src[1]:roi_patch_src[3], roi_patch_src[0]:roi_patch_src[2]]
        img_roi_ref_nocircles = img1_nocircles[roi_patch_ref[1]:roi_patch_ref[3], roi_patch_ref[0]:roi_patch_ref[2]]

        img3 = cv2.resize(img_roi_src, None, fx=upsample_factor, fy=upsample_factor, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("{}".format(module_name[cam_idx]), img3)


        #img4 = cv2.resize(img1[roi_patch_ref[1]:roi_patch_ref[3], roi_patch_ref[0]:roi_patch_ref[2]], None, fx=upsample_factor, fy=upsample_factor)
        #cv2.imshow("{}_{}".format(setupInfo.RigInfo.module_name[0], setupInfo.RigInfo.module_name[cam_idx]), img4)


        # take the ref and source patches, adjust the corner point locations on the patches, upsample the patches and then draw circle and epilines
        # img_src has the epiline and is the ref image ( may need to remove circle and add again at a better location0
        # img_roi_src is the patch that needs to be resized
        img_roi_src_zoom = cv2.resize(img_roi_src_nocircles, None, fx=upsample_factor, fy=upsample_factor, interpolation=cv2.INTER_NEAREST)
        img_roi_ref_zoom = cv2.resize(img_roi_ref_nocircles, None, fx=upsample_factor, fy=upsample_factor, interpolation=cv2.INTER_NEAREST)

        patch_point_ref = (patch_size[0]//2 + corner_point_ref[0,0] - x_ref, patch_size[1]//2 + corner_point_ref[0,1] - y_ref)
        print(patch_point_ref)
        patch_point_ref_zoom = (patch_point_ref[0] * upsample_factor, patch_point_ref[1] * upsample_factor)
        print(patch_point_ref_zoom)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        img_roi_ref_zoom = cv2.circle(img_roi_ref_zoom, (np.int32(patch_point_ref_zoom[0]),np.int32(patch_point_ref_zoom[1])), 2*upsample_factor, color, circle_thickness)
        #cv2.imshow("A1_{}_nocircles".format(setupInfo.RigInfo.module_name[cam_idx]), img_roi_ref_zoom)
        if cam_idx == 1:
            image_patches.append(img_roi_ref_zoom)


        patch_point_src = (patch_size[0]//2 + corner_point_src[0,0] - x_src, patch_size[1]//2 + corner_point_src[0,1] - y_src)
        print(patch_point_src)
        patch_point_src_zoom = (patch_point_src[0] * upsample_factor, patch_point_src[1] * upsample_factor)
        print(patch_point_src_zoom)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        img_roi_src_zoom = cv2.circle(img_roi_src_zoom, (np.int32(patch_point_src_zoom[0]),np.int32(patch_point_src_zoom[1])), 2*upsample_factor, color, circle_thickness)
        cv2.imshow("{}_nocircles".format(module_name[cam_idx]), img_roi_src_zoom)
        image_patches.append(img_roi_src_zoom)


fig, axes = plt.subplots(2, 2)
plot_camera(axes[0, 0], "A1", image_patches[0])
plot_camera(axes[0, 1], "A2", image_patches[1])
plot_camera(axes[1, 0], "A3", image_patches[2])
plot_camera(axes[1, 1], "A4", image_patches[3])
plt.suptitle("Correspondences for point ({:.2f}, {:.2f}) in reference".format(undist_points_stack[0][point_index, 0, 0], undist_points_stack[0][point_index, 0, 1]))
#img_shape = image_patches[0].shape
#imstack = np.concatenate((image_patches[0].reshape(-1), image_patches[1].reshape(-1), image_patches[2].reshape(-1), image_patches[3].reshape(-1)))
#imstack_v = imstack.reshape(-1,img_shape[1],3)
#cv2.imshow('imstack_v', imstack_v)


####################
