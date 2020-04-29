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
from libs.Stereo import *
from libs.parse_calib_data import *
from libs.chart_reproj_errors import *
import matplotlib as matplot
matplot.use('TkAgg')
import matplotlib.pyplot as plt


def plot_camera(ax, title, img):
    ax.imshow(img)
    ax.set_title(title)
    ax.set_axis_off()

def convert_exr_image(exrfile):
    exr = OpenEXR.InputFile(exrfile)
    DW = exr.header()['dataWindow']
    rows, cols = (DW.max.y - DW.min.y + 1, DW.max.x - DW.min.x + 1)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    nptype = (np.float32, (rows, cols))
    chans = [np.frombuffer(exr.channel(c, FLOAT), dtype=nptype, count=1).squeeze() for c in ('R', 'G', 'B')]
    return np.dstack(chans)

def adjust_gamma(image, gamma=1.0):
   invGamma = 1.0 / gamma
   table = np.array([
      ((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)])
   return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))

def estimate_correspondence_based_on_reproj_errors (dir_name, chosen_idx, num_cams=4):

    #reproj_errors = collect_reprojection_errors(args.cal_dir, "rerror",num_cams)
    reproj_errors = collect_reprojection_errors(dir_name, "rerror",num_cams)

    # create a list of frames where corners are detected in all the cameras
    search_frame_list = []
    for frame_num in range(reproj_errors.shape[1]):
         add_index = True
         for cam_idx in range(num_cams):
             if sum(abs(reproj_errors[cam_idx, frame_num,:, :])).all() == 0:
                 add_index = False
         if add_index:
             search_frame_list.append(frame_num)

    norm_array_all = np.linalg.norm(reproj_errors[:,search_frame_list,:,2:4], axis=3)
    norm_array_average = np.average(norm_array_all,axis=0)
    index = np.where(norm_array_average == np.max(norm_array_average))
    index[0][0] = search_frame_list[index[0][0]]
    # extract the corner points
    corners = reproj_errors[:,index[0],:,0:2]
    return index, corners, np.max(norm_array_average), reproj_errors[:, index[0][0], index[1][0],2:4]

# translation of code from LibStereo
def ComputeRefDepthGivenCorrespondingPnt( rotation_ref_wrt_src, translation_ref_wrt_src, kmat_src, kmat_ref, src_pnt, ref_pnt):

    K_ref_inv  = np.linalg.inv(kmat_ref)
    src_pnt_homogeneous = cv2.convertPointsToHomogeneous(src_pnt)
    src_pnt_skew = Stereo.skew_symetric(np.squeeze(src_pnt_homogeneous))
    translationNmT = np.zeros((3,3), dtype=np.float)
    translationNmT[0,2] = translation_ref_wrt_src[0]
    translationNmT[1, 2] = translation_ref_wrt_src[1]
    translationNmT[2, 2] = translation_ref_wrt_src[2]

    ref_pnt_homogeneous = cv2.convertPointsToHomogeneous(ref_pnt)
    K_ref_inv_mult_ref_pnt = np.matmul(K_ref_inv, np.squeeze(ref_pnt_homogeneous))
    src_pnt_skew_mult_k_mat_src = np.matmul(src_pnt_skew, kmat_src)

    A = np.matmul(src_pnt_skew_mult_k_mat_src, np.matmul(rotation_ref_wrt_src, K_ref_inv_mult_ref_pnt))
    b = np.matmul(src_pnt_skew_mult_k_mat_src, np.matmul(translationNmT,K_ref_inv_mult_ref_pnt))
    return -np.dot(b, A) / np.dot(A,A)

# get correcpondence points based on specified methods:
# (1) use reproj charts and take the maximum reprojection error from the specified cam with the contraint that pint is present in all cams
# (2) read the correcpondence points from an npy file
# (3) use the provided option arguments
#def GetCorrespondencePoints( args ):
def GetCorrespondencePoints( cal_dir, read_correspondence_type, correspondence_file, frame_num, point_idx ):
    exr_np = None
    dir_name = cal_dir
    image_read_bin = True
    max_error = 0
    reproj_error = np.zeros((4,2))
    if read_correspondence_type == "max_reproj_error":
        # compute correspondence points based on maximum reprojection error
        [index, exr_np, max_error, reproj_error] = estimate_correspondence_based_on_reproj_errors (dir_name, chosen_idx=0)
        frame_index = index[0][0]
        point_index = index[1][0]
    elif read_correspondence_type == "exr":
        # read EXR correspondence files
        point_index = point_idx
        frame_index = frame_num
        filename = os.path.dirname(cal_dir) + "/f{0:03d}.exr".format(frame_index)
        print("Corners file: ", filename)
        exr_np = convert_exr_image(filename)
    elif read_correspondence_type == "npy":
        # read npy correspondence file
        point_index = point_idx
        frame_index = frame_num
        filename = correspondence_file
        image_read_bin = False
        print("Correspondences file", filename)
        exr_np = np.load(filename, allow_pickle=True)
        print("Correspondences", exr_np)
    else:
        print(" Invalid read_corrrespondence type:{}".format(read_correspondence_type))
        exit(1)
    return image_read_bin, frame_index, point_index, exr_np, max_error, reproj_error

# read images for all the cams based on:
# (1) read from bin files if read_image_bin is True
# (2) Othersise expect that the files are generated in jpg format using splitraw
def ReadImageAllCams( image_dir, setup_file, read_image_bin, frame_index, num_cam=4 ):
    img_stack = []
    if read_image_bin:
        os.unsetenv("PATH_TO_IMAGE_DIR")  # TODO(amaharshi: Remove dependence PATH_TO_IMAGE_DIR dependence from all code)
        assert os.path.exists(image_dir), \
            "Specify either full path or path relative to current working directory for image_dir"
        image_helper = Image(image_dir, rig_setup_file=setup_file)
        display_size = image_helper.display_size(1024)
        num_cam = image_helper.num_cam()
        setupInfo = image_helper.setup_info()
        sensorInfo = setupInfo.SensorInfo
        img_size = (setupInfo.SensorInfo.width, setupInfo.SensorInfo.height)
        # read the bin files
        all_files_read = False
        while not all_files_read:
            for cam_idx in range(image_helper.num_cam()):
                img, gray = image_helper.read_image_file(cam_idx, frame_index, scale_to_8bit=True)
                if img is None:
                    all_files_read = True
                    break
                img_stack.append(img)
                filename = image_dir + "/img_{}.png".format(module_name[cam_idx])
                cv2.imwrite(filename, img)
                if show_image:
                    img2 = cv2.resize(img, None, fx=display_size, fy=display_size)
                    cv2.imshow("{}".format(setupInfo.RigInfo.module_name[cam_idx]), img2)
                    cv2.waitKey(500)
            all_files_read = True  # need to read only one set

    else:
        for cam_idx in range(num_cam):
            filename = image_dir + "/img_{}_0.jpg".format(module_name[cam_idx])
            print("Image file for camera {}:".format(cam_idx), filename)
            try:
                img = cv2.imread(filename)
            except:
                img = None

            if img is None:
                print("Image file {} does not Exist!!".format(filename))
                exit(1)
            #if img_size != img.shape[0:2]:
            #    print("Image size mismatch Please Check!!")
            #    print(img_size)
            #    print(img.shape)
            #    exit(1);

            img_stack.append(img)
    return img_stack


# get new optimal camera matrix for each of the cameras and undistort image and the correspondence sets
def GetOptimalNewCamMatricUndistort( img_stack, exr_np, img_size, read_correspondence_type, do_not_undistort=False, num_cam=4):
    K_new = []
    K1_new = np.empty(K[0].shape)
    undist_img_stack = []
    undist_points_stack = []
    corner_detected = []
    for cam_idx in range(num_cam):
        K1_new, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix=K[cam_idx], distCoeffs = D[cam_idx],
                                                    imageSize=img_size, alpha=0,  newImgSize=img_size)
        K_new.append(K1_new)
        undist_img = cv2.undistort(img_stack[cam_idx], K[cam_idx], D[cam_idx], newCameraMatrix=K_new[cam_idx])
        if do_not_undistort:
            undist_img_stack.append(img_stack[cam_idx])
        else:
            undist_img_stack.append(undist_img)

        # convert to float64 else errors out
        if read_correspondence_type is 'exr':   # YH not sure
            corners = np.ndarray.astype(exr_np[cam_idx, :, 0:2], dtype=np.float64)
        else:
            corners = np.ndarray.astype(np.array(exr_np[cam_idx]), dtype=np.float64)
            corners = corners.reshape(-1,2)

        if(corners[0,0] == -1):
            corner_detected.append(False)
        else:
            corner_detected.append(True)

        undist_points = cv2.undistortPoints(src=corners, cameraMatrix=K[cam_idx], distCoeffs=D[cam_idx], P=K_new[cam_idx])
        if do_not_undistort:
            corners = corners.reshape((836,1,-1)) # YH TODO: FIX magic number 836
            undist_points_stack.append(corners)
        else:
            undist_points_stack.append(undist_points)

    return K_new, corner_detected, undist_img_stack, undist_points_stack, roi


def GetUpsampledPoint(point, factor, size, shift):
    upsampled = [size[0] // 2 + point[0] - shift[0], size[1] // 2 + point[1] - shift[1]]
    return [upsampled[0] * factor, upsampled[1] * factor]


def GetUpsampledLine(line, factor, size, shift):
    if abs(line[1]) > abs(line[2]):
        p0 = [0, -line[2] / line[1]]
        p1 = [size[1], -(line[2] + line[0] * size[1]) / line[1]]
    else:
        p0 = [-line[2] / line[0], 0]
        p1 = [-(line[2] + line[1] * size[0]) / line[0], size[0]]
    up0 = GetUpsampledPoint(p0, factor, size, shift)
    up1 = GetUpsampledPoint(p1, factor, size, shift)
    return [np.cross([up0[0], up0[1], 1], [up1[0], up1[1], 1])]


# Select the patches around the correspondence points, zoom into it based on upsample factor, mark the point with\
# circle and compute and draw epilines.
def SelPatchZoomMarkPointDrawEpilines(undist_img_stack, undist_points_stack, point_index, K_new, T, corner_detected,\
                                      patch_size, upsample_factor, module_name, draw_epilines=True, num_cam=4, show_image=False):

    F_stack = []
    image_patches = []
    estimated_depth = []
    estimated_disparity = []
    # for each baseline
    for cam_idx in range(1,num_cam):

        # select the ROI of size 33x33 around the center point for each image
        corner_point_ref = undist_points_stack[0][point_index]
        y_ref = np.int32(np.round(corner_point_ref[0,1]))
        x_ref = np.int32(np.round(corner_point_ref[0,0]))
        roi_patch_ref = ( x_ref - patch_size[0]//2, y_ref - patch_size[1]//2, x_ref+patch_size[0]//2 + 1, y_ref+patch_size[1]//2 + 1)

        corner_point_src = undist_points_stack[cam_idx][point_index]

        if corner_detected[cam_idx]:

            if draw_epilines:
                estimated_depth.append(ComputeRefDepthGivenCorrespondingPnt( R[cam_idx], T[cam_idx], K_new[cam_idx],\
                                                                         K_new[0], corner_point_src, corner_point_ref))
                estimated_disparity.append( K_new[0][0, 0] * np.linalg.norm(T[1]) /estimated_depth[cam_idx-1] )

            y_src = np.int32(np.round(corner_point_src[0,1]))
            x_src = np.int32(np.round(corner_point_src[0,0]))
            roi_patch_src = ( x_src - patch_size[0]//2, y_src - patch_size[1]//2, x_src+patch_size[0]//2 + 1, y_src+patch_size[1]//2 + 1)

            if draw_epilines:
                # compute fundamental matrix between each pair
                F = Stereo.compute_fundamental_matrix(K_new[0], K_new[cam_idx], R[cam_idx], T[cam_idx])
                F_stack.append(F)

                # covert correspondence points to Homogeneous co-ordinates
                pts_ref = cv2.convertPointsToHomogeneous(corner_point_ref)
                pts_src = cv2.convertPointsToHomogeneous(corner_point_src)

                # compute the epipolar lines
                line_src = cv2.computeCorrespondEpilines(pts_ref, 1, F).reshape(-1, 3)
                line_ref = cv2.computeCorrespondEpilines(pts_src, 2, F).reshape(-1, 3)

            a1_gamma = np.clip(adjust_gamma(undist_img_stack[0].copy(), 2.2),0,255)
            src_im_gamma = np.clip(adjust_gamma(undist_img_stack[cam_idx].copy(), 2.2),0,255)

            if show_image:
                img_src, img1 = Stereo.drawlinesnew(src_im_gamma.copy(), a1_gamma.copy(), line_src, pts_src, pts_ref, circle_thickness)
                img3 = cv2.resize(img_src, None, fx=1 / 2, fy=1 / 2)
                cv2.imshow("{}_full".format(module_name[cam_idx]), img3)
                cv2.waitKey(500)
                img_roi_src = img_src[roi_patch_src[1]:roi_patch_src[3], roi_patch_src[0]:roi_patch_src[2]]
                img3 = cv2.resize(img_roi_src, None, fx=upsample_factor, fy=upsample_factor, interpolation=cv2.INTER_NEAREST)
                cv2.imshow("{}".format(module_name[cam_idx]), img3)

            # select roi
            img_roi_src_nocircles = src_im_gamma[roi_patch_src[1]:roi_patch_src[3], roi_patch_src[0]:roi_patch_src[2]]
            img_roi_ref_nocircles = a1_gamma[roi_patch_ref[1]:roi_patch_ref[3], roi_patch_ref[0]:roi_patch_ref[2]]

            # take the ref and source patches, adjust the corner point locations on the patches, upsample the patches and then draw circle and epilines
            # img_src has the epiline and is the ref image ( may need to remove circle and add again at a better location0
            # img_roi_src is the patch that needs to be resized
            img_roi_src_zoom = cv2.resize(img_roi_src_nocircles, None, fx=upsample_factor, fy=upsample_factor, interpolation=cv2.INTER_NEAREST)
            img_roi_ref_zoom = cv2.resize(img_roi_ref_nocircles, None, fx=upsample_factor, fy=upsample_factor, interpolation=cv2.INTER_NEAREST)

            patch_point_ref = (patch_size[0]//2 + corner_point_ref[0,0] - x_ref, patch_size[1]//2 + corner_point_ref[0,1] - y_ref)
            patch_point_ref_zoom = (patch_point_ref[0] * upsample_factor, patch_point_ref[1] * upsample_factor)
            img_roi_ref_zoom = cv2.circle(img_roi_ref_zoom, (np.int32(patch_point_ref_zoom[0]),
                                                             np.int32(patch_point_ref_zoom[1])), 5, (0,0,255), -1)
            if show_image:
                cv2.imshow("A1_{}_nocircles".format(module_name[cam_idx]), img_roi_ref_zoom)

            if cam_idx == 1:
                image_patches.append(cv2.cvtColor(img_roi_ref_zoom, cv2.COLOR_BGR2RGB))


            patch_point_src = (patch_size[0]//2 + corner_point_src[0,0] - x_src, patch_size[1]//2 + corner_point_src[0,1] - y_src)
            patch_point_src_zoom = (patch_point_src[0] * upsample_factor, patch_point_src[1] * upsample_factor)
            img_roi_src_zoom = cv2.circle(img_roi_src_zoom, (np.int32(patch_point_src_zoom[0]),
                                                             np.int32(patch_point_src_zoom[1])), 5, (0,0,255), -1)
            if draw_epilines:
                img_roi_src_zoom, _ = Stereo.drawlinesnew(img_roi_src_zoom, img_roi_ref_zoom,
                                                          GetUpsampledLine(line_src[0], upsample_factor, patch_size, corner_point_src[0]),
                                                          pts_src, pts_ref, circle_thickness)
            if show_image:
                cv2.imshow("{}_nocircles".format(module_name[cam_idx]), img_roi_src_zoom)

            image_patches.append(cv2.cvtColor(img_roi_src_zoom, cv2.COLOR_BGR2RGB))
    return  image_patches, estimated_depth, estimated_disparity

####################
# Input Parameters
####################

parser = argparse.ArgumentParser(description="Auto Epiviz")
parser.add_argument('--image_dir', default='/Users/amaharshi/debug/autoviz/5_3_raw')
parser.add_argument('--setup_file', default='/Users/amaharshi/debug/autoviz/5_3_raw/setup.py')
parser.add_argument('--cal_dir', default='/Users/amaharshi/debug/z_k_3/zero_k_3_factory/3_5_zero_k_3_save_rt_cv_init_ref_fppd/cv_bypassed')
parser.add_argument('--correspondence_file', default='/Users/amaharshi/debug/autoviz/Pine_3_5_corr/Pine_3_5_corr5.npy')
parser.add_argument('--upsample', type=int, default=15)
parser.add_argument('--patch_size', type=int, default=33)
parser.add_argument('--point_idx', type=int, default=0)
parser.add_argument('--frame_num', type=int, default=0)
parser.add_argument('--image_width', type=int, default=3208)
parser.add_argument('--image_height', type=int, default=2200)
parser.add_argument('--read_correspondence_type', default='max_reproj_error', help='max_reproj_error:\
                    compute from rerrorx.fst, npy: use correspondence npy file, exr: read exr files and bin images\
                     and will need frame_num and point_index ')

# Open a figure to avoid #cv2.imshow crash
plt.figure(1)

args_g, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

print("")
print("Command line options:")
print("  {}".format(args_g))
print("")

image_helper = Image(args_g.image_dir, rig_setup_file=args_g.setup_file)
setupInfo = image_helper.setup_info()
num_cam = len(setupInfo.RigInfo.cam_position_m)
nx = setupInfo.ChartInfo.nx
ny = setupInfo.ChartInfo.ny
module_name  = ["A1", "A2", "A3", "A4"]
circle_thickness = 3
do_not_distort_image = False

show_image = False
upsample_factor_g = args_g.upsample
patch_size_g = (args_g.patch_size, args_g.patch_size)
img_size_g = (args_g.image_height, args_g.image_width)

# read the calibration files
K = []
R = []
T = []
D = []

filename = args_g.cal_dir + "/light_header.json"
for cam_idx in range(num_cam):
    [K1, R1, T1, D1] = parse_light_header( read_lightheader(filename), cam_idx)
    K.append(K1)
    R.append(R1)
    T.append(T1)
    D.append(D1)


#  Get the correspondence points and the frame number and point index
[read_image_bin, frame_index_g, point_index_g, exr_np_g, max_error, reproj_error] = GetCorrespondencePoints( \
    args_g.cal_dir, args_g.read_correspondence_type, args_g.correspondence_file, args_g.frame_num, args_g.point_idx )

# read the images corresponding to the frame number
img_stack_g = ReadImageAllCams( args_g.image_dir, args_g.setup_file, read_image_bin, frame_index_g)

# get new optimal camera matrix for each of the cameras and undistort image and the correspondence sets
[K_new_g, corner_detected_g, undist_img_stack_g, undist_points_stack_g, roi_g] = GetOptimalNewCamMatricUndistort(\
         img_stack_g, exr_np_g, img_size_g, args_g.read_correspondence_type,do_not_distort_image, num_cam)

[image_patches, estimated_depth, estimated_disparity] = SelPatchZoomMarkPointDrawEpilines( undist_img_stack_g, \
         undist_points_stack_g, point_index_g, K_new_g, T,  corner_detected_g, \
         patch_size_g, upsample_factor_g, module_name, not(do_not_distort_image==True), num_cam)

fig, axes = plt.subplots(2, 2, figsize=(10,10))
if  do_not_distort_image:
    for cam_idx in range(num_cam):
        plot_camera(axes[cam_idx//2, cam_idx%2], "{}: reproj_error = ({:.2f}, {:.2f})".format \
            (module_name[cam_idx], reproj_error[cam_idx,0], reproj_error[cam_idx,1]), image_patches[cam_idx])

    plt.suptitle("Correspondences for point ({:.2f}, {:.2f}) in reference at corner point ({}), frame idx = {}, error={:.2f}".format \
                     (undist_points_stack_g[0][point_index_g, 0, 0], undist_points_stack_g[0][point_index_g, 0, 1], \
                      np.unravel_index(point_index_g,(ny,nx)), frame_index_g, max_error))

else:
    plot_camera(axes[0, 0], "A1: reproj_error = ({:.2f}, {:.2f})".format(reproj_error[0,0], reproj_error[0,1]), image_patches[0])
    for cam_idx in range(1, num_cam):
        plot_camera(axes[cam_idx//2, cam_idx%2], "{}: reproj_error = ({:.2f}, {:.2f})\n depth:{:.4f}m\ndisparity:{:.2f}".format \
            (module_name[cam_idx], reproj_error[cam_idx,0], reproj_error[cam_idx,1], estimated_depth[cam_idx-1]/1000, estimated_disparity[cam_idx-1]), image_patches[cam_idx])

    plt.suptitle("Correspondences for point ({:.2f}, {:.2f}) in reference at corner point ({}), frame idx = {}, error={:.2f}".format\
         (undist_points_stack_g[0][point_index_g, 0, 0], undist_points_stack_g[0][point_index_g, 0, 1],\
          np.unravel_index(point_index_g,(ny,nx)), frame_index_g, max_error))

top_dir_name = os.path.basename(os.path.dirname(os.path.dirname(args_g.cal_dir)))
if args_g.read_correspondence_type == "max_reproj_error":

    filename = args_g.image_dir + "/{}_{}_corres_{}_{}.png".format(
        top_dir_name, os.path.basename(args_g.cal_dir), frame_index_g, point_index_g)
else:
    filename = args_g.image_dir + "/{}_{}_corres_{}.png".format(
        top_dir_name, os.path.basename(args_g.cal_dir),os.path.basename\
            (args_g.correspondence_file).split('/')[-1].replace('.npy',''))

print("Correspondence image output to file: ", filename)
plt.savefig(filename)


####################
