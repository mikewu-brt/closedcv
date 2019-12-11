#  Copyright (c) 2019, The LightCo
#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without
#  modification, are strictly prohibited without prior permission of
#  The LightCo.
#
#  @author  cstanski
#  @version V1.0.0
#  @date    Dec 2019
#  @brief
#

import cv2
import argparse
from libs.Image import *
from libs.CalibrationInfo import *
from libs.LensDistortion import *
import matplotlib as matplot
matplot.use('TkAgg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Compute Distortion Map (ASIC)")
parser.add_argument('--cal_dir', default='calibration_nov21_25mm')
parser.add_argument('--image_dir', default='outside_nov21_25mm_f2_8_center')
#parser.add_argument('--cal_dir', default='calibration_nov21_25mm_distortion_map_r1_16')
parser.add_argument('--decimate', type=int, default=16)

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

# Read image file as raw
image_helper = Image(args.image_dir)
img, _ = image_helper.read_image_file(0, 0, scale_to_8bit=False, raw_output=True)

h, w = img.shape
img_quad = np.empty((h // 2, w // 2, 4))
img_quad[:, :, 0] = img[::2, ::2]
img_quad[:, :, 1] = img[::2, 1::2]
img_quad[:, :, 2] = img[1::2, ::2]
img_quad[:, :, 3] = img[1::2, 1::2]

plt.figure(10).clear()
plt.imshow(img_quad[:, :, 0])
plt.title("Original Image")

# Distort image
cal_info = CalibrationInfo(args.cal_dir, "calibration.json")
K = cal_info.K(0)
D = -20.0 * cal_info.D(0)

distort = LensDistortion()
distort.set_radial_distortion_map(K=K, D=D, size=(img.shape[1], img.shape[0]))
distort.set_distortion_map(distortion_map=distort.distortion_map()[::2, ::2, :] / 2.0)
img_dist = np.empty(img_quad.shape)
img_dist[:, :, 0] = distort.correct_distortion(img_quad[:, :, 0], interpolation=cv2.INTER_LANCZOS4)
img_dist[:, :, 1] = distort.correct_distortion(img_quad[:, :, 1], interpolation=cv2.INTER_LANCZOS4)
img_dist[:, :, 2] = distort.correct_distortion(img_quad[:, :, 2], interpolation=cv2.INTER_LANCZOS4)
img_dist[:, :, 3] = distort.correct_distortion(img_quad[:, :, 3], interpolation=cv2.INTER_LANCZOS4)

img_distorted = np.empty(img.shape)
img_distorted[::2, ::2] = img_dist[:, :, 0]
img_distorted[::2, 1::2] = img_dist[:, :, 1]
img_distorted[1::2, ::2] = img_dist[:, :, 2]
img_distorted[1::2, 1::2] = img_dist[:, :, 3]

plt.figure(11).clear()
plt.imshow(img_distorted)
plt.title("Distorted Image - Input")


# Correct distortion
decimate = 16
#D = -1.0 * D
undistort = LensDistortion()
# Create an inverse distortion map from radial
undistort.set_radial_distortion_map(K=K, D=D, size=(img.shape[1], img.shape[0]))
undistort.set_distortion_map(distortion_map=-1.0*undistort.distortion_map())

# decimate for asic - For comparison, don't offset by 1/2 pixel.
asic_dist_map = undistort.asic_distortion_map(pixel_quad_decimate=decimate, pixel_offset=False)

# Reconstruct map from the decimated map
undistort.set_asic_distortion_map(asic_dist_map, (img.shape[1], img.shape[0]), pixel_offset=False)

# Decimate by a factor of 1/2, 1/2 to represent "pixel quads"
undistort.set_distortion_map(undistort.distortion_map()[::2, ::2] / 2.0)

img_quad_undist = np.empty(img_quad.shape)
img_quad_undist[:, :, 0] = undistort.correct_distortion(img_dist[:, :, 0], interpolation=cv2.INTER_LANCZOS4)
img_quad_undist[:, :, 1] = undistort.correct_distortion(img_dist[:, :, 1], interpolation=cv2.INTER_LANCZOS4)
img_quad_undist[:, :, 2] = undistort.correct_distortion(img_dist[:, :, 2], interpolation=cv2.INTER_LANCZOS4)
img_quad_undist[:, :, 3] = undistort.correct_distortion(img_dist[:, :, 3], interpolation=cv2.INTER_LANCZOS4)

img_undistorted = np.empty(img.shape)
img_undistorted[::2, ::2] = img_quad_undist[:, :, 0]
img_undistorted[::2, 1::2] = img_quad_undist[:, :, 1]
img_undistorted[1::2, ::2] = img_quad_undist[:, :, 2]
img_undistorted[1::2, 1::2] = img_quad_undist[:, :, 3]

plt.figure(12).clear()
plt.imshow(img_undistorted)
plt.title("Undistorted Image - Output")


# Generate files for the ASIC team
image_helper.save_np_file("asic_map", asic_dist_map)
image_helper.save_np_file("input", img_distorted)
image_helper.save_np_file("output", img_undistorted)
