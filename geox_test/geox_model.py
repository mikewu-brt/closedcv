#  Copyright (c) 2020, The LightCo
#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without
#  modification, are strictly prohibited without prior permission of
#  The LightCo.
#
#  @author  cstanski
#  @version V1.0.0
#  @date    May 2020
#  @brief
#
#
#  @author  cstanski
#  @version V1.0.0
#  @date    May 2020
#  @brief
#

import numpy as np
import argparse
import os
import commandfile_pb2
import cv2
from libs.LensDistortion import *
from google.protobuf import json_format
import matplotlib as matplot
matplot.use('TkAgg')
import matplotlib.pyplot as plt






parser = argparse.ArgumentParser(description="GEOx OpenCV model")
parser.add_argument('--test_name')
parser.add_argument('--json')
parser.add_argument('--base_dir', default=".")

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

print(args.test_name)
print(args.json)
print(args.base_dir)

json_filename = os.path.join(args.base_dir, args.json)

json_str = open(json_filename, 'r').read()
tests = commandfile_pb2.TestRun()
json_format.Parse(json_str, tests)

# Find requested test case
test_case = []
for case in tests.test_case_run:
    if case.test_name == args.test_name:
        test_case = case
        break

if test_case.test_name != args.test_name:
    print("Could not find test case: {} in: {}".format(args.test_case, json_filename))
    exit(-1)

# Get image size (in PQs)
height_raw = test_case.test_setup[0].image_command[0].image_files[0].height
height_pq = height_raw // 2
width_raw = test_case.test_setup[0].image_command[0].image_files[0].width
width_pq = width_raw // 2

# Create a lens model and populate with ASIC distortion / vignetting map
lens_ref = LensDistortion()
lens_src = LensDistortion()

# Initialize maps to unity
dist_map = np.zeros((height_pq, width_pq, 2))
lens_ref.set_distortion_map(dist_map)
lens_src.set_distortion_map(dist_map)

vig_map = np.ones((height_pq, width_pq, 3))
lens_ref.set_vignetting(vig_map)
lens_src.set_vignetting(vig_map)

# Search test_case for specific distortion and vignetting maps
for distortion in test_case.test_setup[0].distortion_map:
    map = np.fromfile(os.path.join(args.base_dir, distortion.file_path), np.float32).reshape((distortion.height, distortion.width, 2))
    # correct for PQs
    map = map / 2
    if distortion.type == commandfile_pb2.REFERENCE_IMAGE:
        print("Distortion Map - Reference")
        lens_ref.set_asic_distortion_map(map, (width_pq, height_pq))
    else:
        print("Distortion Map - Source")
        lens_src.set_asic_distortion_map(map, (width_pq, height_pq))

for vignetting in test_case.test_setup[0].vignetting_map:
    map = np.fromfile(os.path.join(args.base_dir, vignetting.file_path), np.float32).reshape((vignetting.height, vignetting.width))
    if vignetting.type == commandfile_pb2.REFERENCE_IMAGE:
        print("Vignetting Map - Reference")
        lens_ref.set_asic_vignetting(map, (width_pq, height_pq))
    else:
        print("Vignetting Map - Source")
        lens_src.set_asic_vignetting(map, (width_pq, height_pq))


# Read transform
undist_transform = np.empty((3, 3))
for row in range(3):
    for col in range(3):
        element = compile("test_case.test_setup[0].image_command[0].undistort_params.transform.x{}{}".format(row, col), "", 'eval')
        undist_transform[row, col] = eval(element)

print(undist_transform)

# Read Images
image_ref = []
image_src = []
for im in test_case.test_setup[0].image_command[0].image_files:
    image = np.fromfile(os.path.join(args.base_dir, im.file_path), np.int16).reshape((height_raw, width_raw))
    image_pq = np.empty((height_pq, width_pq, 4))
    image_pq[:, :, 0] = image[::2, ::2]
    image_pq[:, :, 1] = image[::2, 1::2]
    image_pq[:, :, 2] = image[1::2, ::2]
    image_pq[:, :, 3] = image[1::2, 1::2]
    if im.type == commandfile_pb2.REFERENCE_IMAGE:
        image_ref = image_pq.copy()
    else:
        image_src = image_pq.copy()

processed_ref = []
processed_src = []
for im in test_case.test_setup[0].image_command[0].processed_image_files:
    image = np.fromfile(os.path.join(args.base_dir, im.file_path), np.uint16).reshape((height_pq, width_pq, 4))
    if im.type == commandfile_pb2.REFERENCE_IMAGE:
        processed_ref = image.copy()
    else:
        processed_src = image.copy()

# FIXME(Chuck) - Don't assume out was based on reference
processed_pq = processed_ref.copy()

# check command flags to undistort
undist_image = None
undist_lens = None
if test_case.test_setup[0].image_command[0].process_image_req[0] == commandfile_pb2.REF_UNDISTORT_REQUEST:
    print("Undist image Reference")
    undist_image = image_ref
    undist_lens = lens_ref
elif test_case.test_setup[0].image_command[0].process_image_req[0] == commandfile_pb2.SRC_UNDISTORT_REQUEST:
    print("Undist image Source")
    undist_image = image_src
    undist_lens = lens_src

# Geox Model
if (undist_image is None) or (undist_lens is None):
    print("Error, nothing to undistort")
    exit(-1)


# Warp perspective
model_out = undist_image.copy()
model_out = cv2.warpPerspective(model_out, undist_transform, (width_pq, height_pq), flags=cv2.INTER_LANCZOS4)

# Undistort - In PQ domain
for color in range(4):
    model_out[:, :, color] = undist_lens.correct_distortion(model_out[:, :, color], interpolation=cv2.INTER_LANCZOS4)

# Vignetting
for color in range(4):
    model_out[:, :, color] = undist_lens.correct_vignetting(model_out[:, :, color], alpha=1.0)

# Black Level
model_out -= test_case.test_setup[0].config.black_level

# Channel Gains
gains = test_case.test_setup[0].config.channel_gains
model_out = np.multiply(model_out, gains)

# Adjust gain
model_out = model_out * (65535.0 / 4096.0)


# Convert to image pixels and plot
reference = np.empty((height_raw, width_raw))
reference[::2, ::2] = image_ref[:, :, 0]
reference[::2, 1::2] = image_ref[:, :, 1]
reference[1::2, ::2] = image_ref[:, :, 2]
reference[1::2, 1::2] = image_ref[:, :, 3]

processed = np.empty((height_raw, width_raw))
processed[::2, ::2] = processed_pq[:, :, 0]
processed[::2, 1::2] = processed_pq[:, :, 1]
processed[1::2, ::2] = processed_pq[:, :, 2]
processed[1::2, 1::2] = processed_pq[:, :, 3]

expected = np.empty((height_raw, width_raw))
expected[::2, ::2] = model_out[:, :, 0]
expected[::2, 1::2] = model_out[:, :, 1]
expected[1::2, ::2] = model_out[:, :, 2]
expected[1::2, 1::2] = model_out[:, :, 3]

difference = processed - expected
difference[0:5, :] = 0.0
difference[:, 0:4] = 0.0


plt.figure(1).clear()
plt.imshow(expected)
plt.colorbar()
plt.title('Expected')

plt.figure(2).clear()
plt.imshow(processed)
plt.colorbar()
plt.title('HW Output')

plt.figure(3).clear()
plt.imshow(difference)
plt.colorbar()
plt.title('HW Output - Expected')

print("0")

