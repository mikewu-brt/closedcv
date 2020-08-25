#  Copyright (c) 2020, The LightCo
#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without
#  modification, are strictly prohibited without prior permission of
#  The LightCo.
#
#  @author  cstanski
#  @version V1.0.0
#  @date    Aug 2020
#  @brief
#

from libs.parse_calib_data import *
from libs.CalibrationInfo import *
from libs.LensDistortion import *
from libs.Stereo import *
import os
import matplotlib as matplot
import cv2
matplot.use('TkAgg')
import matplotlib.pyplot as plt
import lt_protobuf.light_header.camera_id_pb2 as camera_id_pb2

plt.figure(1)

image_dir = "fpga_test"
cal_dir = "/Users/cstanski/Documents/workspace/rig_calibration/rig_sp_04_00_a/latest"
pixel_quad_decimate = 16
width = 3208
height = 2200
image_idx = 200

decimate = 2 * pixel_quad_decimate
roi = (math.ceil(width / decimate) + 1,  math.ceil(height / decimate) + 1)

# Calibration Info
hw_info_filename = os.path.join(cal_dir, "hwinfo.json")
hw_info = parse_hwinfo(read_lightheader(hw_info_filename))

photometric_cal_filename = os.path.join(cal_dir, "photometric_calib.json")
geometric_cal_filename = os.path.join(cal_dir, "geometric_calib.json")

K = []
R = []
T = []
D = []
V = []
MAP = []
lens = LensDistortion()
for cam_idx in range(len(hw_info.camera)):
    V1 = parse_photometric(read_lightheader(photometric_cal_filename), cam_idx)
    vig = lens.convert_json_vignetting_fpga(roi, V1)
    vig.astype(np.float32).tofile("vignetting_{}.bin".format(camera_id_pb2.CameraID.Name(hw_info.camera[cam_idx].id)))
    lens.set_asic_vignetting(vig, (width, height), pixel_offset=False)
    V.append(lens.vignetting())

    [K1, R1, T1, D1] = parse_geometric(read_lightheader(geometric_cal_filename), cam_idx)
    K.append(K1)
    R.append(R1)
    T.append(T1)
    D.append(D1)

    lens.set_radial_distortion_map(K1, D1, (width, height))
    lens.asic_distortion_map().astype(np.float32).tofile("distortion_{}.bin".format(camera_id_pb2.CameraID.Name(hw_info.camera[cam_idx].id)))
    MAP.append(lens.distortion_map())

cal_info = CalibrationInfo(image_dir, K=K, R=R, T=T, D=D, V=V, MAP=MAP)
stereo = Stereo(image_dir, img_shape=(width, height), cal_info=cal_info)

# Image files
image_helper = Image(image_dir)

a1_idx = camera_id_pb2.CameraID.Value('A1')
a2_idx = camera_id_pb2.CameraID.Value('A2')

img_a1, _ = image_helper.read_image_file(a1_idx, image_idx)
img_a2, _ = image_helper.read_image_file(a2_idx, image_idx)

# Rectify images
img_a1_rect, img_a2_rect, p1, p2, r1, r2 = stereo.rectify_views(img_a1, img_a2, a2_idx)

# Overlay images
rect_overlay = cv2.addWeighted(img_a1_rect, 0.5, img_a2_rect, 0.5, 0.0)

plt.figure(1).clear()
plt.imshow(rect_overlay)
plt.title("Rectified Overlay")

cv2.imshow("A1", img_a1_rect)
cv2.imshow("A2", img_a2_rect)

plt.figure(10).clear()
plt.imshow(img_a1)
plt.title("A1")


# Compute transformation matrices
m1i, m2i = stereo.asic_rectification_transforms(a2_idx, True)
m1, m2 = stereo.rectification_transforms(a2_idx)

img1 = cv2.warpPerspective(img_a1, m1, (width, height))
img2 = cv2.warpPerspective(img_a2, m2, (width, height))

img1_img2 = cv2.addWeighted(img1, 0.5, img2, 0.5, 0.0)
plt.figure(11).clear()
plt.imshow(img1_img2)
plt.title("Img1_Img2")


