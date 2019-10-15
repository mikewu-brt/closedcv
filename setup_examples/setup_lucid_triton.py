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
 *
"""
#  Copyright (c) 2019, The LightCo
#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without
#  modification, are strictly prohibited without prior permission of
#  The LightCo.
#
#  @author  cstanski
#  @version V1.0.0
#  @date    Sep 2019
#  @brief
#  Stereo calibration test script

import numpy as np
import cv2


class SensorInfo:
    type = "SENSOR_IMX428"
    pixel_size_um = 4.5
    width = 3208
    height = 2200
    bits_per_pixel = 12


class LensInfo:
    manufacturer = "Computar"
    model = "V2528_MPY"
    type = "LENS_{}_{}".format(manufacturer.upper(), model.upper())
    fl_mm = 25.0
    focus_distance_mm = 20000.0


class RigInfo:
    rig_manufacturer = "Light Labs - NJ"
    camera_module_manufacturer = "Lucid"
    camera_module_model = "TRI071S"
    camera_module_serial_number = ["SN193500101", "SN193500102", "SN192900007"]
    input_image_filename = np.array(["Image_{}_f{{}}.raw".format(camera_module_serial_number[0].lower()),
                                     "Image_{}_f{{}}.raw".format(camera_module_serial_number[1].lower()),
                                     "Image_{}_f{{}}.raw".format(camera_module_serial_number[2].lower())])
    cam_position_m = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.38, 0.0, 0.0]])
    module_name = ["A1", "A2", "A3"]
    left_justified = False
    packed = True
    cv2_color_conversion = cv2.COLOR_BayerBG2BGR


class ChartInfo:
    name = "Checkerboard"
    nx = 31
    ny = 24
    size_mm = 40.0


# The following class is used for computing the focal length of each camera on the rig by using the knowledge of object distance and its image
class CalibMagInfo:
    fixed_focal_length = np.array([5578.5, 5574, 5573]) # in pixels

    # need to provide two filename arrays and 2 distance measurements for compute_magnification script
    input_image_filename = np.array([["Image_sn193500101_f9.raw",  "Image_sn193500102_f9.raw", "Image_sn192900007_f9.raw"],
                                     ["Image_sn193500101_f2.raw",  "Image_sn193500102_f2.raw", "Image_sn192900007_f2.raw"]])
    obj_dist = np.array([[4582, 4585, 4584],
                         [3426, 3489, 3426]])# - measured distances to object
    delta = 4 # region to serach around the center
