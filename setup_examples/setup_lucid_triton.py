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
    input_image_filename = np.array(["Image_sn193800001_f{}.raw", "Image_sn193800001_f{}.raw"])
    fixed_focal_length = np.array([5580.4, 5573.9, 5575]) # in pixels
    obj_dist = np.array([3426, 3489]) # distance to object  in mm
    delta = 4 # region to serach around the center
