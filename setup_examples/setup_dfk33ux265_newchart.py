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
import numpy as np
import cv2


class SensorInfo:
    type = "SENSOR_IMX265"
    pixel_size_um = 3.45
    width = 2048
    height = 1536
    bits_per_pixel = 10


class LensInfo:
    manufacturer = "Imaging_Source"
    model = "TPL_1220"
    type = "LENS_{}_{}".format(manufacturer.upper(), model.upper())
    fl_mm = 12.0
    focus_distance_mm = 5000.0


class RigInfo:
    rig_manufacturer = "Light Labs - NJ"
    camera_module_manufacturer = "Imagining Source"
    camera_module_model = "DKF33UX265"
    camera_module_serial_number = ["SN1?", "SN2?"]
    input_image_filename = np.array(["cap_rit_{}.rggb", "cap_lft_{}.rggb"])
    cam_position_m = np.array([[0, 0, 0], [1.116, 0, 0]])
    module_name = ["A1", "A2"]
    left_justified = True
    packed = False
    cv2_color_conversion = cv2.COLOR_BayerBG2BGR


class ChartInfo:
    name = "Checkerboard"
    nx = 31
    ny = 24
    size_mm = 40.0
