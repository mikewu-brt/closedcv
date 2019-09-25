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
    type = "IMX428"
    pixel_size_um = 4.5
    width = 3208
    height = 2200
    bits_per_pixel = 12


class LensInfo:
    fl_mm = 25.0


class RigInfo:
    module_name = "Lucid Triton - 25mm"
    cam_position_m = np.array([[0, 0, 0], [0.5, 0, 0]])
    input_image_filename = np.array(["Image_sn193800001_f{}.raw", "Image_sn193800001_f{}.raw"])
    module_name = ["A1", "A2"]
    left_justified = False
    packed = False
    cv2_color_conversion = cv2.COLOR_BayerRG2BGR


class ChartInfo:
    name = "Checkerboard"
    nx = 31
    ny = 24
    size_mm = 40.0
