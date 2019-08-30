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


class SensorInfo:
    type = "IMX265"
    pixel_size_um = 3.45
    width = 2048
    height = 1536
    num_adc_bits = 10


class LensInfo:
    fl_mm = 12.0


class RigInfo:
    module_name = "DKF33UX265"
    cam_position_m = np.array([[0, 0, 0], [1.116, 0, 0]])
    input_image_filename = np.array(["cap_rit_{}.rggb", "cap_lft_{}.rggb"])
    image_filename = np.array(["left{}.npy", "right{}.npy"])
    left_justified = True


class ChartInfo:
    name = "Checkerboard"
    nx = 31
    ny = 24
    size_mm = 40.0
