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
    type = "IMX386"
    pixel_size_um = 1.25
    width = 4032
    height = 3016


class LensInfo:
    fl_mm = 3.95


class RigInfo:
    module_name = "Libra"
    cam_position_m = np.array([[0, 0, 0], [1.027, 0, 0]])
    image_filename = np.array(["left{}_0.npy", "right{}_0.npy"])


class ChartInfo:
    name = "Checkerboard Old"
    nx = 17
    ny = 11
    size_mm = 280.0 / 13.0
