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
    type = "IMX390"
    pixel_size_um = 3.00
    width = 1936
    height = 1100


class LensInfo:
    fl_mm = 17.0


class RigInfo:
    module_name = "4-CamSetup"
    cam_position_m = np.array([[0, 0, 0], [-0.5, 0, 0], [0.5,0,0], [0,0.1,0]])
    input_image_filename = np.array(["{}_Center_{}.raw", "{}_Left_{}.raw", "{}_Right_{}.raw", "{}_Top_{}.raw"])
    image_filename = np.array(["A1_{}.npy", "A2_{}.npy", "A3_{}.npy", "A4_{}.npy"])
    scale = 4.0
    #num_cam = 4
    # the following two parameters are temporary and used only for RWC's way of creating the capture filenames; with non-continous file numbering and multiple captures of same view.
    num_npy_files = 196  
    num_capture_sets = 2


class ChartInfo:
    name = "Checkerboard"
    #nx = 29
    nx = 28
    ny = 17
    size_mm = 40.0
