#  Copyright (c) 2019, The LightCo
#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without
#  modification, are strictly prohibited without prior permission of
#  The LightCo.
#
#  @author  cstanski
#  @version V1.0.0
#  @date    Oct 2019
#  @brief
#

import numpy as np
import scipy
from scipy import interpolate


class Distortion:

    @staticmethod
    def compute_distortion_map(image_shape, corners, distortion_error, step=1):
        dist_map = np.zeros((image_shape[0], image_shape[1], 2))
        grid_x, grid_y = np.mgrid[0:image_shape[1]:step, 0:image_shape[0]:step]
        dist_map[::step, ::step, 0] = scipy.interpolate.griddata((corners[:, 0], corners[:, 1]), distortion_error[:, 0],
                                                                 (grid_x, grid_y), method='linear', fill_value=-999).T
        dist_map[::step, ::step, 1] = scipy.interpolate.griddata((corners[:, 0], corners[:, 1]), distortion_error[:, 1],
                                                                 (grid_x, grid_y), method='linear', fill_value=-999).T

        grid_x1 = grid_x[dist_map[grid_y, grid_x, 0] == -999]
        grid_y1 = grid_y[dist_map[grid_y, grid_x, 0] == -999]
        dist_map[grid_y1, grid_x1, 0] = scipy.interpolate.griddata((corners[:, 0], corners[:, 1]),
                                                                   distortion_error[:, 0], (grid_x1, grid_y1),
                                                                   method='nearest').T

        grid_x2 = grid_x[dist_map[grid_y, grid_x, 1] == -999]
        grid_y2 = grid_y[dist_map[grid_y, grid_x, 1] == -999]
        dist_map[grid_y2, grid_x2, 1] = scipy.interpolate.griddata((corners[:, 0], corners[:, 1]),
                                                                   distortion_error[:, 1], (grid_x2, grid_y2),
                                                                   method='nearest').T
        return dist_map

    def __init(self):
        return