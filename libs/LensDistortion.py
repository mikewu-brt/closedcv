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

import sys
import cv2
import numpy as np
from libs.Image import Image
import scipy
from scipy import interpolate


class LensDistortion:

    __dist_map = None
    __vignetting_map = None

    def compute_distortion_map(self, image_shape, corners, distortion_error, step=1):
        dist_map = np.zeros((image_shape[0], image_shape[1], 2), dtype=np.float32)
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
        self.__dist_map = dist_map
        return self.__dist_map

    def distortion_map(self):
        return self.__dist_map

    def set_distortion_map(self, distortion_map):
        self.__dist_map = distortion_map
        return

    def opencv_distortion_map(self, distortion_map=None):
        if distortion_map is None:
            distortion_map = self.__dist_map

        if distortion_map is None:
            print("Undefined distortion map")
            sys.exit(-1)

        dist_map = distortion_map.copy()
        dist_map[:, :, 0] = np.add(np.arange(dist_map.shape[1]), dist_map[:, :, 0])
        dist_map[:, :, 1] = np.add(np.arange(dist_map.shape[0]).T, dist_map[:, :, 1].T).T
        return dist_map

    def correct_distortion(self, img, distortion_map=None, K=None, D=None):
        if distortion_map is not None and D is not None:
            print("Only a distortion_map or distortion_vector should be used, not both")
            sys.exit(-1)

        if D is None:
            # Use the distortion map
            dist_map = self.opencv_distortion_map(distortion_map)
        else:
            # Create a distortion map based on K and D
            dist_map, _ = cv2.initUndistortRectifyMap(cameraMatrix=K, distCoeffs=D, R=np.identity(3),
                                                      newCameraMatrix=K, size=(img.shape[1], img.shape[0]),
                                                      m1type=cv2.CV_32FC2)

        if img.shape[0:2] != dist_map.shape[0:2]:
            print("Image / Distortion map size difference - {} vs {}".format(img.shape, dist_map.shape))
            sys.exit(-1)

        # Remap image using distortion map
        return cv2.remap(img.copy(), dist_map, None, cv2.INTER_LINEAR)

    def __init__(self, distortion_dir=None):
        # FIXME(Chuck)
        #  - Load vignetting map from file
        if distortion_dir is not None:
            file_helper = Image(distortion_dir)
            self.__dist_map = file_helper.load_np_file("distortion_map.npy")
        return
