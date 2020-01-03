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
import math
import cv2
import numpy as np
from libs.Image import Image
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt


class LensDistortion:

    __dist_map = None
    __vignetting_map = None
    __lens_shade_filter = None
    __vig_setup = None

    def compute_distortion_map(self, image_shape, corners, distortion_error, step=1, extrapolate=True):
        dist_map = np.zeros((image_shape[0], image_shape[1], 2), dtype=np.float32)
        grid_x, grid_y = np.mgrid[0:image_shape[1]:step, 0:image_shape[0]:step]
        if extrapolate:
            fval = -999
        else:
            fval = 0
        dist_map[::step, ::step, 0] = scipy.interpolate.griddata((corners[:, 0], corners[:, 1]), distortion_error[:, 0],
                                                                 (grid_x, grid_y), method='linear', fill_value=fval).T
        dist_map[::step, ::step, 1] = scipy.interpolate.griddata((corners[:, 0], corners[:, 1]), distortion_error[:, 1],
                                                                 (grid_x, grid_y), method='linear', fill_value=fval).T

        if extrapolate:
            grid_x1 = grid_x[dist_map[grid_y, grid_x, 0] == fval]
            grid_y1 = grid_y[dist_map[grid_y, grid_x, 0] == fval]
            dist_map[grid_y1, grid_x1, 0] = scipy.interpolate.griddata((corners[:, 0], corners[:, 1]),
                                                                       distortion_error[:, 0], (grid_x1, grid_y1),
                                                                       method='nearest').T

            grid_x2 = grid_x[dist_map[grid_y, grid_x, 1] == fval]
            grid_y2 = grid_y[dist_map[grid_y, grid_x, 1] == fval]
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

    def vignetting(self):
        return self.__lens_shade_filter

    def set_vignetting(self, lens_shade_filter):
        self.__lens_shade_filter = lens_shade_filter

    def json_vignetting(self, roi):
        if self.__lens_shade_filter is None:
            return

        height, width, _ = self.__lens_shade_filter.shape
        w_step = int(width / roi[0])
        w = w_step * roi[0]
        ws = int((width - w) / 2)
        h_step = int(height / roi[1])
        h = h_step * roi[1]
        hs = int((height - h) / 2)

        vig = np.empty((roi[1], roi[0]))
        for y_idx in range(roi[1]):
            ys = y_idx * h_step + hs
            ye = ys + h_step
            for x_idx in range(roi[0]):
                xs = x_idx * w_step + ws
                xe = xs + w_step
                vig[y_idx, x_idx] = np.average(self.__lens_shade_filter[ys:ye, xs:xe, 0])
        return vig

    def convert_json_vignetting(self, json_vig):
        # Extrapolate the JSON vignetting into a full sensor
        if self.__vig_setup is None:
            print("Vignetting information not specified")
            exit(-1)

        # Back out the knot locations of the JSON vignetting locations
        json_h, json_w = json_vig.shape
        w_step = int(self.__vig_setup.SensorInfo.width / json_w)
        w_start = int((self.__vig_setup.SensorInfo.width - w_step * json_w) / 2)
        x = np.arange(w_start, w_start + w_step * json_w, w_step)
        h_step = int(self.__vig_setup.SensorInfo.height / json_h)
        h_start = int((self.__vig_setup.SensorInfo.height - h_step * json_h) / 2)
        y = np.arange(h_start, h_start + h_step * json_h, h_step)

        f = scipy.interpolate.RegularGridInterpolator((y, x), json_vig, method='linear', bounds_error=False, fill_value=None)

        x = np.arange(self.__vig_setup.SensorInfo.width)
        y = np.arange(self.__vig_setup.SensorInfo.height)
        x, y = np.meshgrid(y, x, indexing='ij')
        pts = np.array([x.reshape(-1), y.reshape(-1)]).T
        self.__lens_shade_filter = np.empty((self.__vig_setup.SensorInfo.height,
                                             self.__vig_setup.SensorInfo.width,
                                             3))

        self.__lens_shade_filter[:, :, 0] = f(pts).reshape(self.__vig_setup.SensorInfo.height,
                                                           self.__vig_setup.SensorInfo.width)
        self.__lens_shade_filter[:, :, 1] = self.__lens_shade_filter[:, :, 0]
        self.__lens_shade_filter[:, :, 2] = self.__lens_shade_filter[:, :, 0]

    def opencv_distortion_map(self, distortion_map=None):
        if distortion_map is None:
            distortion_map = self.__dist_map

        if distortion_map is None:
            print("Undefined distortion map")
            sys.exit(-1)

        dist_map = distortion_map.copy()
        dist_map[:, :, 0] = np.add(np.arange(dist_map.shape[1]), -dist_map[:, :, 0])
        dist_map[:, :, 1] = np.add(np.arange(dist_map.shape[0]).T, -dist_map[:, :, 1].T).T
        return dist_map

    def set_opencv_distortion_map(self, cv_dist_map):
        # Convert an OpenCV map into internal map
        ny, nx, _ = cv_dist_map.shape
        self.__dist_map = np.empty((ny, nx, 2), dtype=np.float32)
        self.__dist_map[:, :, 0] = -np.subtract(cv_dist_map[:, :, 0], np.arange(nx))
        self.__dist_map[:, :, 1] = -np.subtract(cv_dist_map[:, :, 1].T, np.arange(ny).T).T

    def set_radial_distortion_map(self, K, D, size):
        dist_map, _ = cv2.initUndistortRectifyMap(cameraMatrix=K, distCoeffs=D, R=np.identity(3), newCameraMatrix=K,
                                                  size=size, m1type=cv2.CV_32FC2)
        self.set_opencv_distortion_map(dist_map)


    def asic_distortion_map(self, pixel_quad_decimate=16, pixel_offset=True):
        decimate = 2 * pixel_quad_decimate

        # Decimate and extrapolate
        ny, nx, _ = self.__dist_map.shape
        x = np.arange(nx)
        y = np.arange(ny)
        fx = scipy.interpolate.RegularGridInterpolator((y, x), -self.__dist_map[:, :, 0],
                                                       method='linear', bounds_error=False, fill_value=None)
        fy = scipy.interpolate.RegularGridInterpolator((y, x), -self.__dist_map[:, :, 1],
                                                       method='linear', bounds_error=False, fill_value=None)

        dx = math.ceil(nx / decimate)
        dy = math.ceil(ny / decimate)
        y, x = np.meshgrid(np.arange(dy + 1), np.arange(dx + 1), indexing='ij')
        pts = np.array([y.reshape(-1), x.reshape(-1)]).T.astype(np.float64) * decimate
        if pixel_offset:
            pts += np.array([0.5, 0.5])
        asic_map = np.empty((dy+1, dx+1, 2))
        asic_map[:, :, 0] = fx(pts).reshape(dy + 1, dx + 1) / 2.0
        asic_map[:, :, 1] = fy(pts).reshape(dy + 1, dx + 1) / 2.0
        return asic_map

    def set_asic_distortion_map(self, asic_dist_map, img_size, pixel_offset=True):
        nx = img_size[0]
        ny = img_size[1]
        dy, dx, _ = asic_dist_map.shape
        decimate = math.ceil(nx / (dx - 1))

        x = np.arange(dx, dtype=np.float64) * decimate
        y = np.arange(dy, dtype=np.float64) * decimate
        if pixel_offset:
            x += 0.5
            y += 0.5
        fx = scipy.interpolate.RegularGridInterpolator((y, x), -2.0 * asic_dist_map[:, :, 0], method='linear',
                                                       bounds_error=False, fill_value=None)
        fy = scipy.interpolate.RegularGridInterpolator((y, x), -2.0 * asic_dist_map[:, :, 1], method='linear',
                                                       bounds_error=False, fill_value=None)

        y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        pts = np.array([y.reshape(-1), x.reshape(-1)]).T
        self.__dist_map = np.empty((ny, nx, 2), dtype=np.float32)
        self.__dist_map[:, :, 0] = fx(pts).reshape(ny, nx)
        self.__dist_map[:, :, 1] = fy(pts).reshape(ny, nx)
        return

    def correct_distortion(self, img, distortion_map=None, K=None, D=None, interpolation=cv2.INTER_LINEAR):
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
        return cv2.remap(img.copy(), dist_map, None, interpolation=interpolation)

    def correct_vignetting(self, img, lens_shade_filter=None, apply_flag=True, alpha=0.9, max_limit=65535, scale_to_8bit=False):
        if lens_shade_filter is None:
            lens_shade_filter = self.__lens_shade_filter

        if apply_flag and lens_shade_filter is not None:
            if np.ndim(img) == 2:
                lens_shade_filter = lens_shade_filter[:, :, 0]
            image_normalized = img * lens_shade_filter * alpha
            max_val = np.max(image_normalized)
            print("max value: {}".format(max_val))
            image_normalized[image_normalized > max_limit] = max_limit
            print("Normalized max value: {}".format(np.max(image_normalized)))
            img = image_normalized.astype(np.uint16)

        if scale_to_8bit:
            img = (img >> 8).astype(np.uint8)
        return img

    def asic_vignetting_map(self, pixel_quad_decimate=16, pixel_offset=True, alpha=1.0):
        decimate = 2 * pixel_quad_decimate

        # Decimate and extrapolate
        ny, nx, _ = self.__lens_shade_filter.shape
        x = np.arange(nx)
        y = np.arange(ny)
        vig = self.__lens_shade_filter[:, :, 0].copy() * alpha
        f = scipy.interpolate.RegularGridInterpolator((y, x), vig,
                                                       method='linear', bounds_error=False, fill_value=None)

        dx = math.ceil(nx / decimate)
        dy = math.ceil(ny / decimate)
        y, x = np.meshgrid(np.arange(dy + 1), np.arange(dx + 1), indexing='ij')
        pts = np.array([y.reshape(-1), x.reshape(-1)]).T.astype(np.float64) * decimate
        if pixel_offset:
            pts += np.array([0.5, 0.5])
        #asic_vig_map = np.empty((dy+1, dx+1))
        asic_vig_map = f(pts).reshape(dy + 1, dx + 1)
        return asic_vig_map

    def set_asic_vignetting(self, asic_vig_map, img_size, pixel_offset=True):
        nx = img_size[0]
        ny = img_size[1]
        dy, dx = asic_vig_map.shape
        decimate = math.ceil(nx / (dx - 1))

        x = np.arange(dx, dtype=np.float64) * decimate
        y = np.arange(dy, dtype=np.float64) * decimate
        if pixel_offset:
            x += 0.5
            y += 0.5
        f = scipy.interpolate.RegularGridInterpolator((y, x), asic_vig_map[:, :], method='linear',
                                                       bounds_error=False, fill_value=None)

        y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        pts = np.array([y.reshape(-1), x.reshape(-1)]).T
        self.__lens_shade_filter = np.empty((ny, nx, 3), dtype=np.float32)
        self.__lens_shade_filter[:, :, 0] = f(pts).reshape(ny, nx)
        self.__lens_shade_filter[:, :, 1] = self.__lens_shade_filter[:, :, 0]
        self.__lens_shade_filter[:, :, 2] = self.__lens_shade_filter[:, :, 0]
        return

    def correct_distortion_points(self, pts):
        m, n, _ = self.__dist_map.shape
        x = np.arange(n)
        y = np.arange(m)
        new_pts = np.empty(pts.shape)
        in_values = np.vstack((pts[:, 0, 1], pts[:, 0, 0])).T

        f = scipy.interpolate.RegularGridInterpolator((y, x), self.__dist_map[:, :, 0], method='linear')
        new_pts[:, 0, 0] = pts[:, 0, 0] + f(in_values)

        f = scipy.interpolate.RegularGridInterpolator((y, x), self.__dist_map[:, :, 1], method='linear')
        new_pts[:, 0, 1] = pts[:, 0, 1] + f(in_values)
        return new_pts

    def plot_distortion_map(self, figure_num=None, decimation=128):
        m, n, _ = self.__dist_map.shape
        x = np.arange(0, n, decimation)
        y = np.arange(0, m, decimation)
        u = self.__dist_map[::decimation, ::decimation, 0]
        v = self.__dist_map[::decimation, ::decimation, 1]
        if figure_num is not None:
            plt.figure(figure_num).clear()
        plt.gca().invert_yaxis()
        qscale = 4.0
        C = np.hypot(u, -v)
        q = plt.quiver(x, y, u, -v, C, angles='uv', units='inches', minlength=1, scale_units='inches', scale=qscale, pivot='tip')
        plt.quiverkey(q, 0.9, 0.9, qscale / 4, "${:0.1f} pixels$".format(qscale / 4), labelpos='E', coordinates='figure')
        plt.draw()

    def __init__(self, lens_idx=0, distortion_dir=None, vignetting_dir=None):
        if distortion_dir is not None:
            file_helper = Image(distortion_dir)
            setupInfo = file_helper.setup_info()
            self.__dist_map = file_helper.load_np_file("distortion_map_{}.npy".format(setupInfo.RigInfo.module_name[lens_idx]))
        if vignetting_dir is not None:
            file_helper = Image(vignetting_dir)
            self.__vig_setup = file_helper.setup_info()
            self.__lens_shade_filter = file_helper.load_np_file("lens_shading_{}.npy".format(self.__vig_setup.RigInfo.module_name[lens_idx]))
        return
