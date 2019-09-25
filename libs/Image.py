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

import sys
import os
import numpy as np
import cv2
import importlib


class Image:

    def __init__(self, directory):
        path_to_image_dir = os.getenv("PATH_TO_IMAGE_DIR")
        if path_to_image_dir is None:
            path_to_image_dir = '.'
        self.__directory = os.path.join(path_to_image_dir, directory)

        self.__setup = importlib.import_module("{}.setup".format(directory))

        if (len(self.__setup.RigInfo.cam_position_m) != len(self.__setup.RigInfo.input_image_filename)) \
                or (len(self.__setup.RigInfo.cam_position_m) != len(self.__setup.RigInfo.module_name)):
            print("length of cam_position_m {}, input_image_filename {} and module_name {} in \"setup.py\" must be the same".format(
                len(self.__setup.RigInfo.cam_position_m), len(self.__setup.RigInfo.input_image_filename), len(self.__setup.RigInfo.module_name)
            ))
            sys.exit(1)
        return

    def setup_info(self):
        return self.__setup

    @staticmethod
    def unpack(raw, num_bits):
        raw = np.unpackbits(raw, bitorder='little').reshape(-1, num_bits)
        raw_out = np.zeros(raw.shape[0], np.uint16)
        for i in range(num_bits):
            raw_out += raw[:, i] * (1 << i)
        return raw_out

    def num_cam(self):
        if self.__setup is None:
            print("No setup file specified")
            sys.exit(1)
        return len(self.__setup.RigInfo.cam_position_m)

    def display_size(self, max_width):
        scale = 1.0
        while (self.__setup.SensorInfo.width * scale) > float(max_width):
            scale /= 2.0
        return scale

    def read_image_file(self, camera_idx, capture_idx, scale_to_8bit=True):
        """
        Process raw image files into "Numpy" arrays.

        :param camera_idx: Camera Index
        :param capture_idx: Capture Index
        :param store_npy:  Stores processed raw file for future use
        :return: Processed raw image or None
        """
        if self.__setup is None:
            print("No setup file specified")
            sys.exit(1)
        fname = os.path.join(self.__directory, self.__setup.RigInfo.input_image_filename[camera_idx].format(capture_idx))
        print("Reading {}".format(fname))
        try:
            if os.path.splitext(fname)[1] == ".npy":
                raw = np.load(fname)
            elif self.__setup.RigInfo.packed or (self.__setup.SensorInfo.bits_per_pixel <= 8):
                raw = np.fromfile(fname, np.uint8, -1)
            else:
                raw = np.fromfile(fname, np.uint16, -1)
        except:
            raw = None

        if raw is None:
            print("File does not exist, returning None".format(fname))
            img = None
            gray = None
        else:
            # Convert raw to 16 bit data
            if self.__setup.RigInfo.packed:
                raw = self.unpack(raw, self.__setup.SensorInfo.bits_per_pixel)
                if self.__setup.RigInfo.left_justified:
                    raw <<= (16 - self.__setup.SensorInfo.bits_per_pixel)
            elif self.__setup.SensorInfo.bits_per_pixel <= 8:
                raw = raw.astype(np.uint16)
                if self.__setup.RigInfo.left_justified:
                    raw <<= (16 - self.__setup.SensorInfo.bits_per_pixel)

            # Reshape and demosaic
            raw = raw.reshape(self.__setup.SensorInfo.height, self.__setup.SensorInfo.width)
            if not self.__setup.RigInfo.left_justified:
                raw <<= (16 - self.__setup.SensorInfo.bits_per_pixel)

            if scale_to_8bit:
                raw = (raw >> 8).astype(np.uint8)

            img = cv2.cvtColor(raw, self.__setup.RigInfo.cv2_color_conversion)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img, gray

    def load_np_file(self, filename):
        return np.load(os.path.join(self.__directory, filename))

    def save_np_file(self, filename, array):
        np.save(os.path.join(self.__directory, filename), array)
        return

    def save_text_file(self, filename, array):
        np.savetxt(os.path.join(self.__directory, filename), array)
