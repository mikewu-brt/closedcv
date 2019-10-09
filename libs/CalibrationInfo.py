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
import importlib
import os
import json
from libs.Image import *


class CalibrationInfo:

    def K(self, cam_idx=None):
        if cam_idx is None:
            return self.__K
        return self.__K[cam_idx]

    def set_K(self, K, cam_idx=None):
        if cam_idx is None:
            self.__K = K
        else:
            self.__K[cam_idx] = K

    def D(self, cam_idx=None):
        if cam_idx is None:
            return self.__D
        return self.__D[cam_idx]

    def set_D(self, D, cam_idx=None):
        if cam_idx is None:
            self.__D = D
        else:
            self.__D[cam_idx] = D

    def R(self, cam_idx=None):
        if cam_idx is None:
            return self.__R
        return self.__R[cam_idx]

    def set_R(self, R, cam_idx=None):
        if cam_idx is None:
            self.__R = R
        else:
            self.__R[cam_idx] = R

    def T(self, cam_idx=None):
        if cam_idx is None:
            return self.__T
        return self.__T[cam_idx]

    def set_T(self, T, cam_idx=None):
        if cam_idx is None:
            self.__T = T
        else:
            self.__T[cam_idx] = T

    def write_json(self, filename):
        # Build json file
        cal_info = {}
        cal_info["module_calibration"] = []

        num_cam = self.__K.shape[0]
        for cam_idx in range(num_cam):
            cal_info["module_calibration"].append([])
            cal_info["module_calibration"][-1] = {}
            cal_info["module_calibration"][-1]["camera_id"] = self.__setup.RigInfo.module_name[cam_idx]
            cal_info["module_calibration"][-1]["geometry"] = {}

            k_mat = {
                "x00": self.__K[cam_idx, 0, 0],
                "x01": self.__K[cam_idx, 0, 1],
                "x02": self.__K[cam_idx, 0, 2],
                "x10": self.__K[cam_idx, 1, 0],
                "x11": self.__K[cam_idx, 1, 1],
                "x12": self.__K[cam_idx, 1, 2],
                "x20": self.__K[cam_idx, 2, 0],
                "x21": self.__K[cam_idx, 2, 1],
                "x22": self.__K[cam_idx, 2, 2],
            }

            r_mat = {
                "x00": self.__R[cam_idx, 0, 0],
                "x01": self.__R[cam_idx, 0, 1],
                "x02": self.__R[cam_idx, 0, 2],
                "x10": self.__R[cam_idx, 1, 0],
                "x11": self.__R[cam_idx, 1, 1],
                "x12": self.__R[cam_idx, 1, 2],
                "x20": self.__R[cam_idx, 2, 0],
                "x21": self.__R[cam_idx, 2, 1],
                "x22": self.__R[cam_idx, 2, 2],
            }

            t_vec = {
                "x": self.__T[cam_idx, 0, 0],
                "y": self.__T[cam_idx, 1, 0],
                "z": self.__T[cam_idx, 2, 0],
            }

            cal_info["module_calibration"][-1]["geometry"]["per_focus_calibration"] = []
            cal_info["module_calibration"][-1]["geometry"]["per_focus_calibration"].append([])
            cal_info["module_calibration"][-1]["geometry"]["per_focus_calibration"][-1] = {
                "focus_distance": self.__setup.LensInfo.focus_distance_mm,
                "intrinsics": {"k_mat": k_mat},
                "extrinsics": {
                    "canonical": {
                        "rotation": r_mat,
                        "translation": t_vec,
                    },
                },
            }

            cal_info["module_calibration"][-1]["geometry"]["distortion"] = {
                "polynomial": {
                    "distortion_center": {
                        "x": self.__K[cam_idx, 0, 2],
                        "y": self.__K[cam_idx, 1, 2],
                    },
                    "normalization": {
                        "x": self.__K[cam_idx, 0, 0],
                        "y": self.__K[cam_idx, 1, 1],
                    },
                    "coeffs": self.__D[cam_idx][0].tolist()
                },
            }

        json_enc = json.JSONEncoder(sort_keys=True, allow_nan=False, indent=True)
        fd = open(os.path.join(self.__cal_dir, filename), 'w')
        fd.write(json_enc.encode(cal_info))
        fd.close()

    def read_json(self, filename):
        fd = open(os.path.join(self.__cal_dir, filename), 'r')
        raw_cal = json.load(fd)
        fd.close()

        num_cam = len(raw_cal["module_calibration"])
        self.__K = np.zeros((num_cam, 3, 3), dtype=np.float)
        self.__R = np.zeros((num_cam, 3, 3), dtype=np.float)
        self.__T = np.zeros((num_cam, 3, 1), dtype=np.float)
        self.__D = np.zeros((num_cam, 1, 5), dtype=np.float)

        cam_idx = 0
        for module_cal in raw_cal["module_calibration"]:
            for pfc in module_cal["geometry"]["per_focus_calibration"]:

                self.__K[cam_idx, 0, 0] = pfc["intrinsics"]["k_mat"]["x00"]
                self.__K[cam_idx, 0, 1] = pfc["intrinsics"]["k_mat"]["x01"]
                self.__K[cam_idx, 0, 2] = pfc["intrinsics"]["k_mat"]["x02"]
                self.__K[cam_idx, 1, 0] = pfc["intrinsics"]["k_mat"]["x10"]
                self.__K[cam_idx, 1, 1] = pfc["intrinsics"]["k_mat"]["x11"]
                self.__K[cam_idx, 1, 2] = pfc["intrinsics"]["k_mat"]["x12"]
                self.__K[cam_idx, 2, 0] = pfc["intrinsics"]["k_mat"]["x20"]
                self.__K[cam_idx, 2, 1] = pfc["intrinsics"]["k_mat"]["x21"]
                self.__K[cam_idx, 2, 2] = pfc["intrinsics"]["k_mat"]["x22"]

                self.__R[cam_idx, 0, 0] = pfc["extrinsics"]["canonical"]["rotation"]["x00"]
                self.__R[cam_idx, 0, 1] = pfc["extrinsics"]["canonical"]["rotation"]["x01"]
                self.__R[cam_idx, 0, 2] = pfc["extrinsics"]["canonical"]["rotation"]["x02"]
                self.__R[cam_idx, 1, 0] = pfc["extrinsics"]["canonical"]["rotation"]["x10"]
                self.__R[cam_idx, 1, 1] = pfc["extrinsics"]["canonical"]["rotation"]["x11"]
                self.__R[cam_idx, 1, 2] = pfc["extrinsics"]["canonical"]["rotation"]["x12"]
                self.__R[cam_idx, 2, 0] = pfc["extrinsics"]["canonical"]["rotation"]["x20"]
                self.__R[cam_idx, 2, 1] = pfc["extrinsics"]["canonical"]["rotation"]["x21"]
                self.__R[cam_idx, 2, 2] = pfc["extrinsics"]["canonical"]["rotation"]["x22"]

                self.__T[cam_idx, 0, 0] = pfc["extrinsics"]["canonical"]["translation"]["x"]
                self.__T[cam_idx, 1, 0] = pfc["extrinsics"]["canonical"]["translation"]["y"]
                self.__T[cam_idx, 2, 0] = pfc["extrinsics"]["canonical"]["translation"]["z"]

            self.__D[cam_idx, 0] = np.array(module_cal["geometry"]["distortion"]["polynomial"]["coeffs"])
            cam_idx += 1

    def __init__(self, cal_dir, json_fname=None, K=None, D=None, R=None, T=None):
        path_to_image_dir = os.getenv("PATH_TO_IMAGE_DIR")
        if path_to_image_dir is None:
            path_to_image_dir = '.'
        self.__cal_dir = os.path.join(path_to_image_dir, cal_dir)
        self.__setup = importlib.import_module("{}.setup".format(cal_dir))

        if json_fname is None:
            self.__K = K
            self.__D = D
            self.__R = R
            self.__T = T
        else:
            self.read_json(os.path.join(self.__cal_dir, json_fname))
