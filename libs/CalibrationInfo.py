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
from libs.Image import *
import lightheader_pb2
import camera_id_pb2
import sensor_type_pb2
import matrix3x3f_pb2
import point3f_pb2
import point2f_pb2
import hw_info_pb2
from google.protobuf import json_format

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

    def hash_code(self):
        return self.__hash_code

    def vignetting(self, cam_idx=None):
        # roi -> (width, height)
        if cam_idx is None:
            return self.__vig
        return self.__vig[cam_idx]

    def set_vignetting(self, vig, roi, cam_idx=None):
        # roi -> (width, height)
        if cam_idx is None:
            self.__vig = vig
        else:
            self.__vig[cam_idx] = vig

    def dist_map(self, cam_idx=None):
        if cam_idx is None:
            return self.__dist_map
        return self.__dist_map[cam_idx]

    def write_json(self, filename):
        lightheader = lightheader_pb2.LightHeader()
        if self.__imu_info is not None:
            lightheader.hw_info.CopyFrom(self.__imu_info)

        lightheader.hw_info.manufacturer = self.__setup.RigInfo.rig_manufacturer
        lightheader.hw_info.geometric_calib_version = self.__hash_code + '+' +self.__version

        num_cam = self.__K.shape[0]
        for cam_idx in range(num_cam):
            camera = lightheader.hw_info.camera.add()
            camera.id = camera_id_pb2.CameraID.Value(self.__setup.RigInfo.module_name[cam_idx])
            camera.sensor = sensor_type_pb2.SensorType.Value(self.__setup.SensorInfo.type)
            camera.lens = camera.LensType.Value(self.__setup.LensInfo.type)
            camera.focal_length_mm = self.__setup.LensInfo.fl_mm
            camera.pixel_size_mm = self.__setup.SensorInfo.pixel_size_um / 1000.0
            camera.serial_number = self.__setup.RigInfo.camera_module_serial_number[cam_idx]
            camera.manufacturer = self.__setup.RigInfo.camera_module_manufacturer

            cal_info = lightheader.module_calibration.add()
            cal_info.camera_id = camera_id_pb2.CameraID.Value(self.__setup.RigInfo.module_name[cam_idx])

            k_mat = matrix3x3f_pb2.Matrix3x3F()
            k_mat.x00 = self.__K[cam_idx, 0, 0]
            k_mat.x01 = self.__K[cam_idx, 0, 1]
            k_mat.x02 = self.__K[cam_idx, 0, 2]
            k_mat.x10 = self.__K[cam_idx, 1, 0]
            k_mat.x11 = self.__K[cam_idx, 1, 1]
            k_mat.x12 = self.__K[cam_idx, 1, 2]
            k_mat.x20 = self.__K[cam_idx, 2, 0]
            k_mat.x21 = self.__K[cam_idx, 2, 1]
            k_mat.x22 = self.__K[cam_idx, 2, 2]

            r_mat = matrix3x3f_pb2.Matrix3x3F()
            r_mat.x00 = self.__R[cam_idx, 0, 0]
            r_mat.x01 = self.__R[cam_idx, 0, 1]
            r_mat.x02 = self.__R[cam_idx, 0, 2]
            r_mat.x10 = self.__R[cam_idx, 1, 0]
            r_mat.x11 = self.__R[cam_idx, 1, 1]
            r_mat.x12 = self.__R[cam_idx, 1, 2]
            r_mat.x20 = self.__R[cam_idx, 2, 0]
            r_mat.x21 = self.__R[cam_idx, 2, 1]
            r_mat.x22 = self.__R[cam_idx, 2, 2]

            t_vec = point3f_pb2.Point3F()
            t_vec.x = self.__T[cam_idx, 0, 0] * 1000.0
            t_vec.y = self.__T[cam_idx, 1, 0] * 1000.0
            t_vec.z = self.__T[cam_idx, 2, 0] * 1000.0

            per_focus_cal = cal_info.geometry.per_focus_calibration.add()
            per_focus_cal.focus_distance = self.__setup.LensInfo.focus_distance_mm
            per_focus_cal.intrinsics.k_mat.CopyFrom(k_mat)
            per_focus_cal.extrinsics.canonical.rotation.CopyFrom(r_mat)
            per_focus_cal.extrinsics.canonical.translation.CopyFrom(t_vec)

            cal_info.geometry.distortion.polynomial.distortion_center.x = self.__K[cam_idx, 0, 2]
            cal_info.geometry.distortion.polynomial.distortion_center.y = self.__K[cam_idx, 1, 2]
            cal_info.geometry.distortion.polynomial.normalization.x = self.__K[cam_idx, 0, 0]
            cal_info.geometry.distortion.polynomial.normalization.y = self.__K[cam_idx, 1, 1]
            cal_info.geometry.distortion.polynomial.coeffs[:] = self.__D[cam_idx, 0]

            # add distortion map here if the field is defined in the protobuf and distortion mode is set for stereo_calibrate
            list_fields = cal_info.geometry.distortion.DESCRIPTOR.fields_by_name.keys()

            if 'dist_map' in list_fields:
                if self.__dist_map is not None:
                    map_size = self.__dist_map[cam_idx,:,:].shape
                    # save the full image size as width and height (distortion map will need to be expanded to this size)
                    # Actual dist_map size will be determined based on saved array size and then cross-checked against
                    # length estimated from image size and the decimate value
                    cal_info.geometry.distortion.dist_map.width =  self.__setup.SensorInfo.width
                    cal_info.geometry.distortion.dist_map.height = self.__setup.SensorInfo.height
                    cal_info.geometry.distortion.dist_map.pixel_offset = self.__pixel_offset
                    cal_info.geometry.distortion.dist_map.decimate = self.__decimate
                    dist_map_point = point2f_pb2.Point2F()
                    for i in range(map_size[0]):
                        for j in range(map_size[1]):
                            dist_map_point.x = self.__dist_map[cam_idx,i,j,0]
                            dist_map_point.y = self.__dist_map[cam_idx,i,j,1]
                            cal_info.geometry.distortion.dist_map.map.extend([dist_map_point])

            if self.__vig is not None:
                # Generate crosstalk (identity matrix)
                vig_ct = np.empty((self.__vig[cam_idx].shape[1], self.__vig[cam_idx].shape[0], 4, 4))
                vig_ct[:, :] = np.identity(4)

                cal_info.vignetting.crosstalk.width = self.__vig[cam_idx].shape[1]
                cal_info.vignetting.crosstalk.height = self.__vig[cam_idx].shape[0]
                cal_info.vignetting.crosstalk.data_packed[:] = vig_ct.reshape(-1).astype(np.uint)

                vignetting = cal_info.vignetting.vignetting.add()
                vignetting.hall_code = 0
                vignetting.vignetting.width = self.__vig[cam_idx].shape[1]
                vignetting.vignetting.height = self.__vig[cam_idx].shape[0]
                vignetting.vignetting.data[:] = self.__vig[cam_idx].reshape(-1)

        json_str = json_format.MessageToJson(lightheader, preserving_proto_field_name=True, sort_keys=True)
        open(os.path.join(self.__cal_dir, filename), 'w').write(json_str)

    def read_calibration_json(self, filename):
        json_str = open(os.path.join(self.__cal_dir, filename), 'r').read()
        lightheader = lightheader_pb2.LightHeader()
        json_format.Parse(json_str, lightheader)

        num_cam = len(lightheader.module_calibration)
        self.__K = np.zeros((num_cam, 3, 3), dtype=np.float)
        self.__R = np.zeros((num_cam, 3, 3), dtype=np.float)
        self.__T = np.zeros((num_cam, 3, 1), dtype=np.float)
        self.__D = np.zeros((num_cam, 1, 5), dtype=np.float)
        self.__vig = None
        self.__hash_code = lightheader.hw_info.geometric_calib_version[0:64]
        self.__version  =  lightheader.hw_info.geometric_calib_version[65:]
        cam_idx = 0
        self.__dist_map = []
        for module_cal in lightheader.module_calibration:
            for pfc in module_cal.geometry.per_focus_calibration:
                self.__K[cam_idx, 0, 0] = pfc.intrinsics.k_mat.x00
                self.__K[cam_idx, 0, 1] = pfc.intrinsics.k_mat.x01
                self.__K[cam_idx, 0, 2] = pfc.intrinsics.k_mat.x02
                self.__K[cam_idx, 1, 0] = pfc.intrinsics.k_mat.x10
                self.__K[cam_idx, 1, 1] = pfc.intrinsics.k_mat.x11
                self.__K[cam_idx, 1, 2] = pfc.intrinsics.k_mat.x12
                self.__K[cam_idx, 2, 0] = pfc.intrinsics.k_mat.x20
                self.__K[cam_idx, 2, 1] = pfc.intrinsics.k_mat.x21
                self.__K[cam_idx, 2, 2] = pfc.intrinsics.k_mat.x22

                self.__R[cam_idx, 0, 0] = pfc.extrinsics.canonical.rotation.x00
                self.__R[cam_idx, 0, 1] = pfc.extrinsics.canonical.rotation.x01
                self.__R[cam_idx, 0, 2] = pfc.extrinsics.canonical.rotation.x02
                self.__R[cam_idx, 1, 0] = pfc.extrinsics.canonical.rotation.x10
                self.__R[cam_idx, 1, 1] = pfc.extrinsics.canonical.rotation.x11
                self.__R[cam_idx, 1, 2] = pfc.extrinsics.canonical.rotation.x12
                self.__R[cam_idx, 2, 0] = pfc.extrinsics.canonical.rotation.x20
                self.__R[cam_idx, 2, 1] = pfc.extrinsics.canonical.rotation.x21
                self.__R[cam_idx, 2, 2] = pfc.extrinsics.canonical.rotation.x22

                self.__T[cam_idx, 0, 0] = pfc.extrinsics.canonical.translation.x / 1000.0
                self.__T[cam_idx, 1, 0] = pfc.extrinsics.canonical.translation.y / 1000.0
                self.__T[cam_idx, 2, 0] = pfc.extrinsics.canonical.translation.z / 1000.0

            self.__D[cam_idx, 0] = np.array(module_cal.geometry.distortion.polynomial.coeffs[:])

            if module_cal.geometry.distortion.HasField("dist_map"):
                width = 2 +  int((module_cal.geometry.distortion.dist_map.width-1)/(2*module_cal.geometry.distortion.dist_map.decimate))
                height = 2 +  int((module_cal.geometry.distortion.dist_map.height-1)/(2*module_cal.geometry.distortion.dist_map.decimate))
                dist_map = np.zeros((height, width,2), np.float32)
                array_point2f =  module_cal.geometry.distortion.dist_map.map[:]
                count = 0
                for i in range(height):
                    for j in range(width):
                        dist_map[i,j,0] = array_point2f[count].x
                        dist_map[i,j,1] = array_point2f[count].y
                        count = count + 1
                self.__dist_map.append(dist_map)

            if module_cal.HasField("vignetting"):
                if cam_idx == 0:
                    self.__vig = []
                width = module_cal.vignetting.vignetting[0].vignetting.width
                height = module_cal.vignetting.vignetting[0].vignetting.height
                self.__vig.append(np.array(module_cal.vignetting.vignetting[0].vignetting.data).reshape(height, width))

            cam_idx += 1

    def checkin_cal_file(self):
        infile = os.path.join(self.__cal_dir, "calibration.json")

        outfile = "cal"
        for sn in self.__setup.RigInfo.camera_module_serial_number:
            outfile += "_{}".format(sn)
        outfile += "_{}mm.json".format(int(self.__setup.LensInfo.fl_mm))
        cmd = "cp \"{}\" cal-files/{}".format(infile, outfile)
        os.system(cmd)

    def read_imu_json(self, filename):
        json_str = open(os.path.join(self.__cal_dir, filename), 'r').read()
        self.__imu_info = hw_info_pb2.HwInfo()
        json_format.Parse(json_str, self.__imu_info)

    def __init__(self, cal_dir, calibration_json_fname=None, K=None, D=None, R=None, T=None,
                        V=None, MAP=None, VERSION=None, PIXEL_OFFSET=None, DECIMATE=None, HASH_CODE=None):
        path_to_image_dir = os.getenv("PATH_TO_IMAGE_DIR")
        if path_to_image_dir is None:
            path_to_image_dir = '.'
        self.__cal_dir = os.path.join(path_to_image_dir, cal_dir)
        self.__setup = importlib.import_module("{}.setup".format(cal_dir))

        if calibration_json_fname is None:
            self.__K = K
            self.__D = D
            self.__R = R
            self.__T = T
            self.__vig = V
            self.__dist_map = MAP
            self.__version = VERSION
            self.__pixel_offset = PIXEL_OFFSET
            self.__decimate = DECIMATE
            self.__hash_code = HASH_CODE
        else:
            print("Reading calibration from {}".format(calibration_json_fname))
            self.read_calibration_json(os.path.join(self.__cal_dir, calibration_json_fname))

        # Attempt to read the imu.json file
        self.__imu_info = None
        if os.path.isfile(os.path.join(self.__cal_dir, "imu_auto.json")):
            self.read_imu_json("imu_auto.json")


