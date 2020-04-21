import json
import lightheader_pb2
import numpy as np

from google.protobuf import json_format

def write_lightheader(lightheader, filename):
    json_str = json_format.MessageToJson(lightheader, preserving_proto_field_name=True, sort_keys=True)
    open(filename, 'w').write(json_str)


def read_lightheader(filename):
    json_str = open(filename, 'r').read()
    lightheader = lightheader_pb2.LightHeader()
    json_format.Parse(json_str, lightheader)
    return lightheader

def parse_light_header(lightheader, cam_idx):
    K = np.zeros((3, 3))
    R = np.zeros((3, 3))
    T = np.zeros((3, 1))
    D = np.zeros((5, 1))

    pfc = lightheader.module_calibration[cam_idx].geometry.per_focus_calibration[0]

    # Rotation
    R[0, 0] = pfc.extrinsics.canonical.rotation.x00
    R[0, 1] = pfc.extrinsics.canonical.rotation.x01
    R[0, 2] = pfc.extrinsics.canonical.rotation.x02
    R[1, 0] = pfc.extrinsics.canonical.rotation.x10
    R[1, 1] = pfc.extrinsics.canonical.rotation.x11
    R[1, 2] = pfc.extrinsics.canonical.rotation.x12
    R[2, 0] = pfc.extrinsics.canonical.rotation.x20
    R[2, 1] = pfc.extrinsics.canonical.rotation.x21
    R[2, 2] = pfc.extrinsics.canonical.rotation.x22

    # Translation
    T[0, 0] = pfc.extrinsics.canonical.translation.x
    T[1, 0] = pfc.extrinsics.canonical.translation.y
    T[2, 0] = pfc.extrinsics.canonical.translation.z

    # Kmat
    K[0, 0] = pfc.intrinsics.k_mat.x00
    K[0, 1] = pfc.intrinsics.k_mat.x01
    K[0, 2] = pfc.intrinsics.k_mat.x02
    K[1, 0] = pfc.intrinsics.k_mat.x10
    K[1, 1] = pfc.intrinsics.k_mat.x11
    K[1, 2] = pfc.intrinsics.k_mat.x12
    K[2, 0] = pfc.intrinsics.k_mat.x20
    K[2, 1] = pfc.intrinsics.k_mat.x21
    K[2, 2] = pfc.intrinsics.k_mat.x22

    # Distortion
    distortion = lightheader.module_calibration[cam_idx].geometry.distortion
    D[0, 0] = distortion.polynomial.coeffs[0]
    D[1, 0] = distortion.polynomial.coeffs[1]
    D[2, 0] = distortion.polynomial.coeffs[2]
    D[3, 0] = distortion.polynomial.coeffs[3]
    D[4, 0] = distortion.polynomial.coeffs[4]

    return K, R, T, D


def parse_calib_data(calibdata, check_dist_norm = False):
    K = np.zeros((3, 3))
    R = np.zeros((3, 3))
    T = np.zeros((3, 1))
    D = np.zeros((5, 1))

    # Rotation
    R[0, 0] = calibdata["rvec"][0]
    R[0, 1] = calibdata["rvec"][1]
    R[0, 2] = calibdata["rvec"][2]
    R[1, 0] = calibdata["rvec"][3]
    R[1, 1] = calibdata["rvec"][4]
    R[1, 2] = calibdata["rvec"][5]
    R[2, 0] = calibdata["rvec"][6]
    R[2, 1] = calibdata["rvec"][7]
    R[2, 2] = calibdata["rvec"][8]

    # Translation
    T[0, 0] = calibdata["tvec"][0]
    T[1, 0] = calibdata["tvec"][1]
    T[2, 0] = calibdata["tvec"][2]

    # Kmat
    kmat = calibdata["intrinsic_kmat"]["fd_0"]["k_mat"]
    K[0, 0] = kmat[0]
    K[0, 1] = kmat[1]
    K[0, 2] = kmat[2]
    K[1, 0] = kmat[3]
    K[1, 1] = kmat[4]
    K[1, 2] = kmat[5]
    K[2, 0] = kmat[6]
    K[2, 1] = kmat[7]
    K[2, 2] = kmat[8]

    # Distortion
    indist = calibdata["distort_params"]
    if check_dist_norm:
        assert(K[0, 2] == indist["distortion_center"][0])
        assert(K[1, 2] == indist["distortion_center"][1])
        assert(K[0, 0] == indist["norm_x_y"][0])
        assert(K[1, 1] == indist["norm_x_y"][1])
    D[0, 0] = indist["distortion_coeffs"][0]
    D[1, 0] = indist["distortion_coeffs"][1]
    D[2, 0] = indist["distortion_coeffs"][2]
    D[3, 0] = indist["distortion_coeffs"][3]
    D[4, 0] = indist["distortion_coeffs"][4]

    return K, R, T, D


def parse_calib_data_file(filename, check_dist_norm=False):
    return parse_calib_data(json.load(open(filename, 'r')), check_dist_norm)
