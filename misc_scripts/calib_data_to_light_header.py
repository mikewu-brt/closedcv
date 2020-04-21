import argparse
import json
import lightheader_pb2
from libs.parse_calib_data import *

from google.protobuf import json_format


def update(lightheader, calibdata, cam_idx):
    assert( lightheader.module_calibration[cam_idx].camera_id == cam_idx)
    pfc = lightheader.module_calibration[cam_idx].geometry.per_focus_calibration[0]

    # Rotation
    pfc.extrinsics.canonical.rotation.x00 = calibdata["rvec"][0]
    pfc.extrinsics.canonical.rotation.x01 = calibdata["rvec"][1]
    pfc.extrinsics.canonical.rotation.x02 = calibdata["rvec"][2]
    pfc.extrinsics.canonical.rotation.x10 = calibdata["rvec"][3]
    pfc.extrinsics.canonical.rotation.x11 = calibdata["rvec"][4]
    pfc.extrinsics.canonical.rotation.x12 = calibdata["rvec"][5]
    pfc.extrinsics.canonical.rotation.x20 = calibdata["rvec"][6]
    pfc.extrinsics.canonical.rotation.x21 = calibdata["rvec"][7]
    pfc.extrinsics.canonical.rotation.x22 = calibdata["rvec"][8]

    # Translation
    pfc.extrinsics.canonical.translation.x = calibdata["tvec"][0]
    pfc.extrinsics.canonical.translation.y = calibdata["tvec"][1]
    pfc.extrinsics.canonical.translation.z = calibdata["tvec"][2]

    # Kmat
    kmat = calibdata["intrinsic_kmat"]["fd_0"]["k_mat"]
    pfc.intrinsics.k_mat.x00 = kmat[0]
    pfc.intrinsics.k_mat.x01 = kmat[1]
    pfc.intrinsics.k_mat.x02 = kmat[2]
    pfc.intrinsics.k_mat.x10 = kmat[3]
    pfc.intrinsics.k_mat.x11 = kmat[4]
    pfc.intrinsics.k_mat.x12 = kmat[5]
    pfc.intrinsics.k_mat.x20 = kmat[6]
    pfc.intrinsics.k_mat.x21 = kmat[7]
    pfc.intrinsics.k_mat.x22 = kmat[8]

    # Distortion
    distortion = lightheader.module_calibration[cam_idx].geometry.distortion
    indist = calibdata["distort_params"]
    distortion.polynomial.distortion_center.x = indist["distortion_center"][0]
    distortion.polynomial.distortion_center.y = indist["distortion_center"][1]
    distortion.polynomial.normalization.x = indist["norm_x_y"][0]
    distortion.polynomial.normalization.y = indist["norm_x_y"][1]
    distortion.polynomial.coeffs[0] = indist["distortion_coeffs"][0]
    distortion.polynomial.coeffs[1] = indist["distortion_coeffs"][1]
    distortion.polynomial.coeffs[2] = indist["distortion_coeffs"][2]
    distortion.polynomial.coeffs[3] = indist["distortion_coeffs"][3]
    distortion.polynomial.coeffs[4] = indist["distortion_coeffs"][4]

    return lightheader


parser = argparse.ArgumentParser(description="Convert calib data json output to LightHeader json consumable by LRI creator")
parser.add_argument('--cal_dir', default='/Users/amaharshi/results/26_3_cal/exr_files/cv_init_ref_fppd_3_26')
parser.add_argument('--header', default='lightheader.json')

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))


calibration_header = read_lightheader( args.cal_dir + "/" + args.header )
calib0 = json.load( open( args.cal_dir + "/calib0.json", 'r' ))
calib1 = json.load( open( args.cal_dir + "/calib1.json", 'r' ))
calib2 = json.load( open( args.cal_dir + "/calib2.json", 'r' ))
calib3 = json.load( open( args.cal_dir + "/calib3.json", 'r' ))

calibration_header = update( calibration_header, calib0, 0)
calibration_header = update( calibration_header, calib1, 1)
calibration_header = update( calibration_header, calib2, 2)
calibration_header = update( calibration_header, calib3, 3)

write_lightheader( calibration_header, args.cal_dir + "/updated_header.json")
