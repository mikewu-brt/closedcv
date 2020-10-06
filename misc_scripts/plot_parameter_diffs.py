import numpy as np
from matplotlib import pyplot as plt
import argparse
from libs.parse_calib_data import *
import cv2
import json

cam_names = ["A1", "A2", "A3", "A4"]

def read_diff_stats(filename):
    node = json.load(open(filename))
    return node["calib_diff_stats"]

def extract_camera_params(frame_index, camera_id):
    node = json.load(open("online_calib_f{:08d}.json".format(frame_index)))
    return parse_calib_data(node["post_online_calibration"][cam_names[camera_id]])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot parameter differences given folder with online calibration files")
    parser.add_argument('--root', default="/Users/amaharshi/results/video_retry/clean_prism")
    parser.add_argument('--start_index', type=int, default=925)
    parser.add_argument('--end_index', type=int, default=1425)
    parser.add_argument('--use_prism_format', type=bool, default=True)
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Unknown options: {}".format(unknown))

    root = args.root
    start_index = args.start_index
    end_index = args.end_index
    use_prism_format = args.use_prism_format
    os.chdir(root)

    X = np.array([[0., 0., 3208., 3208.], [0., 2200., 2200., 0.], [1., 1., 1., 1.]])

    num_rel = end_index - start_index - 1
    rotations = np.zeros((3, num_rel,))
    raxis = np.zeros((3, num_rel, 3))
    t = np.zeros((3, num_rel, 3))
    p = np.zeros((3, num_rel,))
    focal_length = np.zeros((3, end_index - start_index,))
    for cam_id in range(3):
        if use_prism_format:
            K, R, T, D = extract_camera_params(start_index, cam_id+1)
        else:
            K, R, T, D = parse_calib_data_file(root + "/norm_calib_{}/calib{}.json".format(start_index, cam_id+1))
        focal_length[cam_id, 0] = K[0, 0]
        index = 0
        for i in range(start_index+1, end_index):
            print(i)
            if use_prism_format:
                Kn, Rn, Tn, Dn = extract_camera_params(i, cam_id + 1)
            else:
                Kn, Rn, Tn, Dn = parse_calib_data_file(root + "/norm_calib_{}/calib{}.json".format(i, cam_id + 1))
            rel = Rn @ np.transpose(R)
            Xp = Kn @ rel @ np.linalg.inv(K) @ X
            Xp = Xp / Xp[2, :]
            pdiff = (X - Xp)[:2]
            p[cam_id,  index] = np.max(np.linalg.norm(pdiff, axis=0))
            aa, j = cv2.Rodrigues(rel)
            rotations[cam_id,  index] = np.linalg.norm(aa)
            #aa = aa / rotations[cam_id,  index]
            if rotations[cam_id,  index] != 0:
                raxis[cam_id,  index] = np.squeeze(aa)
            t[cam_id,  index] = np.squeeze(Tn - T)
            index += 1
            focal_length[cam_id,  index] = Kn[0, 0]
            R = Rn
            T = Tn
    print("done")