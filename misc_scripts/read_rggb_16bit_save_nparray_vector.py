"""
 * Copyjright (c) 2018, The LightCo
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are strictly prohibited without prior permission of
 * The LightCo.
 *
 * @author  Yunus Hussain
 * @version V1.0.0
 * @date    August 2019
 * @brief
 *  Stereo calibration test script
 *
 ******************************************************************************/
"""
import cv2
import numpy as np
import os
import argparse
import importlib
import matplotlib as matplot
matplot.use('TkAgg')
import matplotlib.pyplot as plt


####################
# Input Parameters
####################

parser = argparse.ArgumentParser(description="Stereo Calibrate")
parser.add_argument('--image_dir', default='Distance_Aug15_0')
parser.add_argument('--cal_dir', default='Calibration_Aug15_large_board')

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

####################

setup_info = importlib.import_module("{}.setup".format(args.cal_dir))

# check if env variable PATH_TO_IMAGE_DIR is set, if not use relative path
path_to_image_dir = os.getenv("PATH_TO_IMAGE_DIR")
if path_to_image_dir == None:
    path_to_image_dir = '.'


# Capture images file names
image_filename = setup_info.RigInfo.input_image_filename


# Open a figure to avoid cv2.imshow crash
plt.figure(1)

process_image_files = True
sensor_size = [setup_info.SensorInfo.width, setup_info.SensorInfo.height]
num_cam = setup_info.RigInfo.cam_position_m.shape[0]



all_files_read = False
orientation = 0
if process_image_files:
    while True:
        # Convert images
        file_not_present = 0
        print('entering cam_idx_loop')
        for cam_idx in range(num_cam):
            fname = os.path.join(path_to_image_dir,args.image_dir, image_filename[cam_idx].format(orientation))
            print(fname)
            raw = []
            try:
                print(fname)
                raw = np.fromfile(fname, np.uint16, -1)
            except:
                all_files_read = True
                break

            # convert to 2-D
            #print(max(raw))
            raw = raw.reshape(sensor_size[1], sensor_size[0])

            # Save raw as a numpy array

            print("read filename =", fname)
            fname = os.path.join(path_to_image_dir,args.image_dir, setup_info.RigInfo.image_filename[cam_idx].format(orientation))
            print("written filename = ",fname)

            r = raw.copy() / setup_info.RigInfo.scale
            np.save(fname, r.astype(np.uint16))

            raw = raw.astype(np.float32) /(4.0 * setup_info.RigInfo.scale)
            raw = raw.astype(np.uint8)

            cv2.imshow("Raw Image {}".format(cam_idx), raw)
            cv2.waitKey(500)

            img = cv2.cvtColor(raw, cv2.COLOR_BayerBG2BGR)
            cv2.imshow("RBG Image", img)

        if all_files_read:
            break
        orientation += 1
