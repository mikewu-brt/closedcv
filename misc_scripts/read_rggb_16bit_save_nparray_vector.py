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
parser.add_argument('--image_dir', default='Outside_Aug15_0')
parser.add_argument('--cal_dir', default='Calibration_Aug15_large_board')

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

####################


# Capture images file names
image_filename = np.array(["cap_lft_{}.rggb", "cap_rit_{}.rggb"])

setup_info = importlib.import_module("{}.setup".format(args.cal_dir))

# Open a figure to avoid cv2.imshow crash
plt.figure(1)

process_image_files = True
sensor_size = [setup_info.SensorInfo.width, setup_info.SensorInfo.height]
num_cam = 2

#raw = raw.byteswap()


all_files_read = False
orientation = 0
if process_image_files:
    while True:
        # Convert images
        for cam_idx in range(num_cam):
            fname = os.path.join(args.image_dir, image_filename[cam_idx].format(orientation))
            print(fname)
            try:
                raw = np.fromfile(fname, np.uint16, -1)
            except:
                all_files_read = True
                break

            # convert to 2-D
            raw = raw.reshape(sensor_size[1], sensor_size[0])

            # Save raw as a numpy array
            fname = os.path.join(args.image_dir, setup_info.RigInfo.image_filename[cam_idx].format(orientation))

            r = raw.copy() / 64.0
            np.save(fname, r.astype(np.uint16))

            raw = raw.astype(np.float32) * 256.0 /(256.0 * 256.0)
            raw = raw.astype(np.uint8)

            cv2.imshow("Raw Image", raw)
            cv2.waitKey(500)

            img = cv2.cvtColor(raw, cv2.COLOR_BayerBG2BGR)
            cv2.imshow("RBG Image", img)


        if all_files_read:
            break
        orientation += 1
