"""
 * Copyright (c) 2018, The LightCo
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are strictly prohibited without prior permission of
 * The LightCo.
 *
 * @author  Chuck Stanski
 * @version V1.0.0
 * @date    September 2019
 * @brief
 *   Distortion evaluation
 *
"""
import importlib
import argparse
from libs.Stereo import *
import matplotlib as matplot
matplot.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


####################
# Input Parameters
####################

parser = argparse.ArgumentParser(description="Compute distance using random spacial points")
parser.add_argument('--cal_dir', default='Calibration_Aug23')
parser.add_argument('--image_dir', default='Outside_Aug15_0')

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))


####################

image_helper = Image(args.cal_dir)
stereo = Stereo(args.cal_dir)
setup_info = image_helper.setup_info()
nx = setup_info.SensorInfo.width
ny = setup_info.SensorInfo.height

K = stereo.cal_info().K()
D = stereo.cal_info().D()
#D = np.zeros(5)
R = np.identity(3)
mapx1, mapy1 = cv2.initUndistortRectifyMap(cameraMatrix=K[0], distCoeffs=D[0],
                                           R=np.identity(3), newCameraMatrix=K[0],
                                           size=(nx, ny), m1type=cv2.CV_32FC1)
mapx2, mapy2 = cv2.initUndistortRectifyMap(cameraMatrix=K[1], distCoeffs=D[1],
                                           R=np.identity(3), newCameraMatrix=K[1],
                                           size=(nx, ny), m1type=cv2.CV_32FC1)

dx1 = np.subtract(mapx1, np.arange(nx))
dy1 = np.subtract(mapy1.T, np.arange(ny)).T
dx2 = np.subtract(mapx2, np.arange(nx))
dy2 = np.subtract(mapy2.T, np.arange(ny)).T

decimate = 16
dx1 = dx1[0::decimate, 0::decimate]
dy1 = dy1[0::decimate, 0::decimate]
dx2 = dx2[0::decimate, 0::decimate]
dy2 = dy2[0::decimate, 0::decimate]
x, y = np.meshgrid(np.arange(0, nx, decimate), -np.arange(0, ny, decimate))



plt.figure(1).clear()
scale = 20.0  # Quiver lines are "scale" times larger
C = np.hypot(dx1, dy1)
plt.quiver(x, y, dx1, dy1, C, pivot='tip', angles='xy', scale_units='xy', scale=1.0/scale, minlength=0)
plt.title("Distortion - Reference")


plt.figure(2).clear()
C = np.hypot(dx2, dy2)
plt.quiver(x, y, dx2, dy2, C, pivot='tip', angles='xy', scale_units='xy', scale=1.0/scale, minlength=0)
plt.title("Distortion - Source")
