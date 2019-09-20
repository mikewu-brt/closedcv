"""
 * Copyright (c) 2018, The LightCo
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are strictly prohibited without prior permission of
 * The LightCo.
 *
 * @author  Chuck Stanski
 * @version V1.0.0
 * @date    July 2019
 * @brief
 *  Test script to compute various optical parameters related to the depth
 *  test rig version 1
 *
"""
from libs.Optical import *
import numpy as np
import matplotlib as matplot
matplot.use('TkAgg')
import matplotlib.pyplot as plt

# System Configuration setting
#sc = "sc0_2a"
#sc = "lucid_16mm"
#sc = "lucid_25mm"
sc = "lucid_35mm"


if sc == "sc0_2a":
    sensor_type = "IMX265"
    pixel_size_um = 3.45
    h_pixels = 2056
    v_pixels = 1542
    f_num = 2.0
    lens_fl_mm = 12

elif sc == "lucid_16mm":
    sensor_type = "IMX428"
    pixel_size_um = 4.5
    h_pixels = 3208
    v_pixels = 2200
    f_num = 2.8
    lens_fl_mm = 16

elif sc == "lucid_25mm":
    sensor_type = "IMX428"
    pixel_size_um = 4.5
    h_pixels = 3208
    v_pixels = 2200
    f_num = 2.8
    lens_fl_mm = 25

elif sc == "lucid_35mm":
    sensor_type = "IMX428"
    pixel_size_um = 4.5
    h_pixels = 3208
    v_pixels = 2200
    f_num = 2.8
    lens_fl_mm = 35

else:
    print("Unknown config")
    exit(1)


obj_dist_mm = 100 * 1000

baseline_m = 0.5

optical = Optical(h_pixels, v_pixels, pixel_size_um, f_num, lens_fl_mm)
sensor = optical.sensor
lens = optical.lens

print("Sensor information: {}, system config: {}".format(sensor_type, sc))
print("Pixel size {} um -> {:.3f} lp/mm".format(pixel_size_um, sensor.resolution_lp_mm()))

sensor_size = sensor.size_mm()
print("Sensor Size {:.2f} x {:.2f} mm".format(sensor_size['h_mm'], sensor_size['v_mm']))

fl = lens.focal_length_mm()
fl_35mm = optical.focal_length_35mm()
print("")
print("Lens FL = {:.2f} mm".format(fl))
print("FL 35mm = {:.2f} mm".format(fl_35mm))

afov = optical.angular_fov()
print("Optical FOV - hfov: {:.2f} deg, vfov: {:.2f} deg, dfov: {:.2f}".format(
    afov['hfov_deg'], afov['vfov_deg'], afov['dfov_deg']))

obj_res = optical.object_space_resolution(obj_dist_mm)
print("")
print("Pixel size at {} m: {:.2f} mm, {:.2f} lp/mm, {:.2f} lp/m".format(obj_dist_mm / 1000,
                           obj_res['pixel_mm'], obj_res['lp_mm'], obj_res['lp_mm'] * 1000))
print("Scene size at {} m: {:.2f} x {:.2f} (m)".format(obj_dist_mm / 1000,
                           obj_res['h_mm'] / 1000, obj_res['v_mm'] / 1000))

print("Hyper focal distance: {:.2f} m for 1 pixel CoC".format(optical.hyper_focal_dist_mm(1.0) / 1000))
print("Hyper focal distance: {:.2f} m for 3 pixel CoC".format(optical.hyper_focal_dist_mm(3.0) / 1000))

# Estimate the expected disparity vs distance
disp = np.arange(5, 400)
depth = (lens.focal_length_mm() * 1.0e-3) / (sensor.pixel_size_um() * 1.0e-6) * baseline_m / disp
plt.figure(1).clear()
plt.plot(disp, depth)
plt.grid()
plt.title("Disparity vs Distance")
plt.xlabel("Disparity (pixels)")
plt.ylabel("Distance (m)")

one_pixel_err = (lens.focal_length_mm() * 1.0e-3) / (sensor.pixel_size_um() * 1.0e-6) * baseline_m / (disp - 1)
one_half_pixel_err = (lens.focal_length_mm() * 1.0e-3) / (sensor.pixel_size_um() * 1.0e-6) * baseline_m / (disp - 0.5)
depth_err = (one_pixel_err - depth) / depth
depth_err_1_2 = (one_half_pixel_err - depth) / depth
plt.figure(2).clear()
plt.plot(depth, depth_err * 100)
plt.plot(depth, depth_err_1_2 * 100)
plt.grid()
plt.xlabel('Distance (m)')
plt.ylabel("Error (%)")
plt.title("Distance Error per Disparity error")
plt.legend(["1 pixel error", "1/2 pixel error"])
