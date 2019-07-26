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
 ******************************************************************************/
"""
from libs.Optical import *

sensor_type = "IMX265"
pixel_size_um = 3.45
h_pixels = 2056
v_pixels = 1542

f_num = 2
lens_fl_mm = 12

obj_dist_mm = 50 * 1000

optical = Optical(h_pixels, v_pixels, pixel_size_um, f_num, lens_fl_mm)
sensor = optical.sensor
lens = optical.lens

print("Sensor information - {}".format(sensor_type))
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
print("Pixel size at {} m: {:.2f} mm, {:.2f} lp/mm".format(obj_dist_mm / 1000,
                           obj_res['pixel_mm'], obj_res['lp_mm']))
print("Scene size at {} m: {:.2f} x {:.2f} (m)".format(obj_dist_mm / 1000,
                           obj_res['h_mm'] / 1000, obj_res['v_mm'] / 1000))

print("Hyper focal distance: {:.2f} m".format(optical.lens.hyper_focal_dist_mm() / 1000))