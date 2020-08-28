"""
 * Copyright (c) 2020, The LightCo
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are strictly prohibited without prior permission of
 * The LightCo.
 *
 * @author  Chuck Stanski
 * @version V1.0.0
 * @date    June 2020
 * @brief
 *  DRF Output structure definition
 *
"""
from ctypes import *

class DRF_OUTPUT_T(LittleEndianStructure):
    _fields_ = [("best_score", c_uint32, 10),
                ("worst_minima_score", c_uint32, 10),
                ("average_minima_score", c_uint32, 10),
                ("reserved", c_uint32, 2),
                ("best_score_k_index", c_uint32, 11),
                ("best_score_plane_index", c_uint32, 4),
                ("worst_min_score_k_index", c_uint32, 11),
                ("worst_min_score_plane_index", c_uint32, 4),
                ("d_sync_group", c_uint32, 1),
                ("d_sync_start", c_uint32, 1)]


def __log_drf_internal(ctype_instance):
    for field in DRF_OUTPUT_T._fields_:
        data = eval("ctype_instance.{}".format(field[0]))
        print("DRF_OUTPUT_T.{} = {}".format(field[0], data))


def log_drf(ctype_instance):
    if type(ctype_instance) is list:
        for elem in ctype_instance:
            __log_drf_internal(elem)
            print("")
    else:
        __log_drf_internal(ctype_instance)
        print("")

def __log_drf_internal(ctype_instance, fp_file, rtl_best_score_arr, rtl_depth_arr, rtl_plane_arr):
    for field in DRF_OUTPUT_T._fields_:
        data = eval("ctype_instance.{}".format(field[0]))
        #print(field[0])
        if (field[0] == "best_score"):
            print("DRF_OUTPUT_T.{} = {}".format(field[0], data), file=fp_file)
            rtl_best_score_arr.append(data)
        elif (field[0] == "best_score_k_index"):
            print("DRF_OUTPUT_T.{} = {}".format(field[0], data), file=fp_file)
            rtl_depth_arr.append(data)
        elif (field[0] == "best_score_plane_index"):
            print("DRF_OUTPUT_T.{} = {}".format(field[0], data), file=fp_file)
            rtl_plane_arr.append(data)


def log_drf(ctype_instance, fp_file, rtl_best_score_arr, rtl_depth_arr, rtl_plane_arr):
    if type(ctype_instance) is list:
        for elem in ctype_instance:
            __log_drf_internal(elem, fp_file, rtl_best_score_arr, rtl_depth_arr, rtl_plane_arr)
            print("", file=fp_file)
    else:
        __log_drf_internal(ctype_instance, fp_file, rtl_best_score_arr, rtl_depth_arr, rtl_plane_arr)
        print("",file=fp_file)