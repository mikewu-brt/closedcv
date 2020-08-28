"""
 * Copyright (c) 2020, The LightCo
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are strictly prohibited without prior permission of
 * The LightCo.
 *
 * @author  Chuck Stanski
 * @version V1.0.0
 * @date    September 2020
 * @brief
 *  DRF v2 (fDRF) Output structure definition
 *
"""
import ctypes

class DRF_V2_SCORE_T(ctypes.LittleEndianStructure):
    _fields_ = [("k_index", ctypes.c_uint16, 11),
                ("plane_idx", ctypes.c_uint16, 5)]

class DRF_V2_OUTPUT_T(ctypes.LittleEndianStructure):
    _fields_ = [("best_score", ctypes.c_uint8 * 8),
                ("best_score_index", DRF_V2_SCORE_T * 8)]


def __log_drf_v2_internal(ctype_instance):
    print("Best score: {} ".format(ctype_instance.best_score[:]))
    k_index = []
    plane_idx = []
    for i in ctype_instance.best_score_index:
        k_index.append(i.k_index)
        plane_idx.append(i.plane_idx)
    print("k_index:    {} ".format(k_index))
    print("plane_index:{} ".format(plane_idx))

def log_drf_v2(ctype_instance):
    if type(ctype_instance) is list:
        for elem in ctype_instance:
            __log_drf_v2_internal(elem)
            print("")
    else:
        __log_drf_v2_internal(ctype_instance)
        print("")

def __log_drf_v2_internal(ctype_instance, fp_file, rtl_best_score_arr, rtl_depth_arr, rtl_plane_arr):
    print("Best score: {} ".format(ctype_instance.best_score[:]), file=fp_file)
    rtl_best_score_arr.append(ctype_instance.best_score[:])
    k_index = []
    plane_idx = []
    for i in ctype_instance.best_score_index:
        k_index.append(i.k_index)
        plane_idx.append(i.plane_idx)
    rtl_depth_arr.append(k_index)
    rtl_plane_arr.append(plane_idx)
    print("k_index:    {} ".format(k_index), file=fp_file)
    print("plane_index:{} ".format(plane_idx), file=fp_file)

def log_drf_v2(ctype_instance, fp_file, rtl_best_score_arr, rtl_depth_arr, rtl_plane_arr):
    if type(ctype_instance) is list:
        for elem in ctype_instance:
            __log_drf_v2_internal(elem, fp_file, rtl_best_score_arr, rtl_depth_arr, rtl_plane_arr)
            print("", fp_file)
    else:
        __log_drf_v2_internal(ctype_instance, fp_file, rtl_best_score_arr, rtl_depth_arr, rtl_plane_arr)
        print("", fp_file)
