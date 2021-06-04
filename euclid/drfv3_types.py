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

class DRF_V3_RECORD_T(ctypes.LittleEndianStructure):
    _fields_ = [("disparity", ctypes.c_uint32, 10),
                ("plane_idx", ctypes.c_uint32, 6),
                ("score", ctypes.c_uint32, 10),
                ("is_flat", ctypes.c_uint32, 1),
                ("reserved", ctypes.c_uint32, 1),
                ("meta_data", ctypes.c_uint32, 4)]

class DRF_V3_OUTPUT_T(ctypes.LittleEndianStructure):
    _fields_ = [("word", DRF_V3_RECORD_T * 8)]


def __log_drf_v3_internal(ctype_instance):
    for i in ctype_instance.word:
        print("meta_data: {}, is_flat: {}, score: {}, plane_idx: {}, disparity: {}".format(i.meta_data, i.is_flat, \
                                                                       i.score, i.plane_idx, i.disparity))

def log_drf_v3(ctype_instance):
    if type(ctype_instance) is list:
        for elem in ctype_instance:
            __log_drf_v3_internal(elem)
            print("")
    else:
        __log_drf_v3_internal(ctype_instance)
        print("")

def __log_drf_v3_internal(ctype_instance, fp_file, rtl_best_score_arr, rtl_disparity_arr, rtl_plane_arr):
    #print("Best score: {} ".format(ctype_instance.best_score[:]), file=fp_file)
    #rtl_best_score_arr.append(ctype_instance.best_score[:])
    score_arr = []
    disparity_arr = []
    plane_idx = []
    for i in ctype_instance.word:
        score_arr.append(i.score)
        disparity_arr.append(i.disparity)
        plane_idx.append(i.plane_idx)
    rtl_disparity_arr.append(disparity_arr)
    rtl_plane_arr.append(plane_idx)
    print("score:    {} ".format(score_arr), file=fp_file)
    print("disparity:    {} ".format(disparity_arr), file=fp_file)
    print("plane_index:{} ".format(plane_idx), file=fp_file)

def log_drf_v3(ctype_instance, fp_file, rtl_best_score_arr, rtl_disparity_arr, rtl_plane_arr):
    if type(ctype_instance) is list:
        for elem in ctype_instance:
            __log_drf_v3_internal(elem, fp_file, rtl_best_score_arr, rtl_disparity_arr, rtl_plane_arr)
            print("", fp_file)
    else:
        __log_drf_v3_internal(ctype_instance, fp_file, rtl_best_score_arr, rtl_disparity_arr, rtl_plane_arr)
        print("", fp_file)