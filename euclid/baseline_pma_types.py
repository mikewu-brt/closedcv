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
 *  PMA Plane structure definition (ASIC)
 *
"""
import re
import ctypes
MAX_NUM_PLANES = 44
MAX_D_SHIFTS = 8
MAX_PASS_TIMING = 8
MAX_TIMINGS = 3


class BL_PLANE_CONFIGURATION_PARAMS_T(ctypes.LittleEndianStructure):
    _fields_ = [("a", ctypes.c_int8),
                ("b", ctypes.c_int8),
                ("c", ctypes.c_int8),
                ("subpixel_offset", ctypes.c_uint8),
                ("mv_patch_timing_sel", ctypes.c_uint8),
                ("d_shift_dim", ctypes.c_uint8),
                ("gg_map", ctypes.c_uint64 * 32),
                ("d_shift", ctypes.c_uint32 * MAX_D_SHIFTS)]


class BL_PLANE_CONFIG_HEADER_T(ctypes.LittleEndianStructure):
    _fields_ = [("baseline_idx", ctypes.c_uint8, 4),
                ("reserved", ctypes.c_uint8, 3),
                ("is_baseline", ctypes.c_uint8, 1)]


class BL_PLANE_PMAX_INFO_T(ctypes.LittleEndianStructure):
    _fields_ = [("config_idx", ctypes.c_uint8),
                ("num_planes", ctypes.c_uint8),
                ("patch_config_idx", ctypes.c_uint8),
                ("n_xy", ctypes.c_uint16),
                ("disparity", ctypes.c_int16)]


class BL_PLANE_CONFIGURATION_COMMAND_T(ctypes.LittleEndianStructure):
    _fields_ = [("header", BL_PLANE_CONFIG_HEADER_T),
                ("num_planes", ctypes.c_uint8),
                ("ref_patch_offset_q2", ctypes.c_int8),
                ("max_disparity", ctypes.c_int16),
                ("pmax_info", BL_PLANE_PMAX_INFO_T * MAX_PASS_TIMING),
                ("mv_n_xy", ctypes.c_int16 * MAX_TIMINGS),
                ("params", BL_PLANE_CONFIGURATION_PARAMS_T * MAX_NUM_PLANES)]


def pack(ctype_instance):
    buf = ctypes.string_at(ctypes.byref(ctype_instance), ctypes.sizeof(ctype_instance))
    return buf


def unpack(ctype, buf):
    cstring = ctypes.create_string_buffer(buf)
    ctype_instance = ctypes.cast(ctypes.pointer(cstring), ctypes.POINTER(ctype)).contents
    return ctype_instance


def parse_type(field):
    t = re.search(".*types\.(c_\w*)", str(field[1]))
    tc = re.search(".*types\.([\w_]*)", str(field[1]))
    ta = re.search("baseline_pma_types.([\w_]*)_Array_([0-9]*)", str(field[1]))
    c_type = None
    array = 1
    bits = 0

    # Parse array type
    if ta:
        t = [ta.group(0), ta.group(1)]
        array = int(ta.group(2))

        # Check for custom type
        if not re.search("^c_.*", ta.group(1)):
            return ta.group(1).lower(), array, bits

    if t:
        if t[1] == "c_byte":
            c_type = "std::int8_t"
        elif t[1] == "c_ubyte":
            c_type = "std::uint8_t"
        elif t[1] == "c_short":
            c_type = "std::int16_t"
        elif t[1] == "c_ushort":
            c_type = "std::uint16_t"
        elif t[1] == "c_float":
            c_type = "float"
        elif t[1] == "c_uint":
            c_type = "std::uint32_t"
        elif t[1] == "c_int":
            c_type = "std::int32_t"
        elif t[1] == "c_ulong":
            c_type = "std::uint64_t"
        else:
            print("Unknown type: " + t[1])
    else:
        c_type = tc.group(1).lower()

    if len(field) > 2:
        bits = field[2]

    return c_type, array, bits


def log_plane_config(buf):
    plane_command = unpack(BL_PLANE_CONFIGURATION_COMMAND_T, buf)
    for f in BL_PLANE_CONFIGURATION_COMMAND_T._fields_:
        if f[0] == "ref_patch_offset_q2":
            data = eval("plane_command.{}".format(f[0]))
            data = data / 4.
        elif f[0] == "header":
            print("plane_command.header.is_baseline = {}".format(plane_command.header.is_baseline))
            print("plane_command.header.baseline_idx = {}".format(plane_command.header.baseline_idx))
        elif f[0] == "mv_n_xy":
            print("plane_command.{} = {}".format(f[0], plane_command.mv_n_xy[:]))
        else:
            data = eval("plane_command.{}".format(f[0]))
            print("plane_command.{} = {}".format(f[0], data))

    for idx in range(MAX_PASS_TIMING):
        print("")
        for f in BL_PLANE_PMAX_INFO_T._fields_:
            data = eval("plane_command.pmax_info[{}].{}".format(idx, f[0]))
            print("plane_command.pmax_info[{}].{} = {}".format(idx, f[0], data))

    for idx in range(plane_command.num_planes):
        print("")
        for f in BL_PLANE_CONFIGURATION_PARAMS_T._fields_:
            if f[0] == "reserved":
                _, a, _ = parse_type(f)
                for i in range(a):
                    data = eval("plane_command.params[{}].{}[{}]".format(idx, f[0], i))
                    print("plane_command.params[{}].{}[{}] = {}".format(idx, f[0], i, data))
            elif f[0] == "gg_map":
                _, a, _ = parse_type(f)
                for i in range(a):
                    data = hex(eval("plane_command.params[{}].{}[{}]".format(idx, f[0], i)))
                    print("plane_command.params[{}].{}[{}] = {}".format(idx, f[0], i, data))
            elif f[0] == "a" or f[0] == "b":
                data = eval("plane_command.params[{}].{}".format(idx, f[0])) / 256.0
                print("plane_command.params[{}].{} = {} ({})".format(idx, f[0], data, int(data * 256)))
            elif f[0] == "c":
                data = eval("plane_command.params[{}].{}".format(idx, f[0])) / 4.0
                print("plane_command.params[{}].{} = {} ({})".format(idx, f[0], data, int(data * 4)))
            elif f[0] == "d_shift":
                _, a, _ = parse_type(f)
                for i in range(a):
                    data = hex(eval("plane_command.params[{}].{}[{}]".format(idx, f[0], i)))
                    print("plane_command.params[{}].{}[{}] = {}".format(idx, f[0], i, data))
            else:
                data = eval("plane_command.params[{}].{}".format(idx, f[0]))
                print("plane_command.params[{}].{} = {}".format(idx, f[0], data))
