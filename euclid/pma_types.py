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
 *  PMA Plane stricture definition
 *
"""
import re
import ctypes
MAX_NUM_PLANES = 11
MAX_D_SHIFTS = 8
MAX_TIMINGS = 4

class PLANE_CONFIGURATION_PARAMS_T(ctypes.LittleEndianStructure):
    _fields_ = [("a", ctypes.c_int8),
                ("b", ctypes.c_int8),
                ("c", ctypes.c_int8),
                ("offset", ctypes.c_uint8),
                ("patch_timing_sel", ctypes.c_int8),
                ("d_shift_dim", ctypes.c_uint8),
                ("gg_map", ctypes.c_uint32 * (32 * 2)),
                ("d_shift", ctypes.c_uint32 * MAX_D_SHIFTS)]


class PLANE_CONFIGURATION_COMMAND_T(ctypes.LittleEndianStructure):
    _fields_ = [("config_idx", ctypes.c_uint8),
                ("num_planes", ctypes.c_uint8),
                ("ref_patch_offset_q2", ctypes.c_int8),
                ("n_xy", ctypes.c_int16 * MAX_TIMINGS),
                ("params", PLANE_CONFIGURATION_PARAMS_T * MAX_NUM_PLANES)]


def pack(ctype_instance):
    buf = ctypes.string_at(ctypes.byref(ctype_instance), ctypes.sizeof(ctype_instance))
    return buf


def unpack(ctype, buf):
    cstring = ctypes.create_string_buffer(buf)
    ctype_instance = ctypes.cast(ctypes.pointer(cstring), ctypes.POINTER(ctype)).contents
    return ctype_instance


def parse_type(field):
    t = re.search(".*types\.(c_\w*)", str(field[1]))
    ta = re.search("pma_types.([\w_]*)_Array_([0-9]*)", str(field[1]))
    c_type = None
    array = 1

    # Parse array type
    if ta:
        t = [ta.group(0), ta.group(1)]
        array = int(ta.group(2))

        # Check for custom type
        if not re.search("^c_.*", ta.group(1)):
            return ta.group(1).lower(), array

    if t:
        if t[1] == "c_byte":
            c_type = "std::int8_t"
        elif t[1] == "c_ubyte":
            c_type = "std::uint8_t"
        elif t[1] == "c_short":
            c_type = "std::uint16_t"
        elif t[1] == "c_float":
            c_type = "float"
        elif t[1] == "c_uint":
            c_type = "std::uint32_t"
        elif t[1] == "c_int":
            c_type = "std::int32_t"
        else:
            print("Unknown type: " + t[1])
    else:
        print("Parse error: {}".format(field[1]))

    return c_type, array


def log_plane_config(buf):
    plane_command = unpack(PLANE_CONFIGURATION_COMMAND_T, buf)
    for f in PLANE_CONFIGURATION_COMMAND_T._fields_:
        data = eval("plane_command.{}".format(f[0]))
        if f[0] == "ref_patch_offset_q2":
            data = data / 4.

        if f[0] == "n_xy":
            print("plane_command.{} = {}".format(f[0], plane_command.n_xy[:]))
        elif f[0] != "params":
            print("plane_command.{} = {}".format(f[0], data))

    for idx in range(plane_command.num_planes):
        print("")
        for f in PLANE_CONFIGURATION_PARAMS_T._fields_:
            if f[0] == "reserved":
                _, a = parse_type(f)
                for i in range(a):
                    data = eval("plane_command.params[{}].{}[{}]".format(idx, f[0], i))
                    print("plane_command.params[{}].{}[{}] = {}".format(idx, f[0], i, data))
            elif f[0] == "gg_map":
                _, a = parse_type(f)
                for i in range(a // 2):
                    data_lo = eval("plane_command.params[{}].{}[{}]".format(idx, f[0], 2*i))
                    data_hi = eval("plane_command.params[{}].{}[{}]".format(idx, f[0], 2*i+1))
                    data = hex(data_lo + data_hi * 2**32)
                    print("plane_command.params[{}].{}[{}] = {}".format(idx, f[0], i, data))
            elif f[0] == "a" or f[0] == "b":
                data = eval("plane_command.params[{}].{}".format(idx, f[0])) / 256.0
                print("plane_command.params[{}].{} = {}".format(idx, f[0], data))
            elif f[0] == "c":
                data = eval("plane_command.params[{}].{}".format(idx, f[0])) / 4.0
                print("plane_command.params[{}].{} = {}".format(idx, f[0], data))
            else:
                data = eval("plane_command.params[{}].{}".format(idx, f[0]))
                print("plane_command.params[{}].{} = {}".format(idx, f[0], data))
