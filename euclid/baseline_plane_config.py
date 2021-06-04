#!/usr/bin/python3
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
 *  PMA Plane configuration files
 *
"""
import argparse
import re
import datetime
import os
import numpy as np
from grid_map import *
from baseline_pma_types import *
from google.protobuf import json_format


"""
*********************************** MAIN ********************************
"""
parser = argparse.ArgumentParser(description="PMA Plane Configuration")
parser.add_argument('--out_basename', help="<Required> Output base filename")
parser.add_argument('--h_file', action='store_true', default=False, help="<Optional> Generates H file output")
parser.add_argument('--data_file', action='store_true', default=False, help="<Optional> Generate data file")
parser.add_argument('--read_file', action='store_true', default=False, help="<Optional> Read data file")
parser.add_argument('--align_ref_offset', type=float, default=-1.0,
                    help="<Optional> Aligns reference patch to offset.  -1.0 means default alignment")
parser.add_argument('--test_euclid', action='store_true', default=False, help="<Optional> Use testEuclid JSON parser")
parser.add_argument('--json', help="keep scripts consistent")
parser.add_argument('--test_name', help="keep scripts consistent")
parser.add_argument('--index', type=int, help="keep scripts consistent")
parser.add_argument('--hw_scripts', default=".", help="keep scripts consistent.")

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

# Generate H file (if required)
if args.h_file:
    filename = args.out_basename + ".h"
    macro_name = re.sub("/", "_", args.out_basename.upper())
    macro_name = macro_name + "_H"
    namespace_name = re.sub("\.h", "", os.path.basename(filename))
    outfile = open(filename, "wt")

    outfile.write("// AUTOGENERATED FILE - DO NOT EDIT!\n")
    outfile.write("\n")
    outfile.write("/*******************************************************************************\n")
    outfile.write(" * Copyright (c) 2020 Light Labs Inc.\n")
    outfile.write(" * All Rights Reserved\n")
    outfile.write(" * Proprietary and Confidential - Light Labs Inc. \n")
    outfile.write(" * Redistribution and use in source and binary forms, with or without\n")
    outfile.write(" * modification, are strictly prohibited without prior permission of\n")
    outfile.write(" * Light Labs Inc.\n")
    outfile.write(" *\n")
    outfile.write(" * @author  Light Labs Inc\n")
    outfile.write(" * @version V1.0.0\n")
    d = datetime.date.today()
    outfile.write(" * @date    {} {}, {}\n".format(d.strftime('%B'), d.day, d.year))
    outfile.write(" * @brief   \n")
    outfile.write(" *\n")
    outfile.write(" ******************************************************************************/\n")
    outfile.write("\n")
    outfile.write("#ifndef {}_\n".format(macro_name))
    outfile.write("#define {}_\n".format(macro_name))
    outfile.write("\n")
    outfile.write("#include <cstdint>\n\n")
    outfile.write("namespace {}\n".format(namespace_name))
    outfile.write("{\n")

    outfile.write("    struct {}\n".format("BL_PLANE_CONFIG_HEADER_T".lower()))
    outfile.write("    {\n")
    for f in BL_PLANE_CONFIG_HEADER_T._fields_:
        t, a, b = parse_type(f)
        outfile.write("        {} {}".format(t, f[0]))
        if a > 1:
            outfile.write("[{}]".format(a))
        if b > 0:
            outfile.write(":{}".format(b))
        outfile.write(";\n")
    outfile.write("    };\n")
    outfile.write("\n")

    outfile.write("    struct {}\n".format("BL_PLANE_PMAX_INFO_T".lower()))
    outfile.write("    {\n")
    for f in BL_PLANE_PMAX_INFO_T._fields_:
        t, a, b = parse_type(f)
        outfile.write("        {} {}".format(t, f[0]))
        if a > 1:
            outfile.write("[{}]".format(a))
        if b > 0:
            outfile.write(":{}".format(b))
        outfile.write(";\n")
    outfile.write("    };\n")
    outfile.write("\n")

    outfile.write("    struct {}\n".format("BL_PLANE_CONFIGURATION_PARAMS_T".lower()))
    outfile.write("    {\n")
    for f in BL_PLANE_CONFIGURATION_PARAMS_T._fields_:
        t, a, b = parse_type(f)
        outfile.write("        {} {}".format(t, f[0]))
        if a > 1:
            outfile.write("[{}]".format(a))
        if b > 0:
            outfile.write(":{}".format(b))
        outfile.write(";\n")
    outfile.write("    };\n")
    outfile.write("\n")

    outfile.write("    struct {}\n".format("BL_PLANE_CONFIGURATION_COMMAND_T".lower()))
    outfile.write("    {\n")
    for f in BL_PLANE_CONFIGURATION_COMMAND_T._fields_:
        t, a, b = parse_type(f)
        outfile.write("        {} {}".format(t, f[0]))
        if a > 1:
            outfile.write("[{}]".format(a))
        if b > 0:
            outfile.write(":{}".format(b))
        outfile.write(";\n")
    outfile.write("    };\n")

    outfile.write("}}  // namespace {}\n\n".format(namespace_name))
    outfile.write("#endif  // namespace {}_\n\n".format(macro_name))
    outfile.close()

if args.data_file:
    if args.test_euclid:
        import commandfile_pb2

        # Open JSON file
        json_str = open(args.json, 'r').read()
        tests = commandfile_pb2.TestRun()
        json_format.Parse(json_str, tests)

        # Find request test case
        test_case = []
        for case in tests.test_case_run:
            if case.test_name == args.test_name:
                test_case = case
                break

        if test_case.test_name != args.test_name:
            print("Could not find test case: {} in: {}".format(test_case, args.json))
            exit(-1)

        # Determine output filename
        filename = test_case.test_setup[0].pma_plane_configuration[args.index].file_path

        # Read Patch configurations
        patch_config = []
        for i in range(len(test_case.test_setup[0].pma_patch_configuration)):
            patch_config.append(dict())

        Jx = test_case.test_setup[0].pma_patch_configuration[0].jump_x
        Jy = test_case.test_setup[0].pma_patch_configuration[0].jump_x
        for p in test_case.test_setup[0].pma_patch_configuration:
            patch_idx = p.patch_configuration_index
            patch_config[patch_idx]["Cx"] = p.context_x
            patch_config[patch_idx]["Cy"] = p.context_y
            patch_config[patch_idx]["Px"] = p.patch_x
            patch_config[patch_idx]["Py"] = p.patch_y
            patch_config[patch_idx]["Sx"] = p.step_x
            patch_config[patch_idx]["Sy"] = p.step_y
            patch_config[patch_idx]["Jx"] = p.jump_x
            patch_config[patch_idx]["Jy"] = p.jump_y
            if (p.jump_x != Jx) or (p.jump_y != Jy):
                print("All Jx and Jy must be the same for all patch configurations")
                exit(-1)

        # Read PMA Pass information
        pma_pass_info = []
        pass_index = 0
        mv_timing_sel = 0
        for p in test_case.test_setup[0].pma_pass_configuration[args.index].pma_pass_parameters:

            for plane_params_idx in range(len(p.plane_parameters)):
                if plane_params_idx > 0 and p.patch_pma_config[plane_params_idx] == p.patch_pma_config[plane_params_idx - 1]:
                    break

                pma_pass_info.append(dict())
                pma_pass_info[-1]["patch_idx"] = p.patch_pma_config[plane_params_idx]
                pma_pass_info[-1]["plane_config_index"] = p.plane_config_select
                pma_pass_info[-1]["disparity"] = p.max_disparity
                pma_pass_info[-1]["a"] = [p.plane_parameters[plane_params_idx].a.start,
                                          p.plane_parameters[plane_params_idx].a.stop,
                                          p.plane_parameters[plane_params_idx].a.step]
                pma_pass_info[-1]["b"] = [p.plane_parameters[plane_params_idx].b.start,
                                          p.plane_parameters[plane_params_idx].b.stop,
                                          p.plane_parameters[plane_params_idx].b.step]
                pma_pass_info[-1]["c"] = [p.plane_parameters[plane_params_idx].c.start,
                                          p.plane_parameters[plane_params_idx].c.stop,
                                          p.plane_parameters[plane_params_idx].c.step]
                pma_pass_info[-1]["skip_nd"] = p.plane_parameters[plane_params_idx].skip_non_distorted

                if pass_index > 0:
                    for i in range(pass_index):
                        if pma_pass_info[i]["patch_idx"] == pma_pass_info[-1]["patch_idx"]:
                            pma_pass_info[-1]["mv_timing_sel"] = pma_pass_info[i]["mv_timing_sel"]

                if pma_pass_info[-1].get("mv_timing_sel") is None:
                    pma_pass_info[-1]["mv_timing_sel"] = mv_timing_sel
                    mv_timing_sel += 1

                pass_index += 1

        if mv_timing_sel > 3:
            print("Too many timings.  ASIC only support 3 timings")
            exit(-1)

    else:
        import depthhw_config_pb2

        # Open JSON file
        json_str = open(args.json, 'r').read()
        test_case = depthhw_config_pb2.DepthHWConfiguration()
        json_format.Parse(json_str, test_case)

        # Determine output filename
        filename = test_case.pma_plane_configuration[args.index].file_path

        # Read Patch configurations
        patch_config = []
        for i in range(len(test_case.pma_patch_configuration)):
            patch_config.append(dict())

        Jx = test_case.pma_patch_configuration[0].jump_x
        Jy = test_case.pma_patch_configuration[0].jump_x
        for p in test_case.pma_patch_configuration:
            patch_config.append(dict())
            patch_idx = p.patch_configuration_index
            patch_config[patch_idx]["Cx"] = p.context_x
            patch_config[patch_idx]["Cy"] = p.context_y
            patch_config[patch_idx]["Px"] = p.patch_x
            patch_config[patch_idx]["Py"] = p.patch_y
            patch_config[patch_idx]["Sx"] = p.step_x
            patch_config[patch_idx]["Sy"] = p.step_y
            patch_config[patch_idx]["Jx"] = p.jump_x
            patch_config[patch_idx]["Jy"] = p.jump_y
            if (p.jump_x != Jx) or (p.jump_y != Jy):
                print("All Jx and Jy must be the same for all patch configurations")
                exit(-1)

        # Read PMA Pass information
        pma_pass_info = []
        pass_index = 0
        mv_timing_sel = 0
        for p in test_case.pma_parameters[args.index].pma_pass_parameters:

            for idx in range(len(p.plane_parameters)):
                if idx > 0 and p.patch_pma_config_a == p.patch_pma_config_b:
                    break

                if idx == 0:
                    patch_config_idx = p.patch_pma_config_a
                else:
                    patch_config_idx = p.patch_pma_config_b

                pma_pass_info.append(dict())
                pma_pass_info[-1]["patch_idx"] = patch_config_idx
                pma_pass_info[-1]["plane_config_index"] = p.plane_config_select
                pma_pass_info[-1]["disparity"] = p.max_disparity
                pma_pass_info[-1]["a"] = [p.plane_parameters[idx].a.start,
                                          p.plane_parameters[idx].a.stop,
                                          p.plane_parameters[idx].a.step]
                pma_pass_info[-1]["b"] = [p.plane_parameters[idx].b.start,
                                          p.plane_parameters[idx].b.stop,
                                          p.plane_parameters[idx].b.step]
                pma_pass_info[-1]["c"] = [p.plane_parameters[idx].c.start,
                                          p.plane_parameters[idx].c.stop,
                                          p.plane_parameters[idx].c.step]
                pma_pass_info[-1]["skip_nd"] = p.plane_parameters[idx].skip_non_distorted

                if pass_index > 0:
                    for i in range(pass_index):
                        if pma_pass_info[i]["patch_idx"] == pma_pass_info[-1]["patch_idx"]:
                            pma_pass_info[-1]["mv_timing_sel"] = pma_pass_info[i]["mv_timing_sel"]

                if pma_pass_info[-1].get("mv_timing_sel") is None:
                    pma_pass_info[-1]["mv_timing_sel"] = mv_timing_sel
                    mv_timing_sel += 1

                pass_index += 1

        if mv_timing_sel > 3:
            print("Too many timings.  ASIC only support 3 timings")
            exit(-1)

    # Loop over the number of passes
    for pass_index in range(len(pma_pass_info)):

        # Error check
        if pma_pass_info[pass_index]["plane_config_index"] < 0 or pma_pass_info[pass_index]["plane_config_index"] > 7:
            print("plane_config_sel must be between 0 and 7")
            sys.exit(-1)

        if args.index < 0 or args.index > 2:
            print("--index must be between 0 and 2")
            sys.exit(-1)

        pma = []
        P = [patch_config[pma_pass_info[pass_index]["patch_idx"]]["Px"], patch_config[pma_pass_info[pass_index]["patch_idx"]]["Py"]]
        S = [patch_config[pma_pass_info[pass_index]["patch_idx"]]["Sx"], patch_config[pma_pass_info[pass_index]["patch_idx"]]["Sy"]]
        J = [patch_config[pma_pass_info[pass_index]["patch_idx"]]["Jx"], patch_config[pma_pass_info[pass_index]["patch_idx"]]["Jy"]]

        if pma_pass_info[pass_index]["skip_nd"] == 0:
            drop_nd = False
        else:
            drop_nd = True

        if pma_pass_info[pass_index]["a"][2] != 0.0:
            scale_min = pma_pass_info[pass_index]["a"][0] * (P[0] - 1) / 2
            scale_max = pma_pass_info[pass_index]["a"][1] * (P[0] - 1) / 2
            scale_step = pma_pass_info[pass_index]["a"][2]

            if scale_step == 0:
                print("astep cannot be 0")
                sys.exit(-1)

            for scale in np.arange(scale_min, scale_max + scale_step / 2, scale_step):
                if drop_nd and scale == 0.0:
                    continue

                if scale == 0.0:
                    drop_nd = True

                d, gg_map, C, d0, _, d_shift, _, align_err = compute_gg_map(P[0], P[1], S[0], S[1], J[0], J[1],
                    compute_ab(P[0], scale), compute_ab(P[0], 0), 0)

                pma.append(dict())
                pma[-1]['d'] = d
                pma[-1]['gg_map'] = gg_map
                pma[-1]['C'] = C
                pma[-1]['d_shift'] = d_shift
                pma[-1]['subpixel_offset'] = 0

        if pma_pass_info[pass_index]["b"][2] != 0.0:
            shear_min = pma_pass_info[pass_index]["b"][0] * (P[0] - 1) / 2
            shear_max = pma_pass_info[pass_index]["b"][1] * (P[0] - 1) / 2
            shear_step = pma_pass_info[pass_index]["b"][2]

            if shear_step == 0:
                print("bstep cannot be 0")
                sys.exit(-1)

            for shear in np.arange(shear_min, shear_max + shear_step / 2, shear_step):
                if drop_nd and shear == 0.0:
                    continue

                if shear == 0.0:
                    drop_nd = True

                d, gg_map, C, d0, _, d_shift, _, align_err = compute_gg_map(P[0], P[1], S[0], S[1], J[0], J[1],
                    compute_ab(P[0], 0), compute_ab(P[0], shear), 0)

                pma.append(dict())
                pma[-1]['d'] = d
                pma[-1]['gg_map'] = gg_map
                pma[-1]['C'] = C
                pma[-1]['d_shift'] = d_shift
                pma[-1]['subpixel_offset'] = 0

        if pma_pass_info[pass_index]["c"][2] != 0.0:
            offset_min = pma_pass_info[pass_index]["c"][0] * 4
            offset_max = pma_pass_info[pass_index]["c"][1] * 4
            offset_step = pma_pass_info[pass_index]["c"][2] * 4

            if offset_step == 0:
                print("cstep cannot be 0")
                sys.exit(-1)

            for offset in np.arange(offset_min, offset_max + offset_step / 2, offset_step):
                if drop_nd and offset == 0.0:
                    continue

                if offset == 0.0:
                    drop_nd = True

                d, gg_map, C, d0, _, d_shift, _, align_err = compute_gg_map(P[0], P[1], S[0], S[1], J[0], J[1],
                    compute_ab(P[0], 0), compute_ab(P[0], 0), offset)

                pma.append(dict())
                pma[-1]['d'] = d
                pma[-1]['gg_map'] = gg_map
                pma[-1]['C'] = C
                pma[-1]['d_shift'] = d_shift
                pma[-1]['subpixel_offset'] = offset

        # Find the max Cx and use for all planes
        Cx_max = 0
        for p in pma:
            if p['C']['Cx'] > Cx_max:
                Cx_max = p['C']['Cx']

        # Generate data file to push to FW
        if pass_index > 0:
            # Read existing file
            infile = open(filename, "rb")
            plane_command = unpack(BL_PLANE_CONFIGURATION_COMMAND_T, infile.read())
            infile.close()
            plane_start_idx = plane_command.num_planes
            plane_command.num_planes += len(pma)

            # Determine pma_n_xy_idx
            for pma_info_idx in range(len(plane_command.pmax_info)):
                if plane_command.pmax_info[pma_info_idx].n_xy == 0:
                    break

            # Align amended patches to same ref_patch_offset_q2
            args.align_ref_offset = plane_command.ref_patch_offset_q2 / 4
        else:
            plane_command = BL_PLANE_CONFIGURATION_COMMAND_T()
            plane_command.header.is_baseline = 1
            plane_command.header.baseline_idx = args.index
            plane_command.num_planes = len(pma)
            plane_command.ref_patch_offset_q2 = int(d0[2] * 4)
            plane_start_idx = 0
            pma_info_idx = 0

        # Set pma_num_planes and config index
        plane_command.pmax_info[pma_info_idx].num_planes = len(pma)
        plane_command.pmax_info[pma_info_idx].config_idx = pma_pass_info[pass_index]["plane_config_index"]
        plane_command.pmax_info[pma_info_idx].disparity = pma_pass_info[pass_index]["disparity"]
        if abs(pma_pass_info[pass_index]["disparity"]) > abs(plane_command.max_disparity):
            plane_command.max_disparity = pma_pass_info[pass_index]["disparity"]

        # Align planes to ref_offset
        if args.align_ref_offset >= 0.0:
            plane_command.ref_patch_offset_q2 = int(args.align_ref_offset * 4 + 0.5)
            c_diff = plane_command.ref_patch_offset_q2 / 4 - d0[2]
            for plane in pma:
                plane['d'] = (plane['d'][0], plane['d'][1], plane['d'][2] + c_diff)

        n_xy = np.sum(pma[0]['gg_map']).astype(np.uint16)
        plane_command.pmax_info[pma_info_idx].n_xy = n_xy
        if (plane_command.mv_n_xy[pma_pass_info[pass_index]["mv_timing_sel"]] != 0) and \
                (plane_command.mv_n_xy[pma_pass_info[pass_index]["mv_timing_sel"]] != n_xy):
            print("All Nxy for a specific patch timing must be the same")
            exit(-1)
        plane_command.mv_n_xy[pma_pass_info[pass_index]["mv_timing_sel"]] = n_xy

        plane_command.pmax_info[pma_info_idx].patch_config_idx = pma_pass_info[pass_index]["patch_idx"]

        if (plane_start_idx + len(pma)) > MAX_NUM_PLANES:
            print("plane_start_idx + len(pma) exceeds the maximum number of planes")
            print("plane_start_idx: {}, len(pma): {}, MAX_NUM_PLANES: {}".format(plane_start_idx, len(pma),
                                                                                 MAX_NUM_PLANES))
            sys.exit(-1)

        for idx in range(len(pma)):
            plane_idx = plane_start_idx + idx
            plane_command.params[plane_idx].a = int(pma[idx]['d'][0] * 256)
            plane_command.params[plane_idx].b = int(pma[idx]['d'][1] * 256)
            plane_command.params[plane_idx].c = int(pma[idx]['d'][2] * 4)
            plane_command.params[plane_idx].mv_patch_timing_sel = pma_pass_info[pass_index]["mv_timing_sel"]
            plane_command.params[plane_idx].subpixel_offset = int(pma[idx]['subpixel_offset'])
            n_xy = np.sum(pma[idx]['gg_map']).astype(np.uint16)
            for map_idx in range(len(pma[idx]['C']['gg_map_ulong'])):
                plane_command.params[plane_idx].gg_map[map_idx] = pma[idx]['C']['gg_map_ulong'][map_idx]
            if pma[idx]['d_shift'].shape[0] != 1:
                plane_command.params[plane_idx].d_shift_dim = 1
            else:
                plane_command.params[plane_idx].d_shift_dim = 0
            d_shift = np.squeeze(pma[idx]['d_shift'])
            for d_shift_idx in range(d_shift.size):
                plane_command.params[plane_idx].d_shift[d_shift_idx] = d_shift[d_shift_idx]

        outfile = open(filename, "wb")
        outfile.write(pack(plane_command))
        outfile.close()

if args.read_file:
    if args.json == "hwdepth.json":
        # Open JSON file
        json_str = open(args.json, 'r').read()
        test_case = depthhw_config_pb2.DepthHWConfiguration()
        json_format.Parse(json_str, test_case)

        # Determine output filename
        filename = test_case.pma_plane_configuration[args.index].file_path
    else:
        # Open JSON file
        json_str = open(args.json, 'r').read()
        tests = commandfile_pb2.TestRun()
        json_format.Parse(json_str, tests)

        # Find request test case
        test_case = []
        for case in tests.test_case_run:
            if case.test_name == args.test_name:
                test_case = case
                break

        if test_case.test_name != args.test_name:
            print("Could not find test case: {} in: {}".format(test_case, args.json))
            exit(-1)

        filename = test_case.test_setup[0].pma_plane_configuration[args.index].file_path

    infile = open(filename, "rb")
    log_plane_config(infile.read())
    infile.close()