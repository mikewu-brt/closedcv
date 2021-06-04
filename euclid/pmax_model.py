#  Copyright (c) 2020, The LightCo
#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without
#  modification, are strictly prohibited without prior permission of
#  The LightCo.
#
#  @author  yhussain
#  @version V1.0.0
#  @date    Sept 2020 May 2021
#  @brief
#

from libs.Image import *
import os
from math import ceil, floor
import commandfile_pb2
from google.protobuf import json_format
import matplotlib as matplot
import cv2
import numpy as np
from copy import *
import argparse
matplot.use('TkAgg')
import matplotlib.pyplot as plt
import lt_protobuf.light_header.camera_id_pb2 as camera_id_pb2

from bitmap import BitMap

from uint_to_log2 import *
from compress import *
from drf_types import *
from baseline_pma_types import *
from drfv2_types import *
from drfv3_types import *
import ctypes # YH TODO: need to remove this as drf_types already has it

class Pmax_v34:
    def __init__(self ):

        parm_list =dict([
            # main PMA "triple"
            ('K',64), # HW parallelism factor
            ('C', 4), # color channels per pixel
            ('W',12), # bits per color channel
            ('freq',1e8),  # accelerator clock frequency
            # sensor dimentions in quads
            ('imgX' , 1604),
            ('imgY',1100),
            ('ctxtX_max',39),  # virtual address space is 64 (6bits)
            ('ctxtY_max',32),
            ('del2_max', 320),
            ('resW' , 10),  # 10 bits per score
            # ref and source memory sizes
            ('rmemX', 0),
            ('rmemY', 0),
            ('smemX', 0),
            ('smemY',0 ),
            # oversamplng and distortion model
            ('ovrsX' , 4),
            ('Rxm' , 0),
            ('Rym', 0),
            # fp step (skew or compression) per quad, -0.5 to +0.5
            ('distAf', 0.35),
            ('distBf', 0.),
            ('distCf', 0.),
            # scaling and rounding done in SW/firmware
            # A,B are S0.7, C is S5.2 (all 8-bit values)
            ('distA', 0),
            ('distB',0),
            ('distC', 0),
            # patch dimensions (and other parameters) for different verification tests
            # all test cases share the same jump and also use nested grids
            # (not required by HW, but will simplify SW)
            ('jumpX' , 20),
            ('jumpY' , 20),
            ('patchX' , 21),
            ('patchY' , 21),
            ('stepX' , 10),
            ('stepY' , 10),
            ('tileX',0),
            ('tileY',0),
            ('nresX',0),
            ('nresY',0),
            ('ggmap_output_loc',[] ),
            # context size
            ('ctxtY', 0),
            ('ctxtX', 0),
            # size of the patch unpacked into a linear array
            ('patchL', 0),
            # disparity search range
            ('NEGATIVE_DISPARITY_SLACK_PQ', 4),
            ('dispK', 0),
            ('max_disparity_pq', 0),
            ('dispX', 0 ),
            ('rej_thresh', 1e4),
        ])

        self.__num_minima = 8
        #self.__num_minima = 1
        self.__num_passes = 1
        self.__number_drf_sets = 1
        parm_list['rmemX'] = parm_list['ctxtX_max'];
        parm_list['rmemY'] = parm_list['ctxtY_max'];
        parm_list['smemX'] = parm_list['imgX'];
        parm_list['smemY'] = parm_list['ctxtY_max'];
        parm_list['Rxm'] = parm_list['rmemX'] * 2 * parm_list['ovrsX'];
        parm_list['Rym'] = parm_list['rmemY']
        # scaling and rounding done in SW/firmware
        # A,B are S0.7, C is S5.2 (all 8-bit values)
        parm_list['distA'] = floor(parm_list['distAf'] * 256 + 0.5);
        parm_list['distB'] = floor(parm_list['distBf'] * 256 + 0.5);
        parm_list['distC'] = floor(parm_list['distCf'] * 4 + 0.5);

        parm_list['tileX'] = parm_list['jumpX'] - parm_list['stepX'] + 1;
        parm_list['tileY'] = parm_list['jumpY'] - parm_list['stepY'] + 1;
        parm_list['nresX'] = (parm_list['tileX'] - 1) // parm_list['stepX'] + 1;
        parm_list['nresY'] = (parm_list['tileY'] - 1) // parm_list['stepY'] + 1;
        parm_list['ggmap_output_loc'] = np.zeros((parm_list['nresY'], parm_list['nresX'], 2), dtype=np.int32)

        # context size
        parm_list['ctxtY'] = parm_list['patchY'] + parm_list['tileY'] - 1;
        parm_list['ctxtX'] = parm_list['patchX'] + parm_list['tileX'] - 1;
        # increase context X-dim to allow for the full grd map to fit in
        tileX_ext = floor(max((parm_list['tileX'] - 1) * parm_list['distAf'], (parm_list['tileY'] - 1) * parm_list['distBf']) + 0.5);
        parm_list['ctxtX'] += tileX_ext;
        parm_list['tileX'] += tileX_ext;

        if (parm_list['ctxtX'] > parm_list['ctxtX_max']):
            print("WARNING: max context width exceeded", parm_list['ctxtX']);
        if (parm_list['ctxtY'] > parm_list['ctxtY_max']):
            print("WARNING: max context height exceeded", parm_list['ctxtY']);

        # size of the patch unpacked into a linear array
        parm_list['patchL'] = parm_list['patchY'] * parm_list['patchX'] * parm_list['C'];

        # disparity search range
        parm_list['dispK'] = parm_list['imgX'] // (5 * parm_list['K']);
        parm_list['max_disparity_pq'] = parm_list['dispK'] * parm_list['K'] //2
        parm_list['dispX'] = ceil(parm_list['max_disparity_pq'] / parm_list['K']) * parm_list['K']  # relates to number of c-scans in a d-scan

        ############

        self.__MAX_NUM_PLANES = 11
        self.__MAX_NUM_PASSES = 4
        self.__MAX_NUM_TIMINGS = 2
        self.__num_planes = np.zeros((self.__MAX_NUM_PASSES, self.__MAX_NUM_TIMINGS), dtype = np.int16)

        self.__arr_parm_list = []
        for i in range(self.__MAX_NUM_PLANES * self.__MAX_NUM_PASSES):
            self.__arr_parm_list.append(parm_list.copy())


        # --------
        index = 0
        print("\ngeneral info:")
        print("arch: 1xK, parallelism: %d, single plane" % (self.__arr_parm_list[index]['K']));
        print("image size in quads:", self.__arr_parm_list[index]['imgX'], "x", self.__arr_parm_list[index]['imgY']);
        print("disparity search range:", self.__arr_parm_list[index]['dispX'], "c-scans:", self.__arr_parm_list[index]['dispK']);
        print("selected patch size:", self.__arr_parm_list[index]['patchX'], "x", self.__arr_parm_list[index]['patchY']);
        print("selected patch step:", self.__arr_parm_list[index]['stepX'], "x", self.__arr_parm_list[index]['stepY']);
        print("selected tile jump:", self.__arr_parm_list[index]['jumpX'], "x", self.__arr_parm_list[index]['jumpY']);
        print("tile dimensions:", self.__arr_parm_list[index]['tileX'], "x", self.__arr_parm_list[index]['tileY']);
        print("context dimensions:", self.__arr_parm_list[index]['ctxtX'], "x", self.__arr_parm_list[index]['ctxtY']);
        print("number of results:", self.__arr_parm_list[index]['nresX'], "x", self.__arr_parm_list[index]['nresY']);

        del2_rm = self.__arr_parm_list[index]['patchY'] * (self.__arr_parm_list[index]['tileX'] + tileX_ext);
        del2_cm = self.__arr_parm_list[index]['patchX'] * self.__arr_parm_list[index]['tileY'];
        print("delay 2 len for row-maj c-scan", del2_rm);
        print("delay 2 len for col-maj c-scan", del2_cm);

        if (del2_rm > self.__arr_parm_list[index]['del2_max']):
            print("WARNING: max delay 2 len exceeded in row-maj c-scan");
        if (del2_cm > self.__arr_parm_list[index]['del2_max']):
            print("WARNING: max delay 2 len exceeded in col-maj c-scan");

        # --------
        cycles = self.__arr_parm_list[index]['ctxtX'] * self.__arr_parm_list[index]['ctxtY'];
        scores = self.__arr_parm_list[index]['nresX'] * self.__arr_parm_list[index]['nresY'] * self.__arr_parm_list[index]['K'];
        print("\nthroughput for %d-wide PMA, single plane:" % (self.__arr_parm_list[index]['K']));
        print("c-scan cycles", cycles);
        print("d-scan cycles", cycles * self.__arr_parm_list[index]['dispK']);
        print(scores, "scores per c-scan");
        print("%.3f" % (scores / cycles), "scores/cycle", "%.3f" % (self.__arr_parm_list[index]['resW'] * scores / cycles), "packed bits/cycle");
        print("%.3f" % (self.__arr_parm_list[index]['freq'] * self.__arr_parm_list[index]['resW'] * scores / (cycles * 8 * 1024 ** 2)), "packed Mbytes/sec");
        print("%.3f" % (self.__arr_parm_list[index]['freq'] * 16 * scores / (cycles * 8 * 1024 ** 2)), "unpacked Mbytes/sec");

        spf = self.__arr_parm_list[index]['imgX'] * self.__arr_parm_list[index]['imgY'] * self.__arr_parm_list[index]['dispX'] / (self.__arr_parm_list[index]['stepX'] * self.__arr_parm_list[index]['stepY']);
        print("scores per frame: %.2e" % (spf));
        fps = self.__arr_parm_list[index]['freq'] * scores / cycles / spf;
        print("FPS: %.3f, time %.3fms at %.0fMHz" % (fps, 1 / fps, self.__arr_parm_list[index]['freq'] / 1e6));

        # --------
        print("\nC-node config:")
        # for now using row-maj c-scan only
        # note: the del*_sel should be smaller by 2 than the actual delay
        # but for the first two delay lines the actual delay is >=3 so we are safe
        print("delay 1 sel", self.__arr_parm_list[index]['patchX'] - 2);
        print("delay 2 sel", del2_rm - 2);

        # the last (multithreaded acc) delay line has forwarding path to deal with del=1
        if (self.__arr_parm_list[index]['nresX'] - 2 < 0):
            print("delay 3 sel X");
            print("acc fwrd en", 1);
        else:
            print("delay 3 sel", self.__arr_parm_list[index]['nresX'] - 2);
            print("acc fwrd en", 0);

        # row-maj order only for now
        print("c-scan order", 0);

        # --------
        print("\nref_mem info:")
        print("oversampling in X direction:", self.__arr_parm_list[index]['ovrsX']);
        print("distortion a: %3d(%.3f), b: %3d(%.3f), c %d:" % (self.__arr_parm_list[index]['distA'], self.__arr_parm_list[index]['distA'] / 256, self.__arr_parm_list[index]['distB'], self.__arr_parm_list[index]['distB'] / 256, self.__arr_parm_list[index]['distC']));
        # print ("size for max  context dims and spec jumpX:", ceil(ovrsX*(ctxtX_max+(ctxtX_max-1)*0.35+jumpX)), "x", ctxtY_max);
        print("size for spec context dims and spec jumpX:", ceil(self.__arr_parm_list[index]['ovrsX'] * (self.__arr_parm_list[index]['ctxtX'] + (self.__arr_parm_list[index]['ctxtX'] - 1) * 0.35 + self.__arr_parm_list[index]['jumpX'])), "x",
              self.__arr_parm_list[index]['ctxtY']);
        print("steady state load rate:", self.__arr_parm_list[index]['ovrsX'] * self.__arr_parm_list[index]['jumpX'] * self.__arr_parm_list[index]['ctxtY'], "%dx-os quads per d-scan" % (self.__arr_parm_list[index]['ovrsX']));
        print("GEOX utilization at PMA limit of %.3f FPS: %d%%" % (
            fps, ceil(100 * self.__arr_parm_list[index]['ovrsX'] * self.__arr_parm_list[index]['jumpX'] * self.__arr_parm_list[index]['ctxtY'] / (cycles * self.__arr_parm_list[index]['dispK']))));

       ########################end##############


    def K(self, idx):
        return self.__arr_parm_list[idx]['K']
    def dispK(self, idx):
        return self.__arr_parm_list[idx]['dispK']
    def imgY(self, idx ):
        return self.__arr_parm_list[idx]['imgY']
    def imgX(self, idx ):
        return self.__arr_parm_list[idx]['imgX']
    def C(self, idx ):
        return self.__arr_parm_list[idx]['C']
    def ovrsX(self, idx ):
        return self.__arr_parm_list[idx]['ovrsX']
    def rmemX(self, idx ):
        return self.__arr_parm_list[idx]['rmemX']
    def rmemY(self, idx ):
        return self.__arr_parm_list[idx]['rmemY']
    def Rxm(self, idx ):
        return self.__arr_parm_list[idx]['Rxm']
    def Rym(self, idx ):
        return self.__arr_parm_list[idx]['Rym']
    def jumpX(self, idx ):
        return self.__arr_parm_list[idx]['jumpX']
    def jumpY(self, idx ):
        return self.__arr_parm_list[idx]['jumpY']
    def stepX(self, idx ):
        return self.__arr_parm_list[idx]['stepX']
    def stepY(self, idx ):
        return self.__arr_parm_list[idx]['stepY']
    def tileX(self, idx ):
        return self.__arr_parm_list[idx]['tileX']
    def tileY(self, idx ):
        return self.__arr_parm_list[idx]['tileY']
    def ctxtX(self, idx ):
        return self.__arr_parm_list[idx]['ctxtX']
    def ctxtY(self, idx ):
        return self.__arr_parm_list[idx]['ctxtY']
    def ctxtX_max(self, idx ):
        return self.__arr_parm_list[idx]['ctxtX_max']
    def ctxtY_max(self, idx ):
        return self.__arr_parm_list[idx]['ctxtY_max']
    def patchX(self, idx ):
        return self.__arr_parm_list[idx]['patchX']
    def patchY(self, idx ):
        return self.__arr_parm_list[idx]['patchY']
    def nresX(self, idx ):
        return self.__arr_parm_list[idx]['nresX']
    def nresY(self, idx ):
        return self.__arr_parm_list[idx]['nresY']
    def distA(self, idx ):
        return self.__arr_parm_list[idx]['distA']
    def distB(self, idx ):
        return self.__arr_parm_list[idx]['distB']
    def distC(self, idx ):
        return self.__arr_parm_list[idx]['distC']
    def rej_thresh(self, idx ):
        return self.__arr_parm_list[idx]['rej_thresh']
    def dispX(self, idx ):
        return self.__arr_parm_list[idx]['dispX']
    def Rym(self, idx ):
        return self.__arr_parm_list[idx]['Rym']
    def patchL(self, idx ):
        return self.__arr_parm_list[idx]['patchL']
    def NEGATIVE_DISPARITY_SLACK_PQ(self, idx ):
        return self.__arr_parm_list[idx]['NEGATIVE_DISPARITY_SLACK_PQ']
    def ggmap_output_loc(self, idx ):
        return self.__arr_parm_list[idx]['ggmap_output_loc']
    def num_passes(self):
        return self.__num_passes
    def number_drf_sets(self):
        return self.__number_drf_sets
    def num_minima(self):
        return self.__num_minima
    def max_num_planes(self):
        return self.__MAX_NUM_PLANES
    def max_num_timings(self):
        return self.__MAX_NUM_TIMINGS
    def set_num_passes(self, num_pass):
        self.__num_passes = num_pass
    def set_number_drf_sets(self, number_drf_sets):
        self.__number_drf_sets = number_drf_sets
    def set_max_disparity(self, idx,  max_disp):
        self.__arr_parm_list[idx]['max_disparity_pq'] = max_disp/2
        #YH debug not sure self.__arr_parm_list[idx]['dispX'] = ceil(self.__arr_parm_list[idx]['max_disparity_pq'] / self.__arr_parm_list[idx]['K']) * self.__arr_parm_list[idx]['K']  # relates to number of c-scans in a d-scan
    def num_planes(self, pass_num, tim_sel):
        return self.__num_planes[pass_num, tim_sel]
    def inc_num_planes(self, pass_num, tim_sel):
         self.__num_planes[pass_num, tim_sel] = self.__num_planes[pass_num, tim_sel] + 1
    def set_num_planes(self, pass_num, tim_sel, num):
        self.__num_planes[pass_num, tim_sel] = num

    def update_parm_list(self, merge_green_flag,
               plane_command, pma_patch_configuration, current_idx, current_pass_num ):

        index = current_pass_num * self.__MAX_NUM_PLANES + current_idx

        self.__arr_parm_list[index]['patchX'] = pma_patch_configuration.patch_x
        self.__arr_parm_list[index]['patchY'] = pma_patch_configuration.patch_y
        if merge_green_flag:
            #self.__arr_parm_list[index]['patchL'] = self.__arr_parm_list[index]['patchX']*self.__arr_parm_list[index]['patchY']*(self.__arr_parm_list[index]['C']-1)
            self.__arr_parm_list[index]['patchL'] = self.__arr_parm_list[index]['patchX']*self.__arr_parm_list[index]['patchY']*(self.__arr_parm_list[index]['C'])
        else:
            self.__arr_parm_list[index]['patchL'] = self.__arr_parm_list[index]['patchX']*self.__arr_parm_list[index]['patchY']*self.__arr_parm_list[index]['C']

        self.__arr_parm_list[index]['stepX'] = pma_patch_configuration.step_x
        self.__arr_parm_list[index]['stepY'] = pma_patch_configuration.step_y
        self.__arr_parm_list[index]['jumpX'] = pma_patch_configuration.jump_x
        self.__arr_parm_list[index]['jumpY'] = pma_patch_configuration.jump_y
        self.__arr_parm_list[index]['ctxtX'] = pma_patch_configuration.context_x
        self.__arr_parm_list[index]['ctxtY'] = pma_patch_configuration.context_y


        self.__arr_parm_list[index]['distA'] = plane_command.params[current_idx].a
        self.__arr_parm_list[index]['distB'] = plane_command.params[current_idx].b
        self.__arr_parm_list[index]['distC'] = plane_command.params[current_idx].c
        self.__arr_parm_list[index]['distAf'] = self.__arr_parm_list[index]['distA']/256.0
        self.__arr_parm_list[index]['distBf'] = self.__arr_parm_list[index]['distB']/256.0
        self.__arr_parm_list[index]['distCf'] = self.__arr_parm_list[index]['distC']/4.0
        self.__arr_parm_list[index]['dispK'] = ceil(self.__arr_parm_list[index]['max_disparity_pq']/self.__arr_parm_list[index]['K'])
        self.__arr_parm_list[index]['dispX'] = ceil(self.__arr_parm_list[index]['max_disparity_pq'] / self.__arr_parm_list[index]['K']) * self.__arr_parm_list[index]['K']  # relates to number of c-scans in a d-scan
        # increase context X-dim to allow for the full grd map to fit in
        self.__arr_parm_list[index]['tileX'] = self.__arr_parm_list[index]['jumpX'] - self.__arr_parm_list[index]['stepX'] + 1;
        self.__arr_parm_list[index]['tileY'] = self.__arr_parm_list[index]['jumpY'] - self.__arr_parm_list[index]['stepY'] + 1;
        self.__tileX_ext = floor(max((self.__arr_parm_list[index]['tileX']-1)*self.__arr_parm_list[index]['distAf'], (self.__arr_parm_list[index]['tileY']-1)*self.__arr_parm_list[index]['distBf'])+0.5);
        if 0: # YH DEBUG not sure if needed - probably already taken care of in pre-calculations
            self.__ctxtX += self.__tileX_ext;
        self.__arr_parm_list[index]['tileX'] += self.__tileX_ext;

        if (self.__arr_parm_list[index]['ctxtX'] > self.__arr_parm_list[index]['ctxtX_max']):
            print ("WARNING: max context width exceeded", self.__arr_parm_list[index]['ctxtX']);
        if (self.__arr_parm_list[index]['ctxtY'] > self.__arr_parm_list[index]['ctxtY_max']):
            print ("WARNING: max context height exceeded", self.__arr_parm_list[index]['ctxtY']);

        self.__arr_parm_list[index]['rej_thresh'] = pma_patch_configuration.rejection_threshold * 16384.0
        if 1: #YH TODO not sure if it should be tileX or JumpX
            self.__arr_parm_list[index]['nresX'] = (self.__arr_parm_list[index]['jumpX'] - 1) // \
                                                   self.__arr_parm_list[index]['stepX'] + 1
            self.__arr_parm_list[index]['nresY'] = (self.__arr_parm_list[index]['jumpY'] - 1) // \
                                               self.__arr_parm_list[index]['stepY'] + 1
        else:
            self.__arr_parm_list[index]['nresX'] = (self.__arr_parm_list[index]['tileX']-1)//self.__arr_parm_list[index]['stepX'] + 1
            self.__arr_parm_list[index]['nresY'] = (self.__arr_parm_list[index]['tileY']-1)//self.__arr_parm_list[index]['stepY'] + 1
        gg_map =  plane_command.params[current_idx].gg_map

        output_loc = []
        bm = BitMap(32)
        for val in enumerate(gg_map):
            if val[1] != 0:
                print(val)
                str = bin(val[1])
                bm = bm.fromstring(str[2::])
                for idx in range(len(bm.nonzero())):
                    val_1D = val[0] * 32 + bm.nonzero()[idx]
#                    output_loc.append([val_1D//64 - (self.__arr_parm_list[index]['patchY']-1), val_1D%64 - (self.__arr_parm_list[index]['patchX']-1)])
                    output_loc.append([val_1D//32 - (self.__arr_parm_list[index]['patchY']-1), val_1D%32 - (self.__arr_parm_list[index]['patchX']-1)])

        print(output_loc)
        self.__arr_parm_list[index]['ggmap_output_loc'] = np.asarray(output_loc, dtype=np.int32 ).reshape((self.__arr_parm_list[index]['nresY'], self.__arr_parm_list[index]['nresX'],2))


    @staticmethod
    def x2a (x):
	    return floor(4*x+0.5)%Rxm;
    @staticmethod
    def y2a (y):
	    return floor(y+0.5)%Rym;

    # distortion params - (a,b,c)
    #def escan(self, strip_4x1y_ref, strip_src, a, b, c):
    def escan(self, strip_4x1y_ref, strip_src, idx,  plane_command_list, dscan_method, test_name, drf_version):
        ref_4x = strip_4x1y_ref.copy()
        src_1x = strip_src.copy()
        ht,wd,ch = src_1x.shape
        ref_mem = np.zeros((self.Rym(idx), self.Rxm(idx), ch)).astype(np.float32);

        assert self.ovrsX(idx) == 4
        wx = 0.;
        ix = 0;
        rx = 0.;
        ry = 0.;
        dscan_cnt = 0
        drf_scores_in_dscan = [[[] for i in range(self.max_num_planes())]  for j in range(self.num_passes())]
        for current_pass_num in range(self.num_passes()):
            for empty_list_cnt in range(plane_command_list[current_pass_num].num_planes, self.max_num_planes(), 1):
                drf_scores_in_dscan[current_pass_num].pop()
        # YH TODO: need to calculate the maximum  of plane_command_list[].ref_patch_offset_q2 over all passes
        while (rx + self.jumpX(idx)  < wd - plane_command_list[0].ref_patch_offset_q2/self.ovrsX(idx) + self.NEGATIVE_DISPARITY_SLACK_PQ(idx)):   # go over the whole line
            #print("rx = {}, wx = {}".format(rx,wx))
            # GEOX xfer, 4*jumpX columns
            for i in range(4 * self.jumpX(idx)):
                ref_mem[:, self.x2a(wx), :] = ref_4x[:, ix, :];
                wx += 1 / 4;
                if (ix < 4 * wd - 1):
                    ix += 1;
            while (wx - rx > self.ctxtX(idx) + self.jumpX(idx)):
                for current_pass_num in range(self.num_passes()):
                    sx = floor( rx + plane_command_list[current_pass_num].ref_patch_offset_q2 / self.ovrsX(idx) - self.NEGATIVE_DISPARITY_SLACK_PQ( idx));
                    #sx = floor( rx + plane_command_list[0].ref_patch_offset_q2 / self.ovrsX(idx) - self.NEGATIVE_DISPARITY_SLACK_PQ( idx));
                    for index in range(plane_command_list[current_pass_num].num_planes):
                        patch_pma_config = \
                            test_case.test_setup[0].pma_pass_configuration[pma_pass_idx].pma_pass_parameters[
                                current_pass_num].patch_pma_config
                        timing_sel = plane_command_list[current_pass_num].params[index].mv_patch_timing_sel
                        pma_patch_configuration = test_case.test_setup[0].pma_patch_configuration[patch_pma_config[timing_sel]]
                        index_new = current_pass_num * self.max_num_planes() + index
                        #[drf_scores, depth_arr] = self.dscan(ref_mem, src_1x, sx, rx, index_new, dscan_method)
                        drf_scores = self.dscan(ref_mem, src_1x, sx, rx, index_new, dscan_method)
                        #[drf_scores, depth_arr] = self.dscan(ref_mem, src_1x, sx, rx, index, dscan_method)
                        #print("current_pass_num = {}, index = {}".format(current_pass_num, index))
                        #drf_scores_in_escan[current_pass_num][index].append(drf_scores)
                        drf_scores_in_dscan[current_pass_num][index] = drf_scores

                self.drf_model_per_dscan(dscan_cnt, test_name, drf_scores_in_dscan, drf_version)
                dscan_cnt = dscan_cnt + 1
                rx = rx + self.jumpX(idx)

        return


    def dscan(self, rmem_ovrs, smem, sx, rx, idx,  method):

        # decimation with distortion model, rmem_ovrs -> rmem
        # rmem_ovrs is "real" 4x oversampled HW ref memory
        # rmem is "virtual" (does not not exist in HW), it is c-scan space i,j
        if merge_green_flag:
            rmem = np.zeros((self.rmemY(idx), self.rmemX(idx), self.C(idx)), dtype=np.uint64);
            rmem1 = np.zeros((self.rmemY(idx), self.rmemX(idx), self.C(idx)), dtype=np.uint64);
        else:
            rmem = np.zeros((self.rmemY(idx), self.rmemX(idx), self.C(idx)), dtype=np.uint64);

         # fill rmem and smem
        x0 = 0;
        y0 = 0;
        for j in range(self.ctxtY(idx)):
            for i in range(self.ctxtX(idx)):
                # A,B are S0.7, C is S5.2 (all 8-bit values)
                dx = (self.distA(idx) * i + self.distB(idx) * j) // 64 + self.distC(idx);
                dy = 0;
                xp = (self.ovrsX(idx) * (floor(rx) + x0 + i) + dx) % self.Rxm(idx);
                yp = (y0 + j + dy) % self.rmemY(idx);
                rmem[j][i] = rmem_ovrs[yp][xp];
                rmem_ovrs_map[yp][xp] = 1;  # for SW debug only

        if method == "method_1":
            # method -1: based on gg_map
            res1 = np.zeros((self.nresY(idx), self.nresX(idx), self.dispX(idx)), dtype=np.float32);
            rej1 = np.zeros((self.nresY(idx), self.nresX(idx), self.dispX(idx)), dtype=np.bool);
            out1 = np.ones((self.nresY(idx), self.nresX(idx), self.dispX(idx)), dtype=np.int16) * ( 2 ** (expw + sigw) - 1);
            pxi = np.arange(self.patchX(idx));
            pyi = np.arange(self.patchY(idx));
            for ny in range(self.nresY(idx)):
                for nx in range(self.nresX(idx)):
                    # extract a single ref patch from a tile (ty, tx)
                    ty = self.ggmap_output_loc(idx)[ny,nx,0]
                    tx = self.ggmap_output_loc(idx)[ny, nx, 1]
                    ind = np.ix_(pyi + ty, pxi + tx);
                    # patch reshaping order does not matter
                    # as long as it is the same for ref and src
                    if four_channel_flag:
                        R = rmem[ind].reshape(self.patchL(idx));
                    else:
                        R1 = rmem[ind][:,:,0].reshape(self.patchL(idx)//4)
                        R2 = rmem[ind][:, :, 1:3].reshape(self.patchL(idx)//2)
                    # following the entire disparity range...
                    for d in range(min(self.dispX(idx), self.imgX(idx) - (tx + pxi[-1] + sx))):
                        # extract a single src patch from a tile (ty, tx+d) along the disparity range
                        ind = np.ix_(pyi + ty, pxi + sx + tx + d );
                        if four_channel_flag:
                            S = smem[ind].reshape(self.patchL(idx));
                        else:
                            S1 = smem[ind][:,:,0].reshape(self.patchL(idx)//4);
                            S2 = smem[ind][:, :, 1:3].reshape(self.patchL(idx)//2);

                        # comupte ZPF1/NCC metric

                        # linear domain, v3.0..v3.1
                        # nom = np.float64(R@S)**2;
                        # den = np.float64(R@R)*np.float64(S@S);
                        # res1[ty][tx][d] = nom/den;

                        #global max_energy, max_energy1, max_energy2
                        #if(S@S) > max_energy:
                        #    max_energy = S@S
                        #if(R@R) > max_energy1:
                        #    max_energy1 = R@R
                        #if(R@S) > max_energy2:
                        #    max_energy2 = R@S
                        # log domain FP, v3.2+
                        if four_channel_flag:
                            rs_l2 = uint_log2(R @ S);
                            ss_l2 = uint_log2(S @ S);
                            rr_l2 = uint_log2(R @ R);
                            #rs_l2_tmp = uint_log2(R1 @ S);
                            #ss_l2_tmp = uint_log2(S1 @ S);
                            #rr_l2_tmp = uint_log2(R1 @ R);
                        else:
                            rs_l2 = uint_log2(2 * (R1 @ S1) + (R2@S2));
                            ss_l2 = uint_log2(2 * (S1 @ S1) + (S2@S2));
                            rr_l2 = uint_log2(2 * (R1 @ R1) + (R2@R2));

                        res1[ny][nx][d] = rr_l2 + ss_l2 - 2 * rs_l2;
                        #if res1[ny][nx][d] < 0.0:
                        #    print("value cannot be negative but is {} at index = {}, nx = {},ny = {}, d = {}, sx = {}".format(res1[ny][nx][d], idx, nx, ny, d, sx), file=fp_file)
                        rej1[ny][nx][d] = np.abs(rr_l2 - ss_l2) > self.rej_thresh(idx);

                        # truncate res to 0.14, ignore clamps for now (like RTL)
                        if not np.isnan(res1[ny][nx][d]):
                            res_int = floor(res1[ny][nx][d] + 0.5) & (scale - 1);
                            # compress res to custom FP format
                            #out1[ny][nx][d] = compr(res_int);
                            out1[ny][nx][d] = compr_rtl(res_int);
                            # finally apply clamps to the compressed output, reject flag forces clamp
                            if (res1[ny][nx][d] > scale - 1 or rej1[ny][nx][d]):
                                out1[ny][nx][d] = 2 ** (expw + sigw) - 1;
                            elif (res1[ny][nx][d] < 0):
                                out1[ny][nx][d] = 0;

        if method == "method_2":
            # method 1: brute force (reference)
            # always generates dense results
            res1 = np.zeros((self.tileY(idx), self.tileX(idx), self.dispX(idx)), dtype=np.float32);
            rej1 = np.zeros((self.tileY(idx), self.tileX(idx), self.dispX(idx)), dtype=np.bool);
            out1 = np.zeros((self.tileY(idx), self.tileX(idx), self.dispX(idx)), dtype=np.int16);
            res1_1 = np.zeros((self.nresY(idx), self.nresX(idx), self.dispX(idx)), dtype=np.float32);
            rej1_1 = np.zeros((self.nresY(idx), self.nresX(idx), self.dispX(idx)), dtype=np.bool);
            out1_1 = np.zeros((self.nresY(idx), self.nresX(idx), self.dispX(idx)), dtype=np.int16);
            pxi = np.arange(self.patchX(idx));
            pyi = np.arange(self.patchY(idx));
            for ty in range(0,self.tileY(idx),self.stepY(idx)):
                for tx in range(0,self.tileX(idx),self.stepX(idx)):
                    # extract a single ref patch from a tile (ty, tx)
                    ind = np.ix_(pyi + ty, pxi + tx);
                    # patch reshaping order does not matter
                    # as long as it is the same for ref and src
                    R = rmem[ind].reshape(self.patchL(idx));
                    # following the entire disparity range...
                    for d in range(min(self.dispX(idx), self.imgX(idx) - (tx + pxi[-1] + sx))):
                        # extract a single src patch from a tile (ty, tx+d) along the disparity range
                        ind = np.ix_(pyi + ty, pxi +sx + tx + d);
                        S = smem[ind].reshape(self.patchL(idx));

                        # comupte ZPF1/NCC metric

                        # linear domain, v3.0..v3.1
                        # nom = np.float64(R@S)**2;
                        # den = np.float64(R@R)*np.float64(S@S);
                        # res1[ty][tx][d] = nom/den;

                        # log domain FP, v3.2+
                        rs_l2 = np.log2(R @ S) * scale;
                        ss_l2 = np.log2(S @ S) * scale;
                        rr_l2 = np.log2(R @ R) * scale;
                        res1[ty][tx][d] = rr_l2 + ss_l2 - 2 * rs_l2;
                        rej1[ty][tx][d] = np.abs(rr_l2 - ss_l2) > self.rej_thresh(idx);
                        res1_1[ty//self.stepY(idx)][tx//self.stepX(idx)][d] = rr_l2 + ss_l2 - 2 * rs_l2;
                        rej1_1[ty//self.stepY(idx)][tx//self.stepX(idx)][d] = np.abs(rr_l2 - ss_l2) > self.rej_thresh(idx);

                        # truncate res to 0.14, ignore clamps for now (like RTL)

                        if not np.isnan(res1[ty][tx][d]):
                            res_int = floor(res1[ty][tx][d] + 0.5) & (scale - 1);
                            # compress res to custom FP format
                            out1[ty][tx][d] = compr(res_int);
                            # finally apply clamps to the compressed output, reject flag forces clamp
                            if (res1[ty][tx][d] > scale - 1 or rej1[ty][tx][d]):
                                out1[ty][tx][d] = 2 ** (expw + sigw) - 1;
                            elif (res1[ty][tx][d] < 0):
                                out1[ty][tx][d] = 0;
                        if not np.isnan(res1_1[ty//self.stepY(idx)][tx//self.stepX(idx)//self.stepX(idx)][d]):
                            res_int = floor(res1_1[ty//self.stepY(idx)][tx//self.stepX(idx)][d] + 0.5) & (scale - 1);
                            # compress res to custom FP format
                            out1_1[ty//self.stepY(idx)][tx//self.stepX(idx)][d] = compr(res_int);
                            # finally apply clamps to the compressed output, reject flag forces clamp
                            if (res1_1[ty//self.stepY(idx)][tx//self.stepX(idx)][d] > scale - 1 or rej1_1[ty//self.stepY(idx)][tx//self.stepX(idx)][d]):
                                out1_1[ty//self.stepY(idx)][tx//self.stepX(idx)][d] = 2 ** (expw + sigw) - 1;
                            elif (res1_1[ty//self.stepY(idx)][tx//self.stepX(idx)][d] < 0):
                                out1_1[ty//self.stepY(idx)][tx//self.stepX(idx)][d] = 0;

        if method == "method_3":
            # method 1: brute force (reference)
            # always generates dense results
            res1 = np.zeros((self.tileY(idx), self.tileX(idx), self.dispX(idx)), dtype=np.float32);
            rej1 = np.zeros((self.tileY(idx), self.tileX(idx), self.dispX(idx)), dtype=np.bool);
            out1 = np.zeros((self.tileY(idx), self.tileX(idx), self.dispX(idx)), dtype=np.int16);
            pxi = np.arange(self.patchX(idx));
            pyi = np.arange(self.patchY(idx));
            for ty in range(self.tileY(idx)):
                for tx in range(self.tileX(idx)):
                    # extract a single ref patch from a tile (ty, tx)
                    ind = np.ix_(pyi + ty, pxi + tx);
                    # patch reshaping order does not matter
                    # as long as it is the same for ref and src
                    R = rmem[ind].reshape(self.patchL(idx));
                    # following the entire disparity range...
                    #for d in range(dispX):
                    TEST_CONST = 0
                    for d in range(min(self.dispX(idx), imgX - (tx + pxi[-1] + sx))):
                        # extract a single src patch from a tile (ty, tx+d) along the disparity range
                        ind = np.ix_(pyi + ty, pxi +sx + tx + d - TEST_CONST);
                        S = smem[ind].reshape(self.patchL(idx));

                        # comupte ZPF1/NCC metric

                        # linear domain, v3.0..v3.1
                        # nom = np.float64(R@S)**2;
                        # den = np.float64(R@R)*np.float64(S@S);
                        # res1[ty][tx][d] = nom/den;

                        # log domain FP, v3.2+
                        rs_l2 = np.log2(R @ S) * scale;
                        ss_l2 = np.log2(S @ S) * scale;
                        rr_l2 = np.log2(R @ R) * scale;
                        res1[ty][tx][d] = rr_l2 + ss_l2 - 2 * rs_l2;
                        rej1[ty][tx][d] = np.abs(rr_l2 - ss_l2) > self.rej_thresh(idx);

                        # truncate res to 0.14, ignore clamps for now (like RTL)
                        if not np.isnan(res1[ty][tx][d]):
                            res_int = floor(res1[ty][tx][d] + 0.5) & (scale - 1);
                            # compress res to custom FP format
                            out1[ty][tx][d] = compr(res_int);
                            # finally apply clamps to the compressed output, reject flag forces clamp
                            if (res1[ty][tx][d] > scale - 1 or rej1[ty][tx][d]):
                                out1[ty][tx][d] = 2 ** (expw + sigw) - 1;
                            elif (res1[ty][tx][d] < 0):
                                out1[ty][tx][d] = 0;

        if method == "method_4":
            x_dim  = self.ctxtX(idx) - self.patchX(idx) + 1  # YH DEBUG tmp
            # method 2: 1xK with partial sum optimization
            res2 = np.zeros((self.tileY(idx), x_dim, self.dispX(idx)), dtype=np.float32);
            rej2 = np.zeros((self.tileY(idx), x_dim, self.dispX(idx)), dtype=np.bool);
            out2 = np.zeros((self.tileY(idx), x_dim, self.dispX(idx)), dtype=np.int16);
            out1 = np.zeros((self.nresY(idx), self.nresX(idx), self.dispX(idx)), dtype=np.int16);
            delta_dbg = 0
            # this code is vectorized by K
            # *A marks A-node path delay line/acc (K-wide vector)
            # *B marks B-node path delay line/acc from the src mem (K-wide vector)
            # *C marks B-node path delay line/acc from the ref mem (scalar)
            for n in range(self.dispK(idx)):
                # second delay line and accumulator
                #delAy = [np.zeros(self.K(), dtype=np.uint64)] * (self.patchY() * self.tileX());
                #delBy = [np.zeros(self.K(), dtype=np.uint64)] * (self.patchY() * self.tileX());
                #delCy = [np.uint64(0)] * (self.patchY() * self.tileX());
                delAy = [np.zeros(self.K(idx), dtype=np.uint64)] * (self.patchY(idx) * x_dim);
                delBy = [np.zeros(self.K(idx), dtype=np.uint64)] * (self.patchY(idx) * x_dim);
                delCy = [np.uint64(0)] * (self.patchY(idx) * x_dim);
                #accAy = np.zeros((self.tileX(), self.K()), dtype=np.uint64);
                #accBy = np.zeros((self.tileX(), self.K()), dtype=np.uint64);
                #accCy = np.zeros(self.tileX(), dtype=np.uint64);
                accAy = np.zeros((x_dim, self.K(idx)), dtype=np.uint64);
                accBy = np.zeros((x_dim, self.K(idx)), dtype=np.uint64);
                accCy = np.zeros(x_dim, dtype=np.uint64);
                for j in range(self.ctxtY(idx)):
                    # first delay line and accumulator
                    delAx = [np.zeros(self.K(idx), dtype=np.uint64)] * self.patchX(idx);
                    delBx = [np.zeros(self.K(idx), dtype=np.uint64)] * self.patchX(idx);
                    delCx = [np.uint64(0)] * self.patchX(idx);
                    accAx = np.zeros(self.K(idx), dtype=np.uint64);
                    accBx = np.zeros(self.K(idx), dtype=np.uint64);
                    accCx = np.uint64(0);
                    for i in range(self.ctxtX(idx)):
                        if four_channel_flag:
                            r = rmem[j][i];
                            rr = r @ r;  # rr is scalar
                        else:
                            r1 = rmem[j][i][0];
                            r2 = rmem[j][i][1:3];
                            rr = 2*(r1 * r1) + (r2 @ r2);  # rr is scalar
                        rs = np.zeros(self.K(idx), dtype=np.uint64);
                        ss = np.zeros(self.K(idx), dtype=np.uint64);
                        for k in range(self.K(idx)):
                            # vectorised only by C, iterating over K (can be vectorised if needed)
                            if four_channel_flag:
                                s = smem[j, ( sx + i + k + n * self.K(idx))%1604]; # YH new
                                rs[k] = r @ s;
                                ss[k] = s @ s;
                            else:
                                s1 = smem[j, ( sx + i + k + n * self.K(idx))%1604][0]; # YH new
                                s2 = smem[j, ( sx + i + k + n * self.K(idx))%1604][1:3]; # YH new
                                rs[k] = 2*(r1 * s1) + (r2 @ s2);
                                ss[k] = 2 * (s1 * s1) +  s2 @ s2;

                        accAx = accAx + rs - delAx[-1];
                        delAx = [rs.copy()] + delAx[:-1];  # shift
                        accBx = accBx + ss - delBx[-1];
                        delBx = [ss.copy()] + delBx[:-1];  # shift
                        accCx = accCx + rr - delCx[-1];
                        delCx = [rr.copy()] + delCx[:-1];  # shift

                        if (i >= self.patchX(idx) - 1):
                            tx = i - self.patchX(idx) + 1;
                            accAy[tx] = accAy[tx] + accAx - delAy[-1];
                            delAy = [accAx.copy()] + delAy[:-1];  # shift
                            accBy[tx] = accBy[tx] + accBx - delBy[-1];
                            delBy = [accBx.copy()] + delBy[:-1];  # shift
                            accCy[tx] = accCy[tx] + accCx - delCy[-1];
                            delCy = [accCx.copy()] + delCy[:-1];  # shift
                            if (j >= self.patchY(idx) - 1):
                                ty = j - self.patchY(idx) + 1;

                                # comupte ZPF1/NCC metric

                                # linear domain, v3.0..v3.1
                                # nom = np.float64(accAy[tx])*np.float64(accAy[tx]);
                                # nom = np.float64(accAy[tx])**2;
                                # den = np.float64(accBy[tx])*np.float64(accCy[tx]);
                                # res2[ty][tx][n*K:(n+1)*K] = nom/den;

                                # log domain FP, v3.2..v3.3
                                # rs_l2 = np.log2(accAy[tx])*scale;
                                # ss_l2 = np.log2(accBy[tx])*scale;
                                # rr_l2 = np.log2(accCy[tx])*scale;
                                # res2[ty][tx][n*K:(n+1)*K] = rr_l2+ss_l2-2*rs_l2; # >0;
                                # rej2[ty][tx][n*K:(n+1)*K] = np.abs(rr_l2-ss_l2) > rej_thresh;

                                # log domain, HW log2 approximation, not vectorised
                                for k in range(self.K(idx)):
                                    rs_l2 = uint_log2(accAy[tx][k]);
                                    ss_l2 = uint_log2(accBy[tx][k]);
                                    rr_l2 = uint_log2(accCy[tx]);
                                    # if (n == 0 and k == 0):
                                    #   print (rs_l2, ss_l2, rr_l2, rr_l2+ss_l2-2*rs_l2);

                                    d = n * self.K(idx) + k;
                                    res2[ty][tx][d] = rr_l2 + ss_l2 - 2 * rs_l2;
                                    rej2[ty][tx][d] = np.abs(rr_l2 - ss_l2) > self.rej_thresh(idx);

                                    # truncate res to 0.14, ignore clamps for now (like RTL)
                                    res_int = floor(res2[ty][tx][d] + 0.5) & (scale - 1);
                                    # compress res to custom FP format
                                    out2[ty][tx][d] = compr(res_int);
                                    # finally apply clamps to the compressed output, reject flag forces clamp
                                    if (res2[ty][tx][d] > scale - 1 or rej2[ty][tx][d]):
                                        out2[ty][tx][d] = 2 ** (expw + sigw) - 1;
                                    elif (res2[ty][tx][d] < 0):
                                        out2[ty][tx][d] = 0;
            for ny in range(self.nresY(idx)):
                for nx in range(self.nresX(idx)):
                    ty = self.ggmap_output_loc(idx)[ny,nx,0]
                    tx = self.ggmap_output_loc(idx)[ny, nx, 1]
                    out1[ny][nx] = out2[ty][tx]
        return out1

    def drf_compute_min(self, num_minima, out1, drf_scores, depth_arr, plane_arr):
        dmin = 0
        pmin = 1.0
        class Score:
            score = 1023.;
            # smin = 0.;
            smax = 0.;
            disp = -1;
            plane = -1;

        tlen = num_minima
        scmp = [0] * tlen;
        pcmp = [0] * tlen;
        dcmp = [0] * tlen;
        clrd = [0] * tlen;

        assert len(out1.shape) == 4
        dlen = out1.shape[3]
        for y in range(out1.shape[1]):
            for x in range(out1.shape[2]):
                tops = [Score() for i in range(tlen)];
                scores = np.min(out1[:,y,x,:],axis=0)
                scores_argmin = np.argmin(out1[:,y,x,:],axis=0)
                for d in range(dlen):

                    # pipeline stage 1 (adders / comparators)
                    for t in range(tlen):
                        scmp[t] = scores[d] < tops[t].score;
                        dcmp[t] = (d - tops[t].disp) >= dmin;
                        pcmp[t] = scores[d] <= pmin * tops[t].smax;
                        clrd[t] = (pcmp[t] and dcmp[t]) or (tops[t].disp == -1);
                        tops[t].smax = max(tops[t].smax, scores[d]);

                    # pipeline stage 2 (priority encoders/ muxes)
                    for t in range(tlen):
                        if (scmp[t]):
                            # new peak candidate found
                            if (clrd[t]):
                                # check entries for min distance/prominence fail starting from the lowest score
                                # shift entries down from "t" to the lowest score failed (removed by shift)
                                # if none failed shift out the lowest score entry
                                b = tlen - 1;
                                for i in range(t + 1, b):
                                    if (not clrd[i]):
                                        b = i;
                                for i in range(b, t, -1):
                                    tops[i] = copy(tops[i - 1]);
                            # insert/replace the entry at "t" with the new one
                            tops[t].score = scores[d];
                            tops[t].smax = scores[d];
                            tops[t].disp = d;
                            tops[t].plane = scores_argmin[d]
                            # stop checking entries on the first tops update
                            break;
                        # stop checking entries on the first distance/prominence fail
                        if (not clrd[t]):
                            break;
                for t in range(tlen):
                    drf_scores[y,x,t] = tops[t].score
                    depth_arr[y,x,t] = tops[t].disp
                    plane_arr[y,x,t] = tops[t].plane

        return

    # DRF processing over each dscan but multipass
    def drf_model_per_dscan(self, dscan_cnt, test_name, drf_scores, version=1, print_logs=True ):
        tmp_index = 0

        for current_pass_cnt in range(self.num_passes()):
            # covert to numpy array
            drf_simple_np = np.asarray(drf_scores[current_pass_cnt])
            assert len(drf_simple_np.shape) == 4
            ny = drf_simple_np.shape[1]
            nx = drf_simple_np.shape[2]
            drf_scores_tmp = np.ones((ny, nx, self.num_minima()), dtype = np.float) *(1023)
            depth_arr_tmp = np.ones((ny, nx, self.num_minima()), dtype = np.int) *(1023)
            plane_arr_tmp = np.ones((ny, nx, self.num_minima()), dtype = np.int) *(-1)
            start_plane = 0;
            for timing_sel in range(self.max_num_timings()):
                num_planes = self.num_planes(current_pass_cnt, timing_sel)
                if num_planes > 0:
                    self.drf_compute_min( self.num_minima(), drf_simple_np[start_plane:start_plane + num_planes], drf_scores_tmp, depth_arr_tmp, plane_arr_tmp)
                    start_plane = start_plane + num_planes

                    drf_shape = drf_scores_tmp.shape
                    if print_logs:
                        print("*************drf count = {}***********".format(dscan_cnt), file=fp_file)
                    if version == 1:
                        list_drf = drf[dscan_cnt + drf_rtl_idx]
                    else:
                        list_drf = drf[ drf_rtl_idx * self.number_drf_sets() + dscan_cnt * self.number_drf_sets() + current_pass_cnt + timing_sel ]
                    print_idx = 0;
                    for ny in range(drf_shape[0]):
                         for nx in range(drf_shape[1]):
                            best_score_arr[tmp_index].append(drf_scores_tmp[ny,nx].copy())
                            best_depth_arr[tmp_index].append(depth_arr_tmp[ny, nx].copy())
                            best_plane_arr[tmp_index].append((plane_arr_tmp[ny, nx].copy() + start_plane - num_planes))
                            if print_logs:
                                print("", file=fp_file)
                                print("best score = {}".format(
                                    #drf_scores_tmp[ny, nx]//4), file=fp_file)   # divided by 4 to match 8 bit hardware output
                                    drf_scores_tmp[ny, nx]), file=fp_file)   # divided by 4 to match 8 bit hardware output
                                print("best score k index = {}".format(depth_arr_tmp[ny, nx]), file=fp_file)

                                print("best score plane index  = {}".format( plane_arr_tmp[ny, nx] + start_plane - num_planes), file=fp_file)
                                if version == 1:
                                    log_drf(list_drf[print_idx], fp_file, rtl_best_score_arr[current_pass_cnt], rtl_depth_arr[current_pass_cnt], rtl_plane_arr[current_pass_cnt])
                                elif version == 2:
                                    log_drf_v2(list_drf[print_idx], fp_file, rtl_best_score_arr[tmp_index],
                                           rtl_depth_arr[tmp_index], rtl_plane_arr[tmp_index])
                                elif version == 3:
                                    log_drf_v3(list_drf[print_idx], fp_file, rtl_best_score_arr[tmp_index],
                                               rtl_depth_arr[tmp_index], rtl_plane_arr[tmp_index])
                                print("", file=fp_file)
                                print("", file=fp_file)
                                print_idx = print_idx + 1
                    tmp_index = tmp_index + 1

        for current_pass_cnt in range(self.number_drf_sets()):
            best_score_np = np.asarray(best_score_arr[current_pass_cnt])
            np.save("best_score_{}_{}.npy".format(current_pass_cnt, test_name), best_score_np)
            depth_np = np.asarray(best_depth_arr[current_pass_cnt])
            np.save("depth_{}_{}.npy".format(current_pass_cnt, test_name), depth_np)
            plane_np = np.asarray(best_plane_arr[current_pass_cnt])
            np.save("plane_{}_{}.npy".format(current_pass_cnt, test_name), plane_np)
#
            rtl_best_score_np = np.asarray(rtl_best_score_arr[current_pass_cnt])
            np.save("rtl_best_score_{}_{}.npy".format(current_pass_cnt, test_name), rtl_best_score_np)
            rtl_depth_np = np.asarray(rtl_depth_arr[current_pass_cnt])
            np.save("rtl_depth_{}_{}.npy".format(current_pass_cnt, test_name), rtl_depth_np)
            rtl_plane_np = np.asarray(rtl_plane_arr[current_pass_cnt])
            np.save("rtl_plane_{}_{}.npy".format(current_pass_cnt, test_name), rtl_plane_np)
#

    def read_images(self, imgX, imgY, base_dir, image_dir, image_type,  test_case, ):
        # Image files
        image_helper = Image(image_dir)

        a1_idx = test_case.test_setup[0].image_command[0].image_files[0].camera_id
        a2_idx = test_case.test_setup[0].image_command[0].image_files[1].camera_id

        # 4 modes of reading input images
        # mode 1 - packed 12 bit format - image_helper
        # mode 2 - unpacked 16 bit format binary - use image_helper
        # mode 3 - unpacked 16 bit format but output of RTL Geox organized  in PQ format described in Geox document
        # mode 4 - ref upsamples 4x and src upsampled 1x and are generated by geox python model as npy file  - for now the file names will be hard coded

        filename_ref = os.path.join(base_dir, test_case.test_setup[0].image_command[0].image_files[0].file_path)
        filename_src = os.path.join(base_dir, test_case.test_setup[0].image_command[0].image_files[1].file_path)
        if image_type == "mode_1":
            img_a1, _ = image_helper.read_image_file(a1_idx, image_idx, file_name=filename_ref, scale_to_8bit=False,
                                                     raw_output=True)
            img_a2, _ = image_helper.read_image_file(a2_idx, image_idx, file_name=filename_src, scale_to_8bit=False,
                                                     raw_output=True)
        elif image_type == "mode_2":
            img_a1 = np.fromfile(filename_ref, np.uint16).reshape((2 * imgY, 2 * imgX))
            img_a2 = np.fromfile(filename_src, np.uint16).reshape((2 * imgY, 2 * imgX))
        #    img_a1 = np.fromfile(filename_ref, np.uint16).reshape((imgY, imgX, 4))
        #    img_a2 = np.fromfile(filename_src, np.uint16).reshape((imgY, imgX, 4)
        elif image_type == "mode_3":
            img_a1 = np.empty((height, width))
            img_a1_tmp = np.fromfile(filename_ref, np.uint16)
            img_a1_tmp = img_a1_tmp.reshape((-1, 4))
            img_a1[::2, ::2] = img_a1_tmp[:, 0].reshape((imgY, imgX))
            img_a1[1::2, ::2] = img_a1_tmp[:, 1].reshape((imgY, imgX))
            img_a1[0::2, 1::2] = img_a1_tmp[:, 2].reshape((imgY, imgX))
            img_a1[1::2, 1::2] = img_a1_tmp[:, 3].reshape((imgY, imgX))
            img_a2 = np.empty((height, width))
            img_a2_tmp = np.fromfile(filename_src, np.uint16)
            img_a2_tmp = img_a2_tmp.reshape((-1, 4))
            img_a2[::2, ::2] = img_a2_tmp[:, 0].reshape((imgY, imgX))
            img_a2[1::2, ::2] = img_a2_tmp[:, 1].reshape((imgY, imgX))
            img_a2[0::2, 1::2] = img_a2_tmp[:, 2].reshape((imgY, imgX))
            img_a2[1::2, 1::2] = img_a2_tmp[:, 3].reshape((imgY, imgX))
        elif image_type == "mode_4":
            input_ref_image_file_nx = filename_ref.replace(".bin", "_{}X.npy".format(4))
            input_src_image_file_nx = filename_src.replace(".bin", "_{}X.npy".format(1))
            #    img_a1 = (np.load("/Users/yhussain/project/closedcv/fpga_test/undistorted_image_ref_1245_out_geox_passthru_4X.npy"))
            #    img_a2 = (np.load( "/Users/yhussain/project/closedcv/fpga_test/undistorted_image_src_1245_out_geox_passthru_1X.npy"))
            #img_a1 = (np.load( "/Users/yhussain/project/euclid_fw/euclid_hw_api/test/tests-json/tests-images/circle_ref_4X.npy"))
            #img_a2 = (np.load( "/Users/yhussain/project/euclid_fw/euclid_hw_api/test/tests-json/tests-images/circle_src_30_1X.npy"))
            img_a1 = np.load( input_ref_image_file_nx)
            img_a2 = np.load( input_src_image_file_nx)
        else:
            print("Invalid Image read mode.... exiting.......")
            exit(0)

        C = self.C(0)
        ovrsX = self.ovrsX(0)

        if merge_green_flag:
            assert C == 4
            img_ref_4x = np.empty((imgY, imgX * ovrsX, C), dtype=np.float64)
            image_ref_pq = np.empty((imgY, imgX, C), dtype=np.uint32)
            image_src_pq = np.empty((imgY, imgX, C))
            if image_type != "mode_4":
                if four_channel_flag:
                    image_ref_pq[:, :, 0] = img_a1[::2, ::2]
                    image_ref_pq[:, :, 1] = (img_a1[::2, 1::2] + img_a1[1::2, ::2]) // 2
                    image_ref_pq[:, :, 2] = image_ref_pq[:, :, 1]
                    image_ref_pq[:, :, 3] = img_a1[1::2, 1::2]

                    image_src_pq[:, :, 0] = img_a2[::2, ::2] // 16
                    image_src_pq[:, :, 1] = 0.5 * (img_a2[::2, 1::2] // 16 + img_a2[1::2, ::2] // 16)
                    image_src_pq[:, :, 2] = image_src_pq[:, :, 1]
                    image_src_pq[:, :, 3] = img_a2[1::2, 1::2] // 16
                else:
                    #image_ref_pq[:, :, 0] = (img_a1[::2, 1::2] + img_a1[1::2, ::2]) // 2
                    image_ref_pq[:, :, 0] = img_a1[::2, 1::2]
                    image_ref_pq[:,:,0] = image_ref_pq[:,:,0] +  img_a1[1::2, ::2]
                    image_ref_pq[:,:,0] = np.round(image_ref_pq[:,:,0]/2.0)

                    image_ref_pq[:, :, 1] = img_a1[::2, ::2]
                    image_ref_pq[:, :, 2] = img_a1[1::2, 1::2]
                    image_ref_pq[:, :, 3] = 0

                    image_src_pq[:, :, 0] =  np.round((np.round(img_a2[::2, 1::2] / 16.0) + np.round(img_a2[1::2, ::2] / 16.0)) / 2.0)
                    image_src_pq[:, :, 1] = np.round(img_a2[::2, ::2] / 16.0)
                    image_src_pq[:, :, 2] = np.round(img_a2[1::2, 1::2] / 16.0)
                    image_src_pq[:, :, 3] = 0

                img_ref_4x[:, ::4, :] = np.round(image_ref_pq / 16.0)
                img_ref_4x[:, 1:-4:4, :] = np.round((image_ref_pq[:, :-1, :] * .75 + .25 * image_ref_pq[:, 1::, :]) / 16.0)
                img_ref_4x[:, 2:-4:4, :] = np.round((image_ref_pq[:, :-1, :] * .5 + .5 * image_ref_pq[:, 1::, :]) / 16.0)
                img_ref_4x[:, 3:-4:4, :] = np.round((image_ref_pq[:, :-1, :] * .25 + .75 * image_ref_pq[:, 1::, :]) / 16.0)

            else:
                if four_channel_flag:
                    img_ref_4x[:, :, 0] = img_a1[:, :, 0] // 16
                    img_ref_4x[:, :, 1] = (img_a1[:, :, 1]//16 + img_a1[:, :, 2]//16) // 2
                    img_ref_4x[:, :, 2] = img_ref_4x[:, :, 1]
                    img_ref_4x[:, :, 3] = img_a1[:, :, 3] // 16

                    image_src_pq[:, :, 0] = img_a2[:, :, 0] // 16
                    image_src_pq[:, :, 1] = (img_a2[:, :, 1]//16 + img_a2[:, :, 2]//16) // 2
                    image_src_pq[:, :, 2] = image_src_pq[:, :, 1]
                    image_src_pq[:, :, 3] = img_a2[:, :, 3] // 16
                else:
                    img_ref_4x[:, :, 0] = (img_a1[:, :, 1]//16 + img_a1[:, :, 2]//16)//2
                    img_ref_4x[:, :, 1] = img_a1[:, :, 0] // 16
                    img_ref_4x[:, :, 2] = img_a1[:, :, 3] // 16
                    img_ref_4x[:, :, 3] = 0

                    image_src_pq[:, :, 0] = (img_a2[:, :, 1]//16 + img_a2[:, :, 2]//16)//2
                    image_src_pq[:, :, 1] = img_a2[:, :, 0] // 16
                    image_src_pq[:, :, 2] = img_a2[:, :, 3] // 16
                    image_src_pq[:, :, 3] = 0
        else:
            img_ref_4x = np.empty((imgY, imgX * ovrsX, C), dtype=np.float64)
            image_ref_pq = np.empty((imgY, imgX, C))
            image_src_pq = np.empty((imgY, imgX, C))
            if image_type != "mode_4":
                image_ref_pq[:, :, 0] = img_a1[::2, ::2] // 16
                image_ref_pq[:, :, 1] = img_a1[::2, 1::2] // 16
                image_ref_pq[:, :, 2] = img_a1[1::2, ::2] // 16
                image_ref_pq[:, :, 3] = img_a1[1::2, 1::2] // 16

                image_src_pq[:, :, 0] = img_a2[::2, ::2] // 16
                image_src_pq[:, :, 1] = img_a2[::2, 1::2] // 16
                image_src_pq[:, :, 2] = img_a2[1::2, ::2] // 16
                image_src_pq[:, :, 3] = img_a2[1::2, 1::2] // 16
            else:
                img_ref_4x[:, :, 0] = img_a1[:, :, 0]
                img_ref_4x[:, :, 1] = img_a1[:, :, 1]
                img_ref_4x[:, :, 2] = img_a1[:, :, 2]
                img_ref_4x[:, :, 3] = img_a1[:, :, 3]

                image_src_pq[:, :, 0] = img_a2[:, :, 0]
                image_src_pq[:, :, 1] = img_a2[:, :, 1]
                image_src_pq[:, :, 2] = img_a2[:, :, 2]
                image_src_pq[:, :, 3] = img_a2[:, :, 3]
        return img_ref_4x, image_src_pq

# start of main code

parser = argparse.ArgumentParser(description="PMAx python model")
parser.add_argument('--test_name')
parser.add_argument('--json')
#parser.add_argument('--index')
parser.add_argument('--base_dir', default=".")
parser.add_argument('--path_to_image_dir', default="/Users/yhussain/project/closedcv")
parser.add_argument('--image_dir', default="fpga_test")
parser.add_argument('--cal_dir', default="/Users/yhussain/project/rig_calibration/rig_sp_04_00_a/latest")
parser.add_argument('--image_type', default='', help='mode_1: packed 12-bit format, \
                    mode_2: unpacked 16 bit binary format , \
                    mode_3: unpacked 16 bit binary format and is output of RTL Geox oganized in PQ format described in HW Geox doc , \
                    mode_4: reference image 4x upsamples, src image 1x upsampled and generated by geox python model as npy files hardcoded in name for now')
parser.add_argument('--dscan_method', default='method_1', help='method_1: computes brute force metric at ggmap_loc only - this is the used method, \
                    method_2: used for debugging only  , \
                    method_3: used for debug only. Full brute force computation of metric and is very slow. Used to look at all value of score for every value od disparity to debug \
                    method_4: used for debug only. Implements delya line and dense map and all scores can be loooked at here for debug')
parser.add_argument('--log_file', default="pmax_model_out_dbg_tmp.txt")
parser.add_argument('--drf_version', type=int, default=1)
parser.add_argument('--num_dscans_per_escan_rtl', type=int, default=77)

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

json_filename = args.json

json_str = open(json_filename, 'r').read()
tests = commandfile_pb2.TestRun()
json_format.Parse(json_str, tests)

# Find requested test case
test_case = []
for case in tests.test_case_run:
    if case.test_name == args.test_name:
        test_case = case
        break

if test_case.test_name != args.test_name:
    print("Could not find test case: {} in: {}".format(test_case, json_filename))
    exit(-1)


##############YH DEBUG CODE to print out DRF output for comparison

#global merge_green_flag
merge_green_flag = True # if true then the ywo green channels are merged and 4 channel to 3 color channel conversion happens
four_channel_flag = False
debug = False
print_logs = True

fp_file = open(os.path.join(args.base_dir,args.log_file), "w")

pmax_v34 = Pmax_v34()

# Open DRF File
drf_filename =  os.path.join(args.base_dir, test_case.test_setup[0].image_command[0].drf_response.file_path)
drf_infile = open(drf_filename, "rb")

if args.drf_version == 1:
    print("Reading {}".format(drf_filename))
    drf = []
    while True:
        elem = drf_infile.read(ctypes.sizeof(DRF_OUTPUT_T))
        if elem == b'':
            break
        elem = unpack(DRF_OUTPUT_T, elem)
        if elem.d_sync_start == 1:
            drf.append([])
        drf[-1].append(elem)

    drf_infile.close()
    print("Done")
    print("")
    print("drf_len = {}".format(len(drf)))

##############YH DEBUG CODE to print out DRF output for comparison


if debug:
    plt.figure(1)

os.environ['PATH_TO_IMAGE_DIR'] = args.path_to_image_dir
path_to_image_dir = os.getenv("PATH_TO_IMAGE_DIR")
image_dir = args.image_dir

width = test_case.test_setup[0].image_command[0].image_files[0].width
height = test_case.test_setup[0].image_command[0].image_files[0].height

imgY = pmax_v34.imgY(0)
imgX = pmax_v34.imgX(0)
ovrsX = pmax_v34.ovrsX(0)
C = pmax_v34.C(0)

# read the 4X ref and 1X src images
[img_ref_4x, image_src_pq] = pmax_v34.read_images(imgX, imgY, args.base_dir, image_dir, args.image_type, test_case, )

# get the baseline index
pma_pass_idx = test_case.test_setup[0].image_command[0].pma_parameters.pma_pass_index
pma_pass_configuration = test_case.test_setup[0].pma_pass_configuration[pma_pass_idx]
baseline_index = pma_pass_configuration.index
baseline = pma_pass_configuration.baseline
# get the num of passes for the chosen baseline
num_passes = pma_pass_configuration.number_of_pma_passes

pmax_v34.set_num_passes(num_passes)
num_drf_sets = num_passes

#current_pass_num = 0 # normally will loop over num_passes
# loop over number of passes
plane_command_list = []
for current_pass_num in range(num_passes):
    plane_config_select = test_case.test_setup[0].pma_pass_configuration[pma_pass_idx].pma_pass_parameters[current_pass_num].plane_config_select
    pma_patch_config0 = pma_pass_configuration.pma_pass_parameters[current_pass_num].patch_pma_config[0]
    pma_patch_config1 = pma_pass_configuration.pma_pass_parameters[current_pass_num].patch_pma_config[1]
    if pma_patch_config0 != pma_patch_config1:
        num_drf_sets = num_drf_sets + 1
    max_disparity_selected = test_case.test_setup[0].pma_pass_configuration[pma_pass_idx].pma_pass_parameters[current_pass_num].max_disparity

    script_path = test_case.test_setup[0].pma_plane_configuration[plane_config_select].script_path
    script_args = test_case.test_setup[0].pma_plane_configuration[plane_config_select].script_args
    #print(script_path," ", script_args)
    if script_path != '':
        for script_args_index in range(len(script_args)):
            system_command = 'python3 ' + script_path + script_args[script_args_index] + '--json ' + args.json + ' --test_name ' + args.test_name + ' --index 0'
            print(system_command)
            os.system(system_command)

    filename = test_case.test_setup[0].pma_plane_configuration[plane_config_select].file_path


    infile = open(filename, "rb")
    plane_command_list.append(unpack(BL_PLANE_CONFIGURATION_COMMAND_T, infile.read()))
    infile.close()

    for index in range(plane_command_list[current_pass_num].num_planes):
        patch_pma_config = \
            test_case.test_setup[0].pma_pass_configuration[pma_pass_idx].pma_pass_parameters[
                current_pass_num].patch_pma_config
        timing_sel = plane_command_list[current_pass_num].params[index].mv_patch_timing_sel
        pmax_v34.inc_num_planes(current_pass_num, timing_sel)
        pma_patch_configuration = test_case.test_setup[0].pma_patch_configuration[patch_pma_config[timing_sel]]
        pmax_v34.set_max_disparity( current_pass_num*pmax_v34.max_num_planes() + index, max_disparity_selected)
        pmax_v34.update_parm_list(merge_green_flag, plane_command_list[current_pass_num], pma_patch_configuration, index, current_pass_num)

pmax_v34.set_number_drf_sets(num_drf_sets)

# Read DRF Data file
num_pass_timings = pmax_v34.num_passes()
if args.drf_version == 2:
    print("Reading {}".format(drf_filename))
    drf = []
    p = 0
    num_drf = 0
    num_drf_in_pass = (pmax_v34.jumpX(p)//pmax_v34.stepX(p)) * (pmax_v34.jumpY(p)//pmax_v34.stepY(p))
    toggle_flag = False
    while True:
        elem = drf_infile.read(ctypes.sizeof(DRF_V2_OUTPUT_T))
        if elem == b'':
            break
        elem = unpack(DRF_V2_OUTPUT_T, elem)
        if num_drf == 0:
            drf.append([])
        drf[-1].append(elem)
        num_drf += 1

        # Check to advance pass
        if num_drf == num_drf_in_pass:
            num_drf = 0
            if pmax_v34.num_planes(p, 1) == 0:
                p += 1
                toggle_flag = False
            elif toggle_flag == False:
                toggle_flag = True
            elif toggle_flag == True:
                p += 1
                toggle_flag = False
            if p == num_pass_timings:
                p = 0
            num_drf_in_pass = (pmax_v34.jumpX(p*pmax_v34.max_num_planes())//pmax_v34.stepX(p*pmax_v34.max_num_planes())) * (pmax_v34.jumpY(p*pmax_v34.max_num_planes())//pmax_v34.stepY(p*pmax_v34.max_num_planes()))

    drf_infile.close()
    print("Done")
    print("")

    print("drf_len = {}".format(len(drf)))
if args.drf_version == 3:
    print("Reading {}".format(drf_filename))
    drf = []
    p = 0
    num_drf = 0
    num_drf_in_pass = (pmax_v34.jumpX(p)//pmax_v34.stepX(p)) * (pmax_v34.jumpY(p)//pmax_v34.stepY(p))
    toggle_flag = False
    while True:
        elem = drf_infile.read(ctypes.sizeof(DRF_V3_OUTPUT_T))
        if elem == b'':
            break
        elem = unpack(DRF_V3_OUTPUT_T, elem)
        if num_drf == 0:
            drf.append([])
        drf[-1].append(elem)
        num_drf += 1

        # Check to advance pass
        if num_drf == num_drf_in_pass:
            num_drf = 0
            if pmax_v34.num_planes(p, 1) == 0:
                p += 1
                toggle_flag = False
            elif toggle_flag == False:
                toggle_flag = True
            elif toggle_flag == True:
                p += 1
                toggle_flag = False
            if p == num_pass_timings:
                p = 0

            num_drf_in_pass = (pmax_v34.jumpX(p*pmax_v34.max_num_planes())//pmax_v34.stepX(p*pmax_v34.max_num_planes())) * (pmax_v34.jumpY(p*pmax_v34.max_num_planes())//pmax_v34.stepY(p*pmax_v34.max_num_planes()))

    drf_infile.close()
    print("Done")
    print("")

    print("drf_len = {}".format(len(drf)))

current_pass_num = 0 # normally will loop over num_passes
drf_simple = [[] for _ in range(pmax_v34.num_passes())]
depth_simple = [[] for _ in range(pmax_v34.num_passes())]

rmemY = pmax_v34.rmemY(0)
Rxm = pmax_v34.Rxm(0)
#plane_command has all the parameters
rmem_ovrs_map = np.zeros((rmemY, Rxm)); # image of the ref fetch sequence for debug

img_ref_4x_strip = np.zeros((rmemY,imgX*ovrsX,C))
img_src_1x_strip = np.zeros((rmemY,imgX,C), dtype=np.uint64)


dscans_per_escan = floor((imgX - plane_command_list[0].ref_patch_offset_q2 / pmax_v34.ovrsX(0)  + pmax_v34.NEGATIVE_DISPARITY_SLACK_PQ(0))/pmax_v34.jumpX(0))
# for now pass the num_dscans_per_escan for Firmware/FPGA case via arg.
dscans_per_escan_rtl = args.num_dscans_per_escan_rtl

best_depth_arr = [[] for _ in range(pmax_v34.number_drf_sets())]
best_plane_arr = [[] for _ in range(pmax_v34.number_drf_sets())]
best_score_arr = [[] for _ in range(pmax_v34.number_drf_sets())]
rtl_depth_arr = [[] for _ in range(pmax_v34.number_drf_sets())]
rtl_plane_arr = [[] for _ in range(pmax_v34.number_drf_sets())]
rtl_best_score_arr = [[] for _ in range(pmax_v34.number_drf_sets())]
#target_escan_num = 18 # debug code
drf_rtl_idx = 0
jumpY = pmax_v34.jumpY(0)
for num_escans in range(imgY//jumpY):
    #if num_escans == target_escan_num:
        deltaY = jumpY * num_escans
        #num_escans = 0 # YH DEBUG MUST remove this
        if deltaY + rmemY < imgY:
            img_ref_4x_strip[:,:,:] = img_ref_4x[deltaY + 0 : deltaY + rmemY, :, :]
            img_src_1x_strip[:,:,:] = image_src_pq[deltaY + 0:deltaY + rmemY, :, :].astype(np.uint64)
            pmax_v34.escan(img_ref_4x_strip.copy(), img_src_1x_strip.copy(), 0,   plane_command_list,
                                             args.dscan_method, args.test_name, args.drf_version)

            print("done with escan for all planes and pass number: {} ".format(num_escans), file=fp_file)
            print("done with escan for all planes and pass number: {} ".format(num_escans))

            drf_rtl_idx = drf_rtl_idx + dscans_per_escan_rtl

print("", file=fp_file)
fp_file.close()
