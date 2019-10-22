#  Copyright (c) 2019, The LightCo
#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without
#  modification, are strictly prohibited without prior permission of
#  The LightCo.
#
#  @author  cstanski
#  @version V1.0.0
#  @date    Oct 2019
#  @brief
#

import argparse
from libs.CalibrationInfo import *


####################
# Input Parameters
####################

parser = argparse.ArgumentParser(description="Check-in Calibration File")
parser.add_argument('--cal_dir', default='Oct2_cal')

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

####################

cal_info = CalibrationInfo(args.cal_dir)
cal_info.checkin_cal_file()
