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
#  Generate Checkboard

from Checkerboard import *
from Tv import *
import cv2
import argparse
import matplotlib as matplot
matplot.use('TkAgg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Generate Distortion Charts")
parser.add_argument('--chart_type', default='circle', help="\'circle\' or \'checkerboard\'")
parser.add_argument('--tv', default='main', help="\'main\', \'hubble\', \'webb\' or \'kepler\'")
parser.add_argument('--num_x', default=52, type=int, help="Number of checkerboards / circles along X")
parser.add_argument('--num_y', default=34, type=int, help="Number of checkerboards / circles along Y")
parser.add_argument('--size', default=60, type=int, help="Checkerboard size (or circle spacing)")
parser.add_argument('--shift', default=12, type=int, help="Pixel shift per display")
parser.add_argument('--offset_x', default=0, type=int, help="Start x offset")
parser.add_argument('--offset_y', default=0, type=int, help="Start y offset")
parser.add_argument('--circle_radius_ratio', default=4.0, type=float, help="Checkerboard size to circle radius ratio")
parser.add_argument('--vignetting', action="store_true", default=False,
                    help='Displays blank screen for vignetting estimation.')

args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

plt.figure(1).clear()

# Assume 4k TV
tv_width = 3840
tv_height = 2160

# Determine chart type
if args.chart_type == "circle":
    chart = Checkerboard.CIRCLE
elif args.chart_type == "checkerboard":
    chart = Checkerboard.CHECKERBOARD
else:
    print("Unknown chart type")
    sys.exit(-1)

if args.tv == "main":
    diag = 85.6
elif args.tv == "hubble":
    diag = 54.6
elif args.tv == "webb" or args.tv == "kepler":
    diag = 42.5
else:
    print("Unknown tv")
    sys.exit(-1)

if args.shift > 0:
    pixel_shift = args.shift
else:
    pixel_shift = tv_width

# Plot to use to advance distortion charts
tmp = Checkerboard(canvas_size_px=(100, 100), num_checkerboards=(3, 3), checker_size_px=(10, 10))
img = tmp.generate()
cv2.imshow("Tmp", img)

ck = Checkerboard(canvas_size_px=(tv_width, tv_height),
                  num_checkerboards=(args.num_x, args.num_y),
                  checker_size_px=(args.size, args.size),
                  circle_radius_ratio=args.circle_radius_ratio)

pixel_size = pixel_size_mm(tv_width, tv_height, diag)
checker_width, checker_height = ck.get_checker_size()
print("Pixel Size {} mm".format(pixel_size))
print("Checker Size {} mm".format(pixel_size * checker_width))

# Blank canvas for vignetting
img = ck.blank_canvas(tv_width, tv_height)
cv2.imshow("Checkerboard", img)
cv2.setWindowProperty("Checkerboard", prop_id=cv2.WND_PROP_FULLSCREEN, prop_value=cv2.WINDOW_FULLSCREEN)
cv2.moveWindow("Checkerboard", -1230, -2160)
if args.vignetting:
    # Pause
    cv2.waitKey(0)

# Project checkerboard or circle chart
for y in range(args.offset_y, args.size + args.offset_y, pixel_shift):
    for x in range(args.offset_x, args.size + args.offset_x, pixel_shift):
        img = ck.generate(shift_x=x, shift_y=y, chart=chart)
        print("x: {}, y: {}".format(x, y))

        cv2.imshow("Checkerboard", img)
        cv2.waitKey(0)
