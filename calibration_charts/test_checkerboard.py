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
import matplotlib as matplot
matplot.use('TkAgg')
import matplotlib.pyplot as plt

plt.figure(1).clear()

tv_width = 3840
tv_height = 2160


#ck = Checkerboard(canvas_size_px=(tv_width-50, tv_height - 100), num_checkerboards=(60, 33), checker_size_px=(60, 60))
ck = Checkerboard(canvas_size_px=(tv_width, tv_height), num_checkerboards=(60, 35), checker_size_px=(60, 60))
img = ck.generate(shift_x=15, shift_y=15)


pixel_size = pixel_size_mm(tv_width, tv_height, 85.6)
checker_width, checker_height = ck.get_checker_size()
print("Pixel Size {} mm".format(pixel_size))
print("Checker Size {} mm".format(pixel_size * checker_width))

cv2.imshow("Checkerboard", img)
cv2.setWindowProperty("Checkerboard", prop_id=cv2.WND_PROP_FULLSCREEN, prop_value=cv2.WINDOW_FULLSCREEN)
cv2.moveWindow("Checkerboard", -1200, -2160)
