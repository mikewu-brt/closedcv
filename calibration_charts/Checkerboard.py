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
#  Checkerboard chart construction

import sys
import numpy as np
import cv2


class Checkerboard:

    # Constants
    CHECKERBOARD = 0
    CIRCLE = 1

    # Variables
    __canvas_width = 3840
    __canvas_height = 2160
    __checker_width = 160
    __checker_height = 160
    __nx = 5
    __ny = 4

    # Methods
    @staticmethod
    def blank_canvas(nx, ny):
        blank = np.full((ny, nx, 3), fill_value=255, dtype=np.uint8)
        return blank

    def __center(self):
        return int(self.__canvas_width / 2), int(self.__canvas_height / 2)

    def generate(self, shift_x=0, shift_y=0, chart=CHECKERBOARD, color=(0, 0, 0)):
        img = self.blank_canvas(self.__canvas_width, self.__canvas_height)
        xc, yc = self.__center()

        xs = int(xc - self.__nx / 2.0 * self.__checker_width + shift_x)
        ys = int(yc - self.__ny / 2.0 * self.__checker_height + shift_y)

        if chart == Checkerboard.CHECKERBOARD:
            for y_idx in range(self.__ny):
                y = ys + y_idx * self.__checker_height
                x_start = xs
                if y_idx & 1:
                    x_start += self.__checker_width
                for x in range(x_start, xs + self.__nx * self.__checker_width, 2 * self.__checker_width):
                    pt1 = (x, y)
                    pt2 = (x + self.__checker_width - 1, y + self.__checker_height - 1)
                    cv2.rectangle(img, pt1, pt2, color=color, thickness=cv2.FILLED)

        elif chart == Checkerboard.CIRCLE:
            for y in range(ys, ys + self.__ny * self.__checker_height, self.__checker_height):
                for x in range(xs, xs + self.__nx * self.__checker_width, self.__checker_width):
                    cv2.circle(img, (x, y), int(self.__checker_width / 4), color=color, thickness=cv2.FILLED)
        else:
            print("Unknown chart type")
            sys.exit(-1)

        return img

    def get_canvas_size(self):
        return self.__canvas_width, self.__canvas_height

    def set_canvas_size(self, width, height):
        self.__canvas_width = int(width)
        self.__canvas_height = int(height)
        return

    def get_checker_size(self):
        return self.__checker_width, self.__checker_height

    def set_checker_size(self, width, height):
        self.__checker_width = int(width)
        self.__checker_height = int(height)
        return

    def get_num_checkers(self):
        return self.__nx, self.__ny

    def set_number_checkers(self, nx, ny):
        self.__nx = int(nx)
        self.__ny = int(ny)
        return

    def __init__(self, canvas_size_px=None, checker_size_px=None, num_checkerboards=None):
        if canvas_size_px is not None:
            self.set_canvas_size(canvas_size_px[0], canvas_size_px[1])

        if checker_size_px is not None:
            self.set_checker_size(checker_size_px[0], checker_size_px[1])

        if num_checkerboards is not None:
            self.set_number_checkers(num_checkerboards[0], num_checkerboards[1])
        return
