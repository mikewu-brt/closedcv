"""
 * Copyright (c) 2018, The LightCo
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are strictly prohibited without prior permission of
 * The LightCo.
 *
 * @brief
 *  Subpixel chessboard corner drawing
 *
"""

import numpy as np
import cv2
import math

# Similar to cv2.drawChessboardCorners but with subpixel precision
def drawCornersSubPix(image, patternSize, corners, found, shift = 4):

    radius = 5
    r = radius << shift
    line_type = cv2.LINE_AA

    colors = [[0, 0, 255, 0],
              [0, 128, 255, 0],
              [0, 200, 200, 0],
              [0, 255, 0, 0],
              [200, 200, 0, 0],
              [255, 0, 0, 0],
              [255, 0, 255, 0]] # copied from opencv/modules/calib3d/src/calibinit.cpp

    if image.dtype != np.uint8:
        print("Error: It seems OpenCV does not support subpixel drawing for non uint8 images.")
        raise

    color = colors[0]

    if corners is None:
        return image

    prev_pt = (0, 0)
    for i in range(0, len(corners)):
        p = corners[i][0] * (1 << shift)
        pt = tuple(np.round_(p).astype(int))

        if found and i != 0:
            row = math.floor(i / patternSize[0])
            color = colors[row % 7]
            cv2.line(image, prev_pt, pt, color, 1, line_type, shift)

        cv2.line(image, (pt[0] - r, pt[1] - r), (pt[0] + r, pt[1] + r), color, 1, line_type, shift)
        cv2.line(image, (pt[0] - r, pt[1] + r), (pt[0] + r, pt[1] - r), color, 1, line_type, shift)
        cv2.circle(image, pt, r, color, 1, line_type, shift)

        prev_pt = pt

    return image
