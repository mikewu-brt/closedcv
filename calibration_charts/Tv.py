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


def pixel_size_mm(width_px, height_px, diagonal_in):
    dp = (width_px ** 2 + height_px ** 2) ** 0.5
    d_mm = diagonal_in * 25.4
    return d_mm / dp
