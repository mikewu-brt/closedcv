#  Copyright (c) 2020, The LightCo
#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without
#  modification, are strictly prohibited without prior permission of
#  The LightCo.
#
#  @author  cstanski
#  @version V1.0.0
#  @date    May 2020
#  @brief
#

from libs.LensDistortion import *
import numpy as np
import numpy.polynomial.polynomial as poly

outfile = "vignetting_radial.map"

width = 3208
height = 2200

lens = LensDistortion()

c = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]]) / (width * width / 4)
c[0, 0] = 1
x = np.arange(-width/2, width/2)
y = np.arange(-height/2, height/2)
(ym, xm) = np.meshgrid(y, x, indexing='ij')
vig_map = np.zeros((height, width, 3))
vig_map[:, :, 0] = 1.0 / poly.polyval2d(ym, xm, c)
vig_map[:, :, 1] = vig_map[:, :, 0]
vig_map[:, :, 2] = vig_map[:, :, 0]

lens.set_vignetting(vig_map)

asic_vig_map = lens.asic_vignetting_map()
asic_vig_map.astype(np.float32).tofile(outfile)
