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

outfile = "distortion_radial.map"

width = 3208
height = 2200
img_size = (width, height)

K = np.identity(3)
K[0, 2] = width / 2
K[1, 2] = height / 2

D = np.zeros(5)
D[0] = 1e-9

lens = LensDistortion()

lens.set_radial_distortion_map(K=K, D=D, size=img_size)

asic_dist_map = lens.asic_distortion_map()
asic_dist_map.astype(np.float32).tofile(outfile)


