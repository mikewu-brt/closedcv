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

# Generate a flat vignetting map (1.0) for all pixels


from libs.LensDistortion import *
import numpy as np
import numpy.polynomial.polynomial as poly
import math

outfile = "vignetting_flat.map"

width = 3208
height = 2200

vig_map = np.ones((height, width, 3))

lens = LensDistortion()
lens.set_vignetting(vig_map)

asic_vig_map = lens.asic_vignetting_map()
asic_vig_map.astype(np.float32).tofile(outfile)



