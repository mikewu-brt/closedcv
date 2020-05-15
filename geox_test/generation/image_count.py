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
#
#  @author  cstanski
#  @version V1.0.0
#  @date    May 2020
#  @brief
#

import numpy as np
import matplotlib as matplot
matplot.use('TkAgg')
import matplotlib.pyplot as plt


width = 3208
height = 2200
num_bits = 12
M = pow(2, 12)

values = np.remainder(np.arange(width * height), M)
image = values.reshape((height, width))

outfile = "image_count.bin"
image.astype(np.int16).tofile(outfile)

plt.figure(1).clear()
plt.imshow(image.astype(np.int16))
