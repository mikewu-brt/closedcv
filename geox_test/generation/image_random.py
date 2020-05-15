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
matplot.use('Qt5agg')
import matplotlib.pyplot as plt


width = 3208
height = 2200
num_bits = 12
M = pow(2, num_bits)

image = np.random.rand(height, width) * M

outfile = "image_random.bin"
image.astype(np.int16).tofile(outfile)

plt.figure(1)
plt.imshow(image.astype(np.int16)[:, :, 0])
plt.title("Channel 0")

plt.figure(2)
plt.imshow(image.astype(np.int16)[:, :, 1])
plt.title("Channel 1")

plt.figure(3)
plt.imshow(image.astype(np.int16)[:, :, 2])
plt.title("Channel 2")

plt.figure(4)
plt.imshow(image.astype(np.int16)[:, :, 3])
plt.title("Channel 3")
