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


from geox_test.from_vlad.lanczos_test import *

La = 4
Ls = 32
L = Lanczos_1Dn(La, Ls)

outfile = open("lanczos_kernel.h", "w")

for i in range(Ls):
    for j in range(La):
        outfile.write("{}f, ".format(L[i, j].astype(np.float32)))
    outfile.write("\n")
    for j in range(La, 2*La):
        outfile.write("{}f, ".format(L[i, j].astype(np.float32)))
    outfile.write("\n\n")

outfile.close()

