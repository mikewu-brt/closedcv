from math import ceil, floor
import numpy as np

# main PMA "triple"
K = 64;			# HW parallelism factor
C = 4;			# color channels per pixel
W = 12;			# bits per color channel

# accelerator clock frequency
# freq = 1e9;		
freq = 1e8;

# sensor dimentions in quads
imgX = 1604;
imgY = 1100;

ctxtX_max = 32+7; #virtual address space is 64 (6bits) 
ctxtY_max = 32;

del2_max = 320;

resW = 10; # 10 bits per score

# ref and source memory sizes (single plane test only for now)
rmemX = ctxtX_max; 
rmemY = ctxtY_max; 
smemX = imgX; 
smemY = ctxtY_max;
# YH debug not sure - ref memory size in Pmax


# oversamplng and distortion model
ovrsX = 4;

Rxm = rmemX * 2 * ovrsX;
Rym = rmemY;

# fp step (skew or compression) per quad, -0.5 to +0.5
global distAf, distBf, distCf
distAf = 0.35;
distBf = 0.;
distCf = 0.;
# scaling and rounding done in SW/firmware
# A,B are S0.7, C is S5.2 (all 8-bit values)
global distA, distB, distC
distA = floor(distAf*256+0.5);
distB = floor(distBf*256+0.5);
distC = floor(distCf*4+0.5);

# patch dimensions (and other parameters) for different verification tests
# all test cases share the same jump and also use nested grids
# (not required by HW, but will simplify SW)

global jumpX, jumpY, patchX, patchY, stepX, stepY
jumpX = 20;
jumpY = 20;

patchX = 21;
patchY = 21;
stepX = 10;
stepY = 10;

#patchX = 7;
#patchY = 7;
#stepX = 2;
#stepY = 2;

# patchX = 5;
# patchY = 5;
# stepX = 1;
# stepY = 1;

global tileX, tileY, nresX, nresY
tileX = jumpX-stepX+1;
tileY = jumpY-stepY+1;
nresX = (tileX-1)//stepX+1;
nresY = (tileY-1)//stepY+1;

global g_output_loc
g_output_loc = np.zeros((nresY, nresX))

# context size
global ctxtX, ctxtY
ctxtY = patchY+tileY-1;
ctxtX = patchX+tileX-1;
# increase context X-dim to allow for the full grd map to fit in
tileX_ext = floor(max((tileX-1)*distAf, (tileY-1)*distBf)+0.5);
ctxtX += tileX_ext;
tileX += tileX_ext;

if (ctxtX > ctxtX_max):
	print ("WARNING: max context width exceeded", ctxtX);
if (ctxtY > ctxtY_max):
	print ("WARNING: max context height exceeded", ctxtY);

#size of the patch unpacked into a linear array
global patchL
patchL = patchY*patchX*C;

# disparity search range
NEGATIVE_DISPARITY_SLACK_PQ = 4
global dispK, dispX
dispK = imgX//(5*K);
dispX_pix = dispK*K;

max_disparity_pq = dispX_pix//2

dispX = ceil(max_disparity_pq/K) * K # relates to number of c-scans in a d-scan


# r-node reject threshold
# rej_thresh = 1e3;

# # D.V. accuracy threshold
# ver_thresh = 10;

#--------	
print ("\ngeneral info:")
print ("arch: 1xK, parallelism: %d, single plane"%(K));
print ("image size in quads:", imgX, "x", imgY);
print ("disparity search range:", dispX, "c-scans:", dispK);
print ("selected patch size:", patchX, "x", patchY);
print ("selected patch step:", stepX, "x", stepY);
print ("selected tile jump:", jumpX, "x", jumpY);
print ("tile dimensions:", tileX, "x", tileY);
print ("context dimensions:", ctxtX, "x", ctxtY);
print ("number of results:", nresX, "x", nresY);

del2_rm = patchY*(tileX+tileX_ext);
del2_cm = patchX*tileY;
print ("delay 2 len for row-maj c-scan", del2_rm);
print ("delay 2 len for col-maj c-scan", del2_cm);

if (del2_rm > del2_max):
	print ("WARNING: max delay 2 len exceeded in row-maj c-scan");
if (del2_cm > del2_max):
	print ("WARNING: max delay 2 len exceeded in col-maj c-scan");

#--------
cycles = ctxtX*ctxtY;
scores = nresX*nresY*K;
print ("\nthroughput for %d-wide PMA, single plane:"%(K));
print ("c-scan cycles", cycles);
print ("d-scan cycles", cycles*dispK);
print (scores, "scores per c-scan");
print ("%.3f"%(scores/cycles), "scores/cycle", "%.3f"%(resW*scores/cycles), "packed bits/cycle");
print ("%.3f"%(freq*resW*scores/(cycles*8*1024**2)), "packed Mbytes/sec");
print ("%.3f"%(freq*16*scores/(cycles*8*1024**2)), "unpacked Mbytes/sec");

spf = imgX*imgY*dispX/(stepX*stepY);
print ("scores per frame: %.2e"%(spf));
fps = freq*scores/cycles/spf;
print ("FPS: %.3f, time %.3fms at %.0fMHz"%(fps, 1/fps, freq/1e6));	

#--------
print ("\nC-node config:")
# for now using row-maj c-scan only
# note: the del*_sel should be smaller by 2 than the actual delay
# but for the first two delay lines the actual delay is >=3 so we are safe
print ("delay 1 sel", patchX-2);
print ("delay 2 sel", del2_rm-2);

# the last (multithreaded acc) delay line has forwarding path to deal with del=1
if (nresX-2 < 0):
	print ("delay 3 sel X");
	print ("acc fwrd en", 1);
else:
	print ("delay 3 sel", nresX-2);
	print ("acc fwrd en", 0);

# row-maj order only for now
print ("c-scan order", 0);

#--------
print ("\nref_mem info:")
print ("oversampling in X direction:", ovrsX);
print ("distortion a: %3d(%.3f), b: %3d(%.3f), c %d:"%(distA, distA/256, distB, distB/256, distC));
# print ("size for max  context dims and spec jumpX:", ceil(ovrsX*(ctxtX_max+(ctxtX_max-1)*0.35+jumpX)), "x", ctxtY_max);
print ("size for spec context dims and spec jumpX:", ceil(ovrsX*(ctxtX+(ctxtX-1)*0.35+jumpX)), "x", ctxtY);
print ("steady state load rate:", ovrsX*jumpX*ctxtY, "%dx-os quads per d-scan"%(ovrsX));	
print ("GEOX utilization at PMA limit of %.3f FPS: %d%%"%(fps, ceil(100*ovrsX*jumpX*ctxtY/(cycles*dispK))));	

#r-node reject threshold
rej_thresh = 1e4;

# D.V. accuracy threshold
ver_thresh = 10;

