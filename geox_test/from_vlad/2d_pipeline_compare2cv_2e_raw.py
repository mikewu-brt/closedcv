from math import *
import numpy as np;
import matplotlib as matplot
matplot.use('Qt5agg')
import matplotlib.pyplot as plt
import cv2;
from lanczos_test import Lanczos_1Dn
from scale2x import *

# def LmouseCallback(event,x,y,flags,param):
# 	global img_c, img_o;
# 	if event == cv2.EVENT_LBUTTONDOWN:
# 		print(x, y);
# 		pc = img_c[x//2][y//2];
# 		po = img_o[x//2][y//2];
# 		print(pc);
# 		print(po);
# 		print(pc-po);
# 		print("%", 100*(pc-po)/pc);

# wd,ht must be divisible by 16 (for now)
#img_i = plt.imread('Lenna_(test_image).png');

dirname = "../Depth Captures/outside_nov21_25mm_f2_8_center"
img_in_raw = np.load(os.path.join(dirname, "input.npy"))
img_in_raw = img_in_raw / 65536.0
img_i = np.empty((img_in_raw.shape[0] // 2, img_in_raw.shape[1] // 2, 4))
img_i[:, :, 0] = img_in_raw[::2, ::2]
img_i[:, :, 1] = img_in_raw[::2, 1::2]
img_i[:, :, 2] = img_in_raw[1::2, ::2]
img_i[:, :, 3] = img_in_raw[1::2, 1::2]
plt.figure(0).clear()
plt.imshow(img_in_raw)
plt.title("In")

# img_i = plt.imread('raw_50cm_0140_b_0_g_0_mtf_0_seg.png');
# img_i = plt.imread('IMG_20181129_140851_seg.png');
# img_i = plt.imread('Left_0.jpg').astype(np.float32)/256;
ht,wd,ch = img_i.shape;
#ht = ceil(ht/16)*16;
#wd = ceil(wd/16)*16;

# Chuck - Updated below lines
img_o = np.zeros((ht,wd,ch)).astype(np.float32);
gain = np.zeros((ht,wd)).astype(np.float32);
#ht = floor(ht/16)*16;
#wd = floor(wd/16)*16;
print (wd,ht,ch);


# identity OK
m3x3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32);
# translate OK
# m3x3 = np.array([[1, 0, 10.13], [0, 1, 21.87], [0, 0, 1]]).astype(np.float32);
# scale X OK
# m3x3 = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32);
# shear OK
# m3x3 = np.array([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32);
# perspective OK
#m3x3 = np.array([[1, 0, 0], [0, 1, 0], [0.0001, 0.0001, 1]]).astype(np.float32);
# mixes OK
# m3x3 = np.array([[2, 0, 10], [0.5, 1, 20], [0.0001, -0.0005, 1]]).astype(np.float32);
# a = 0.1;
# m3x3 = np.array([[cos(a), -sin(a), 10.3], [sin(a), cos(a), 20.7], [0, 0, 1]]).astype(np.float32);
# m3x3 = np.array([[cos(a)*0.9, -sin(a), 55.3], [sin(a), cos(a)*0.8, 10.7], [0.0001, -0.0002, 1]]).astype(np.float32);
print();
print (m3x3);
m3x3i = np.linalg.inv(m3x3.astype(np.float64)).astype(np.float32);
print (m3x3i);

Ls = 32;
La = 4;
LK = Lanczos_1Dn(La,Ls);
offs = 1./(2*Ls);
# print (LK);

mstep = 16;
#dmap = np.zeros((ceil(ht/mstep)+1, ceil(wd/mstep)+1, 2));
dmap = np.load(os.path.join(dirname, "asic_dist_map.npy"))
# print (dmap);
#vmap = np.ones((ceil(ht/mstep)+1, ceil(wd/mstep)+1));
vmap = np.load(os.path.join(dirname, "asic_vig_map.npy"))
# for y in range (0,ht+mstep,mstep):
# 	for x in range(0,wd+mstep,mstep):
# 		dy = (y-ht/2)**2;
# 		dx = (x-wd/2)**2;
# 		vmap[y//mstep][x//mstep] = 1.0 - 0.005*sqrt(dx+dy);
# print (vmap);

blk_lev = np.float32(0);
ch_gain = np.ones(4).astype(np.float32);

for y in range (ht):
	for x in range(wd):
		xy1 = np.array([x, y, 1]).astype(np.float32);
		
		#apply combined 2D xform
		uvw = m3x3i @ xy1;

		#u,v,w -> yp,xp,1
		#normalize back to homogenious coordinates
		wi = 1./uvw[2];
		xp = uvw[0]*wi;
		yp = uvw[1]*wi;

		#clip coordinates after the combined xform, set px to 0 outside
		if(xp < 0 or xp >= wd or yp < 0 or yp >= ht):
			continue;

		#apply distortion map (coordinate domain)
		#read 4-neghborhood of x',y'
		xm = xp/mstep;
		ym = yp/mstep;
		xi = floor(xm);
		yi = floor(ym);
		xf = xm-xi;
		yf = ym-yi;

		a = dmap[yi][xi];
		b = dmap[yi][xi+1];
		c = dmap[yi+1][xi];
		d = dmap[yi+1][xi+1];

		#bilinear interpolation
		e = a*(1-xf)+b*xf;
		f = c*(1-xf)+d*xf;
		g = e*(1-yf)+f*yf;

		xpp = xp+offs+g[0]; #add x-component of the distortion correction
		ypp = yp+offs+g[1]; #add y-component of the distortion correction

		#clip coordinates again, set px to 0 outside
		if(xpp < La-1 or xpp >= wd-La or ypp < La-1 or ypp >= ht-La):
			continue;

		#Lanczos interpolation in the signal domain
		#read 16-neighborhood of x'', y''
		xi = floor(xpp);
		yi = floor(ypp);
		xl = floor((xpp-xi)*Ls);
		yl = floor((ypp-yi)*Ls);

		nb = img_i[yi-La+1:yi+La+1, xi-La+1:xi+La+1];

		for c in range(ch):
			img_o[y][x][c] = nb[:,:,c] @ LK[xl] @ LK[yl];

		#apply vignetting map
		#read 4-neghborhood of x'',y''
		xm = xpp/mstep;
		ym = ypp/mstep;
		xi = floor(xm);
		yi = floor(ym);
		xf = xm-xi;
		yf = ym-yi;

		a = vmap[yi][xi];
		b = vmap[yi][xi+1];
		c = vmap[yi+1][xi];
		d = vmap[yi+1][xi+1];

		#bilinear interpolation
		e = a*(1-xf)+b*xf;
		f = c*(1-xf)+d*xf;
		g = e*(1-yf)+f*yf;
		gain[y, x] = g
		#print("({}, {}), gain: {}".format(x, y, g))

		for c in range(ch):
			#elementary ISP functions
			img_o[y][x][c] -= blk_lev;
			img_o[y][x][c] *= g;
			img_o[y][x][c] *= ch_gain[c];

			#clipping
			if (img_o[y][x][c] < 0.):
				img_o[y][x][c] = 0.;
			if (img_o[y][x][c] > 1.):
				img_o[y][x][c] = 1.;

			#convert to 12-bits for PMA
			#...

# plt.figure();
# plt.imshow(img_i);
# plt.figure();
# plt.imshow(img_o);
# plt.show();

#img_c = cv2.warpPerspective(img_i, m3x3, (wd,ht), flags=cv2.INTER_LANCZOS4);

#plt.figure();
# cv2.imshow("inp", cv2.cvtColor(scale2x(img_i), cv2.COLOR_BGR2RGB));
#cv2.imshow("out", cv2.cvtColor(scale2x(img_o), cv2.COLOR_BGR2RGB));
#cv2.imshow("cv2", cv2.cvtColor(scale2x(img_c), cv2.COLOR_BGR2RGB));
# cv2.imshow("out", cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB));
# cv2.imshow("cv2", cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB));

plt.figure(1).clear();
img_out_raw = np.empty(img_in_raw.shape)
img_out_raw[::2, ::2] = img_o[:, :, 0]
img_out_raw[::2, 1::2] = img_o[:, :, 1]
img_out_raw[1::2, ::2] = img_o[:, :, 2]
img_out_raw[1::2, 1::2] = img_o[:, :, 3]
plt.imshow(img_out_raw)
plt.title("Out")
img_m_raw = np.load(os.path.join(dirname, "output.npy"))
img_m_raw = img_m_raw / 65536.0
plt.figure(2).clear()
plt.imshow(img_m_raw)
plt.title("CV")

img_d_raw = np.sqrt(np.sqrt(np.abs(img_m_raw - img_out_raw)))
#img_d_raw = img_m_raw - img_out_raw
plt.figure(3).clear()
plt.imshow(img_d_raw)
plt.colorbar()
plt.title("diff - |m-o|^(1/4)")

img_e_raw = np.abs(img_m_raw - img_out_raw)
plt.figure(4).clear()
plt.imshow(img_e_raw)
plt.colorbar()
plt.title("diff")

img_f_raw = np.divide(img_m_raw, img_out_raw)
plt.figure(5).clear()
plt.imshow((1-img_f_raw) * (1-img_f_raw))
plt.colorbar()
plt.title("gain")


#img_d = np.sqrt(np.sqrt(np.abs(img_c-img_o)));
#cv2.namedWindow("dif", 1);
# cv2.setMouseCallback("dif", LmouseCallback);
#cv2.imshow("dif", cv2.cvtColor(scale2x(img_d), cv2.COLOR_BGR2RGB));
# cv2.imshow("dif", cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB));

print ("done");


