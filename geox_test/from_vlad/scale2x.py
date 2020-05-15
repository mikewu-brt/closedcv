import numpy as np;

def scale2x(img):
	w,h,c = img.shape;
	img_2x = np.zeros((w*2, h*2, c)).astype(np.float32);
	img_2x [0::2, 0::2] = img;
	img_2x [1::2, 0::2] = img;
	img_2x [0::2, 1::2] = img;
	img_2x [1::2, 1::2] = img;
	return img_2x;
