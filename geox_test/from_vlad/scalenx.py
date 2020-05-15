import numpy as np;

def scalenx(img, n):
	h,w,c = img.shape;
	# print(w,h,c);

	img_nx = np.zeros((n*h, n*w, c)).astype(np.float32);
	for i in range(n):
		for j in range(n):
			img_nx [i::n, j::n] = img;			
	return img_nx;
