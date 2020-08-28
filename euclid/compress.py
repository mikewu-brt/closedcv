# compress r-node score bit field using custom FP format:
# no sign, no exp bias, exp == 0 -> denormalized
# monotonic, smooth uint->uint map,
# close to log2 for large input, linear for small inputs

from math import log2

expw = 4;
sigw = 6; #4 to 6

def compr (v):
	for e in range (2**expw-1):
		if (v>>(sigw+1) == 0):
			break;
		v >>= 1;
	v += e<<sigw;
	return v;

def tbit (v, pos):
	return (v&(1<<pos));

def compr_rtl (v):
	e = 0;
	s = 0;
	for i in range (2**expw-1):
		if (tbit(v, i+sigw)):
			e = i+1;
			s = i;

	r = (e<<sigw)|((v>>s)&(2**sigw-1));
	return r;

# print ();

# for i in range (100):
# 	print(i, compr(i), compr_rtl(i));

# for i in range (15):
# 	print();
# 	v = 2**i-1;
# 	print(v, compr(v), compr_rtl(v));
# 	v = 2**i;
# 	print(v, compr(v), compr_rtl(v));
# 	# print(2**sigw*(log2(v)-sigw+1)); #approx for large v
# 	v = 2**i+1;
# 	print(v, compr(v), compr_rtl(v));

