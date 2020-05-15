from math import *
import numpy as np;

def Lanczos_1Dn_old (a, n):
	# "n+1" entries in the table maintained only for compatibility
	# with experimental linear interpolated kernel python variant
	# "n" entries is used in the current HW, extra entry never accessed
	L = np.zeros((n+1, 2*a));
	for j in range (n+1):
		sum = 0.;
		for i in range (-a, a):
			x = i+j/n;
			if (x == 0):
				L[j][a-i-1] = 1.;
			else:
				L[j][a-i-1] = a*sin(pi*x)*sin(pi*x/a)/(pi*x)**2;
			sum += L[j][a-i-1];
		sum = 1./sum;
		for i in range (-a, a):
			L[j][a-i-1] *= sum;
	return L;

def Lanczos_1Dn (a, n):
        L = np.zeros((n, 2*a)).astype(np.float32);
        for j in range (n):
                sum = 0.;
                for i in range (-a, a):
                        x = i+j/n;
                        if (x == 0):
                                L[j][a-i-1] = 1.;
                        else:
                                L[j][a-i-1] = a*sin(pi*x)*sin(pi*x/a)/(pi*x)**2;
                        sum += L[j][a-i-1];
                sum = 1./sum;
                for i in range (-a, a):
                        L[j][a-i-1] *= sum;
        return L;

def Lanczos_1Dx (a, xf):
	L = np.zeros(2*a);
	sum = 0.;
	for i in range (-a, a):
		x = i+xf;
		if (x == 0):
			L[a-i-1] = 1.;
		else:
			L[a-i-1] = a*sin(pi*x)*sin(pi*x/a)/(pi*x)**2;
		sum += L[a-i-1];
	sum = 1./sum;
	for i in range (-a, a):
		L[a-i-1] *= sum;
	return L;

# print ();
# print (Lanczos_1Dn (2, 64));

