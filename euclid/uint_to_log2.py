import numpy as np;
from math import floor;

np.random.seed(314159);
print();

expw = 7;
tabw = 6;
extw = 8;
sigw = 14;
xlen = 1<<expw;
tlen = 1<<tabw;
elen = 1<<extw;
scale = tlen*elen;
print ("log scale factor", scale);
print ();

# init approximation tables a,b
ka = [0]*tlen;
kb = [0]*tlen;
l2p = 0;
for i in range (tlen):
  f = np.float64(i+1)/tlen+1;
  l2 = np.log2(f);
  ka[i] = floor(l2p*scale+0.5);
  kb[i] = floor((l2-l2p)*scale/2+0.5); #/2 to keep kb within 8bits
  l2p = l2;

# uncomment to export tables to RTL
# print("ka");
# for i in range(tlen):
#   print ("%d,"%(ka[i]));
# print("kb");
# for i in range(tlen):
#   print ("%d,"%(kb[i]));


def uint_log2 (v):
  #convert to float, then get uint representation of the float
  f = np.float32(v);
  u = f.view(np.uint32);

  # remove the FP exponent bias
  # not needed for PMA since r-node uses the differences of logs
  # u -= 127<<23;
  # correct the bias to match RTL EXPW=7 (vs IEEE EXPW=8)
  u -= 64<<23;
  
  # extract bit fields, sign bit ignored
  u >>= 23-tabw-extw;
  ext = u&(elen-1);
  u >>= extw;
  idx = u&(tlen-1);
  u >>= tabw;
  exp = u&(xlen-1);

  # approximation
  l2 = exp*scale + ka[idx] + (kb[idx]*ext>>(extw-1)); #-1 to compensate for /2 (kb)
  return(l2);

# sanity check
# for i in range (0, 36):
#   if (i == 0):
#     u = 0x2c395df0;
#     # u = 0x2dac1add;
#   else:
#     u = (1<<i)>>1;
#   v = uint_log2(u);
#   print (hex(u), hex(v));

# verify error
# err = 0.;
# e16 = 0.;
# eav = 0.;
# for t in range (1000):
#   v1 = np.power(2, np.random.uniform()*36).astype(np.uint64);
#   v2 = np.power(2, np.random.uniform()*36).astype(np.uint64);

#   #exact
#   d1 = scale*np.log2(v1) - scale*np.log2(v2);
#   #FP16 for comparison
#   d2 = scale*np.log2(v1).astype(np.float16) - scale*np.log2(v2).astype(np.float16);
#   #HW approx
#   d3 = uint_log2(v1) - uint_log2(v2);

#   err += abs(d1-d3);
#   eav += d1-d3;
#   e16 += abs(d1-d2);

# print ("\nerr %.2e"%(err/t));
# print ("eav %.2e"%(eav/t));
# print ("e16 %.2e"%(e16/t));

def setBitNumber(n):
  if n == 0:
    return 0;

  msb = 0;
  n = n // 2;
  while n != 0:
    n = n // 2;
    msb = msb + 1

  return msb

def uint_log2_rtl (data_in):

  #data_in = np.uint64(6405090)
  #  data_in = np.uint64(7467485)
  #data_in = 1149452
  leading_one = setBitNumber(data_in)

  real_e = leading_one
  bias_e = real_e + 63

  mant =  (data_in.astype(np.int64) << 14) >> leading_one # mant is 14 bits
  #z_r = (bias_e << 15) + mant
  z_r = (bias_e << 14) + mant

  float_data = z_r & ((1 << 22) -1)

  #ext = float_data[0 +:EXTW];
  # ext is bits 0,1,2...EXTW-1 of float_data
  #idx = float_data[EXTW +:TABW];
  # idx is bits EXTW, EXTW+1, EXTW+2,.... EXTW+TABW-1

  ext = z_r & ((1<<extw) -1)
  idx = (z_r >> extw) & ((1<<tabw)-1)

  kbxext = kb[idx]*ext
  # extra 1 bit on kakbext, to get any overflow
  # kbxext[extw-1+:extw+1]  with extw = 8 is kbxext[7+:9] - bits 7, 8, 9,...15 i.e kbxext[7:15]
  kakbxext = ka[idx] + ((kbxext >> (extw-1)) & (((1<<(extw+2)) - 1)))
  kakbxext_sigw = kakbxext & ((1 << (sigw)) -1)
  kakbxext_ovf = kakbxext & (1<< sigw)
  data_out = kakbxext_sigw
  flag_upper_part = (1<<21) - (1<<14)
  data_out = data_out + kakbxext_ovf + (float_data & flag_upper_part)
  return data_out

#uint_log2_rtl (np.uint64(6405090))
#uint_log2_rtl (np.uint64(7467485))
