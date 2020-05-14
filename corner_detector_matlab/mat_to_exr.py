import numpy as np
from scipy.io import loadmat
import argparse
import os
import sys

def write_exr(pixels_rgb, filename):
    import OpenEXR
    import Imath
    pshape = pixels_rgb.shape
    assert len(pshape) == 3 and pshape[2] == 3, "Only RGB image conversion supported"
    HEADER = OpenEXR.Header(pshape[1], pshape[0])
    chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    HEADER['channels'] = dict([(c, chan) for c in "RGB"])
    exr = OpenEXR.OutputFile(filename, HEADER)
    exr.writePixels({'R': pixels_rgb[:, :, 0].tostring(), 'G': pixels_rgb[:, :, 1].tostring(), 'B': pixels_rgb[:, :, 2].tostring()})
    exr.close()

def imageio_write(mlab, folder):
    import imageio
    for i in range(mlab.shape[0]):
        write_exr(mlab[i, :, :, :], folder + "/f{:03d}.exr".format(i))
#        imageio.imwrite(folder + "/f{:03d}.exr".format(i), mlab[i, :, :, :])

def reshape_matlab_corners(matlab):
    mshape = matlab.shape
    matlab = np.swapaxes(matlab, 2, 3)
    matlab = np.reshape(matlab, (mshape[0], mshape[1], mshape[3], 44, 19))
    matlab = np.reshape(matlab, (mshape[0], mshape[1], mshape[3], 836))
    matlab = np.swapaxes(matlab, 2, 3)
    matlab = np.swapaxes(matlab, 0, 1)
    mshape = matlab.shape
    retval = np.zeros((mshape[0], mshape[1], mshape[2], mshape[3] + 1))
    retval[:, :, :, :2] = matlab
    retval[:, :, :, 2] = matlab[:, :, :, 1]
    return retval.astype(np.float32)

parser = argparse.ArgumentParser(description="mat_t_npy ")
parser.add_argument('--input_mat_file', default='Oct2_cal/input_for_extrinsics_function.mat')
parser.add_argument('--output_exr_dir', default='Oct2_cal/matlab_exr')
#parser.add_argument('--exr_dir_basename', default='matlab_exr')
#parser.add_argument('--npy_dir', default='Oct2_cal')
parser.add_argument('--boardsize_x', type=int, default=19)
parser.add_argument('--boardsize_y', type=int, default=44)
parser.add_argument('--num_cam', type=int, default=4)
args, unknown = parser.parse_known_args()
if unknown:
    print("Unknown options: {}".format(unknown))

num_cam = args.num_cam
nx = args.boardsize_x
ny = args.boardsize_y
#corner_detector_matlab = loadmat(args.input_mat_dir + '/input_for_extrinsincs_function.mat')
corner_detector_matlab = loadmat(args.input_mat_file)
print(corner_detector_matlab.keys())

#print(corner_detector_matlab['A1_imagesUsed'])
A1_imagesUsed = corner_detector_matlab['A1_imagesUsed']
A1_imagePoints = corner_detector_matlab['A1_imagePoints']
A2_imagesUsed = corner_detector_matlab['A2_imagesUsed']
A2_imagePoints = corner_detector_matlab['A2_imagePoints']
A3_imagesUsed = corner_detector_matlab['A3_imagesUsed']
A3_imagePoints = corner_detector_matlab['A3_imagePoints']
A4_imagesUsed = corner_detector_matlab['A4_imagesUsed']
A4_imagePoints = corner_detector_matlab['A4_imagePoints']


corners_detected_matlab = np.zeros((num_cam,A1_imagesUsed.shape[0],nx*ny,2), dtype=np.float) - np.ones((num_cam,A1_imagesUsed.shape[0], nx*ny,2), dtype=np.float)

count = [0, 0, 0, 0]
for indx in range(A1_imagesUsed.shape[0]):
    if A1_imagesUsed[indx] == 1:
        corners_detected_matlab[0, indx] = A1_imagePoints[:,:,count[0]]
        count[0] = count[0] + 1
    if A2_imagesUsed[indx] == 1:
        corners_detected_matlab[1, indx] = A2_imagePoints[:,:,count[1]]
        count[1] = count[1] + 1
    if A3_imagesUsed[indx] == 1:
        corners_detected_matlab[2, indx] = A3_imagePoints[:,:,count[2]]
        count[2] = count[2] + 1
    if A4_imagesUsed[indx] == 1:
        corners_detected_matlab[3, indx] = A4_imagePoints[:,:,count[3]]
        count[3] = count[3] + 1



print(corners_detected_matlab.shape)
corners_detected_matlab = reshape_matlab_corners(corners_detected_matlab)
print(corners_detected_matlab.shape)

if not os.path.exists(args.output_exr_dir):
    os.mkdir(args.output_exr_dir)
else:
    print("directory {} already exists!!".format(args.output_exr_dir))
    exit(1)

imageio_write(corners_detected_matlab, args.output_exr_dir)
