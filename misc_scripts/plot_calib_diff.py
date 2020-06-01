import os
import numpy as np
from matplotlib import pyplot as plt
from decimal import Decimal
import sys
import json

import cv2
import glob
from libs.exr_convert import convert_exr_image

def create_video(folder, filepattern, suffix="rig"):
    img_array = []
    size = (0,0)
    files = sorted(glob.glob(folder + '/diffs/diffs_*/*' + filepattern + '.png'))
    jpegs = sorted(glob.glob(folder + '/jpgs/*/*A1*.jpg'))
    rots = sorted(glob.glob(folder + '/rotations/*/rotations_{}.png'.format(suffix)))
    for file, jpeg, rot in zip(files, jpegs, rots):
        diff_img = cv2.imread(file)
        height, width, layers = diff_img.shape
        tmp = cv2.imread(rot)
        jpgsize = (int(3200*height/2200), height)
        size = (width + jpgsize[0], height + tmp.shape[0])
        rimgshape = (tmp.shape[0], size[0], tmp.shape[2])
        rimg = np.ones(rimgshape, dtype='uint8') * 255
        rimg[:, :tmp.shape[1], :] = tmp
        cam_img = cv2.resize(cv2.imread(jpeg), jpgsize)
        hcon = cv2.hconcat([cam_img, diff_img])
        img_array.append(cv2.vconcat([hcon, rimg]))
    print(size)

    out = cv2.VideoWriter('rotations_{}.mp4'.format(suffix), cv2.VideoWriter_fourcc(*'mp4v'), 5, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def plot_calib_diff(folder):
    stats = json.load(open(folder + "/diff_calib_stats.json", 'r'))
    worst_error = stats["worst_error_src_pixels"]
    worst_x = stats["worst_pt_ref_x"]
    worst_y = stats["worst_pt_ref_y"]
    worst_depth = stats["worst_pt_ref_depth_mm"]
    titlestr = "Worst error(A1, A2): {:.2f}, Depth: {:.2E}, Ref: ({}, {})".format(
                    worst_error, Decimal(worst_depth), worst_x, worst_y)
    basename = os.path.basename(folder)
    arr = convert_exr_image(folder + '/calib_diff_img.exr')
    pixel_diff = arr[:, :, 0]
    distance = arr[:, :, 1]
    plt.figure(figsize=(8, 4))
    plt.imshow(pixel_diff, vmin=0., vmax=2.)
    plt.colorbar()
    plt.title(titlestr)
    plt.xlabel(basename)
    plt.savefig(folder + '/' + basename + '_enorm.png')
    plt.close()
    plt.figure(figsize=(8, 4))
    plt.imshow(pixel_diff)
    plt.colorbar()
    plt.title(titlestr)
    plt.xlabel(basename)
    plt.savefig(folder + '/' + basename + '_error.png')
    plt.close()
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(pixel_diff)
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(distance)
    plt.colorbar()
    plt.suptitle(basename + "\n" + titlestr)
    plt.savefig(folder + '/' + basename + '_distance.png')
    plt.close()

def get_worst_error(folder):
    stats = json.load(open(folder + "/diff_calib_stats.json", 'r'))
    return stats["worst_error_src_pixels"]



if __name__ == '__main__':
    root = '/Users/amaharshi/results/video_sequence/multiframe/diffs/'
    start_index = 555
    end_index = 651
    suffix = "rig"
    os.chdir(root)

    # folder = root + '/diffs/diffs_{:03d}_{:03d}'.format(start_index, start_index)
    # plot_calib_diff(folder)
    # for i in range(start_index, end_index):
    #     folder = root + '/diffs/diffs_{:03d}_{:03d}'.format(i, i+1)
    #     plot_calib_diff(folder)
    # # create_video(root, 'enorm', suffix)

    plt.figure()
    a = list()
    for i in range(start_index, end_index):
        folder = root + '/diff_{:03d}'.format(i)
        a.append(get_worst_error(folder))

    plt.plot(range(start_index, end_index), a)
    plt.title("Difference in calib (A1, A2)")
    plt.xlabel("Frame number")
    plt.ylabel("Pixels")
    plt.savefig("calib_difference_A1A2.png")

    print("done")

