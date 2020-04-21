import os
import numpy as np
from matplotlib import pyplot as plt
from libs.read_fst import load_fst

def plotreproj(filename, outdir):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    arr = load_fst(filename)
    num_frames = arr.shape[0]

    for frame in range(num_frames):
        reproj = arr[frame, :, [0, 3]].T
        image_coord = arr[frame, :, [2, 1]].T
        plt.figure()
        avg_10x = 10. * np.average(np.linalg.norm(reproj, axis=1))
        Q = plt.quiver(image_coord[:, 0], image_coord[:, 1], reproj[:, 0], reproj[:, 1])
        plt.quiverkey(Q, 0.1, 0.1, avg_10x, np.format_float_scientific(avg_10x, precision=2), labelpos='N', coordinates='axes')
        plt.xlabel(filename + "_frame_" + str(frame))
        plt.savefig(outdir + "/frame_" + str(frame) + "_rerrors.png")
        plt.close()
    plt.figure()
    reproj = arr[:, :, [0, 3]]
    plt.plot(np.average(np.linalg.norm(reproj, axis=2), axis=1))
    plt.xlabel("View error for " + filename )
    plt.savefig( outdir + "/view_errors.png")
    plt.close()
    print("done " + filename)



def pproj(dirname, file_prefix, outdir_prefix, num_cams=4):
    for cam in num_cams:
        plotreproj(dirname + "/" + file_prefix + str(cam) + ".fst", dirname + "/" + outdir_prefix + str(cam))


def collect_reprojection_errors(dirname, file_prefix, num_cams=4):
    cam_reproj = load_fst(os.path.join(dirname, file_prefix + str(0) + ".fst"))
    rshape = cam_reproj.shape
    reproj_errors = np.zeros((num_cams, rshape[0], rshape[1], rshape[2]))
    reproj_errors[0, :, :, :] = cam_reproj[:, :, [2, 1, 0, 3]]  # Switch indices to account for BGRA fst format
    for cam in range(1, num_cams):
        cam_reproj = load_fst(os.path.join(dirname, file_prefix + str(cam) + ".fst"))
        reproj_errors[cam, :, :, :] = cam_reproj[:, :, [2, 1, 0, 3]]  # Switch indices to account for BGRA fst format
    return reproj_errors


def get_valid_points(reproj_errors):
    rerrors = reproj_errors[:, :, :, 2:]
    valid = list()
    for cam in range(rerrors.shape[0]):
        rcam = np.squeeze(rerrors[cam, :, :, :])
        invalid = list()
        for frame in range(rerrors.shape[1]):
            if np.count_nonzero(rerrors[cam, frame, 0, :]) == 0:
                invalid.append(frame)
        valid.append(np.delete(rcam, invalid, axis=0))
    return valid

def plot_reprojection_errors(folder, prefix, chart_shape=(44,19)):
    reproj_errors = collect_reprojection_errors(folder, prefix)
    valid = get_valid_points(reproj_errors)
    cam_index = 0
    fig1, together = plt.subplots()
    fig2, axes = plt.subplots(2, 4, figsize=(24, 8))
    for cam_errors in valid:
        nerrors = np.average(np.linalg.norm(cam_errors, axis=2), axis=0)
        index = np.argmax(nerrors)
        corner = np.unravel_index(index, chart_shape)
        max_error = nerrors[index]
        together.plot(nerrors)
        x = np.arange(0, chart_shape[1], 1)
        y = np.arange(0, chart_shape[0], 1)
        xx, yy = np.meshgrid(x, y)
        scatter = axes[0, cam_index].scatter(xx, yy, s=nerrors*100)
        axes[0, cam_index].set_title("Average absolute reprojection errors for camera {}\nMax Error {:.2f} at {}".format(cam_index, max_error, corner))
        yerrors = np.average(cam_errors[:, :, 0], axis=0)
        xerrors = np.average(cam_errors[:, :, 1], axis=0)
        Q = axes[1, cam_index].quiver(xx, yy, xerrors, yerrors)
        axes[1, cam_index].quiverkey(Q, -0.1, -0.1, max_error, np.format_float_scientific(max_error, precision=2), labelpos='N', coordinates='axes')
        axes[1, cam_index].set_title("Average for camera {}, chart error: {:.2E}, {:.2E}".format(cam_index, np.average(xerrors), np.average(yerrors)))
        cam_index += 1
    fig1.suptitle(folder)
    fig2.suptitle(folder)
    fig1.savefig(folder + "/linear_plot.png")
    fig2.savefig(folder + "/scatter_quiver.png")
