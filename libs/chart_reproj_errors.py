import os
import numpy as np
from matplotlib import pyplot as plt
from libs.read_fst import load_fst


module_name = ["A1", "A2", "A3", "A4", "A5", "A6"]


def chart_mesh_grid(chart_shape=(44,19)):
    x = np.arange(0, chart_shape[1], 1)
    y = np.arange(0, chart_shape[0], 1)
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def black_list(chart_shape=(44,19), markers=[390, 428, 389, 407, 426, 409, 427]):
    xx, yy = chart_mesh_grid(chart_shape)
    xend = chart_shape[1]-1
    yend = chart_shape[0]-1
    return np.unique(np.hstack(
        [np.ravel_multi_index(np.where(yy == yend), chart_shape),
         np.ravel_multi_index(np.where(xx == xend), chart_shape),
         np.ravel_multi_index(np.where(yy == 0), chart_shape),
         np.ravel_multi_index(np.where(xx == 0), chart_shape),
         markers])).astype(int)


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
        q = plt.quiver(image_coord[:, 0], image_coord[:, 1], reproj[:, 0], reproj[:, 1])
        plt.quiverkey(q, 0.1, 0.1, avg_10x, np.format_float_scientific(avg_10x, precision=2),
                      labelpos='N', coordinates='axes')
        plt.xlabel(filename + "_frame_" + str(frame))
        plt.savefig(outdir + "/frame_" + str(frame) + "_rerrors.png")
        plt.close()
    plt.figure()
    reproj = arr[:, :, [0, 3]]
    plt.plot(np.average(np.linalg.norm(reproj, axis=2), axis=1))
    plt.xlabel("View error for " + filename)
    plt.savefig(outdir + "/view_errors.png")
    plt.close()
    print("done " + filename)


def pproj(dirname, file_prefix, outdir_prefix, num_cams=4):
    for cam in range(num_cams):
        plotreproj(dirname + "/" + file_prefix + str(cam) + ".fst", dirname + "/" + outdir_prefix + str(cam))


def plot_view_errors_orientations(folder, title, num_cams = 6,
        prefix_rerror="rerror", prefix_rot="chart_rotations", prefix_trans="chart_translation"):
    rerrors = collect_reprojection_errors(folder, prefix_rerror, num_cams)[:, :, :, 2:]
    verrs = np.average(np.linalg.norm(rerrors, axis=3), axis=2)
    max_verrs = np.max(verrs, axis=1)
    rotations = np.linalg.norm(collect_rotations(folder, prefix_rot), axis=1)
    translations = np.linalg.norm(collect_translations(folder, prefix_trans), axis=1)
    max_rot = np.max(rotations)
    rotations = rotations / max_rot
    max_trans = np.max(translations)
    translations = translations / max_trans
    num_cams = verrs.shape[0]
    fig, axes = plt.subplots(2, int((num_cams + 1)/2), figsize=(12, 8))
    for cam in range(num_cams):
        pindex = np.unravel_index(cam, axes.shape)
        axes[pindex].plot(verrs[cam]/max_verrs[cam], label="view_error")
        axes[pindex].plot(rotations, label="rotations")
        axes[pindex].plot(translations, label="translations")
        axes[pindex].set_title("Max error: {:.2f}".format(max_verrs[cam]))
        axes[pindex].legend()
    fig.suptitle("{} - max rotation(degrees): {:.2f}, max translation: {:.2f}".format(title, max_rot * 180. / np.pi, max_trans))
    return fig


def compare_view_errors(rerrors, labels, title, metric="max"):
    num_cams = rerrors[0].shape[0]
    fig, axes = plt.subplots(2, int((num_cams + 1)/2), figsize=(24, 16))
    for cam in range(num_cams):
        pindex = np.unravel_index(cam, axes.shape)
        max_val = 0.
        min_val = 100.
        for ir in range(len(rerrors)):
            r = rerrors[ir][cam, :, :, 2:]
            if metric == "max":
                values = np.max(np.linalg.norm(r, axis=2), axis=1)
                axes[pindex].plot(values, label=labels[ir])
            else:
                values = np.average(np.linalg.norm(r, axis=2), axis=1)
                axes[pindex].plot(values, label=labels[ir])
            max_val = max(max_val, np.max(values))
            min_val = min(min_val, np.min(np.where(values == 0., 100., values)))
        axes[pindex].set_ylim(min_val, max_val)
        axes[pindex].legend()
        axes[pindex].set_title(title + " for camera {}".format(module_name[cam]))
    return fig


def plot_view_errors(rerrors, title):
    num_cams = rerrors.shape[0]
    fig, ax = plt.subplots()
    for cam in range(num_cams):
        r = rerrors[cam, :, :, 2:]
        ax.plot(np.average(np.linalg.norm(r, axis=2), axis=1), label=module_name[cam])
    ax.legend()
    ax.set_title(title)
    return fig


def collect_translations(dirname, file_prefix):
    return load_fst(os.path.join(dirname, file_prefix + ".fst"))


def collect_rotations(dirname, file_prefix):
    return load_fst(os.path.join(dirname, file_prefix + ".fst"))


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


def plot_reprojection_errors(folder, file_prefix, num_cams = 6, chart_shape=(44, 19)):
    reproj_errors = collect_reprojection_errors(folder, file_prefix, num_cams)
    valid = get_valid_points(reproj_errors)
    cam_index = 0
    fig1, together = plt.subplots()
    fig2, axes = plt.subplots(2, num_camsi, figsize=(24, 8))
    for cam_errors in valid:
        nerrors = np.average(np.linalg.norm(cam_errors, axis=2), axis=0)
        index = np.argmax(nerrors)
        corner = np.unravel_index(index, chart_shape)
        max_error = nerrors[index]
        together.plot(nerrors, label=module_name[cam_index])
        x = np.arange(0, chart_shape[1], 1)
        y = np.arange(0, chart_shape[0], 1)
        xx, yy = np.meshgrid(x, y)
        axes[0, cam_index].scatter(xx, yy, s=nerrors*100)
        axes[0, cam_index].set_title(
            "Average absolute reprojection errors for camera {}\nMax Error {:.2f} at {}".format(
                cam_index, max_error, corner))
        xerrors = np.average(cam_errors[:, :, 0], axis=0)
        yerrors = np.average(cam_errors[:, :, 1], axis=0)
        q = axes[1, cam_index].quiver(xx, yy, xerrors, yerrors, scale=7.5)
        axes[1, cam_index].quiverkey(q, -0.1, -0.1, 0.75, np.format_float_scientific(0.75, precision=2),
                                     labelpos='N', coordinates='axes')
        axes[1, cam_index].set_title(
            "Average for camera {}, chart error: {:.2E}, {:.2E}".format(
                cam_index, np.average(xerrors), np.average(yerrors)))
        cam_index += 1
    together.legend()
    fig1.suptitle(folder)
    fig2.suptitle(folder)
    fig1.savefig(folder + "/linear_plot.png")
    fig2.savefig(folder + "/quiver_scatter.png")


def plot_chart_deviations(folder, short_title, filename="chart_warp.fst"):
    chart = load_fst(os.path.join(folder, filename))
    orig = chart[:, :2]
    error = chart[:, 2:]
    plt.figure(figsize=(5,11))
    Q = plt.quiver(orig[:, 0], orig[:, 1], -error[:, 0], -error[:, 1], scale=50)
    plt.quiverkey(Q, -0.1, -0.1, 2, "2mm", labelpos='N', coordinates='axes')
    plt.title(short_title)
    plt.savefig(os.path.join(folder, short_title + "_chart.png"))
    plt.close()
    return orig, error


def view_errors(folders, labels, title="Max view errors", metric="max"):
    prefix = "rerror"
    rerrors = list()
    for f in folders:
        rerrors.append(collect_reprojection_errors(f, prefix))
    return compare_view_errors(rerrors, labels, title, metric)
