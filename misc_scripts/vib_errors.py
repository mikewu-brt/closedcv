from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import cv2
from libs.parse_calib_data import *

fig_index = [4, 1, 6, 10]
module_name = ["A1", "A2", "A3", "A4"]

def plot_camera(w, h, fi, R, T, ax, Rbase, factor):
    R = np.squeeze(Rbase).transpose() @ R
    rod = cv2.Rodrigues(R)[0]
    R = cv2.Rodrigues(rod * factor)[0]
    rodri = cv2.Rodrigues(R)[0]
    theta = np.linalg.norm(rodri)
    if theta != 0.:
        rodri = rodri / theta

    T = T.reshape((3, 1))
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    xx = xx - w / 2.
    yy = yy - h / 2.
    sshape = xx.shape
    zz = np.ones(sshape) * fi
    surface = np.vstack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]).transpose()
    surface = (R.transpose() @ (surface.transpose() - T)).transpose()

    top = np.array([[-w/2,  w/2, 0],
                    [-h/2, -h/2, 0],
                    [  fi,   fi, 0]])
    top = R.transpose() @ (top - T)
    pc = Poly3DCollection([top.transpose()], alpha=0.5, linewidths=1)
    pc.set_facecolor('yellow')
    pc.set_edgecolor('black')
    ax.add_collection3d(pc)
    left = np.array([[-w/2, -w/2, 0],
                     [-h/2,  h/2, 0],
                     [  fi,   fi, 0]])
    left = R.transpose() @ (left - T)
    pc = Poly3DCollection([left.transpose()], alpha=0.5, linewidths=1)
    pc.set_facecolor('yellow')
    pc.set_edgecolor('black')
    ax.add_collection3d(pc)
    bottom = np.array([[w/2, -w/2, 0],
                       [h/2,  h/2, 0],
                       [ fi,   fi, 0]])
    bottom = R.transpose() @ (bottom - T)
    pc = Poly3DCollection([bottom.transpose()], alpha=0.5, linewidths=1)
    pc.set_facecolor('yellow')
    pc.set_edgecolor('black')
    ax.add_collection3d(pc)
    right = np.array([[ w/2,  w/2, 0],
                      [-h/2,  h/2, 0],
                      [  fi,   fi, 0]])
    right = R.transpose() @ (right - T)
    pc = Poly3DCollection([right.transpose()], alpha=0.5, linewidths=1)
    pc.set_facecolor('yellow')
    pc.set_edgecolor('black')
    ax.add_collection3d(pc)
    ax.plot_surface(surface[:, 0].reshape(sshape), surface[:, 1].reshape(sshape), surface[:, 2].reshape(sshape), alpha=1)
    # ax.set_ylim3d(-50, 50)
    # ax.set_xlim3d(-50, 50)
    ax.view_init(-90., -90.)
    ax.set_axis_off()
    titlestr = 'Axis/Angle(degrees)\n({:.2f}, {:.2f}, {:.2f}), {:.2f}'.format(rodri[0, 0], rodri[1, 0], rodri[2, 0], theta*180./np.pi)
    ax.set_title(titlestr)

def plot_rotation(fig, index, R, Rbase, factor=1000., T = np.zeros((3,1))):
    p = .0045
    sx = 7
    w = 3200. * p * sx
    h = 2200. * p * sx

    f = 5560.
    fi = f * p

    ax = fig.add_subplot(2, 6, index, projection='3d')
    plot_camera(w, h, fi, R, T, ax, Rbase, factor)


def plot_rotations(folder, Rbase, ref_cam):
    fig = plt.figure(figsize=(12, 4))
    _, Rref, Tref, _ = parse_calib_data_file(folder + "/calib{}.json".format(ref_cam))
    plot_rotation(fig, fig_index[ref_cam], Rref.transpose() @ Rref, Rbase[ref_cam, :, :])

    for i in range(4):
        if i != ref_cam:
            _, R, T, _ = parse_calib_data_file(folder + "/calib{}.json".format(i))
            R = Rref.transpose() @ R
            plot_rotation(fig, fig_index[i], R, Rbase[i, :, :])
    fig.suptitle("Relative rotations shown w.r.t. {} (Rotation factor: 1000x)".format(module_name[ref_cam]), x=0.6, y=0.1)
    plt.savefig(folder + '/rotations_{}.png'.format(module_name[ref_cam]))
    plt.close()


def get_camera_centers(R, T):
    num_cams = R.shape[0]
    centers = np.zeros((num_cams, 3, 1))
    for cam in range(num_cams):
        centers[cam] = -R[cam].transpose() @ T[cam]
    return centers

def get_rotation_rig(R, T):
    cc_a2 = R[1].transpose() @ T[1] #Use x-axis as the negative direction of A2 camera center
    cc_a4 = -R[3].transpose() @ T[3]
    x = cc_a2 / np.linalg.norm(cc_a2)
    z = np.cross(np.squeeze(x), np.squeeze(cc_a4))
    z = z / np.linalg.norm(z)
    y = np.cross(z, np.squeeze(x))
    return np.hstack([x, np.expand_dims(y, axis=1), np.expand_dims(z, axis=1)])


def plot_rotations_rig(folder, outfolder, Rbase):
    _, R, T, _ = parse_calib_data_folder(folder)
    Rrig = get_rotation_rig(R, T)

    fig = plt.figure(figsize=(12, 4))
    for i in range(4):
        plot_rotation(fig, fig_index[i], Rrig.transpose() @ R[i], Rbase[i], 10.)
    fig.suptitle("Rotations shown w.r.t. rig (Rotation factor: 10)", x=0.6, y=0.1)
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    filename = outfolder + '/rotations_rig.png'
    #assert not os.path.exists(filename), "File exists, are you sure you want to override?"
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    root = '/Users/amaharshi/results/video_sequence'
    start_index = 550
    end_index = 651
    for ref_cam in [0, -1]:
        folder = root + '/norm_calib/norm_calib_{:03d}'.format(start_index)

        _, Rbase, Tbase, _ = parse_calib_data_folder(folder)

        if ref_cam < 0:
            Rrig = get_rotation_rig(Rbase, Tbase)
            for i in range(4):
                Rbase[i, :, :] = Rrig.transpose() @ Rbase[i, :, :]

            for i in range(start_index, end_index):
                print(i)
                plot_rotations_rig(root + '/norm_calib/norm_calib_{:03d}'.format(i),
                                   root + '/rotations/f{:03d}'.format(i), Rbase)
        else:
            RrefBase = Rbase[ref_cam, :, :].copy()
            for i in range(4):
                Rbase[i, :, :] = RrefBase.transpose() @ Rbase[i, :, :]

            for i in range(start_index, end_index):
                print(i)
                plot_rotations(root + '/norm_calib/norm_calib_{:03d}'.format(i), Rbase, ref_cam)


    print("done")
