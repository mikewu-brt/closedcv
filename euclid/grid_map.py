# Compute the grid map for a given set of parameters
import numpy as np
import math
import sys

def ij2xy(i, j, d):
    a = d[0]
    b = d[1]
    c = d[2]

    x = i + a*i + b*j + c
    y = j
    return x, y

def compute_gg_map(Px, Py, Sx, Sy, Jx, Jy, a, b, offset):
    """
    :param Px: Patch X size in pixels
    :param Py: Patch Y size in pixels
    :param Sx: Skip X size in pixels
    :param Sy: Skip Y size in pixels
    :param Jx: Jump X size in pixels
    :param Jy: Jump Y size in pixels
    :param a: Distortion scale  [-0.35, 0.35]
    :param b: Distortion scale  [-0.35, 0.35]
    :param offset: Plane offset [0, 0.75]
    :return:
        (a, b, c) - plane distortion parameters
        gg_map - plane gg map
        Conext - Dictionary of various context parameters
        (a0, b0, c0) - reference plane (ie undistorted plane) parameters
        gg0_map - reference plane gg map
        align_err - Patch center alignment error
    """

    # Convert a and b to discrete floats - HW is Q-1 notation
    a = int(a * 256) / 256
    b = int(b * 256) / 256

    # Oversample Rate
    OSx = 4

    # Max distortion
    a_max = 0.35
    b_max = -0.35
    o_min = 0

    # Compute tile and context size
    Tx = Jx - Sx + 1
    Ty = Jy - Sy + 1
    Nx = Jx // Sx
    Ny = Jy // Sy

    Cx = 64
    Cy = Py + Ty - 1

    ###
    # Compute c parameters - pivot bottom left or top right
    ###

    # Compute minimum c to keep worst case shear in reference memory
    c_wc_min = -a_max * 0 - b_max * (Cy-1) - o_min / OSx
    d_wc_min = (a_max, b_max, c_wc_min)
    wc_tl_x, _ = ij2xy(0, 0, d_wc_min)
    wc_br_x, _ = ij2xy(Px-1, Py-1, d_wc_min)
    pc_wc_max = (wc_tl_x + wc_br_x) / 2

    # Compute the patch location of the undistorted patch at o_min
    c0_min = math.floor((pc_wc_max - (Px-1)/2) * OSx + 0.5) / OSx

    # Round c0_min to a whole pixel offset
    c0_min = math.ceil(c0_min)

    # Compute the patch location of the desired patch at min offset
    if b <= 0:
        c_min = -a * 0 - b * (Cy-1) - o_min / OSx
        d_min = (a, b, c_min)
        tl_min_x, tl_min_y = ij2xy(0, 0, d_min)
        br_min_x, br_min_y = ij2xy(Px-1, Py-1, d_min)
        pc_min = (tl_min_x + br_min_x) / 2
        c_min += c0_min + (Px-1)/2 - pc_min
    else:
        c_min = -a * 0 - b * 0 - o_min / OSx
        d_min = (a, b, c_min)
        tl_min_x, tl_min_y = ij2xy(0, Cy-Py, d_min)
        br_min_x, br_min_y = ij2xy(Px-1, Cy-1, d_min)
        pc_min = (tl_min_x + br_min_x) / 2
        c_min += c0_min + (Px-1)/2 - pc_min

    # Adjust the undistorted patch and desired patch by offset
    c0 = c0_min + (offset - o_min) / OSx
    c = math.floor(c_min * OSx + (offset - o_min)) / OSx

    # Alignment
    x_align = np.zeros((Ny, Nx))
    for y in range(Ny):
        j = y * Sy
        for x in range(Nx):
            i = x * Sx
            px_tl = math.floor((i + a * i + b * j + c) * OSx + 0.5) / OSx
            px_br = math.floor((i + (Px-1) + a * (i + Px-1) + b * (j + Py-1) + c) * OSx + 0.5) / OSx
            pc = (px_tl + px_br) / 2

            p0c = c0 + i + (Px-1) / 2

            i_new = (px_br + (p0c - pc) - b * (j + Py-1) - c) / (1 + a)
            x_align[y, x] = i_new - (i + Px-1)

    # Compute the alignment error to a pixel boundary
    x_align_pix = np.floor(x_align + 0.5).astype(np.int32)

    # GG Maps
    gg0_map = np.zeros((Cy, Cx))
    gg_map = gg0_map.copy()
    d_shift = np.zeros((64, 64))
    for y in range(Ny):
        j0 = y * Sy + Py-1
        for x in range(Nx):
            i0 = x * Sx + Px-1
            gg0_map[j0, i0] = 1

            if (i0 + x_align_pix[y, x]) < Cx:
                gg_map[j0, i0 + x_align_pix[y, x]] = 1
                d_shift[j0, i0 + x_align_pix[y, x]] = -x_align_pix[y, x]

    # Reduce Cx to a minimum
    col_sum = np.sum(gg_map, axis=0) + np.sum(gg0_map, axis=0)
    Cx_max = np.argwhere(col_sum != 0)[-1][0] + 1
    gg_map = gg_map[:, :Cx_max]
    gg0_map = gg0_map[:, :Cx_max]

    align_err = x_align_pix - x_align + offset / OSx
    d = (a, b, c)
    d0 = (0, 0, c0 - offset / OSx)

    Context = dict();
    Context["Cx"] = Cx_max
    Context["Cy"] = Cy
    Context["Px"] = Px
    Context["Py"] = Py
    Context["Tx"] = Cx_max - Px + 1
    Context["Ty"] = Cy - Py + 1

    # Convert gg_map into gg_map_ulong
    gg_map_ulong = np.zeros(Cy)
    for i in range(Cx_max):
        gg_map_ulong = gg_map_ulong + gg_map[:, i] * pow(2, i)
    Context["gg_map_ulong"] = gg_map_ulong.astype(np.uint64)

    # Compute the disp_correction and make d_shift suitable for HW
    if np.all(x_align_pix[0, :] == x_align_pix):
        idx = np.argwhere(gg_map != 0)[0, 0]
        d_shift = np.expand_dims(d_shift[idx, :], axis=0)
    elif np.all(x_align_pix[:, 0].T == x_align_pix.T):
        d_shift = np.expand_dims(np.sum(d_shift, axis=1) / Nx, axis=1)
    disp_corr = (-x_align_pix.astype(np.float64) * (1 + a)).astype(np.int)
    d_shift = (d_shift.astype(np.float64) * (1 + a)).astype(np.int)

    # Offset d_shift by 8 and compress
    d_shift = d_shift + 8
    if np.any(d_shift < 0) or np.any(d_shift > 15):
        print("d_shift exceeds HW capability")
        sys.exit(-1)
    if d_shift.shape[1] != 1:
        d_shift_hw = d_shift[:, 0::8] * (2**0)
        for i in range(1, 8):
            d_shift_hw += d_shift[:, i::8] * (2**(i*4))
    else:
        d_shift_hw = d_shift[0::8, :] * (2**0)
        for i in range(1, 8):
            d_shift_hw += d_shift[i::8, :] * (2**(i*4))

    return d, gg_map, Context, d0, gg0_map, d_shift_hw, disp_corr, align_err


def compute_ab(P, s):
    ab = s / ((P-1)/2)
    ab = math.floor(ab * 256 + 0.5) / 256
    return ab


def compute_pma_distortions(Px, Py, Sx, Sy, Jx, Jy):
    # Compute the maximum shear
    ab_max = 0.35
    dist_step = 0.5

    # Create PMA parameters for an undistorted plane
    pma = []
    d, gg_map, C, d0, gg0_map, d_shift, disp_corr, align_err = compute_gg_map(Px, Py, Sx, Sy, Jx, Jy,
                                                       compute_ab(Px, 0.0), compute_ab(Py, 0.0), 0)
    pma.append(dict())
    pma[-1]['d'] = d
    pma[-1]['gg_map'] = gg_map
    pma[-1]['ND'] = 0

    # Compute shears
    if Sx > 1:
        max_shear = math.ceil((Px - 1) * ab_max + 0.5) / 2
        for shear in np.arange(-dist_step, -max_shear-dist_step, -dist_step):
            d, gg_map, C, d0, gg0_map, d_shift, disp_corr, align_err = compute_gg_map(Px, Py, Sx, Sy, Jx, Jy,
                                                               compute_ab(Px, 0), compute_ab(Px, shear), 0)
            pma.append(dict())
            pma[-1]['d'] = d
            pma[-1]['gg_map'] = gg_map
            pma[-1]['shear'] = shear

    # Compute expansion / compression
    if Sx > 1:
        max_scale = math.floor((Px - 1) * ab_max + 0.5) / 2
        for scale in np.arange(-max_scale, max_scale + dist_step, dist_step):
            if scale == 0:
                continue

            d, gg_map, C, d0, gg0_map, d_shift, disp_corr, align_err = compute_gg_map(Px, Py, Sx, Sy, Jx, Jy, compute_ab(Px, scale),
                                                               compute_ab(Px, 0), 0)
            pma.append(dict())
            pma[-1]['d'] = d
            pma[-1]['gg_map'] = gg_map
            pma[-1]['scale'] = scale

    # Compute offset cases
    if Sx == 1:
        for offset in range(-3, 4, 1):
            if offset == 0:
                continue

            d, gg_map, C, d0, gg0_map, d_shift, disp_corr, align_err = compute_gg_map(Px, Py, Sx, Sy, Jx, Jy, compute_ab(Px, 0),
                                                               compute_ab(Px, 0), offset)
            pma.append(dict())
            pma[-1]['d'] = d
            pma[-1]['gg_map'] = gg_map
            pma[-1]['offset'] = offset

    return pma
