import numpy as np


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


def convert_exr_image(exrfile):
    import OpenEXR
    import Imath
    exr = OpenEXR.InputFile(exrfile)
    DW = exr.header()['dataWindow']
    rows, cols = (DW.max.y - DW.min.y + 1, DW.max.x - DW.min.x + 1)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    nptype = (np.float32, (rows, cols))
    chans = [np.frombuffer(exr.channel(c, FLOAT), dtype=nptype, count=1).squeeze() for c in ('R', 'G', 'B')]
    return np.dstack(chans)


def reshape_matlab_corners(matlab):
    mshape = matlab.shape
    matlab = np.swapaxes(matlab, 2, 3)
    matlab = np.reshape(matlab, (mshape[0], mshape[1], mshape[3], 44, 19))
    matlab = np.flip(matlab, axis=4)
    matlab = np.reshape(matlab, (mshape[0], mshape[1], mshape[3], 836))
    matlab = np.swapaxes(matlab, 2, 3)
    matlab = np.swapaxes(matlab, 0, 1)
    mshape = matlab.shape
    retval = np.zeros((mshape[0], mshape[1], mshape[2], mshape[3] + 1))
    retval[:, :, :, :2] = matlab
    retval[:, :, :, 2] = matlab[:, :, :, 1]
    return retval.astype(np.float32)


def imageio_write(mlab, folder):
    import imageio
    for i in range(mlab.shape[0]):
        imageio.imwrite(folder + "/f{:03d}.exr".format(i), mlab[i, :, :, :])
