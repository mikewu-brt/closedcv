# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 11:02:46 2017
This is a script file for loading single channel dp/fst floating data.

@author: feng@light.co
"""

import numpy as np
import os

dp_header_dtype = np.dtype([("magic_id", ("S1", 2)),
                            ("tag_header", "S4"),
                            ("tag_align", "<u4"),
                            ("tag_header_size", "<u8"),
                            ("width", "<u4"),
                            ("height", "<u4"),
                            ("channel_count", "<u4"),
                            ("image_format", "<u4"),
                            ("tag_data", "S4"),
                            ("tag_align2", "<u4"),
                            ("data_size", "<u8")]);


def load_fst(fn):
    f = open(fn, 'r')
    dp_header = np.fromfile(f, dtype=dp_header_dtype, count=1)
    rows = dp_header['height'][0]
    cols = dp_header['width'][0]
    channel_count = dp_header['channel_count'][0]

    f.seek(dp_header.itemsize, os.SEEK_SET)
    dp_data = np.fromfile(f, dtype=(np.float32, (rows, cols, channel_count)), count=1)
    f.close()

    return np.squeeze(dp_data[0])


def write_fst(filename, data, is_bgra=False):
    # make sure input is a numpy array
    assert type(data) == np.ndarray
    # check data shape
    shp = data.shape
    assert len(shp) == 2 or len(shp) == 3
    if len(shp) == 2:
        data = data[:,:,np.newaxis]
    num_channels = data.shape[2]
    assert num_channels <= 4
    # expand to 2 or 3-channel output to 4 channels
    if num_channels == 2 or num_channels == 3:
        new_data = np.zeros((data.shape[0], data.shape[1], 4), dtype=data.dtype)
        new_data[:,:,:num_channels] = data
        data = new_data
        num_channels = 4
    # Convert assumed RGBA to BGRA format expected by ParserFST
    if num_channels == 4 and not is_bgra:
        data = data[:, :, [2, 1, 0, 3]]

    sizeoffloat = 4
    dp_header = np.array([(['D', 'X'], 'head', 32764, 16, data.shape[1],
                           data.shape[0], data.shape[2], 0, 'data', 32764, data.size * sizeoffloat)],
                         dtype=dp_header_dtype)

    with open(filename, 'wb') as f:
        f.write(dp_header.tobytes())
        f.write(data.astype(np.float32).tobytes())

# dp/fst file header definitions

## magic number id
#uint8 header_id0
#uint8 header_id1
#
## tag header
#char    id[4] = 'head'
#uint64  size  =
#
## header
#uint width
#uint height
#uint channel_count
#uint format
#
## data tag
#char    id[4] = 'data'
#uint64  size = width * height * channel_count * sizeof(float)
#
## data
#

