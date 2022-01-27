"""raw.py - read and write .raw files


"""

import numpy as np
import tifffile


def raw_imread(path, dtype=None, shape=None):
    """Read a .raw file

    params
        path: path to the file
        dtype: data type
        shape: (height, width) of the image
    returns: a Numpy read-only array mapping the file as an image
    """

    if dtype is None or shape is None:
        as_uint32 = np.memmap(
            path,
            dtype=">u4",
            mode="r", shape=(2,))
        width_be, height_be = as_uint32[:2]
        del as_uint32
        as_uint32 = np.memmap(
            path,
            dtype="<u4",
            mode="r", shape=(2,))
        width_le, height_le = as_uint32[:2]
        del as_uint32

        # Heuristic, detect endian by assuming that the smaller width is the right one. Works for widths < 64K
        if width_le < width_be:
            width, height = width_le, height_le
            dtype = "<u2"
        else:
            width, height = width_be, height_be
            dtype = ">u2"
        shape = (height, width)

    try:
        return np.memmap(
            path,
            dtype=dtype,
            mode="r",
            offset=8,
            shape=shape
        )
    except Exception:
        print(f"Bad path: {path}, height = {shape[0]}, width = {shape[1]}")
        # raise


def raw_imsave(path, img):
    """
    Write a .raw file

    :param path: path to the file
    :param img: a Numpy 2d array
    """

    as_uint32 = np.memmap(
        path,
        dtype=np.uint32,
        mode="w+", shape=(2,)
    )
    as_uint32[0] = img.shape[1]
    as_uint32[1] = img.shape[0]
    del as_uint32
    as_uint16 = np.memmap(
        path,
        dtype=np.uint16,
        mode="r+",
        offset=8,
        shape=img.shape
    )
    as_uint16[:] = img
    del as_uint16
