from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, TimeoutError
from concurrent.futures.process import BrokenProcessPool
from math import ceil, floor, sqrt
from multiprocessing import Queue, Process, Manager, freeze_support
from pathlib import Path
from queue import Empty
from time import time, sleep
from typing import List, Tuple, Union, Callable
import numpy as np

import h5py
import hdf5plugin
from numpy import floor as np_floor
from numpy import max as np_max
from numpy import mean as np_mean
from numpy import sqrt as np_sqrt
from numpy import round as np_round
from numpy import zeros, float32, dstack, rollaxis, savez_compressed, array, maximum, rot90, arange, uint8, uint16
from psutil import cpu_count, virtual_memory
from skimage.measure import block_reduce
from skimage.transform import resize, resize_local_mean
from tifffile import natural_sorted
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from pystripe.core import (imread_tif_raw_png, imsave_tif, progress_manager, is_uniform_2d, is_uniform_3d,
                           convert_to_8bit_fun,  convert_to_16bit_fun)
from supplements.cli_interface import PrintColors, date_time_now
from tsv.volume import TSVVolume, VExtent
import os
import sys
import argparse

def generate_voxel_spacing(
        shape: Tuple[int, int, int],
        source_voxel: Tuple[float, float, float],
        target_shape: Tuple[int, int, int],
        target_voxel: float):
    voxel_locations = [
        np.arange(axis_shape) * float(axis_v_size) - (axis_shape - 1) / 2.0 * float(axis_v_size)
        for axis_shape, axis_v_size in zip(shape, source_voxel)
    ]
    axis_spacing = []
    for i, axis_vals in enumerate(voxel_locations):
        # Get Downsampled starting value
        start = np_round(resize_local_mean(axis_vals, (int(target_shape[i]),)))[0]
        # Create target_voxel spaced list
        axis_spacing.append(array([start + target_voxel * val for val in range(target_shape[i])]))
    return axis_spacing

def gen_npz(downsampled_path, destination, target_voxel, source_voxel, max_processors):
    # down-sample on z accurately
    print('creating npz')
    npz_file = destination / f"{destination.stem}_zyx{target_voxel:.1f}um.npz"
    images = natural_sorted([str(f) for f in downsampled_path.iterdir() if f.is_file() and f.suffix.lower() in (
            ".tif", ".tiff", ".raw", ".png")])
    init_shape = Path(downsampled_path/images[0])
    im = Image.open (init_shape)
    width, height = im.size
    shape = [height, width]
    num_images = len(images)

    if npz_file.exists():
        print('this line shouldnt appear but: ' + str(npz_file))
        return 0
        return return_code
    print(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
            f"{PrintColors.BLUE}down-sampling: {PrintColors.ENDC}"
            f"resizing on the z-axis accurately ...")
    
    print 
    target_shape_3d = [
        # denominator = 
        int(floor(num_images / (target_voxel / float(source_voxel[0])))),
        int(round(shape[0] / (target_voxel / float(source_voxel[1])))),
        int(round(shape[1] / (target_voxel / float(source_voxel[2]))))
    ]
    # if rotation in (90, 270):
    #     target_shape_3d[1], target_shape_3d[2] = target_shape_3d[2], target_shape_3d[1]
    #     print('rotating')
    print('debug prints:')
    print('exact value: ' + str(num_images / (target_voxel / float(source_voxel[0]))))
    print('target_shape_3d: ' + str(target_shape_3d))
    print('num_images: ' + str(num_images))
    print('target_voxel: ' + str(target_voxel))
    print('source voxels (z,y,x): ' + str(source_voxel))

    files = sorted(downsampled_path.glob("*.tif"))
    if len(files) == 0:
        files = sorted(downsampled_path.glob("*.tiff"))
    print(f"Debug: Number of files loaded = {len(files)}") 
    print(f"Debug: path used: {downsampled_path}")
        # Using a ThreadPoolExecutor to read and process files concurrently
    with ThreadPoolExecutor(max_processors) as pool:
        img_stack = list(pool.map(imread_tif_raw_png, tqdm(files, desc="loading", unit="images")))
        print(img_stack)
        img_stack = dstack(img_stack)
        img_stack = rollaxis(img_stack, -1) 
        print(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
                f"{PrintColors.BLUE}down-sampling: {PrintColors.ENDC}"
                f"resizing the z-axis ...")
        img_stack = resize(img_stack, target_shape_3d, preserve_range=True, anti_aliasing=True)
        axes_spacing = generate_voxel_spacing(
            (num_images, shape[0], shape[1]),
            source_voxel,
            target_shape_3d,
            target_voxel)
        print(f"{PrintColors.GREEN}{date_time_now()}:{PrintColors.ENDC}"
                f"{PrintColors.BLUE} down-sampling: {PrintColors.ENDC}"
                f"saving as npz.")
        savez_compressed(
            npz_file,
            I=img_stack,
            xI=array(axes_spacing, dtype='object')  # note specify object to avoid "ragged" warning
        )


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="Linux path to downsampled TIFF images")
parser.add_argument("-o", "--output", required=True, help="Linux path to NPZ output")
parser.add_argument("-dt", "--downsampled_voxel", required=True, help="Source downsample (target) voxel value used in micron (um)")
parser.add_argument("-dx", "--voxel_x", required=True, help="Source x voxel value")
parser.add_argument("-dy", "--voxel_y", required=True, help="Source y voxel value")
parser.add_argument("-dz", "--voxel_z", required=True, help="Source z voxel value")

args = parser.parse_args()
source = [args.voxel_z, args.voxel_y,  args.voxel_x] # zyx format
max_processors = cpu_count(logical=False)

gen_npz(Path(args.input), Path(args.output), float(args.downsampled_voxel), source, max_processors)

# Ex: python npz_downsample.py -i /in/ -o /out/ -dt 10 -dx 10 -dy 10 -dz 9.6
