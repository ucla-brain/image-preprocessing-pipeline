"""convert.py - programs to convert stacks to output formats"""

import argparse
import itertools
import multiprocessing
import numpy as np
from tifffile import imsave
from .volume import VExtent, TSVVolume
import sys
import tqdm
import os

try:
    from blockfs.directory import Directory
    from mp_shared_memory import SharedMemory
    from precomputed_tif.blockfs_stack import BlockfsStack

    blockfs_present = True
except:
    blockfs_present = False

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'


def worker(args):
    fun = convert_one_plane
    fun(*args)


def convert_to_2D_tif(
        v,
        output_pattern,
        mipmap_level=None,
        volume=None,
        dtype=None,
        compression=('ZLIB', 1),
        cores=multiprocessing.cpu_count(),
        chunks=1024,
        rotation=0):
    """Convert a terastitched volume to TIF

    v: the volume to convert
    output_pattern:
        File naming pattern. output_pattern.format(z=z) is
        called to get the path names for each TIF plane. The directory must
        already exist.
    mipmap_level:
        mipmap decimation level, e.g. "2" to output files
        at 1/4 resolution.
    volume:
        an optional VExtent giving the volume to output
    dtype:
        an optional numpy dtype, defaults to the dtype indicated by the bit depth
    compression:
        between 0 and 9
    cores:
        # of processes to run simultaneously
    chunks: int
        chunk size of multiprocessing pool
    rotation: Rotate image by 0, 90, 180, or 270 degrees
    """
    if volume is None:
        volume = v.volume
    if dtype is None:
        dtype = v.dtype
    if mipmap_level is not None:
        decimation = 2 ** mipmap_level
    else:
        decimation = 1

    # futures = []
    # with multiprocessing.Pool(cores) as pool:
    #     for z in range(volume.z0, volume.z1, decimation):
    #         futures.append(pool.apply_async(
    #             convert_one_plane,
    #             (v, compression, decimation, dtype, output_pattern, volume, z, rotation)))
    #     for future in tqdm.tqdm(futures):
    #         future.get()

    arg_list = []
    for z in range(volume.z0, volume.z1, decimation):
        arg_list.append((v, compression, decimation, dtype, output_pattern, volume, z, rotation))
    num_images_need_processing = len(arg_list)
    while num_images_need_processing//chunks < cores and chunks > 1:
        # print(f"chunks={chunks}")
        chunks //= 2
    print(f"tsv is converting {num_images_need_processing} images using {cores} cores and {chunks} chunks")

    with multiprocessing.Pool(processes=cores) as pool:
        worker.fun = convert_one_plane
        list(tqdm.tqdm(
            pool.imap_unordered(
                worker,
                arg_list,
                chunksize=chunks),
            total=num_images_need_processing,
            ascii=True
        ))


def convert_one_plane(v, compression, decimation, dtype, output_pattern, volume, z, rotation):
    path = output_pattern.format(z=z)
    if os.path.exists(path):
        return

    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    mini_volume = VExtent(
        volume.x0, volume.x1, volume.y0, volume.y1, z, z + 1)
    plane = v.imread(mini_volume, dtype)[0]
    if decimation > 1:
        plane = plane[::decimation, ::decimation]
    if rotation == 90:
        plane = np.rot90(plane, 1)
    elif rotation == 180:
        plane = np.rot90(plane, 2)
    elif rotation == 270:
        plane = np.rot90(plane, 3)

    for _ in range(10):
        try:
            imsave(path, plane, compress=compression)
        except OSError:
            continue
        break


V: TSVVolume = None

if blockfs_present:

    def do_plane(volume: VExtent,
                 z0: int,
                 z: int,
                 sm: SharedMemory,
                 path: str,
                 compression: int):
        mini_volume = VExtent(
            volume.x0, volume.x1, volume.y0, volume.y1, z, z + 1)
        plane = V.imread(mini_volume, sm.dtype)[0]
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        imsave(path, plane, compress=compression)
        with sm.txn() as memory:
            memory[z - z0] = plane


    def convert_to_tif_and_blockfs(
            precomputed_path,
            output_pattern: str,
            volume: VExtent = None,
            dtype=None,
            compression=4,
            cores=multiprocessing.cpu_count(),
            io_cores=multiprocessing.cpu_count(),
            voxel_size=(1800, 1800, 2000),
            n_levels: int = 5):
        if volume is None:
            volume = V.volume
        if dtype is None:
            dtype = V.dtype

        blockfs_stack = BlockfsStack(volume.shape, precomputed_path)
        blockfs_stack.write_info_file(n_levels, voxel_size)
        directory = blockfs_stack.make_l1_directory(io_cores)
        directory.create()
        directory.start_writer_processes()
        sm = SharedMemory((directory.z_block_size,
                           volume.y1 - volume.y0,
                           volume.x1 - volume.x0), dtype)
        with multiprocessing.Pool(cores) as pool:
            for z0 in tqdm.tqdm(
                    range(volume.z0, volume.z1, directory.z_block_size)):
                z1 = min(volume.z1, z0 + directory.z_block_size)
                futures = []
                for z in range(z0, z1):
                    futures.append(pool.apply_async(
                        do_plane,
                        (volume, z0, z, sm, output_pattern % z, compression)))
                for future in futures:
                    future.get()
                x0 = np.arange(0, sm.shape[2], directory.x_block_size)
                x1 = np.minimum(sm.shape[2], x0 + directory.x_block_size)
                y0 = np.arange(0, sm.shape[1], directory.y_block_size)
                y1 = np.minimum(sm.shape[1], y0 + directory.y_block_size)
                with sm.txn() as memory:
                    for (x0a, x1a), (y0a, y1a) in itertools.product(
                            zip(x0, x1), zip(y0, y1)):
                        directory.write_block(memory[:z1 - z0, y0a:y1a, x0a:x1a],
                                              x0a, y0a, z0)
        directory.close()
        for level in range(2, n_levels + 1):
            blockfs_stack.write_level_n(level, n_cores=io_cores)


def make_diag_stack(xml_path, output_pattern,
                    mipmap_level=None,
                    volume=None,
                    dtype=None,
                    silent=False,
                    compression=4,
                    cores=multiprocessing.cpu_count()):
    v = TSVVolume.load(xml_path)
    if volume is None:
        volume = v.volume
    if dtype is None:
        dtype = v.dtype
    if mipmap_level is not None:
        decimation = 2 ** mipmap_level
    else:
        decimation = 1
    if cores == 1:
        for z in tqdm.tqdm(range(volume.z0, volume.z1, decimation)):
            make_diag_plane(v, compression, decimation, dtype, mipmap_level,
                            output_pattern, volume, z)
        return

    futures = []
    with multiprocessing.Pool(cores) as pool:
        for z in range(volume.z0, volume.z1, decimation):
            futures.append(pool.apply_async(
                make_diag_plane,
                (v, compression, decimation, dtype, mipmap_level,
                 output_pattern, volume, z)))
        for future in tqdm.tqdm(futures):
            future.get()


def make_diag_plane(v, compression, decimation, dtype, mipmap_level, output_pattern, volume, z):
    mini_volume = VExtent(
        volume.x0, volume.x1, volume.y0, volume.y1, z, z + 1)
    plane = v.make_diagnostic_img(mini_volume)[0].astype(dtype)
    if plane.shape[2] > 3:
        plane = plane[:, :, :3]
    if mipmap_level is not None:
        plane = plane[::decimation, ::decimation]
    if plane.shape[2] < 3:
        plane = np.dstack(
            list(plane.transpose(2, 0, 1)) +
            [np.zeros(plane.shape[:2], plane.dtype)] * (3 - plane.shape[2]))
    path = output_pattern.format(z=z)
    imsave(path, plane, compress=compression, photometric="rgb")


def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Make a z-stack out of a Terastitcher volume"
    )
    args, mipmap_level, volume = parse_args(parser, args)
    v = TSVVolume.load(args.xml_path, args.ignore_z_offsets, args.input)

    if not blockfs_present or args.precomputed_path is None:
        convert_to_2D_tif(v,
                          args.output_pattern,
                          mipmap_level=mipmap_level,
                          volume=volume,
                          silent=args.silent,
                          compression=args.compression,
                          cores=args.cpus,
                          ignore_z_offsets=args.ignore_z_offsets,
                          rotation=args.rotation)
    else:
        global V
        voxel_size = [float(_) for _ in args.voxel_size.split(",")]
        V = v
        convert_to_tif_and_blockfs(args.precomputed_path,
                                   args.output_pattern,
                                   volume,
                                   dtype=np.uint16,
                                   compression=args.compression,
                                   cores=args.cpus,
                                   io_cores=args.n_io_cpus,
                                   voxel_size=voxel_size,
                                   n_levels=args.levels)


def parse_args(parser: argparse.ArgumentParser, args=sys.argv[1:]):
    """Standardized argument parser for convert functions

    :param parser: an argument parser, possibly configured for the application
    :returns: the parsed argument dictionary, the mipmap level and the volume \
    (or None for the entire volume)
    """
    parser.add_argument(
        "--xml-path",
        required=True,
        help="Path to the XML file generated by Terastitcher")
    parser.add_argument(
        "--output-pattern",
        required=True,
        help='Pattern for tif files, e.g. "output/img_{z:04d}.tif"')
    parser.add_argument(
        "--mipmap-level",
        default=0,
        type=int,
        help="Image decimation level, e.g. --mipmap-level=2 means 4x4x4 "
             "smaller image")
    parser.add_argument(
        "--volume",
        default="",
        help='Volume to be captured. Format is "<x0>,<x1>,<y0>,<y1>,<z0>,<z1>".'
             ' Default is whole volume.')
    parser.add_argument(
        "--compression",
        default=4,
        type=int,
        help="TIFF compression level (0-9, default=3)")
    parser.add_argument(
        "--silent",
        action="store_true")
    parser.add_argument(
        "--cpus",
        default=multiprocessing.cpu_count(),
        type=int,
        help="Number of CPUs to use for multiprocessing")
    parser.add_argument(
        "--ignore-z-offsets",
        action="store_true",
        help="Ignore any z offsets in the stitching XML file."
    )
    parser.add_argument(
        "--input",
        help="Optional input location for unstitched stacks. Default is to "
             "use the value encoded in the --xml-path file"
    )
    parser.add_argument(
        "--rotation",
        type=int,
        default=0,
        help="Rotate each plane by the given number of degrees. Only 0, 90, "
             "180 and 270 are supported"
    )
    if blockfs_present:
        parser.add_argument(
            "--precomputed-path",
            help="Path to precomputed neuroglancer volume to be created. "
                 "Default is not to create a neuroglancer volume"
        )
        parser.add_argument(
            "--levels",
            help="# of neuroglancer mipmap levels to create. Default = 5",
            type=int,
            default=5
        )
        parser.add_argument(
            "--n-io-cpus",
            help="# of CPUs used when creating volume",
            type=int,
            default=min(12, multiprocessing.cpu_count())
        )
        parser.add_argument(
            "--voxel-size",
            help="Voxel size in microns in format x,y,z",
            default="1.8,1.8,2.0"
        )

    args = parser.parse_args(args)
    if args.mipmap_level == 0:
        mipmap_level = None
    else:
        mipmap_level = args.mipmap_level
    if args.volume != "":
        x0, x1, y0, y1, z0, z1 = map(int, args.volume.split(","))
        volume = VExtent(x0, x1, y0, y1, z0, z1)
    else:
        volume = None
    return args, mipmap_level, volume


def diag():
    """Produce a diagnostic image"""
    parser = argparse.ArgumentParser(
        description="Make a false-color diagnostic image stack"
    )
    args, mipmap_level, volume = parse_args(parser)
    make_diag_stack(args.xml_path,
                    args.output_pattern,
                    mipmap_level=mipmap_level,
                    volume=volume,
                    silent=args.silent,
                    compression=args.compression,
                    cores=args.cpus)


if __name__ == "__main__":
    import os

    if os.environ.get("TSV_DIAG", False):
        diag()
    else:
        main()
