"""simple.py - Simple mode for stitching a stack"""

import argparse
import multiprocessing
from tsv.volume import TSVSimpleVolume, VExtent
from tsv.convert import convert_to_2D_tif
import sys


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        required=True,
                        help="Path to the root directory of the stack")
    parser.add_argument("--voxel-size-xy",
                        type=float,
                        help="The pixel size in the X and Y direction in "
                        "microns")
    parser.add_argument("--voxel-size-x",
                        type=float,
                        help="The pixel size in the X direction in microns")
    parser.add_argument("--voxel-size-y",
                        type=float,
                        help="The pixel size in the Y direction in microns")
    parser.add_argument("--voxel-size-z",
                        type=float,
                        default=1,
                        help="The pixel size in the Z direction in microns")
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

    return parser.parse_args(args)


err_msg = """Error: you must either specify --voxel-size-xy or
if --voxel-size-xy is not specified, both --voxel-size-x and --voxel-size-y"""


def main(args=sys.argv[1:]):
    args = parse_args(args)
    if args.voxel_size_xy is not None:
        if args.voxel_size_x is not None or \
           args.voxel_size_y is not None:
            print(err_msg, file=sys.stderr)
            exit(-1)
        voxel_size_x = voxel_size_y = args.voxel_size_xy
    elif args.voxel_size_x is not None and args.voxel_size_y is not None:
        voxel_size_x = args.voxel_size_x
        voxel_size_y = args.voxel_size_y
    else:
        print(err_msg, file=sys.stderr)
        exit(-1)

    if args.mipmap_level == 0:
        mipmap_level = None
    else:
        mipmap_level = args.mipmap_level
    if args.volume != "":
        x0, x1, y0, y1, z0, z1 = map(int, args.volume.split(","))
        volume = VExtent(x0, x1, y0, y1, z0, z1)
    else:
        volume = None

    v = TSVSimpleVolume(args.path, voxel_size_x, voxel_size_y,
                        args.voxel_size_z)
    convert_to_2D_tif(v,
                      args.output_pattern,
                      mipmap_level=mipmap_level,
                      volume=volume,
                      silent=args.silent,
                      compression=args.compression,
                      cores=args.cpus,
                      ignore_z_offsets=True)


if __name__=="__main__":
    main()
