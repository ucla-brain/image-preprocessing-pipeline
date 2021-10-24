import argparse
import json
import logging
import multiprocessing
import numpy as np
import os
import pathlib
import sys
import tifffile
import tqdm

from .scan import Scanner, AverageDrift
from .volume import VExtent


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        help="The root of the input stack tree",
                        required=True)
    parser.add_argument("--output-pattern",
                        help="Output pattern for file names, e.g. "
                        "/path-to/img_%%04d.tiff",
                        required=True)
    parser.add_argument("--voxel-size",
                        help="Three, comma separated numbers for the voxel "
                             "size in microns: x,y,z",
                        default="1.8,1.8,2.0")
    parser.add_argument("--z-step",
                        help="The number of microns of a z stepper motor step",
                        default=300.0,
                        type=float)
    parser.add_argument("--piezo-distance",
                        help="The number of microns of travel for the piezo-"
                             "electric crystal. Default is 300.",
                        default=300.0,
                        type=float)
    parser.add_argument("--threshold",
                        help="The threshold for a good match between stacks, "
                             "a number between 0 and 1, default = .75",
                        default=.75,
                        type=float)
    parser.add_argument("--x-slop",
                        help="Maximum error in the X direction. This "
                             "constrains the search for a match in x to "
                             "the range, -x-slop to x-slop inclusive",
                        type=int,
                        default=30)
    parser.add_argument("--y-slop",
                        help="Maximum error in the Y direction. This "
                             "constrains the search for a match in y to "
                             "the range, -y-slop to y-slop inclusive",
                        type=int,
                        default=30)
    parser.add_argument("--z-slop",
                        help="Maximum error in the Z direction. This "
                             "constrains the search for a match in z to "
                             "the range, -z-slop to z-slop inclusive",
                        type=int,
                        default=6)
    parser.add_argument("--z-skip",
                        help="For x and y alignments, do one plane out of every z-skip planes. Defaults to doing "
                        "one plane per alignment.",
                        default="middle")
    parser.add_argument("--dark",
                        help="Voxels with values less than this will be "
                             "considered background. Tiles that are almost "
                             "all background will be aligned using average "
                             "values from other alignments.",
                        type=int,
                        default=200)
    parser.add_argument("--min-support",
                        help="Minimum number of examples accepted for a composite alignment",
                        type=int,
                        default=5)
    parser.add_argument("--n-cores",
                        help="The number of cores to use when calculating",
                        type=int,
                        default=os.cpu_count())
    parser.add_argument("--n-io-cores",
                        help="The number of cores to use when writing files.",
                        type=int,
                        default=min(os.cpu_count(), 12))
    parser.add_argument("--log-level",
                        help="The logging output level, one of \"DEBUG\", "
                             "\"INFO\", \"WARNING\" or \"CRITICAL\".",
                        default="WARNING")
    parser.add_argument("--compression",
                        help="The compression level (0 to 9)",
                        default=3,
                        type=int)
    parser.add_argument("--stack-offset-output",
                        help="If present, write a .json file containing "
                             "the calculated stack offsets and scores")
    parser.add_argument("--stack-offset-input",
                        help="If present, use the contents of this file "
                             "instead of calculating the stack offsets.")
    parser.add_argument("--stacks",
                        help="A JSON dictionary giving the final alignment of "
                             "the stacks")
    parser.add_argument("--loose-x",
                        help="Allow for loose, per-Y interpretation of "
                        "x-offsets",
                        action="store_true")
    return parser.parse_args(args)


scanner = None


def do_one_plane(z, path, compress, width, height):
    plane = scanner.imread(VExtent(0, width,
                                   0, height,
                                   z, z + 1), np.uint16)
    tifffile.imsave(path, plane.reshape(plane.shape[1], plane.shape[2]),
                    compress=compress)


def dump_round(fd):
    json.dump(dict(
        x=dict([(",".join([str(_) for _ in k]), scanner.alignments_x[k])
                for k in scanner.alignments_x]),
        y=dict([(",".join([str(_) for _ in k]), scanner.alignments_y[k])
                for k in scanner.alignments_y]),
        z=dict([(",".join([str(_) for _ in k]), scanner.alignments_z[k])
                for k in scanner.alignments_z])), fd, indent=2)


def load_round(fd):
    d = json.load(fd)
    scanner.alignments_x = \
        dict([(tuple(int(_) for _ in k.split(",")), d["x"][k])
              for k in d["x"]])
    scanner.alignments_y = \
        dict([(tuple(int(_) for _ in k.split(",")), d["y"][k])
              for k in d["y"]])
    scanner.alignments_z = \
        dict([(tuple(int(_) for _ in k.split(",")), d["z"][k])
              for k in d["z"]])



def main(args=sys.argv[1:]):
    global scanner
    opts = parse_args(args)
    logging.basicConfig(level=getattr(logging, opts.log_level))
    drift = AverageDrift(0, 0, 0, 0, 0, 0, 0, 0, 0)
    voxel_size = [float(_) for _ in opts.voxel_size.split(",")]
    z_skip = opts.z_skip if opts.z_skip == "middle" else int(opts.z_skip)
    scanner = Scanner(pathlib.Path(opts.input),
                      voxel_size=voxel_size,
                      z_skip=z_skip,
                      x_slop=opts.x_slop,
                      y_slop=opts.y_slop,
                      z_slop=opts.z_slop,
                      decimate=8,
                      dark=opts.dark,
                      drift=drift,
                      min_support=opts.min_support,
                      n_cores=opts.n_cores,
                      loose_x=opts.loose_x)
    if opts.stack_offset_input:
        with open(opts.stack_offset_input) as fd:
            load_round(fd)
    else:
        scanner.align_all_stacks()
    if opts.stack_offset_output:
        with open(opts.stack_offset_output, "w") as fd:
            dump_round(fd)
    scanner.calculate_next_round_parameters(threshold=opts.threshold)
    scanner.rebase_stacks()
    if opts.stacks:
        stacks = [stack.as_dict() for stack in scanner.stacks[0]]
        with open(opts.stacks, "w") as fd:
            json.dump(stacks, fd, indent=2)
    width = scanner.volume.x1
    height = scanner.volume.y1
    futures = []
    done_paths = set()
    logging.info("Writing output images")
    with multiprocessing.Pool(opts.n_io_cores) as pool:
        for z in range(scanner.volume.z0, scanner.volume.z1):
            path = opts.output_pattern % z
            path_dir = os.path.dirname(path)
            if path_dir not in done_paths and not os.path.exists(path_dir):
                os.mkdir(path_dir)
                done_paths.add(path_dir)
            futures.append(pool.apply_async(
                do_one_plane, (z, path, opts.compression, width, height)))
        for future in tqdm.tqdm(futures):
            future.get()

if __name__ == "__main__":
    main()