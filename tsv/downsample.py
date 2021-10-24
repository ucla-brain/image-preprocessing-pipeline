import tifffile
from skimage.measure import block_reduce
import glob
import multiprocessing
import argparse
import os
import tqdm
import sys


def downsample(src, dest, factor=2, compress=4):
    img = tifffile.imread(src)
    downsampled = block_reduce(img, block_size=(factor, factor))
    tifffile.imsave(dest, downsampled.astype(img.dtype), compress=compress)


def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",
                        required=True,
                        help="Source glob expression")
    parser.add_argument("--dest",
                        required=True,
                        help="Destination directory")
    parser.add_argument("--downsample-factor",
                        type=int,
                        default=2)
    parser.add_argument("--n-cores",
                        type=int,
                        default=20)
    parser.add_argument("--compression",
                        type=int,
                        default=4)
    parser.add_argument("--silent",
                        action="store_true",
                        help="Perform operation without progress bar")
    args = parser.parse_args(args=args)
    src_files = sorted(glob.glob(args.src))
    if not os.path.isdir(args.dest):
        os.makedirs(args.dest)
    with multiprocessing.Pool(args.n_cores) as pool:
        futures = []
        for src_file in src_files:
            filename = os.path.split(src_file)[1]
            dest_file = os.path.join(args.dest, filename)
            futures.append(pool.apply_async(
                downsample,
                (src_file, dest_file, args.downsample_factor,
                 args.compression)))
        for future in tqdm.tqdm(futures, disable=args.silent):
            future.get()


if __name__ == "__main__":
    main()
