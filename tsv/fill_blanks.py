import argparse
import io
import itertools
import numpy as np
import os
import sys
import tifffile
import tqdm
from .raw import raw_imread


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Fill missing stacks in microscope output.\n"
        "This program scans the source directory tree to come up with a "
        "complete list of X, Y and Z coordinates for the planes in the "
        "acquired stacks. It then fills in missing planes with tif files "
        "componsed of all zeros.")
    parser.add_argument("--src",
                        required=True,
                        help="Path to root of source tree, e.g. Ex_0_Em_0")
    parser.add_argument("--dest",
                        required=False,
                        help="Path to root of destriped destination for "
                        "blanks. Defaults to src + \"_destriped\".")
    parser.add_argument("--silent",
                        action="store_true",
                        help = "Don't display progress bar")
    return parser.parse_args(args)


def main(args=sys.argv[1:]):
    args = parse_args(args)
    if args.dest is None:
        dest = args.src + "_destriped"
    else:
        dest = args.dest
    src = args.src
    #
    # Discover all xs, ys, zs
    xs = set()
    ys = set()
    zs = set()
    img_shape = None
    img_dtype = None
    for dx in os.listdir(src):
        src_path_x = os.path.join(src, dx)
        if os.path.isdir(src_path_x):
            n_digits = len(dx)
            dest_path_x = os.path.join(dest, dx)
            if not os.path.exists(dest_path_x):
                os.mkdir(dest_path_x)
            try:
                x = int(dx)
            except ValueError:
                continue
            xs.add(x)
            for dy in os.listdir(src_path_x):
                src_path_y = os.path.join(src_path_x, dy)
                if os.path.isdir(src_path_y):
                    try:
                        x, y = [int(_) for _ in dy.split("_")]
                    except:
                        continue
                    ys.add(y)
                    for img_file in os.listdir(src_path_y):
                        try:
                            z = int(img_file.split(".")[0])
                        except ValueError:
                            continue
                        zs.add(z)
                        if img_shape is None:
                            z_format = "%%0%dd.tif" % len(img_file.split(".")[0])
                            img_path = os.path.join(src_path_y, img_file)
                            if img_file.endswith(".raw"):
                                img = raw_imread(img_path)
                            else:
                                img = tifffile.imread(img_path)
                            img_shape = img.shape
                            img_dtype = img.dtype
    x_format = "%%0%dd" % n_digits
    y_format = x_format + "_" + x_format
    for x, y in itertools.product(xs, ys):
        dest_path = os.path.join(dest, x_format % x, y_format % (x, y))
        if not os.path.exists(dest_path):
            os.mkdir(dest_path)
    blank_img = np.zeros(img_shape, img_dtype)
    blank_stream = io.BytesIO()
    tifffile.imsave(blank_stream, blank_img, compress=9)
    for x, y, z in tqdm.tqdm(itertools.product(xs, ys, zs),
                             disable=args.silent,
                             total=len(xs) * len(ys) * len(zs)):
        dest_path = os.path.join(dest, x_format % x, y_format % (x, y),
                                 z_format % z)
        if not os.path.exists(dest_path):
            blank_stream.seek(0)
            with open(dest_path, "wb") as fd:
                fd.write(blank_stream.getvalue())


if __name__ == "__main__":
    main()