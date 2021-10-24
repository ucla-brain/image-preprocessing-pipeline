# renumber_directories.py
#
#     The Smart Spim can create directory names with negative values which
#     are not handled well by Terastitcher. This script renumbers directories
#     so that they are always positive, adding an offset to accomplish that.
#
import argparse
import os
import sys


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        help="Path to the root of the smartspim output")
    return parser.parse_args(args)


def main(args=sys.argv[1:]):
    args = parse_args(args)
    path = args.path
    directories = []
    min_x = 0
    min_y = 0
    all_x = set()
    for root, dirnames, filenames in os.walk(path):
        try:
            int(os.path.split(root)[-1])
        except:
            continue
        for dirname in dirnames:
            if "_" in dirname:
                try:
                    x, y = [int(_) for _ in dirname.split("_")]
                except TypeError:
                    continue
                all_x.add(x)
                directories.append((x, y))
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
    if min_x == 0 and min_y == 0:
        return
    for x, y in directories:
        xnew = x - min_x
        ynew = y - min_y
        src = os.path.join(path, "%06d" % x, "%06d_%06d" % (x, y))
        dest = os.path.join(path, "%06d" % x, "%06d_%06d" % (xnew, ynew))
        os.rename(src, dest)
    if min_x < 0:
        for x in all_x:
            xnew = x - min_x
            src = os.path.join(path, "%06d" % x)
            dest = os.path.join(path, "%06d" % xnew)
            os.rename(src, dest)


if __name__=="__main__":
    main()
