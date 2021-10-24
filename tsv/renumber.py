"""renumber.py - a utility for renumbering the files in a stack hierarchy


This utility renames files so that they are alphabetically in order as well as
numerically by zero-padding the files to the left. The files should be in
the format, "nnnnn.tiff".

Usage:
tsv-renumber [--n-digits=<digits>] <path-to-hierarchy-root>
where:
    <digits> is the number of digits in the file name. Default is 6.
    <path-to-hierarchy-root> is the path to the root directory (2 folders down
    from where the files are).
"""

import argparse
import glob
import os
import sys
import tqdm


def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-digits", default=6, type=int,
                        help="The number of digits in the final file name.")
    parser.add_argument("root", metavar="path-to-hierarchy-root",
                        help="The root directory of the stack tree")
    args = parser.parse_args(args=args)
    pattern = "%0" + str(args.n_digits) + "d.tiff"
    for path in tqdm.tqdm(
            glob.glob(os.path.join(args.root, "*", "*", "*.tiff"))):
        folder, filename = os.path.split(path)
        idx = int(filename.split(".")[0])
        dest = os.path.join(folder, pattern % idx)
        os.rename(path, dest)

if __name__=="__main__":
    main(sys.argv[1:])


