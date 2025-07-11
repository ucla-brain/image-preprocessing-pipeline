import os
import sys
import psutil
from pathlib import Path
from multiprocessing import freeze_support
from platform import uname
from process_images import merge_all_channels
from argparse import RawDescriptionHelpFormatter, ArgumentParser, BooleanOptionalAction
from supplements.cli_interface import PrintColors


if __name__ == '__main__':
    freeze_support()
    if sys.platform == "win32":
        fnt_cube2video = Path(r".") / "fnt" / "Windows" / "fnt-cube2video.exe"
        psutil.Process().nice(getattr(psutil, "IDLE_PRIORITY_CLASS"))
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
    elif sys.platform == 'linux' and 'microsoft' not in uname().release.lower():
        fnt_cube2video = Path(r".") / "fnt" / "Linux" / "fnt-cube2video"
        psutil.Process().nice(value=19)
        os.environ["NUMPY_MADVISE_HUGEPAGE"] = "1"
        os.environ["TERM"] = "xterm"
    else:
        print("yet unsupported OS")
        raise RuntimeError
    parser = ArgumentParser(
        description="Process FNT cubes in parallel\n\n",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="Developed 2023 by Keivan Moradi at UCLA, Hongwei Dong Lab (B.R.A.I.N.) \n"
    )
    parser.add_argument("--cyan", "-c", type=str, required=False, default="",
                        help="Cyan channel path.")
    parser.add_argument("--magenta", "-m", type=str, required=False, default="",
                        help="Magenta channel path.")
    parser.add_argument("--yellow", "-y", type=str, required=False, default="",
                        help="Yellow channel path.")
    parser.add_argument("--black", "-k", type=str, required=False, default="",
                        help="key (black) channel path.")
    parser.add_argument("--output_path", "-o", type=str, required=True,
                        help="Output path for the merged data.")
    parser.add_argument("--num_processes", "-n", type=int, required=False,
                        default=psutil.cpu_count(logical=False) + 4,
                        help="Number of CPU cores.")
    parser.add_argument("--resume", default=True, action=BooleanOptionalAction,
                        help="Resume processing remaining images. In order to disable use --no_resume.")
    parser.add_argument("--convert_to_8bit", default=False, action=BooleanOptionalAction,
                        help="Image pre-processing: convert to 8-bit. Default is --no-convert_to_8bit")
    parser.add_argument("--bit_shift", type=int, default=8,
                        help="bit_shift for 8-bit conversion. An number between 0 and 8. "
                             "Smaller values make images brighter compared with he original image. "
                             "Default is 8 (no change in brightness).\n"
                             "0 any value larger than   255 will be set to 255 in 8 bit, values smaller than 255 "
                             "will not change.\n"
                             "1 any value larger than   511 will be set to 255 in 8 bit, 0-  1 will be set to 0,"
                             "   2-  3 to 1.\n"
                             "2 any value larger than  1023 will be set to 255 in 8 bit, 0-  3 will be set to 0,"
                             "   4-  7 to 1.\n"
                             "3 any value larger than  2047 will be set to 255 in 8 bit, 0-  7 will be set to 0,"
                             "   8- 15 to 1.\n"
                             "4 any value larger than  4095 will be set to 255 in 8 bit, 0- 15 will be set to 0,"
                             "  16- 31 to 1.\n"
                             "5 any value larger than  8191 will be set to 255 in 8 bit, 0- 31 will be set to 0,"
                             "  32- 63 to 1.\n"
                             "6 any value larger than 16383 will be set to 255 in 8 bit, 0- 63 will be set to 0,"
                             "  64-127 to 1.\n"
                             "7 any value larger than 32767 will be set to 255 in 8 bit, 0-127 will be set to 0,"
                             " 128-255 to 1.\n"
                             "8 any value larger than 65535 will be set to 255 in 8 bit, 0-255 will be set to 0,"
                             " 256-511 to 1.")
    args = parser.parse_args()
    img_paths = []
    order_of_colors = ""
    if args.cyan:
        order_of_colors += "c"
        img_paths += [Path(args.cyan)]
    if args.magenta:
        order_of_colors += "m"
        img_paths += [Path(args.magenta)]
    if args.yellow:
        order_of_colors += "y"
        img_paths += [Path(args.yellow)]
    if args.black:
        order_of_colors += "k"
        img_paths += [Path(args.black)]
    for color in "cmyk":
        if color not in order_of_colors:
            order_of_colors += color
    if img_paths:
        merge_all_channels(
            tif_paths=img_paths,
            z_offsets=[0,] * (len(img_paths) - 1),
            merged_tif_path=Path(args.output_path),
            order_of_colors=order_of_colors,
            resume=args.resume,
            right_bit_shifts=None
        )
    else:
        print(f"{PrintColors.FAIL}at least one of --red or --green or --blue is required.{PrintColors.ENDC}")
        raise RuntimeError

