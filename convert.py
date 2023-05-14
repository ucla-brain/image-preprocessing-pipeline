import os
import sys
import psutil
import subprocess
from pystripe.core import process_img
from time import time
from platform import uname
from multiprocessing import freeze_support, Queue
from numpy import max as np_max
from numpy import min as np_min
from numpy import mean as np_mean
from numpy import median as np_median
from pathlib import Path
from process_images import get_imaris_command, MultiProcessCommandRunner, commands_progress_manger
from argparse import RawDescriptionHelpFormatter, ArgumentParser, Namespace, BooleanOptionalAction
from parallel_image_processor import parallel_image_processor
from supplements.cli_interface import PrintColors
from cpufeature.extension import CPUFeature
from tqdm import tqdm
from re import compile

# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['NUMEXPR_NUM_THREADS'] = '1'
# os.environ['OMP_NUM_THREADS'] = '1'


def main(args: Namespace):
    input_path = Path(args.input)
    if not args.input or not input_path.exists():
        print(f"{PrintColors.FAIL}Input path is not valid.{PrintColors.ENDC}")
        raise RuntimeError

    tif_2d_folder = Path(args.tif)
    if args.tif and not tif_2d_folder.exists():
        tif_2d_folder.mkdir(exist_ok=True, parents=True)

    dir_tera_fly = Path(args.teraFly)
    if args.teraFly and not dir_tera_fly.exists():
        dir_tera_fly.mkdir(exist_ok=True, parents=True)

    return_code = 0

    down_sample = None
    if args.downsample_y > 0 or args.downsample_x > 0:
        down_sample = (
            args.downsample_y if args.downsample_y else 1,
            args.downsample_x if args.downsample_x else 1,
        )
    if args.downsample_method.lower() == 'min':
        downsample_method = np_min
    elif args.downsample_method.lower() == 'max':
        downsample_method = np_max
    elif args.downsample_method.lower() == 'mean':
        downsample_method = np_mean
    elif args.downsample_method.lower() == 'median':
        downsample_method = np_median
    else:
        print(f"{PrintColors.FAIL}unsupported down-sampling method: {args.downsample_method}{PrintColors.ENDC}")
        raise RuntimeError

    new_size = None
    if args.new_size_y and args.new_size_x:
        new_size = (args.new_size_y, args.new_size_x)
    elif args.new_size_y or args.new_size_x:
        print(f"{PrintColors.FAIL}both new_size_x and new_size_y are needed!{PrintColors.ENDC}")
        raise RuntimeError
    compression = (args.compression_method, args.compression_level) if args.compression_level > 0 else None

    if (
            input_path.is_file() and input_path.suffix.lower() == ".ims"
    ) or (
            input_path.is_dir() and (
            args.dark > 0 or args.convert_to_8bit or new_size or down_sample or args.rotation or args.gaussian or
            args.background_subtraction or args.de_stripe or args.flip_upside_down or args.bleach_correction or
            args.compression_level > 0)
    ):
        if not args.tif:
            print(f"{PrintColors.FAIL}tif path is needed to continue.{PrintColors.ENDC}")
            raise RuntimeError

        return_code = parallel_image_processor(
            source=input_path,
            destination=tif_2d_folder,
            fun=process_img,
            kwargs={
                "gaussian_filter_2d": args.gaussian,
                "down_sample": down_sample,
                "downsample_method": downsample_method,
                "new_size": new_size,
                "sigma": (200, 200) if args.de_stripe else (0, 0),
                "dark": args.dark,
                "lightsheet": args.background_subtraction,
                "bleach_correction_frequency": 0.0005 if args.bleach_correction else None,
                "rotate": 0,
                "flip_upside_down": args.flip_upside_down,
                "convert_to_8bit": args.convert_to_8bit,
                "bit_shift_to_right": args.bit_shift
            },
            max_processors=args.nthreads,
            channel=args.channel,
            compression=compression,
            timeout=args.timeout,
            source_voxel=(args.voxel_size_z, args.voxel_size_y, args.voxel_size_x),
            target_voxel=args.voxel_size_target,
            rotation=args.rotation,
        )
    elif input_path.is_dir():
        tif_2d_folder = input_path

    assert return_code == 0
    progress_queue = Queue()
    if args.teraFly:
        command = [
            f"mpiexec -np {min(12, args.nthreads)} python -m mpi4py {paraconverter}",
            "--sfmt=\"TIFF (series, 2D)\"",
            "--dfmt=\"TIFF (tiled, 3D)\"",
            "--resolutions=\"012345\"",
            "--clist=0",
            "--halve=mean",
            # "--noprogressbar",
            # "--sparse_data",
            # "--fixed_tiling",
            # "--height=256",
            # "--width=256",
            # "--depth=256",
            f"-s={tif_2d_folder}",  # destination_folder
            f"-d={dir_tera_fly}",
        ]
        command = " ".join(command)

        if args.imaris:
            MultiProcessCommandRunner(progress_queue, command).start()
        else:
            start_time = time()
            subprocess.call(command, shell=True)
            print(f"elapsed time = {round((time() - start_time) / 60, 1)}")

    if args.imaris:
        ims_file = Path(args.imaris)
        files = list(tif_2d_folder.glob("*.tif"))
        files += list(tif_2d_folder.glob("*.tiff"))
        is_renamed: bool = False
        if compile(r"^\d+$").findall(files[0].name[:-len(files[0].suffix)]):
            files = [file.rename(file.parent / ("_" + file.name)) for file in files]
            is_renamed = True

        command = get_imaris_command(
            imaris_path=imaris_converter, input_path=tif_2d_folder, output_path=ims_file,
            voxel_size_x=args.voxel_size_x,
            voxel_size_y=args.voxel_size_y,
            voxel_size_z=args.voxel_size_z,
            workers=args.nthreads
        )
        print(f"\t{PrintColors.BLUE}tiff to ims conversion command:{PrintColors.ENDC}\n\t\t{command}\n")

        MultiProcessCommandRunner(progress_queue, command,
                                  pattern=r"(WriteProgress:)\s+(\d*.\d+)\s*$", position=0).start()
        progress_bars = [tqdm(total=100, ascii=True, position=0, unit=" %", smoothing=0.01, desc=f"imaris")]
        commands_progress_manger(progress_queue, progress_bars, running_processes=2 if args.teraFly else 1)

        if is_renamed:
            [file.rename(file.parent / file.name[1:]) for file in files]

    if args.movie:
        movie_file = Path(args.movie)
        duration = args.movie_frame_duration

        with open(tif_2d_folder/"ffmpeg_input.txt", "wb") as ffmpeg_input:
            for file in sorted(tif_2d_folder.glob("*.tif"))[args.movie_start:args.movie_end]:
                ffmpeg_input.write(f"file '{file.absolute().as_posix()}'\n".encode())
                ffmpeg_input.write(f"duration {duration}\n".encode())

        command = " ".join([
            f"ffmpeg -r 60 -f concat -safe 0",
            f"-i {tif_2d_folder/'ffmpeg_input.txt'}",
            f"{movie_file.absolute()}"
        ])
        print(f"{PrintColors.BLUE}{command}{PrintColors.ENDC}")

        pipe = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout
        pipe.read().decode()
        pipe.close()


if __name__ == '__main__':
    freeze_support()
    cpu_instruction = "SSE2"
    for item in ["SSE2", "AVX", "AVX2", "AVX512f"]:
        cpu_instruction = item if CPUFeature[item] else cpu_instruction
    PyScriptsPath = Path(r".") / "TeraStitcher" / "pyscripts"
    if sys.platform == "win32":
        # print("Windows is detected.")
        psutil.Process().nice(getattr(psutil, "IDLE_PRIORITY_CLASS"))
        TeraStitcherPath = Path(r"TeraStitcher") / "Windows" / cpu_instruction
        os.environ["PATH"] = f"{os.environ['PATH']};{TeraStitcherPath.as_posix()}"
        os.environ["PATH"] = f"{os.environ['PATH']};{PyScriptsPath.as_posix()}"
        terastitcher = "terastitcher.exe"
        teraconverter = "teraconverter.exe"
    elif sys.platform == 'linux' and 'microsoft' not in uname().release.lower():
        print("Linux is detected.")
        psutil.Process().nice(value=19)
        TeraStitcherPath = Path(r".") / "TeraStitcher" / "Linux" / cpu_instruction
        os.environ["PATH"] = f"{os.environ['PATH']}:{TeraStitcherPath.as_posix()}"
        os.environ["PATH"] = f"{os.environ['PATH']}:{PyScriptsPath.as_posix()}"
        terastitcher = "terastitcher"
        teraconverter = "teraconverter"
        os.environ["TERM"] = "xterm"
    else:
        print("yet unsupported OS")
        raise RuntimeError

    paraconverter = PyScriptsPath / "paraconverter.py"
    if not paraconverter.exists():
        print(paraconverter)
        print("Error: paraconverter not found")
        raise RuntimeError

    imaris_converter = Path(r"imaris") / "ImarisConvertiv.exe"
    if not imaris_converter.exists():
        print("Error: ImarisConvertiv.exe not found")
        raise RuntimeError

    parser = ArgumentParser(
        description="Imaris to tif and TeraFly converter (version 0.1.0)\n\n",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="Developed 2022 by Keivan Moradi, Hongwei Dong Lab (B.R.A.I.N) at UCLA\n"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to input image or path.")
    parser.add_argument("--tif", "-t", type=str, required=False, default='',
                        help="Path to tif output path.")
    parser.add_argument("--teraFly", "-f", type=str, required=False, default='',
                        help="Path to output teraFly path.")
    parser.add_argument("--imaris", "-o", type=str, required=False, default='',
                        help="Path to 8-bit imaris output file.")
    parser.add_argument("--movie", "-m", type=str, required=False, default='',
                        help="Path to mp4 output file")
    parser.add_argument("--nthreads", "-n", type=int, default=psutil.cpu_count(logical=False),
                        help="number of threads. default is all physical cores for tif conversion and 12 for terafly.")
    parser.add_argument("--channel", "-c", type=int, default=0,
                        help="channel to be converted. Default is 0.")
    parser.add_argument("--gaussian", "-g", default=False, action=BooleanOptionalAction,
                        help="image pre-processing: apply Gaussian filter to denoise. Default is --no-gaussian.")
    parser.add_argument("--de_stripe", default=False, action=BooleanOptionalAction,
                        help="image pre-processing: apply de-striping algorithm. Default is --no-de_stripe")
    parser.add_argument("--downsample_x", "-dsx", type=int, default=0,
                        help="image pre-processing: down-sampling factor for x-axis. Default is 0.")
    parser.add_argument("--downsample_y", "-dsy", type=int, default=0,
                        help="image pre-processing: down-sampling factor for y-axis. Default is 0.")
    parser.add_argument("--downsample_method", "-dsm", type=str, default='max',
                        help="image pre-processing: down-sampling method. "
                             "options are max, min, mean, median. Default is max.")
    parser.add_argument("--new_size_x", "-nsx", type=int, default=0,
                        help="image pre-processing: new size of x-axis. Default is 0.")
    parser.add_argument("--new_size_y", "-nsy", type=int, default=0,
                        help="image pre-processing: new size of y-axis. Default is 0.")
    parser.add_argument("--dark", "-d", type=int, default=0,
                        help="image pre-processing: background vs foreground threshold. Default is 0.")
    parser.add_argument("--background_subtraction", default=False, action=BooleanOptionalAction,
                        help="image pre-processing: apply lightsheet cleaning algorithm. "
                             "Default is --no-background_subtraction")
    parser.add_argument("--bleach_correction", default=False, action=BooleanOptionalAction,
                        help="image pre-processing: correct image bleaching. Default is --no-bleach_correction.")
    parser.add_argument("--convert_to_8bit", default=False, action=BooleanOptionalAction,
                        help="image pre-processing: convert to 8-bit. Default is --no-convert_to_8bit")
    parser.add_argument("--bit_shift", "-b", type=int, default=8,
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
    parser.add_argument("--rotation", "-r", type=int, default=0,
                        help="image pre-processing: rotate the image. "
                             "one of 0, 90, 180 or 270 degree values are accepted. Default is 0.")
    parser.add_argument("--flip_upside_down", default=False, action=BooleanOptionalAction,
                        help="image pre-processing: flip the y-axis. Default is --no-flip_upside_down")
    parser.add_argument("--compression_level", "-zl", type=int, default=0,
                        help="image pre-processing: compression level for tif files. Default is 0.")
    parser.add_argument("--compression_method", "-zm", type=str, default="ADOBE_DEFLATE",
                        help="image pre-processing: compression method for tif files. Default is ADOBE_DEFLATE. "
                             "LZW and PackBits are also supported.")
    parser.add_argument("--movie_start", type=int, default=0,
                        help="start frame counting from 0. Default is 0.")
    parser.add_argument("--movie_end", type=int, default=None,
                        help="end frame counting from 0. Default is the last frame.")
    parser.add_argument("--movie_frame_duration", type=int, default=5,
                        help="duration of each frame. should be a positive integer. Default is 5.")
    parser.add_argument("--voxel_size_x", "-dx", type=float, default=1,
                        help="x voxel size in µm. Default is 1.")
    parser.add_argument("--voxel_size_y", "-dy", type=float, default=1,
                        help="y voxel size in µm. Default is 1.")
    parser.add_argument("--voxel_size_z", "-dz", type=float, default=1,
                        help="z voxel size in µm. Default is 1.")
    parser.add_argument("--voxel_size_target", "-dt", type=float, default=None,
                        help="target voxel size in µm for down-sampling.")
    parser.add_argument("--timeout", type=float, default=None,
                        help="timeout in seconds for image reading. applies to image series and tsv volumes (not ims). "
                             "adds 30% overhead for copying the data from one process to another.")

    main(parser.parse_args())
