import os
import subprocess
import sys
from argparse import RawDescriptionHelpFormatter, ArgumentParser, Namespace, BooleanOptionalAction
from multiprocessing import freeze_support, Queue
from pathlib import Path
from platform import uname
from re import compile
from time import time

import psutil
from cpufeature.extension import CPUFeature
from tqdm import tqdm
from tifffile import natural_sorted

from parallel_image_processor import parallel_image_processor
from process_images import get_imaris_command, MultiProcessCommandRunner, commands_progress_manger
from pystripe.core import process_img, imread_tif_raw_png
from supplements.cli_interface import PrintColors


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

    dir_fnt = Path(args.fnt)
    if args.fnt and not dir_fnt.exists():
        dir_fnt.mkdir(exist_ok=True, parents=True)

    return_code = 0

    down_sample = None
    if args.downsample_y > 0 or args.downsample_x > 0:
        down_sample = (
            args.downsample_y if args.downsample_y else 1,
            args.downsample_x if args.downsample_x else 1,
        )
    if args.downsample_method.lower() not in ('min', 'max', 'mean', 'median'):
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
            input_path.is_file() and input_path.suffix.lower() in (".ims", ".xml")
    ) or (
            input_path.is_dir() and (
            args.dark > 0 or args.convert_to_8bit or
            new_size or down_sample or args.voxel_size_target or
            args.rotation or args.flip_upside_down or
            args.gaussian or args.background_subtraction or args.de_stripe or args.bleach_correction)
    ):
        if not args.tif:
            print(f"{PrintColors.FAIL}tif path is needed to continue.{PrintColors.ENDC}")
            raise RuntimeError

        need_pre_processing = False
        if args.dark > 0 or args.convert_to_8bit or down_sample or args.flip_upside_down or args.gaussian or \
                args.background_subtraction or args.de_stripe or args.bleach_correction:
            need_pre_processing = True

        de_striping_sigma = (0, 0)
        if args.de_stripe:
            de_striping_sigma = (250, 250)
        if args.bleach_correction:
            de_striping_sigma = (4000, 4000)

        return_code = parallel_image_processor(
            source=input_path,
            destination=tif_2d_folder,
            fun=process_img if need_pre_processing else None,
            kwargs={
                "gaussian_filter_2d": args.gaussian,
                "down_sample": down_sample,
                "down_sample_method": args.downsample_method,
                "new_size": new_size,
                "sigma": de_striping_sigma,
                "padding_mode": args.padding_mode,
                "bidirectional": True if args.bleach_correction else False,
                "dark": args.dark,
                "lightsheet": args.background_subtraction,
                "bleach_correction_frequency": 1 / args.bleach_correction_period if args.bleach_correction else None,
                "bleach_correction_max_method": False,
                "bleach_correction_clip_min": args.bleach_correction_clip_min,
                "bleach_correction_clip_max": args.bleach_correction_clip_max,
                "exclude_dark_edges_set_them_to_zero": False,
                "rotate": 0,
                "flip_upside_down": args.flip_upside_down,
                "convert_to_16bit": args.convert_to_16bit,
                "convert_to_8bit": args.convert_to_8bit,
                "bit_shift_to_right": args.bit_shift
            } if args.channel >= 0 else None,
            rename=args.rename,
            max_processors=args.nthreads,
            channel=args.channel,
            compression=compression,
            timeout=args.timeout,
            source_voxel=(args.voxel_size_z, args.voxel_size_y, args.voxel_size_x),
            target_voxel=args.voxel_size_target,
            rotation=args.rotation,
            resume=args.resume,
            needed_memory=args.needed_memory * 1024 ** 3,
            save_images=args.save_images
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
            f"--clist={args.channel}",
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

    progressbar_position = 0
    progress_bars = []
    if args.fnt:
        file_txt_sorted_tif_list = tif_2d_folder / "files.txt"
        files = list(tif_2d_folder.glob("*.tif"))
        with open(file_txt_sorted_tif_list, "w") as f:
            f.write("\n".join(natural_sorted(list(map(str, files)))))
        command = [
            f"{fnt_slice2cube}",
            f"-i {file_txt_sorted_tif_list}",
            f"-o {args.fnt}",
            f"-j{args.nthreads}",
            f"-r {args.voxel_size_x}:{args.voxel_size_y}:{args.voxel_size_z}",
            "-Y"
        ]
        command = " ".join(command)
        print(command)
        MultiProcessCommandRunner(progress_queue, command,
                                  pattern=r"(\d+)(?=\)\s*finished)\.\s*$",
                                  position=progressbar_position,
                                  percent_conversion=100 / len(files)).start()
        progress_bars += [tqdm(total=100, ascii=True, position=progressbar_position, unit=" %", smoothing=0.01,
                              desc=f"FNT")]
        progressbar_position += 1

        # if args.imaris:
        #     MultiProcessCommandRunner(progress_queue, command,
        #                               pattern=r"(WriteProgress:)\s+(\d*.\d+)\s*$",
        #                               position=progressbar_position).start()
        # else:
        #     start_time = time()
        #     subprocess.call(command, shell=True)
        #     print(f"elapsed time = {round((time() - start_time) / 60, 1)}")

    is_renamed: bool = False
    if args.imaris:
        ims_file = Path(args.imaris)
        files = list(tif_2d_folder.glob("*.tif"))
        files += list(tif_2d_folder.glob("*.tiff"))
        assert len(files) > 0

        if compile(r"^\d+$").findall(files[0].name[:-len(files[0].suffix)]):
            files = [file.rename(file.parent / ("_" + file.name)) for file in files]
            is_renamed = True

        command = get_imaris_command(
            imaris_path=imaris_converter, input_path=tif_2d_folder, output_path=ims_file,
            voxel_size_x=args.voxel_size_x,
            voxel_size_y=args.voxel_size_y,
            voxel_size_z=args.voxel_size_z,
            workers=args.nthreads,
            dtype=imread_tif_raw_png(files[0]).dtype
        )
        print(f"\t{PrintColors.BLUE}tiff to ims conversion command:{PrintColors.ENDC}\n\t\t{command}\n")

        MultiProcessCommandRunner(progress_queue, command,
                                  pattern=r"(WriteProgress:)\s+(\d*.\d+)\s*$", position=progressbar_position).start()
        progress_bars += [tqdm(total=100, ascii=True, position=progressbar_position, unit=" %", smoothing=0.01,
                               desc=f"imaris")]
        progressbar_position += 1

    if progressbar_position > 0:
        commands_progress_manger(progress_queue, progress_bars, running_processes=2 if args.teraFly else 1)
        if is_renamed:
            [file.rename(file.parent / file.name[1:]) for file in files]

    if args.movie:
        movie_file = Path(args.movie)
        duration = args.movie_frame_duration

        with open(tif_2d_folder / "ffmpeg_input.txt", "wb") as ffmpeg_input:
            for file in sorted(tif_2d_folder.glob("*.tif"))[args.movie_start:args.movie_end]:
                ffmpeg_input.write(f"file '{file.absolute().as_posix()}'\n".encode())
                ffmpeg_input.write(f"duration {duration}\n".encode())

        command = " ".join([
            f"ffmpeg -r 60 -f concat -safe 0",
            f"-i {tif_2d_folder / 'ffmpeg_input.txt'}",
            f"{movie_file.absolute()}"
        ])
        print(f"{PrintColors.BLUE}{command}{PrintColors.ENDC}")

        pipe = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout
        pipe.read().decode()
        pipe.close()


if __name__ == '__main__':
    freeze_support()
    os.environ["GLIBC_TUNABLES"] = "glibc.malloc.hugetlb=2"
    os.environ["NUMPY_MADVISE_HUGEPAGE"] = "1"
    cpu_instruction = "SSE2"
    for item in ["SSE2", "AVX", "AVX2", "AVX512f"]:
        cpu_instruction = item if CPUFeature[item] else cpu_instruction
    PyScriptsPath = Path(r".") / "TeraStitcher" / "pyscripts"
    if sys.platform == "win32":
        # print("Windows is detected.")
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        psutil.Process().nice(getattr(psutil, "IDLE_PRIORITY_CLASS"))
        TeraStitcherPath = Path(r"TeraStitcher") / "Windows" / cpu_instruction
        os.environ["PATH"] = f"{os.environ['PATH']};{TeraStitcherPath.as_posix()}"
        os.environ["PATH"] = f"{os.environ['PATH']};{PyScriptsPath.as_posix()}"
        terastitcher = "terastitcher.exe"
        teraconverter = "teraconverter.exe"
        fnt_slice2cube = Path(r".") / "fnt" / "Windows" / "fnt-slice2cube.exe"
    elif sys.platform == 'linux' and 'microsoft' not in uname().release.lower():
        print("Linux is detected.")
        os.environ["NUMPY_MADVISE_HUGEPAGE"] = "1"
        psutil.Process().nice(value=19)
        TeraStitcherPath = Path(r".") / "TeraStitcher" / "Linux" / cpu_instruction
        os.environ["PATH"] = f"{os.environ['PATH']}:{TeraStitcherPath.as_posix()}"
        os.environ["PATH"] = f"{os.environ['PATH']}:{PyScriptsPath.as_posix()}"
        terastitcher = "terastitcher"
        teraconverter = "teraconverter"
        os.environ["TERM"] = "xterm"
        fnt_slice2cube = Path(r".") / "fnt" / "Linux" / "fnt-slice2cube"
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
                        help="path to input image. Input path can be: a tera-stitcher stage 5 volume (xml) file"
                             ", Imaris (ims) image, or a 2D tif[f]?, png, raw series.")
    parser.add_argument("--tif", "-t", type=str, required=False, default='',
                        help="path to tif output files.")
    parser.add_argument("--teraFly", "-f", type=str, required=False, default='',
                        help="path to output teraFly files.")
    parser.add_argument("--fnt", "-fnt", type=str, required=False, default='',
                        help="path to output Fast Neurite Tracer files.")
    parser.add_argument("--imaris", "-o", type=str, required=False, default='',
                        help="path to imaris output file.")
    parser.add_argument("--movie", "-m", type=str, required=False, default='',
                        help="path to mp4 output file")
    parser.add_argument("--nthreads", "-n", type=int, default=psutil.cpu_count(logical=False),
                        help="number of threads. default is all physical cores for tif conversion and 12 for TeraFly.")
    parser.add_argument("--channel", "-c", type=int, default=0,
                        help="channel to be converted. Default is 0. "
                             "negative vales mean RGB image. Most operations do not work on RGB. "
                             "Only compression is tested for RGB.")
    parser.add_argument("--gaussian", "-g", default=False, action=BooleanOptionalAction,
                        help="image pre-processing: apply Gaussian filter to denoise. Default is --no-gaussian.")
    parser.add_argument("--de_stripe", default=False, action=BooleanOptionalAction,
                        help="image pre-processing: apply de-striping algorithm. Default is --no-de_stripe")
    parser.add_argument("--padding_mode", "-w", type=str, default='reflect',
                        help="Padding method affects the edge artifacts during de-stripping and bleach correction. "
                             "The default mode is reflect, but in some cases wrap method works better. "
                             "Options: constant, edge, linear_ramp, maximum, mean, median, minimum, reflect, "
                             "symmetric, wrap, and empty")
    parser.add_argument("--downsample_x", "-dsx", type=int, default=0,
                        help="image pre-processing: 2D down-sampling factor for x-axis. Default is 0.")
    parser.add_argument("--downsample_y", "-dsy", type=int, default=0,
                        help="image pre-processing: 2D down-sampling factor for y-axis. Default is 0.")
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
    parser.add_argument("--bleach_correction_period", type=float, default=2000,
                        help="inverse of low-pass filter frequency used for bleach correction. "
                             "Try camera tile size first. Default is 2000.")
    parser.add_argument("--bleach_correction_clip_min", type=float, default=20,
                        help="foreground vs background threshold. Default is 20.")
    parser.add_argument("--bleach_correction_clip_max", type=float, default=255,
                        help="max of the image without outliers. Default is 255.")
    parser.add_argument("--convert_to_16bit", default=False, action=BooleanOptionalAction,
                        help="Image pre-processing: convert to 16-bit. Default is --no-convert_to_16bit")
    parser.add_argument("--convert_to_8bit", default=False, action=BooleanOptionalAction,
                        help="Image pre-processing: convert to 8-bit. Default is --no-convert_to_8bit")
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
    parser.add_argument("--compression_level", "-zl", type=int, default=1,
                        help="image pre-processing: compression level for tif files. Default is 1.")
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
                        help="target voxel size in µm for 3D down-sampling.")
    parser.add_argument("--timeout", type=float, default=None,
                        help="timeout in seconds for image reading. applies to image series and tsv volumes (not ims). "
                             "adds up to 30 percent overhead for copying the data from one process to another.")
    parser.add_argument("--rename", default=False, action=BooleanOptionalAction,
                        help="applies to tif to tif conversion only. "
                             "sorts input files and renumbers them like img_000000.tif. Default is --no-rename. ")
    parser.add_argument("--resume", default=False, action=BooleanOptionalAction,
                        help="applies to tif conversion only. "
                             "resume processing remaining files. Default is --no-resume.")
    parser.add_argument("--needed_memory", type=int, default=1,
                        help="Memory needed per thread in GB. Default is 1 GB.")
    parser.add_argument("--save_images", default=True, action=BooleanOptionalAction,
                        help="save the processed images. Default is --save_images. "
                             "if you just need to do downsampling use --no-save_images.")
    main(parser.parse_args())
