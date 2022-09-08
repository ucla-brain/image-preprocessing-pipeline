import os
import sys
import psutil
import subprocess
from pystripe.core import batch_filter
from time import time
from platform import uname
from multiprocessing import freeze_support, Queue
from numpy import ndarray, where
from pathlib import Path
from pystripe.core import convert_to_8bit_fun
from process_images import get_imaris_command, MultiProcessCommandRunner, commands_progress_manger
from argparse import RawDescriptionHelpFormatter, ArgumentParser, Namespace, BooleanOptionalAction
from parallel_image_processor import parallel_image_processor
from supplements.cli_interface import PrintColors
from cpufeature.extension import CPUFeature
from tqdm import tqdm

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'


def process_img(image: ndarray, dark: int = 120, bit_shift: int = 0):
    # image = filter_streaks(image, (128, 256), wavelet='db2')
    if dark > 0:
        image = where(image > dark, image - dark, 0)
    image = convert_to_8bit_fun(image, bit_shift_to_right=bit_shift)
    return image


def main(args: Namespace):
    input_path = Path(args.input)
    if not args.input or not input_path.exists():
        print(f"{PrintColors.FAIL}Input path is not valid.{PrintColors.ENDC}")
        raise RuntimeError

    tif_2d_folder = Path(args.tif)
    if not args.tif:
        tif_2d_folder.mkdir(exist_ok=True, parents=True)

    dir_tera_fly = Path(args.teraFly)
    if args.teraFly:
        dir_tera_fly.mkdir(exist_ok=True, parents=True)

    return_code = 0
    if input_path.is_file() and input_path.suffix.lower() == ".ims":
        if not args.tif:
            print(f"{PrintColors.FAIL} tif path is needed for ims to any format{PrintColors.ENDC}")
            raise RuntimeError
        return_code = parallel_image_processor(
            process_img,
            input_path,
            tif_2d_folder,
            (),
            {"dark": args.dark, "bit_shift": args.bit_shift},
            max_processors=args.nthreads,
            channel=args.channel
        )
    elif input_path.is_dir() and (args.dark > 0 or args.convert_to_8bit):
        if not args.tif:
            print(f"{PrintColors.FAIL} tif path is needed for processing 2D tif series{PrintColors.ENDC}")
            raise RuntimeError
        return_code = batch_filter(
            input_path,
            tif_2d_folder,
            workers=args.nthreads,
            # sigma=[foreground, background] Default is [0, 0], indicating no de-striping.
            sigma=(0, 0),
            # level=0,
            wavelet="db10",
            crossover=10,
            # threshold=-1,
            compression=('ZLIB', 0),  # ('ZSTD', 1) conda install imagecodecs
            flat=None,
            dark=args.dark,
            # z_step=voxel_size_z,  # z-step in micron. Only used for DCIMG files.
            # rotate=False,
            lightsheet=False,
            artifact_length=150,
            # percentile=0.25,
            # convert_to_16bit=False,  # defaults to False
            convert_to_8bit=args.convert_to_8bit,
            bit_shift_to_right=args.bit_shift,
            continue_process=True,
            dtype='uint16',
            tile_size=None,
            down_sample=None,
            new_size=None,
            timeout=None
        )
    elif input_path.is_dir():
        tif_2d_folder = input_path

    assert return_code == 0

    if args.teraFly:
        command = [
            f"mpiexec -np {args.nthreads} python -m mpi4py {paraconverter}",
            "--sfmt=\"TIFF (series, 2D)\"",
            "--dfmt=\"TIFF (tiled, 3D)\"",
            "--resolutions=\"012345\"",
            "--clist=0",
            "--halve=max",
            # "--noprogressbar",
            # "--sparse_data",
            # "--fixed_tiling",
            # "--height=256",
            # "--width=256",
            # "--depth=256",
            f"-s={tif_2d_folder}",  # destination_folder
            f"-d={dir_tera_fly}",
        ]
        start_time = time()
        subprocess.call(" ".join(command), shell=True)
        print(f"elapsed time = {round((time() - start_time) / 60, 1)}")

    if args.imaris:
        ims_file = Path(args.imaris)
        files = [file.rename(file.parent / ("_" + file.name)) for file in tif_2d_folder.glob("*.tif")]

        command = get_imaris_command(imaris_path=imaris_converter, input_path=tif_2d_folder, output_path=ims_file)
        progress_queue = Queue()
        MultiProcessCommandRunner(progress_queue, command,
                                  pattern=r"(WriteProgress:)\s+(\d*.\d+)\s*$", position=0).start()
        progress_bars = [tqdm(total=100, ascii=True, position=0, unit=" %", smoothing=0.01, desc=f"imaris")]
        commands_progress_manger(progress_queue, progress_bars, running_processes=1)

        [file.rename(file.parent / file.name[1:]) for file in files]

    if args.movie:
        movie_file = Path(args.movie)
        duration = 0.05

        with open(tif_2d_folder/"ffmpeg_input.txt", "wb") as ffmpeg_input:
            for file in list(tif_2d_folder.glob("*.tif"))[args.movie_start:args.movie_end]:
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
    PyScriptsPath = Path(r"./TeraStitcher/pyscripts")
    if sys.platform == "win32":
        # print("Windows is detected.")
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
        TeraStitcherPath = Path(r"TeraStitcher/Windows")/cpu_instruction
        os.environ["PATH"] = f"{os.environ['PATH']};{TeraStitcherPath.as_posix()}"
        os.environ["PATH"] = f"{os.environ['PATH']};{PyScriptsPath.as_posix()}"
        terastitcher = "terastitcher.exe"
        teraconverter = "teraconverter.exe"
    elif sys.platform == 'linux' and 'microsoft' not in uname().release.lower():
        print("Linux is detected.")
        psutil.Process().nice(value=19)
        TeraStitcherPath = Path(r"./TeraStitcher/Linux")/cpu_instruction
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

    imaris_converter = Path(r"./imaris9.7/ImarisConvertiv.exe")
    if not imaris_converter.exists():
        print("Error: ImarisConvertiv.exe not found")
        raise RuntimeError

    parser = ArgumentParser(
        description="Imaris to tif and TeraFly converter (version 0.1.0)\n\n",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="Developed 2022 by Keivan Moradi, Hongwei Dong Lab (B.R.A.I.N) at UCLA\n"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to input image or path")
    parser.add_argument("--tif", "-t", type=str, required=False, default='',
                        help="Path to tif output path")
    parser.add_argument("--teraFly", "-f", type=str, required=False, default='',
                        help="Path to output teraFly path")
    parser.add_argument("--imaris", "-o", type=str, required=False, default='',
                        help="Path to 8-bit imaris output file")
    parser.add_argument("--movie", "-m", type=str, required=False, default='',
                        help="Path to mp4 output file")
    parser.add_argument("--nthreads", "-n", type=int, default=12,
                        help="number of threads")
    parser.add_argument("--channel", "-c", type=int, default=0,
                        help="channel to be converted")
    parser.add_argument("--dark", "-d", type=int, default=0,
                        help="background vs foreground threshold")
    parser.add_argument("--convert_to_8bit", default=False, action=BooleanOptionalAction,
                        help="convert to 8-bit")
    parser.add_argument("--bit_shift", "-b", type=int, default=8,
                        help="bit_shift for 8-bit conversion")
    parser.add_argument("--movie_start", type=int, default=0,
                        help="start frame counting from 0")
    parser.add_argument("--movie_end", type=int, default=None,
                        help="end frame counting from 0")

    main(parser.parse_args())
