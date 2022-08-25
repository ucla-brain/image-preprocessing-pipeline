import os
import sys
import psutil
import subprocess
from pystripe.core import batch_filter
from time import time
from platform import uname
from multiprocessing import freeze_support, cpu_count
from numpy import zeros, ndarray, where
from pathlib import Path
from pystripe.core import convert_to_8bit_fun
from process_images import correct_path_for_cmd, correct_path_for_wsl
from argparse import RawDescriptionHelpFormatter, ArgumentParser, Namespace
from parallel_image_processor import parallel_image_processor

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
    source = Path(args.input)
    tif_2d_folder = Path(args.tif)
    tif_2d_folder.mkdir(exist_ok=True, parents=True)
    dir_tera_fly = None
    if args.teraFly:
        dir_tera_fly = Path(args.teraFly)
        dir_tera_fly.mkdir(exist_ok=True, parents=True)
    ims_file = None
    if args.imaris:
        ims_file = Path(args.imaris)

    return_code = 0
    if source.is_file() and source.suffix.lower() == ".ims":
        return_code = parallel_image_processor(
            process_img,
            str(source),
            str(tif_2d_folder),
            (),
            {"dark": args.dark, "bit_shift": args.bit_shift},
            max_processors=args.nthreads,
            channel=args.channel
        )
    elif source.is_dir():
        return_code = batch_filter(
            source,
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
            convert_to_8bit=True,
            bit_shift_to_right=args.bit_shift,
            continue_process=True,
            dtype='uint16',
            tile_size=None,
            down_sample=None,
            new_size=None,
            timeout=None
        )
    assert return_code == 0

    if dir_tera_fly is not None:
        command = [
            f"mpiexec -np {args.nthreads} python -m mpi4py {paraconverter}",
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
        start_time = time()
        subprocess.call(" ".join(command), shell=True)
        print(f"elapsed time = {round((time() - start_time) / 60, 1)}")

    if ims_file is not None:
        file = tif_2d_folder / sorted(tif_2d_folder.glob("*.tif"))[0]
        command = [
            f"" if sys.platform == "win32" else "wine",
            f"{correct_path_for_cmd(imaris_converter)}",
            f"--input {correct_path_for_cmd(file)}",
            f"--output {ims_file}",
        ]
        if sys.platform == "linux" and 'microsoft' in uname().release.lower():
            command = [
                f'{correct_path_for_cmd(imaris_converter)}',
                f'--input {correct_path_for_wsl(file)}',
                f"--output {correct_path_for_wsl(ims_file)}",
            ]

        command += [
            "--inputformat TiffSeries",
            f"--nthreads {args.nthreads}",
            f"--compression 1"
        ]
        subprocess.call(" ".join(command), shell=True)


if __name__ == '__main__':
    freeze_support()
    PyScriptsPath = Path(r"./TeraStitcher/pyscripts")
    if sys.platform == "win32":
        # print("Windows is detected.")
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
        TeraStitcherPath = Path(r"TeraStitcher/Windows/avx512")
        os.environ["PATH"] = f"{os.environ['PATH']};{TeraStitcherPath.as_posix()}"
        os.environ["PATH"] = f"{os.environ['PATH']};{PyScriptsPath.as_posix()}"
        terastitcher = "terastitcher.exe"
        teraconverter = "teraconverter.exe"
    elif sys.platform == 'linux' and 'microsoft' not in uname().release.lower():
        print("Linux is detected.")
        psutil.Process().nice(value=19)
        TeraStitcherPath = Path(r"./TeraStitcher/Linux/AVX512")
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

    imaris_converter = Path(r"./imaris") / "ImarisConvertiv.exe"
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
    parser.add_argument("--tif", "-t", type=str, required=True,
                        help="Path to tif output path")
    parser.add_argument("--teraFly", "-f", type=str, default='',
                        help="Path to output teraFly path")
    parser.add_argument("--imaris", "-o", type=str, default='',
                        help="Path to 8-bit imaris output file")
    parser.add_argument("--nthreads", "-n", type=int, default=cpu_count(),
                        help="number of threads")
    parser.add_argument("--channel", "-c", type=int, default=0,
                        help="channel to be converted")
    parser.add_argument("--dark", "-d", type=int, default=0,
                        help="background vs foreground threshold")
    parser.add_argument("--bit_shift", "-b", type=int, default=8,
                        help="bit_shift for 8-bit conversion")

    main(parser.parse_args())
