import os
import sys
import psutil
import subprocess
from pystripe.core import batch_filter, convert_to_8bit_fun
from multiprocessing import freeze_support
from pathlib import Path
from time import time
from platform import uname
import re
import h5py
from tifffile import imsave
from tqdm import tqdm
from numpy import where


if __name__ == '__main__':
    source = Path(sys.argv[1])
    tif_2d_folder = Path(sys.argv[2])
    dir_tera_fly = Path(sys.argv[3])

    tif_2d_folder.mkdir(exist_ok=True)
    dir_tera_fly.mkdir(exist_ok=True)

    freeze_support()
    # dir_tera_fly.mkdir(exist_ok=True, parents=True)
    PyScriptsPath = Path(r"./TeraStitcher/pyscripts")
    if sys.platform == "win32":
        # print("Windows is detected.")
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
        TeraStitcherPath = Path(r"./TeraStitcher/windows/avx512")
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


    def correct_path_for_cmd(filepath):
        if sys.platform == "win32":
            return f"\"{filepath}\""
        else:
            return str(filepath).replace(" ", r"\ ").replace("(", r"\(").replace(")", r"\)")


    def correct_path_for_wsl(filepath):
        p = re.compile(r"/mnt/(.)/")
        new_path = p.sub(r'\1:\\\\', str(filepath))
        new_path = new_path.replace(" ", r"\ ").replace("(", r"\(").replace(")", r"\)").replace("/", "\\\\")
        return new_path


    if source.is_file() and source.suffix.lower() == ".ims":
        with h5py.File(source, "r") as file:
            dark = 120
            bit_shift = 0
            images = file[f"DataSet/ResolutionLevel 0/TimePoint 0/Channel {1}/Data"]
            list(tqdm(
                map(lambda args:
                    imsave(
                        tif_2d_folder / f"img_{args[0]:04}.tif",
                        convert_to_8bit_fun(
                            where(args[1] > dark, args[1] - dark, 0),
                            bit_shift_to_right=bit_shift)
                    ),
                    enumerate(images)),
                total=len(images), ascii=True, smoothing=0.05, mininterval=1.0, unit="img", desc="ims2tif"
            ))
    else:
        batch_filter(
            source,
            tif_2d_folder,
            workers=96,
            # sigma=[foreground, background] Default is [0, 0], indicating no de-striping.
            sigma=(0, 0),
            # level=0,
            wavelet="db10",
            crossover=10,
            # threshold=-1,
            compression=('ZLIB', 0),  # ('ZSTD', 1) conda install imagecodecs
            flat=None,
            dark=100,
            # z_step=voxel_size_z,  # z-step in micron. Only used for DCIMG files.
            # rotate=False,
            lightsheet=False,
            artifact_length=150,
            # percentile=0.25,
            # convert_to_16bit=False,  # defaults to False
            convert_to_8bit=True,
            bit_shift_to_right=1,
            continue_process=True,
            dtype='uint16',
            tile_size=None,
            down_sample=None,
            new_size=None,
            timeout=None
        )

    command = [
        f"mpiexec -np 4 python -m mpi4py {paraconverter}",
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

    # ims_file_path = destination_folder / f'{destination_folder.name}.ims'
    # file = destination_folder / "SP210729-01_Z000.tif"
    # command = [
    #     f"" if sys.platform == "win32" else "wine",
    #     f"{correct_path_for_cmd(imaris_converter)}",
    #     f"--input {correct_path_for_cmd(file)}",
    #     f"--output {ims_file_path}",
    # ]
    # if sys.platform == "linux" and 'microsoft' in uname().release.lower():
    #     command = [
    #         f'{correct_path_for_cmd(imaris_converter)}',
    #         f'--input {correct_path_for_wsl(file)}',
    #         f"--output {correct_path_for_wsl(ims_file_path)}",
    #     ]
    #
    # command += [
    #     "--inputformat TiffSeries",
    #     f"--nthreads 96",
    #     f"--compression 1",
    #     # f"--defaultcolorlist BBRRGG"
    # ]
    # subprocess.call(" ".join(command), shell=True)
