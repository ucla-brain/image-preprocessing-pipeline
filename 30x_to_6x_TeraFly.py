import os
import sys
import psutil
import subprocess
from pystripe.core import batch_filter
from multiprocessing import freeze_support
from pathlib import Path
from time import time


# source_folder = Path(r'X:\3D_stitched\20210910_SM210705_01_R_PFC_LS_6x_1000z\tif')
final_tiff_folder = Path(r'D:\20210910_SM210705_01_R_PFC_LS_6x_1000z\tif')
dir_tera_fly = Path(r'D:\20210910_SM210705_01_R_PFC_LS_6x_1000z\TeraFly_C2')

dir_tera_fly.mkdir(exist_ok=True)
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
    TeraStitcherPath = Path(r"./TeraStitcher/linux")
    os.environ["PATH"] = f"{os.environ['PATH']}:{TeraStitcherPath.as_posix()}"
    os.environ["PATH"] = f"{os.environ['PATH']}:{PyScriptsPath.as_posix()}"
    terastitcher = "terastitcher"
    teraconverter = "teraconverter"
    os.environ["TERM"] = "xterm"
elif sys.platform == 'linux' and 'microsoft' in uname().release.lower():
    print("Windows subsystem for Linux is detected.")
    psutil.Process().nice(value=19)
    CacheDriveExample = "/mnt/d/"
    TeraStitcherPath = Path(r"./TeraStitcher/linux")
    os.environ["PATH"] = f"{os.environ['PATH']}:{TeraStitcherPath.as_posix()}"
    os.environ["PATH"] = f"{os.environ['PATH']}:{PyScriptsPath.as_posix()}"
    terastitcher = "terastitcher"
    teraconverter = "teraconverter"
    os.environ["TERM"] = "xterm"
else:
    log.error("yet unsupported OS")
    raise RuntimeError

paraconverter = PyScriptsPath / "paraconverter.py"
if not paraconverter.exists():
    print(paraconverter)
    print("Error: paraconverter not found")
    raise RuntimeError

imaris_converter = Path(r"./imaris") / "ImarisConvertiv.exe"
if not imaris_converter.exists():
    log.error("Error: ImarisConvertiv.exe not found")
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


if __name__ == '__main__':
    freeze_support()
    # batch_filter(
    #     source_folder,
    #     final_tiff_folder,
    #     workers=61,
    #     chunks=10,
    #     # sigma=[foreground, background] Default is [0, 0], indicating no de-striping.
    #     sigma=(0, 0),
    #     # level=0,
    #     wavelet="db10",
    #     crossover=10,
    #     # threshold=-1,
    #     compression=('ZLIB', 0),  # ('ZSTD', 1) conda install imagecodecs
    #     flat=None,
    #     dark=None,
    #     # z_step=voxel_size_z,  # z-step in micron. Only used for DCIMG files.
    #     # rotate=False,
    #     lightsheet=False,
    #     artifact_length=150,
    #     # percentile=0.25,
    #     # convert_to_16bit=False,  # defaults to False
    #     convert_to_8bit=False,
    #     bit_shift_to_right=3,
    #     continue_process=True,
    #     dtype='uint16',
    #     tile_size=None,
    #     down_sample=5,
    #     new_size=None
    # )

    command = [
        f"mpiexec -np 48 python -m mpi4py {paraconverter}",
        # f"{teraconverter}",
        "--sfmt=\"TIFF (series, 2D)\"",
        "--dfmt=\"TIFF (tiled, 3D)\"",
        "--resolutions=\"012345\"",
        "--clist=2",
        "--halve=max",
        # "--noprogressbar",
        # "--sparse_data",
        # "--fixed_tiling",
        "--height=256",
        "--width=256",
        "--depth=256",
        f"-s={final_tiff_folder}",  # destination_folder
        f"-d={dir_tera_fly}",
    ]
    start_time = time()
    subprocess.call(" ".join(command), shell=True)
    print(f"elapsed time = {(time() - start_time) / 60}")

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

