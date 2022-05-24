import os
import sys
import psutil
import subprocess
import h5py
import hdf5plugin
from pystripe.core import batch_filter
from time import time
from platform import uname
from multiprocessing import Queue, Process, Manager, freeze_support, cpu_count
from queue import Empty
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from concurrent.futures.process import BrokenProcessPool
from typing import List, Tuple, Union, Callable
from numpy import zeros, ndarray, where
from pathlib import Path
from supplements.cli_interface import PrintColors
from pystripe.core import imread_tif_raw, convert_to_8bit_fun, imsave_tif, progress_manager
from process_images import correct_path_for_cmd, correct_path_for_wsl

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'


class MultiProcess(Process):
    def __init__(
            self,
            progress_queue: Queue,
            args_queue: Queue,
            function: Callable,
            images: Union[List[str], str],
            save_path: Path,
            tif_prefix: str,
            args: tuple,
            kwargs: dict,
            shape: Tuple[int, int],
            dtype: str,
            channel: int = 0,
            timeout: float = 300,
            resume: bool = True
    ):
        Process.__init__(self)
        self.daemon = False
        self.progress_queue = progress_queue
        self.args_queue = args_queue
        self.function = function
        self.is_ims = False
        if isinstance(images, str):
            images = Path(images)
            assert images.suffix.lower() == ".ims"
            self.is_ims = True
        else:
            assert Path(images[0]).suffix.lower() in (".tif", ".tiff", ".raw")
        self.channel = channel
        self.images = images
        self.save_path = save_path
        self.tif_prefix = tif_prefix
        self.args = args
        self.kwargs = kwargs
        self.die = False
        self.timeout = timeout
        self.default_img = zeros(shape, dtype=dtype)
        self.shape = shape
        self.dtype = dtype
        self.resume = resume

    def run(self):
        running = True
        pool = ProcessPoolExecutor(max_workers=1)
        function = self.function
        images = self.images
        is_ims = self.is_ims
        timeout = self.timeout
        args = self.args
        kwargs = self.kwargs
        save_path = self.save_path
        tif_prefix = self.tif_prefix
        shape = self.shape
        dtype = self.dtype
        channel = self.channel
        resume = self.resume
        file = None
        if is_ims:
            file = h5py.File(images)
            images = file[f"DataSet/ResolutionLevel 0/TimePoint 0/Channel {channel}/Data"]
        while not self.args_queue.empty():
            if self.die:
                break
            try:
                idx = self.args_queue.get(block=True, timeout=10)
                tif_save_path = save_path / f"{tif_prefix}_{idx:04}.tif"
                if resume and tif_save_path.exists():
                    continue
                try:
                    if is_ims:
                        img = images[idx]
                    else:
                        future = pool.submit(imread_tif_raw, (Path(images[idx], )), {"dtype": dtype, "shape": shape})
                        img = future.result(timeout=timeout)
                        if len(img.shape) == 3:
                            img = img[:, :, channel]
                    img = function(img, *args, **kwargs)
                    imsave_tif(tif_save_path, img)
                except (BrokenProcessPool, TimeoutError):
                    message = f"\nwarning: {timeout}s timeout reached for processing input file number: {idx}\n"
                    if not tif_save_path.exists():
                        message += f"\ta dummy (zeros) image is saved as output instead:\n\t\t{tif_save_path}\n"
                        imsave_tif(tif_save_path, self.default_img)
                    print(f"{PrintColors.WARNING}{message}{PrintColors.ENDC}")
                    pool.shutdown()
                    pool = ProcessPoolExecutor(max_workers=1)
                except (KeyboardInterrupt, SystemExit):
                    # print(f"{PrintColors.WARNING}dying from {idx}{PrintColors.ENDC}")
                    self.die = True
                except Exception as inst:
                    print(
                        f"{PrintColors.WARNING}"
                        f"\nwarning: process failed for image index {idx}."
                        f"\n\targs: {(tif_save_path, *args)}"
                        f"\n\tkwargs: {kwargs}"
                        f"\n\texception instance: {type(inst)}"
                        f"\n\texception arguments: {inst.args}"
                        f"\n\texception: {inst}"
                        f"{PrintColors.ENDC}")
                self.progress_queue.put(running)
            except Empty:
                self.die = True
        pool.shutdown()
        if is_ims and isinstance(file, h5py.File):
            file.close()
        running = False
        self.progress_queue.put(running)


def parallel_image_processor(
        fun: Callable,
        source: Union[Path, str],
        destination: Union[Path, str],
        args: tuple = None,
        kwargs: dict = None,
        tif_prefix: str = "img",
        channel: int = 0,
        timeout: int = 300,
        max_processors: int = cpu_count(),
        progress_bar_name: str = "ImgProc",
):
    """
    fun: Callable
        is a function that process images
    source: Path or str
        path to a folder contacting 2d tif or raw series or path to an ims file. Hierarchical model is not supported.
    destination: Path or str
        destination folder.
    args: Tuple
        arguments of given function in correct order
    kwargs:
        keyboard arguments of the given function
    tif_prefix: str
        prefix of the processed tif file
    channel: int
        The channel of multichannel tif or ims file
    timeout: int
        max time in seconds to waite for each image to be processed not including the save time
    max_processors: int
        maximum number of processors
    chunks: int
        the number images from the list each process handles
    progress_bar_name: str (optional)
        the name next to the progress bar
    """
    if isinstance(source, str):
        source = Path(source)
    if isinstance(destination, str):
        destination = Path(destination)
    if source.is_file() and source.suffix.lower() == ".ims":
        print(f"ims file detected. hdf5plugin=v{hdf5plugin.version}")
        with h5py.File(source) as ims_file:
            img = ims_file[f"DataSet/ResolutionLevel 0/TimePoint 0/Channel {channel}/Data"]
            shape = img.shape
            dtype = img.dtype
        num_images = shape[0]
        shape = (shape[1], shape[2])
        images = str(source)
    elif source.is_dir():
        images = [str(f) for f in source.iterdir() if f.is_file() and f.suffix.lower() in (".tif", ".tiff", ".raw")]
        num_images = len(images)
        img = imread_tif_raw(Path(images[0]))
        shape = img.shape
        dtype = img.dtype
        manager = Manager()
        images = manager.list(images)
    else:
        print("source can be either one ims file or 2D tif series")
        raise RuntimeError

    Path(destination).mkdir(exist_ok=True)

    args_queue = Queue()
    for idx in range(num_images):
        args_queue.put(idx)

    process_list: List[MultiProcess, ...] = []
    progress_queue = Queue()
    workers = min(max_processors, num_images)
    for worker in range(workers):
        process_list += [MultiProcess(
            progress_queue, args_queue, fun, images, destination, tif_prefix, args, kwargs, shape, dtype,
            channel=channel, timeout=timeout)
        ]
        process_list[-1].start()

    return_code = progress_manager(process_list, progress_queue, workers, args_queue.qsize(), desc=progress_bar_name)
    args_queue.close()
    progress_queue.close()
    return return_code


def process_img(image: ndarray, dark: int = 120, bit_shift: int = 0):
    # image = filter_streaks(image, (128, 256), wavelet='db2')
    if dark > 0:
        image = where(image > dark, image - dark, 0)
    image = convert_to_8bit_fun(image, bit_shift_to_right=bit_shift)
    return image


def main(source: str, tif_2d_folder: str, dir_tera_fly: str, ims_file: str = None):
    source = Path(source)
    tif_2d_folder = Path(tif_2d_folder)
    dir_tera_fly = Path(dir_tera_fly)
    if ims_file is not None:
        ims_file = Path(ims_file)
    tif_2d_folder.mkdir(exist_ok=True, parents=True)
    dir_tera_fly.mkdir(exist_ok=True, parents=True)

    return_code = 0
    if source.is_file() and source.suffix.lower() == ".ims":
        return_code = parallel_image_processor(
            process_img,
            str(source),
            str(tif_2d_folder),
            (),
            {"dark": 0, "bit_shift": 0},
            max_processors=48,
            channel=1
        )
    elif source.is_dir():
        return_code = batch_filter(
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
    assert return_code == 0
    command = [
        f"mpiexec -np {cpu_count()} python -m mpi4py {paraconverter}",
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
            f"--nthreads 96",
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

    try:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    except IndexError:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
