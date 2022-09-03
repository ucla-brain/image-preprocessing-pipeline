import os
import h5py
import hdf5plugin
from multiprocessing import Queue, Process, Manager, freeze_support, cpu_count
from queue import Empty
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from concurrent.futures.process import BrokenProcessPool
from typing import List, Tuple, Union, Callable
from numpy import zeros, ulonglong
from pathlib import Path
from supplements.cli_interface import PrintColors
from pystripe.core import imread_tif_raw, imsave_tif, progress_manager
from tsv.volume import TSVVolume, VExtent

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'


def imread_tsv(tsv_volume: TSVVolume, extent, volume):
    return tsv_volume.imread(extent, volume)


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
            timeout: Union[float, None] = 900,
            resume: bool = True,
            compression: Tuple[str, int] = ("ZLIB", 1)
    ):
        Process.__init__(self)
        self.daemon = False
        self.progress_queue = progress_queue
        self.args_queue = args_queue
        self.function = function
        self.is_ims = False
        self.is_tsv = False
        if isinstance(images, TSVVolume):
            self.is_tsv = True
        elif isinstance(images, str) or isinstance(images, Path):
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
        self.compression = compression

    def run(self):
        running = True
        pool = ProcessPoolExecutor(max_workers=1)
        function = self.function
        images = self.images
        is_tsv = self.is_tsv
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
        compression = self.compression
        file = None
        img_sum = None
        counter = 0
        if save_path is None:
            img_sum = zeros(shape, ulonglong)
        x0, x1, y0, y1 = 0, 0, 0, 0
        if is_tsv:
            x0, x1, y0, y1 = images.volume.x0, images.volume.x1, images.volume.y0, images.volume.y1
        if is_ims:
            file = h5py.File(images)
            images = file[f"DataSet/ResolutionLevel 0/TimePoint 0/Channel {channel}/Data"]
        while not self.args_queue.empty():
            if self.die:
                break
            try:
                idx = self.args_queue.get(block=True, timeout=10)
                tif_save_path = None
                if isinstance(save_path, Path):
                    tif_save_path = save_path / f"{tif_prefix}_{idx:06}.tif"
                    if resume and tif_save_path.exists():
                        continue
                try:
                    if is_ims:
                        img = images[idx]
                    else:
                        if is_tsv:
                            future = pool.submit(imread_tsv, images, VExtent(x0, x1, y0, y1, idx, idx + 1), dtype)
                        else:
                            future = pool.submit(imread_tif_raw, (Path(images[idx],)), {"dtype": dtype, "shape": shape})
                        img = future.result(timeout=timeout)
                        if is_tsv:
                            img = img[0]
                        if len(img.shape) == 3:
                            img = img[:, :, channel]
                    if function is not None:
                        img = function(img, *args, **kwargs)
                    if img_sum is not None:
                        img_sum += img
                        counter += 1
                    else:
                        imsave_tif(tif_save_path, img, compression=compression)
                except (BrokenProcessPool, TimeoutError):
                    message = f"\nwarning: {timeout}s timeout reached for processing input file number: {idx}\n"
                    if tif_save_path is not None and not tif_save_path.exists():
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
        if counter > 0:
            running = (img_sum / counter).astype(dtype)
        else:
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
        timeout: Union[float, None] = 900,
        max_processors: int = cpu_count(),
        progress_bar_name: str = "ImgProc",
        compression: Tuple[str, int] = ("ZLIB", 1),
        resume: bool = True
):
    """
    fun: Callable
        is a function that process images
    source: Path or str
        path to a folder contacting 2d tif or raw series or path to an ims file. Hierarchical model is not supported.
    destination: Path, str or None
        destination folder. If destination is None an average image will be generated.
    args: Tuple
        arguments of given function in correct order
    kwargs:
        keyboard arguments of the given function
    tif_prefix: str
        prefix of the processed tif file
    channel: int
        The channel of multichannel tif or ims file
    timeout: float
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

    args_queue = Queue()
    if isinstance(source, TSVVolume):
        images = source
        for idx in range(source.volume.z0, source.volume.z1, 1):
            args_queue.put(idx)
        num_images = source.volume.z1 - source.volume.z0
        shape = source.volume.shape[1:3]
        dtype = source.dtype
    elif source.is_file() and source.suffix.lower() == ".ims":
        print(f"ims file detected. hdf5plugin=v{hdf5plugin.version}")
        with h5py.File(source) as ims_file:
            img = ims_file[f"DataSet/ResolutionLevel 0/TimePoint 0/Channel {channel}/Data"]
            num_images = img.shape[0]
            shape = img.shape[1:3]
            dtype = img.dtype
        for idx in range(num_images):
            args_queue.put(idx)
        images = str(source)
    elif source.is_dir():
        images = [str(f) for f in source.iterdir() if f.is_file() and f.suffix.lower() in (".tif", ".tiff", ".raw")]
        num_images = len(images)
        for idx in range(num_images):
            args_queue.put(idx)
        img = imread_tif_raw(Path(images[0]))
        shape = img.shape
        dtype = img.dtype
        manager = Manager()
        images = manager.list(images)
    else:
        print("source can be either a tsv volume, a ims file path, or 2D tif series folder")
        raise RuntimeError

    if destination is not None:
        Path(destination).mkdir(exist_ok=True)

    process_list: List[MultiProcess, ...] = []
    progress_queue = Queue()
    workers = min(max_processors, num_images)
    for worker in range(workers):
        process_list += [MultiProcess(
            progress_queue, args_queue, fun, images, destination, tif_prefix, args, kwargs, shape, dtype,
            channel=channel, timeout=timeout, compression=compression, resume=resume)
        ]
        process_list[-1].start()

    return_code = progress_manager(process_list, progress_queue, workers, num_images, desc=progress_bar_name)
    args_queue.close()
    progress_queue.close()
    return return_code


if __name__ == '__main__':
    freeze_support()
