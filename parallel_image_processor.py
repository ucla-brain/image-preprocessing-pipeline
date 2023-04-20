import os
import h5py
import hdf5plugin
from multiprocessing import Queue, Process, Manager, freeze_support, cpu_count
from queue import Empty
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from concurrent.futures.process import BrokenProcessPool
from typing import List, Tuple, Union, Callable
from numpy import zeros
from pathlib import Path
from supplements.cli_interface import PrintColors, date_time_now
from pystripe.core import imread_tif_raw_png, imsave_tif, progress_manager
from tsv.volume import TSVVolume, VExtent
from time import time
from tqdm import tqdm

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'


def imread_tsv(tsv_volume: TSVVolume, extent: VExtent, d_type: str):
    return tsv_volume.imread(extent, d_type)[0]


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
            timeout: Union[float, None] = 1800,
            resume: bool = True,
            compression: Tuple[str, int] = ("ADOBE_DEFLATE", 1)
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
        self.shape = shape
        self.d_type = dtype
        self.resume = resume
        self.compression = compression

    def run(self):
        running: bool = True
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
        post_processed_shape = self.shape
        d_type = self.d_type
        post_processed_d_type = self.d_type
        channel = self.channel
        resume = self.resume
        compression = self.compression
        file = None
        x0, x1, y0, y1 = 0, 0, 0, 0
        if is_tsv:
            x0, x1, y0, y1 = images.volume.x0, images.volume.x1, images.volume.y0, images.volume.y1
        if is_ims:
            file = h5py.File(images)
            images = file[f"DataSet/ResolutionLevel 0/TimePoint 0/Channel {channel}/Data"]
        queue_time_out = 20
        while not self.die and self.args_queue.qsize() > 0:
            try:
                queue_start_time = time()
                idx = self.args_queue.get(block=True, timeout=queue_time_out)
                queue_time_out = max(queue_time_out, 0.9 * queue_time_out + 0.3 * (time() - queue_start_time))
                tif_save_path = None
                if isinstance(save_path, Path):
                    tif_save_path = save_path / f"{tif_prefix}_{idx:06}.tif"
                    if resume and tif_save_path.exists():
                        continue
                try:
                    if is_ims:
                        img = images[idx]
                    else:
                        start_time = time()
                        if is_tsv:
                            future = pool.submit(imread_tsv, images, VExtent(x0, x1, y0, y1, idx, idx + 1), d_type)
                        else:
                            future = pool.submit(
                                imread_tif_raw_png,
                                (Path(images[idx]), ),
                                {"dtype": d_type, "shape": shape}
                            )
                        img = future.result(timeout=timeout)
                        if timeout is not None:
                            timeout = max(timeout, 0.9 * timeout + 0.3 * (time() - start_time))
                        if len(img.shape) == 3:
                            img = img[:, :, channel]
                    if function is not None:
                        img = function(img, *args, **kwargs)
                        if img.shape != post_processed_shape or img.dtype != post_processed_d_type:
                            post_processed_shape = img.shape
                            post_processed_d_type = img.dtype
                    imsave_tif(tif_save_path, img, compression=compression)
                except (BrokenProcessPool, TimeoutError):
                    message = f"\nwarning: {timeout}s timeout reached for processing input file number: {idx}\n"
                    if tif_save_path is not None and not tif_save_path.exists():
                        message += f"\ta dummy (zeros) image is saved as output instead:\n\t\t{tif_save_path}\n"
                        imsave_tif(tif_save_path, zeros(post_processed_shape, dtype=post_processed_d_type))
                    print(f"{PrintColors.WARNING}{message}{PrintColors.ENDC}")
                    pool.shutdown()
                    pool = ProcessPoolExecutor(max_workers=1)
                except KeyboardInterrupt:
                    self.die = True
                    # while not self.args_queue.qsize() == 0:
                    #     try:
                    #         self.args_queue.get(block=True, timeout=10)
                    #     except Empty:
                    #         continue
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
            except (Empty, TimeoutError):
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
        timeout: Union[float, None] = 1800,
        max_processors: int = cpu_count(),
        progress_bar_name: str = " ImgProc",
        compression: Tuple[str, int] = ("ADOBE_DEFLATE", 1),
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
        # to test stitching quality first a sample from every 100 z-step will be stitched
        top_list = []
        for step in (1000, 100, 10, 1):
            for idx in range(source.volume.z0, source.volume.z1, step):
                if idx not in top_list:
                    args_queue.put(idx)
                if step > 1:
                    top_list += [idx]
        del top_list
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
        assert num_images > 0
        for idx in range(num_images):
            args_queue.put(idx)
        img = imread_tif_raw_png(Path(images[0]))
        shape = img.shape
        dtype = img.dtype
        manager = Manager()
        images = manager.list(images)
    else:
        print("source can be either a tsv volume, a ims file path, or 2D tif series folder")
        raise RuntimeError

    if destination is not None:
        Path(destination).mkdir(exist_ok=True)

    progress_queue = Queue()
    workers = min(max_processors, num_images)
    print(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}starting workers ...")
    for worker in tqdm(range(workers), desc=' workers'):
        if progress_queue.qsize() < num_images - worker:
            MultiProcess(
                progress_queue, args_queue, fun, images, destination, tif_prefix, args, kwargs, shape, dtype,
                channel=channel, timeout=timeout, compression=compression, resume=resume).start()
        else:
            print('\n the existing workers finished the job! no more worker is needed.')
            workers = worker
            break

    return_code_or_img_list = progress_manager(progress_queue, workers, num_images, desc=progress_bar_name)
    args_queue.cancel_join_thread()
    args_queue.close()
    progress_queue.cancel_join_thread()
    progress_queue.close()
    return return_code_or_img_list


if __name__ == '__main__':
    freeze_support()
