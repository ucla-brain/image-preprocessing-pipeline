from concurrent.futures import ProcessPoolExecutor, TimeoutError
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from math import ceil, floor, sqrt
from multiprocessing import Queue, Process, Manager, freeze_support
from pathlib import Path
from queue import Empty
from time import time, sleep
from typing import List, Tuple, Union, Callable

import h5py
import hdf5plugin
from numpy import floor as np_floor
from numpy import max as np_max
from numpy import mean as np_mean
from numpy import sqrt as np_sqrt
from numpy import round as np_round
from numpy import zeros, float32, dstack, rollaxis, savez_compressed, array, maximum, rot90, arange
from psutil import cpu_count, virtual_memory
from skimage.measure import block_reduce
from skimage.transform import resize, resize_local_mean
from tifffile import natural_sorted
from tqdm import tqdm

from pystripe.core import imread_tif_raw_png, imsave_tif, progress_manager, is_uniform_2d, is_uniform_3d
from supplements.cli_interface import PrintColors, date_time_now
from tsv.volume import TSVVolume, VExtent


def imread_tsv(tsv_volume: TSVVolume, extent: VExtent, d_type: str):
    return tsv_volume.imread(extent, d_type)[0]


class MultiProcess(Process):
    def __init__(
            self,
            progress_queue: Queue,
            args_queue: Queue,
            semaphore: Queue,
            function: Callable,
            images: Union[List[str], str],
            save_path: Path,
            args: tuple,
            kwargs: dict,
            shape: Tuple[int, int],
            dtype: str,
            rename: bool = False,
            tif_prefix: str = 'img',
            source_voxel: Union[Tuple[float, float, float], None] = None,
            target_voxel: Union[int, float, None] = None,
            down_sampled_path: Union[Path, None] = None,
            rotation: int = 0,
            channel: int = 0,
            timeout: Union[float, None] = 1800,
            resume: bool = True,
            compression: Tuple[str, int] = ("ADOBE_DEFLATE", 1),
            needed_memory: int = None
    ):
        Process.__init__(self)
        self.daemon = False
        self.progress_queue = progress_queue
        self.args_queue = args_queue
        self.semaphore = semaphore
        self.needed_memory = needed_memory
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
            assert Path(images[0]).suffix.lower() in (".tif", ".tiff", ".raw", ".png")
        self.channel = channel
        self.images = images
        self.save_path = save_path
        self.rename = rename
        self.tif_prefix = tif_prefix
        self.args = args
        self.kwargs = kwargs
        self.die = False
        self.timeout = timeout
        self.shape = shape
        self.d_type = dtype
        self.resume = resume
        self.compression = compression
        self.source_voxel = source_voxel
        self.target_voxel = target_voxel
        self.down_sampled_path = down_sampled_path
        self.target_shape = None
        self.down_sampling_methods = None
        if self.target_voxel is not None and self.source_voxel is not None and shape is not None:
            if rotation in (90, 270):
                self.calculate_down_sampling_target((shape[1], shape[0]), True)
            else:
                self.calculate_down_sampling_target(shape, False)
        self.rotation = rotation

    def calculate_down_sampling_target(self, new_shape: Tuple[int, int], is_rotated: bool):
        # calculate voxel size change
        new_shape: array = array(new_shape)
        new_voxel_size: list = list(self.source_voxel)
        if is_rotated:
            new_voxel_size[1] *= self.shape[0] / new_shape[1]
            new_voxel_size[2] *= self.shape[1] / new_shape[0]
            new_voxel_size[1], new_voxel_size[2] = new_voxel_size[2], new_voxel_size[1]
        else:
            new_voxel_size[1] *= self.shape[0] / new_shape[0]
            new_voxel_size[2] *= self.shape[1] / new_shape[1]
        new_voxel_size: tuple = tuple(new_voxel_size)
        if new_voxel_size != self.source_voxel:
            print(f"image processing function changed the voxel size from {self.source_voxel} to {new_voxel_size}")

        reduction_times = self.target_voxel / array(new_voxel_size[1:3])
        target_shape = new_shape / reduction_times
        self.target_shape = tuple(target_shape.round().astype(int))

        reduction_factors = np_floor(np_sqrt(reduction_times)).astype(int)
        down_sampling_method_y = list(np_max if i % 2 == 0 else np_mean for i in range(reduction_factors[0]))
        down_sampling_method_x = list(np_mean if i % 2 == 0 else np_max for i in range(reduction_factors[1]))
        if reduction_factors[0] > reduction_factors[1]:
            down_sampling_method_x += [None, ] * (reduction_factors[0] - reduction_factors[1])
        elif reduction_factors[0] < reduction_factors[1]:
            down_sampling_method_y += [None, ] * (reduction_factors[1] - reduction_factors[0])
        self.down_sampling_methods = tuple(zip(down_sampling_method_y, down_sampling_method_x))

    def imsave_tif(self, path, img, compression=None):
        die = imsave_tif(path, img, compression=compression)
        if die:
            self.die = True

    def tif_save_path(self, idx: int, images: List[Path]):
        if self.is_tsv or self.is_ims or self.rename:
            return self.save_path / f"{self.tif_prefix}_{idx:06}.tif"
        else:
            return self.save_path / Path(images[idx]).name

    def free_ram_is_not_enough(self):
        self.semaphore.get(block=True)
        free_ram_is_not_enough = False
        if self.needed_memory is not None and virtual_memory().available < self.needed_memory:
            free_ram_is_not_enough = True
            sleep(1)
        self.semaphore.put(1)
        return free_ram_is_not_enough

    def run(self):
        running_next: bool = True
        function = self.function
        images = self.images
        is_tsv = self.is_tsv
        is_ims = self.is_ims
        timeout = self.timeout
        # TODO: if no timeout was needed directly run functions without using pool
        if timeout:
            pool = ProcessPoolExecutor(max_workers=1)
        else:
            pool = ThreadPoolExecutor(max_workers=1)
        args = self.args
        kwargs = self.kwargs
        tif_prefix = self.tif_prefix
        channel = self.channel
        resume = self.resume
        compression = self.compression
        down_sampled_path = self.down_sampled_path
        d_type = self.d_type
        post_processed_d_type = self.d_type
        shape = self.shape
        rotation = self.rotation
        post_processed_shape = self.shape
        if rotation in (90, 270):
            post_processed_shape = (shape[1], shape[0])
        need_down_sampling = False
        down_sampling_method_z = None
        if self.source_voxel is not None and self.target_voxel is not None and shape is not None:
            need_down_sampling = True
            reduction_factor_z = ceil(sqrt(self.target_voxel / self.source_voxel[0]))
            # the last down-sampling for z step should be based on np_max to ensure max brightness
            down_sampling_method_z = tuple(np_max if i % 2 == 0 else np_mean for i in range(reduction_factor_z))

        file = None
        x0, x1, y0, y1 = 0, 0, 0, 0
        if is_tsv:
            x0, x1, y0, y1 = images.volume.x0, images.volume.x1, images.volume.y0, images.volume.y1
        if is_ims:
            file = h5py.File(images)
            images = file[f"DataSet/ResolutionLevel 0/TimePoint 0/Channel {channel}/Data"]
        queue_time_out = 20
        while not self.die and self.args_queue.qsize() > 0:
            if self.free_ram_is_not_enough():
                continue
            try:
                queue_start_time = time()
                idx_down_sampled, indices = self.args_queue.get(block=True, timeout=queue_time_out)
                queue_time_out = max(queue_time_out, 0.9 * queue_time_out + 0.3 * (time() - queue_start_time))
                z_stack = None
                down_sampled_tif_path = Path()
                if need_down_sampling and down_sampled_path is not None:
                    down_sampled_tif_path = down_sampled_path / f"{tif_prefix}_{idx_down_sampled:06}.tif"
                    if resume and down_sampled_tif_path.exists():
                        exist_count = 0
                        for idx_z, idx in enumerate(indices):
                            if self.tif_save_path(idx, images).exists():
                                exist_count += 1
                        if len(indices) == exist_count:
                            for _ in range(exist_count):
                                self.progress_queue.put(running_next)
                            continue
                    z_stack = zeros((len(indices),) + self.target_shape, dtype=float32)

                for idx_z, idx in enumerate(indices):
                    if self.die:
                        break
                    while self.free_ram_is_not_enough():
                        continue
                    if self.die:
                        break
                    tif_save_path = self.tif_save_path(idx, images)
                    if resume and tif_save_path.exists() and not need_down_sampling:  # function is not None and
                        self.progress_queue.put(running_next)
                        continue
                    try:
                        if resume and tif_save_path.exists():
                            img = None
                            if need_down_sampling:
                                img = imread_tif_raw_png(tif_save_path)
                        else:
                            if is_ims:
                                img = images[idx]
                            else:
                                # the pool protects the process in case of timeout errors in imread_* functions
                                start_time = time()
                                if is_tsv:
                                    future = pool.submit(
                                        imread_tsv, images, VExtent(x0, x1, y0, y1, idx, idx + 1), d_type)
                                else:
                                    future = pool.submit(
                                        imread_tif_raw_png, Path(images[idx]), dtype=d_type, shape=shape)
                                img = future.result(timeout=timeout)
                                if timeout is not None:
                                    timeout = max(timeout, 0.9 * timeout + 0.3 * (time() - start_time))
                                if len(img.shape) == 3 and 0 <= channel < 3:
                                    img = img[:, :, channel]

                            # apply function
                            if function is not None:
                                if args is not None and kwargs is not None:
                                    img = function(img, *args, **kwargs)
                                elif args is not None:
                                    img = function(img, *args)
                                elif kwargs is not None:
                                    img = function(img, **kwargs)
                                else:
                                    img = function(img)

                            if rotation == 90:
                                img = rot90(img, 1)
                            elif rotation == 180:
                                img = rot90(img, 2)
                            elif rotation == 270:
                                img = rot90(img, 3)

                            if is_tsv or is_ims or function is not None or rotation in (90, 180, 270):
                                self.imsave_tif(tif_save_path, img, compression=compression)

                            if img.dtype != post_processed_d_type:
                                post_processed_d_type = img.dtype

                            if rotation in (90, 270) or img.shape != post_processed_shape:
                                post_processed_shape = img.shape
                                if need_down_sampling:
                                    self.calculate_down_sampling_target(post_processed_shape, rotation in (90, 270))
                                    z_stack = zeros((len(indices),) + self.target_shape, dtype=float32)

                        # down-sampling on xy
                        if need_down_sampling and self.target_shape is not None and \
                                self.down_sampling_methods is not None and img is not None:
                            if is_uniform_2d(img):
                                z_stack[idx_z] = zeros(self.target_shape, dtype=float32)
                            else:
                                img = img.astype(float32)
                                for y_method, x_method in self.down_sampling_methods:
                                    if y_method is not None and ceil(img.shape[0] / 2) >= self.target_shape[0]:
                                        img = block_reduce(img, block_size=(2, 1), func=y_method)
                                    if x_method is not None and ceil(img.shape[1] / 2) >= self.target_shape[1]:
                                        img = block_reduce(img, block_size=(1, 2), func=x_method)
                                # print(img.shape, end='')
                                img = resize(img, self.target_shape, preserve_range=True, anti_aliasing=True)
                                z_stack[idx_z] = img.astype(float32)

                    except (BrokenProcessPool, TimeoutError):
                        message = f"\nwarning: {timeout}s timeout reached for processing input file number: {idx}\n"
                        if tif_save_path is not None and not tif_save_path.exists():
                            message += f"\ta dummy (zeros) image is saved as output instead:\n\t\t{tif_save_path}\n"
                            self.imsave_tif(tif_save_path, zeros(post_processed_shape, dtype=post_processed_d_type))
                        print(f"{PrintColors.WARNING}{message}{PrintColors.ENDC}")
                        if isinstance(pool, ProcessPoolExecutor):
                            pool.shutdown()
                            pool = ProcessPoolExecutor(max_workers=1)
                    except KeyboardInterrupt:
                        self.die = True
                        break
                    except Exception as inst:
                        print(
                            f"{PrintColors.WARNING}"
                            f"\nwarning: process failed for image index {idx}."
                            f"\n\targs: {tif_save_path if args is None else (tif_save_path, *args)}"
                            f"\n\tkwargs: {kwargs}"
                            f"\n\texception instance: {type(inst)}"
                            f"\n\texception arguments: {inst.args}"
                            f"\n\texception: {inst}"
                            f"{PrintColors.ENDC}")

                    self.progress_queue.put(running_next)

                # approximate down-sampling on the z-axis
                if need_down_sampling and down_sampling_method_z is not None and z_stack is not None:
                    if is_uniform_3d(z_stack):
                        self.imsave_tif(down_sampled_tif_path, zeros(self.target_shape, dtype=float32),
                                        compression=compression)
                    else:
                        for z_method in down_sampling_method_z:
                            if z_method is not None and z_stack.shape[0] > 1:
                                z_stack = block_reduce(z_stack, block_size=(2, 1, 1), func=z_method)
                        assert z_stack.shape[0] == 1
                        self.imsave_tif(down_sampled_tif_path, z_stack[0], compression=compression)

            except (Empty, TimeoutError):
                self.die = True
        if is_ims and isinstance(file, h5py.File):
            file.close()
        if isinstance(pool, ProcessPoolExecutor):
            pool.shutdown()
        self.progress_queue.put(not running_next)


def calculate_downsampling_z_ranges(start, end, steps):
    z_list_list = []
    for idx in range(start, end, steps):
        z_range = list(range(idx, idx + steps))
        if z_range[-1] > end:
            while z_range[-1] >= end:
                del z_range[-1]
        z_list_list += [z_range]
    return z_list_list


def generate_voxel_spacing(
        shape: Tuple[int, int, int],
        source_voxel: Tuple[int, int, int],
        target_shape: Tuple[int, int, int],
        target_voxel: float):
    voxel_locations = [arange(axis_shape) * axis_v_size - (axis_shape - 1) /
                       2.0 * axis_v_size for axis_shape, axis_v_size in zip(shape, source_voxel)]
    axis_spacing = []
    for i, axis_vals in enumerate(voxel_locations):
        # Get Downsampled starting value
        start = np_round(resize_local_mean(axis_vals, (int(target_shape[i]),)))[0]
        # Create target_voxel spaced list
        axis_spacing.append(array([start + target_voxel * val for val in range(target_shape[i])]))
    return axis_spacing


def jumpy_step_range(start, end):
    distance = end - start
    steps = [1, ]
    while distance / steps[-1] > 0:
        steps += [steps[-1] * 10]
    steps.reverse()
    top_list = []
    for step in steps:
        for idx in range(start, end, step):
            if idx not in top_list:
                top_list += [idx]
    return top_list


def parallel_image_processor(
        source: Union[Path, str],
        destination: Union[Path, str],
        fun: Union[Callable, None] = None,
        args: tuple = None,
        kwargs: dict = None,
        rename: bool = False,
        tif_prefix: str = "img",
        channel: int = 0,
        source_voxel: Union[Tuple[float, float, float], None] = None,
        target_voxel: Union[int, float, None] = None,
        rotation: int = 0,
        timeout: Union[float, None] = None,
        max_processors: int = cpu_count(logical=False),
        progress_bar_name: str = " ImgProc",
        compression: Tuple[str, int] = ("ADOBE_DEFLATE", 1),
        resume: bool = True,
        needed_memory: int = None
):
    """
    fun: Callable
        is a function that process images.
        Note: the function should not rotate the image if down-sampling by parallel image processor is required.
        Use rotate option of parallel image processor instead, which is safe for down-sampling.
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
    source_voxel: tuple
        voxel sizes of the image in um and zyx order.
    target_voxel: float
        down-sampled isotropic voxel size in um.
    rotation: int
        Rotate the image. One of 0, 90, 180 or 270 degree values are accepted. Default is 0 (no rotation).
    timeout: float
        max time in seconds to waite for each image to be processed not including the save time.
        Note: requesting timeout has some computational overhead in the current implementation.
    max_processors: int
        maximum number of processors
    chunks: int
        the number images from the list each process handles
    progress_bar_name: str
        the name next to the progress bar
    needed_memory: int
        needed_memory in bytes to run the function. if provided the workers try to avoid out of memory condition.
    """
    if isinstance(source, str):
        source = Path(source)
    if isinstance(destination, str):
        destination = Path(destination)

    if destination is not None:
        Path(destination).mkdir(exist_ok=True)

    down_sampling_z_steps: int = 1
    need_down_sampling: bool = False
    if source_voxel is not None and target_voxel is not None:
        need_down_sampling = True
        down_sampling_z_steps = max(1, floor(target_voxel / source_voxel[0]))

    args_queue = Queue()
    if isinstance(source, TSVVolume):
        images = source
        num_images = source.volume.z1 - source.volume.z0
        shape = source.volume.shape[1:3]
        dtype = source.dtype
        # to test stitching quality first a sample from every 100 z-step will be stitched
        if need_down_sampling and down_sampling_z_steps > 1:
            for ds_z_idx, z_range in enumerate(
                    calculate_downsampling_z_ranges(source.volume.z0, source.volume.z1, down_sampling_z_steps)):
                args_queue.put((ds_z_idx, z_range))
        else:
            for idx in jumpy_step_range(source.volume.z0, source.volume.z1):
                args_queue.put((idx, [idx]))

    elif source.is_file() and source.suffix.lower() == ".ims":
        print(f"ims file detected. hdf5plugin=v{hdf5plugin.version}")
        with h5py.File(source) as ims_file:
            img = ims_file[f"DataSet/ResolutionLevel 0/TimePoint 0/Channel {channel}/Data"]
            num_images = img.shape[0]
            shape = img.shape[1:3]
            dtype = img.dtype
        if need_down_sampling and down_sampling_z_steps > 1:
            for ds_z_idx, z_range in enumerate(calculate_downsampling_z_ranges(0, num_images, down_sampling_z_steps)):
                args_queue.put((ds_z_idx, z_range))
        else:
            for idx in range(num_images):
                args_queue.put((idx, [idx]))
        images = str(source)
    elif source.is_dir():
        images = natural_sorted([str(f) for f in source.iterdir() if f.is_file() and f.suffix.lower() in (
            ".tif", ".tiff", ".raw", ".png")])
        num_images = len(images)
        assert num_images > 0
        if need_down_sampling and down_sampling_z_steps > 1:
            for ds_z_idx, z_range in enumerate(calculate_downsampling_z_ranges(0, num_images, down_sampling_z_steps)):
                args_queue.put((ds_z_idx, z_range))
        else:
            for idx in range(num_images):
                args_queue.put((idx, [idx]))
        img = imread_tif_raw_png(Path(images[0]))
        shape = img.shape
        dtype = img.dtype
        manager = Manager()
        images = manager.list(images)
        del img
    else:
        print("source can be either a tsv volume, an ims file path, or a 2D tiff series folder")
        raise RuntimeError

    down_sampled_path = None
    if need_down_sampling:
        shape_3d = array((num_images,) + shape)
        new_source_voxel = source_voxel
        if rotation in (90, 270):
            shape_3d = array((num_images, shape[1], shape[0]))
            new_source_voxel = (source_voxel[0], source_voxel[2], source_voxel[1])

        reduction_times = target_voxel / array(new_source_voxel)
        target_shape = shape_3d / reduction_times
        target_shape_remainder = target_shape - target_shape.round()
        target_voxel_actual = maximum(target_voxel + target_shape_remainder / target_shape.round(), new_source_voxel)
        print(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
              f"{PrintColors.BLUE}down-sampling: {PrintColors.ENDC}\n"
              f"\tpost-processed shape zyx: {' '.join(target_shape.round(0).astype(str))}\n"
              f"\tactual voxel sizes zyx: {' '.join(target_voxel_actual.round(3).astype(str))}")

        down_sampled_path = destination / f"down_sampled_" \
                                          f"z{down_sampling_z_steps * new_source_voxel[0]:.1f}_" \
                                          f"yx{target_voxel:.1f}um"
        down_sampled_path.mkdir(exist_ok=True)

    progress_queue = Queue()
    semaphore = Queue()
    semaphore.put(1)
    workers = min(max_processors, args_queue.qsize())
    print(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}starting workers ...")
    for worker in tqdm(range(workers), desc=' workers'):
        if progress_queue.qsize() + worker < num_images:
            MultiProcess(
                progress_queue, args_queue, semaphore, fun, images, destination, args, kwargs, shape, dtype,
                rename=rename, tif_prefix=tif_prefix,
                source_voxel=source_voxel, target_voxel=target_voxel, down_sampled_path=down_sampled_path,
                rotation=rotation, channel=channel, timeout=timeout, compression=compression, resume=resume,
                needed_memory=needed_memory).start()
        else:
            print('\n the existing workers can finish the job! no more workers are needed.')
            workers = worker
            break

    return_code = progress_manager(progress_queue, workers, num_images, desc=progress_bar_name)
    args_queue.cancel_join_thread()
    args_queue.close()
    progress_queue.cancel_join_thread()
    progress_queue.close()

    # down-sample on z accurately
    if return_code == 0 and need_down_sampling:
        print(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
              f"{PrintColors.BLUE}down-sampling: {PrintColors.ENDC}"
              f"resizing on the z-axis accurately ...")
        target_shape_3d = [
            round(num_images / (target_voxel / source_voxel[0])),
            round(shape[0] / (target_voxel / source_voxel[1])),
            round(shape[1] / (target_voxel / source_voxel[2]))
        ]
        if rotation in (90, 270):
            target_shape_3d[1], target_shape_3d[2] = target_shape_3d[2], target_shape_3d[1]
        files = sorted(down_sampled_path.glob("*.tif"))
        with ThreadPoolExecutor(max_processors) as pool:
            img_stack = pool.map(imread_tif_raw_png, tqdm(files, desc="loading", unit="images"))
            img_stack = dstack(img_stack)  # yxz format
            img_stack = rollaxis(img_stack, -1)  # zyx format
            print(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
                  f"{PrintColors.BLUE}down-sampling: {PrintColors.ENDC}"
                  f"resizing the z-axis ...")
            img_stack = resize(img_stack, target_shape_3d, preserve_range=True, anti_aliasing=True)
            axes_spacing = generate_voxel_spacing(
                (num_images, shape[0], shape[1]),
                source_voxel,
                target_shape_3d,
                target_voxel)
            print(f"{PrintColors.GREEN}{date_time_now()}:{PrintColors.ENDC}"
                  f"{PrintColors.BLUE} down-sampling: {PrintColors.ENDC}"
                  f"saving as npz.")
            savez_compressed(
                destination / f"down_sampled_zyx{target_voxel:.1f}um.npz",
                I=img_stack,
                xI=array(axes_spacing, dtype='object')  # note specify object to avoid "ragged" warning
            )

    return return_code


if __name__ == '__main__':
    freeze_support()
