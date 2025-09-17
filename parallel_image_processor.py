import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, TimeoutError
from concurrent.futures.process import BrokenProcessPool
from imaris_ims_file_reader.ims import ims
from math import ceil, floor, sqrt
from multiprocessing import Queue, Process, Manager, freeze_support
from pathlib import Path
from queue import Empty
from time import time, sleep
from typing import Any, Callable, Optional, Tuple, Union, Literal

from contextlib import contextmanager
from numpy import floor as np_floor
from numpy import max as np_max
from numpy import mean as np_mean
from numpy import round as np_round
from numpy import sqrt as np_sqrt
from numpy import percentile as np_percentile
from numpy import (zeros, float32, dstack, rollaxis, savez_compressed, array, maximum, rot90, arange, uint8, uint16, flip,
                   stack, fliplr, flipud, ndarray)
from psutil import cpu_count, virtual_memory
from skimage.measure import block_reduce
from skimage.transform import resize, resize_local_mean
from tifffile import natural_sorted
from tqdm import tqdm

from pystripe.core import (imread_tif_raw_png, imsave_tif, progress_manager, is_uniform_2d, is_uniform_3d,
                           convert_to_8bit_fun, convert_to_16bit_fun)
from supplements.cli_interface import PrintColors, date_time_now
from tsv.volume import TSVVolume, VExtent


def imread_tsv(tsv_volume: TSVVolume, extent: VExtent, d_type: str):
    return tsv_volume.imread(extent, d_type)[0]


class ImarisZWrapper:
    @staticmethod
    @contextmanager
    def _suppress_stdout():
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout

    def __init__(self, ims_path, timepoint=0, channel=0):
        with self._suppress_stdout():
            self.imaris_data = ims(ims_path)
        self.timepoint = timepoint
        self.channel = channel
        self.num_z = self.imaris_data.shape[2]

        x_min = float(self.imaris_data.read_numerical_dataset_attr('ExtMin0'))
        x_max = float(self.imaris_data.read_numerical_dataset_attr('ExtMax0'))
        y_min = float(self.imaris_data.read_numerical_dataset_attr('ExtMin1'))
        y_max = float(self.imaris_data.read_numerical_dataset_attr('ExtMax1'))
        z_min = float(self.imaris_data.read_numerical_dataset_attr('ExtMin2'))
        z_max = float(self.imaris_data.read_numerical_dataset_attr('ExtMax2'))
        self.flip_x = abs(x_min) > abs(x_max)
        self.flip_y = abs(y_min) > abs(y_max)
        self.flip_z = abs(z_min) > abs(z_max)

    def __getitem__(self, z):
        if isinstance(z, slice):
            indices = range(*z.indices(self.num_z))
            if self.flip_z:
                indices = reversed(list(indices))
            img = stack([self.imaris_data[self.timepoint, self.channel, zi, :, :] for zi in indices])
        else:
            if self.flip_z:
                img = self.imaris_data[self.timepoint, self.channel, self.num_z - z - 1, :, :]
            else:
                img = self.imaris_data[self.timepoint, self.channel, z, :, :]
        if self.flip_x:
            img = fliplr(img)
        if self.flip_y:
            img = flipud(img)
        return img

    def __len__(self):
        return self.num_z

    def close(self):
        with self._suppress_stdout():
            self.imaris_data.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            print(f"Exception during __del__: {e}")
            pass


def crop_2d_image_px(image: ndarray, x_start: int, x_end: int, y_start: int, y_end: int) -> ndarray:
    """Crop a 2D image using pixel coordinates (shape [y, x])."""
    y_max, x_max = image.shape
    x_start = max(x_start, 0)
    y_start = max(y_start, 0)
    x_end = min(x_end, x_max)
    y_end = min(y_end, y_max)
    return image[y_start:y_end, x_start:x_end]


class MultiProcess(Process):
    """Worker that reads, optional-processes, crops, rotates, (optionally flips), saves, and (optionally) downsamples."""

    # NOTE: to reduce PyCharm type noise, keep hints simple and permissive here.
    def __init__(
            self,
            progress_queue: Queue,
            args_queue: Queue,
            semaphore: Queue,
            function: Optional[Callable[..., ndarray]],
            images: Any,  # list[str] | list[Path] | TSVVolume | Path(str to .ims) | ImarisZWrapper (set at runtime)
            save_path: Path,
            args: Optional[tuple],
            kwargs: Optional[dict],
            shape: Tuple[int, int],
            dtype: Union[str, Any],
            rename: bool = False,
            tif_prefix: str = 'img',
            source_voxel: Optional[Tuple[float, float, float]] = None,
            target_voxel: Optional[Union[int, float]] = None,
            down_sampled_path: Optional[Path] = None,
            rotation: int = 0,
            channel: int = 0,
            timeout: Optional[float] = 1800,
            resume: bool = True,
            compression: Tuple[str, int] = ("ADOBE_DEFLATE", 1),
            needed_memory: Optional[int] = None,
            save_images: bool = True,
            alternating_downsampling_method: Union[bool, Literal["contrast"]] = True,
            down_sampled_dtype: Union[str, type] = "float32",
            crop_bbox: Optional[Tuple[int, int, int, int]] = None,
    ):
        super().__init__()
        self.daemon = False
        self.progress_queue = progress_queue
        self.args_queue = args_queue
        self.semaphore = semaphore
        self.needed_memory = needed_memory
        self.function = function
        self.is_ims = False
        self.is_tsv = False
        self.images_count: int = 0  # used for naming when we don't have a list

        if isinstance(images, TSVVolume):
            # TSV path: pass the TSVVolume object (we will avoid process-pools later)
            self.is_tsv = True
            self.images = images
            try:
                self.images_count = int(images.volume.z1 - images.volume.z0)
            except Exception as e:
                print(f"{PrintColors.WARNING}TSVVolume z count unavailable: {e}{PrintColors.ENDC}")
                self.images_count = 0

        elif isinstance(images, (str, Path)):
            # IMS path must be a single .ims file, not a directory and not a list
            ims_path = Path(images)
            if ims_path.suffix.lower() != ".ims":
                raise TypeError(
                    f"For IMS mode, 'images' must be a path to a .ims file; got: {ims_path}"
                )
            self.is_ims = True
            self.images = ims_path  # keep path only; open ImarisZWrapper in run()
            # images_count will be determined in run() after opening the wrapper

        else:
            # TIF/PNG/RAW path list: must be list/tuple of file paths
            if not isinstance(images, (list, tuple)):
                raise TypeError(
                    "For TIF/PNG/RAW mode, 'images' must be a list/tuple of file paths."
                )
            if len(images) == 0:
                raise ValueError("Empty image list for TIF/PNG/RAW mode.")

            allowed = {".tif", ".tiff", ".raw", ".png"}
            # (Lightweight validation; avoid walking the whole list for speed)
            for i, fp in enumerate(images[:8]):
                if Path(fp).suffix.lower() not in allowed:
                    raise ValueError(f"Unsupported extension in images[{i}]: {fp}")

            self.images = images  # list of paths (pickleable)
            self.images_count = len(images)

        self.channel = channel
        self.images = images
        self.save_path = save_path
        self.save_images = save_images
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
        self.down_sampled_dtype = down_sampled_dtype
        self.target_shape: Optional[Tuple[int, int]] = None
        self.down_sampling_methods: Optional[Tuple[Tuple[Optional[Callable], Optional[Callable]], ...]] = None
        self.alternating_downsampling_method = alternating_downsampling_method
        if self.target_voxel is not None and self.source_voxel is not None and shape is not None:
            if rotation in (90, 270):
                self.calculate_down_sampling_target((shape[1], shape[0]), True, alternating_downsampling_method)
            else:
                self.calculate_down_sampling_target(shape, False, alternating_downsampling_method)
        self.rotation = rotation
        self.crop_bbox = crop_bbox

    @staticmethod
    def _p90(arr):
        """Percentile-90 pooling for contrast-enhancing downsampling."""
        return float(np_percentile(arr, 90.0))

    def calculate_down_sampling_target(
            self,
            new_shape: Tuple[int, int],
            is_rotated: bool,
            alternating_downsampling_method: Union[bool, Literal["contrast"]],
    ) -> None:
        # compute voxel size change
        new_shape_arr = array(new_shape)
        new_voxel_size: list = list(self.source_voxel)  # type: ignore[arg-type]
        if is_rotated:
            new_voxel_size[1] *= self.shape[0] / new_shape_arr[1]
            new_voxel_size[2] *= self.shape[1] / new_shape_arr[0]
            new_voxel_size[1], new_voxel_size[2] = new_voxel_size[2], new_voxel_size[1]
        else:
            new_voxel_size[1] *= self.shape[0] / new_shape_arr[0]
            new_voxel_size[2] *= self.shape[1] / new_shape_arr[1]
        new_voxel_size_t = (float(new_voxel_size[0]), float(new_voxel_size[1]), float(new_voxel_size[2]))
        if new_voxel_size_t != self.source_voxel:
            print(f"image processing function changed the voxel size from {self.source_voxel} to {new_voxel_size_t}")

        reduction_times = self.target_voxel / array(new_voxel_size_t[1:3])  # type: ignore[operator]
        target_shape_arr = new_shape_arr / reduction_times
        target_shape_2 = (int(round(target_shape_arr[0])), int(round(target_shape_arr[1])))
        self.target_shape = target_shape_2  # explicit (int,int) for PyCharm

        # choose reduction “passes” per axis
        reduction_factors = np_floor(np_sqrt(reduction_times)).astype(int)
        n_y, n_x = int(reduction_factors[0]), int(reduction_factors[1])
        n_steps = max(n_y, n_x)

        if isinstance(alternating_downsampling_method, str) and alternating_downsampling_method.lower() == "contrast":
            # Contrast-enhancing mode: prefer P90 pooling in both axes to retain highlights
            y_methods = [self._p90] * n_y + [None] * (n_steps - n_y)
            x_methods = [self._p90] * n_x + [None] * (n_steps - n_x)
        else:
            # Original alternating mean/max scheme (bias towards preserving peaks on z; alternate in xy)
            y_seq = [np_max if i % 2 == 0 else np_mean for i in range(n_y)]
            x_seq = [np_mean if i % 2 == 0 else np_max for i in range(n_x)]
            y_methods = y_seq + [None] * (n_steps - n_y)
            x_methods = x_seq + [None] * (n_steps - n_x)

        self.down_sampling_methods = tuple(zip(y_methods, x_methods))

    def imsave_tif(self, path, img, compression=None):
        die = imsave_tif(path, img, compression=compression)
        if die:
            self.die = True

    def tif_save_path(self, idx: int, images: Any, flip_z: bool = False) -> Path:
        """Build output path; works for TSV/IMS/list sources without strict typing."""
        # ensure we have a count for naming
        total = self.images_count
        if total <= 0:
            try:
                total = len(images)  # type: ignore[arg-type]
            except Exception as e:
                print(f"{PrintColors.WARNING}Exception during len(images): {e}{PrintColors.ENDC}")
                total = 0

        if self.is_tsv or self.is_ims or self.rename:
            if flip_z and total > 0:
                return self.save_path / f"{self.tif_prefix}_{(total - idx - 1):06}.tif"
            else:
                return self.save_path / f"{self.tif_prefix}_{idx:06}.tif"
        else:
            file = Path(images[idx])  # type: ignore[index]
            if file.suffix.lower() in (".png", ".raw"):
                return self.save_path / file.with_suffix(".tif")
            else:
                return self.save_path / file.name

    def free_ram_is_not_enough(self) -> bool:
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
        pool = ProcessPoolExecutor(max_workers=1) if timeout else ThreadPoolExecutor(max_workers=1)
        args = self.args
        kwargs = self.kwargs
        tif_prefix = self.tif_prefix
        channel = self.channel
        resume = self.resume
        save_images = self.save_images
        compression = self.compression
        down_sampled_path = self.down_sampled_path
        d_type = self.d_type
        post_processed_d_type = self.d_type
        shape = self.shape
        crop_x0 = crop_x1 = crop_y0 = crop_y1 = None
        x0 = y0 = x1 = y1 = 0  # only used for TSV
        if self.crop_bbox is not None:
            crop_y0, crop_y1, crop_x0, crop_x1 = self.crop_bbox
            shape = (crop_y1 - crop_y0, crop_x1 - crop_x0)
        rotation = self.rotation
        post_processed_shape = self.shape
        if rotation in (90, 270):
            post_processed_shape = (shape[1], shape[0])

        need_down_sampling = False
        down_sampling_method_z = None
        if self.source_voxel is not None and self.target_voxel is not None and shape is not None:
            need_down_sampling = True
            reduction_factor_z = ceil(sqrt(self.target_voxel / self.source_voxel[0]))
            down_sampling_method_z = tuple(np_max if i % 2 == 0 else np_mean for i in range(reduction_factor_z))

        flip_x = flip_y = flip_z = False

        # TSV global extents for VExtent mapping
        if is_tsv:
            x0, x1, y0, y1 = images.volume.x0, images.volume.x1, images.volume.y0, images.volume.y1
            # name count for reverse indexing if needed
            self.images_count = int(images.volume.z1 - images.volume.z0)
        if is_ims:
            images = ImarisZWrapper(images, timepoint=0, channel=channel)
            self.images_count = len(images)

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
                    if flip_z and self.images_count > 0:
                        down_sampled_tif_path = down_sampled_path / f"{tif_prefix}_{(self.images_count - idx_down_sampled - 1):06}.tif"
                    else:
                        down_sampled_tif_path = down_sampled_path / f"{tif_prefix}_{idx_down_sampled:06}.tif"
                    if resume and down_sampled_tif_path.exists():
                        exist_count = 0
                        for idx_z, idx in enumerate(indices):
                            if self.tif_save_path(idx, images, flip_z=flip_z).exists():
                                exist_count += 1
                        if len(indices) == exist_count:
                            for _ in range(exist_count):
                                self.progress_queue.put(running_next)
                            continue

                    if self.target_shape is None:
                        # safety: should be set earlier; fall back to current shape
                        self.target_shape = (shape[0], shape[1])
                    z_stack = zeros((len(indices),) + self.target_shape, dtype=float32)

                for idx_z, idx in enumerate(indices):
                    if self.die:
                        break
                    while self.free_ram_is_not_enough():
                        continue
                    if self.die:
                        break

                    tif_save_path = self.tif_save_path(idx, images, flip_z=flip_z)
                    if resume and tif_save_path.exists() and not need_down_sampling:
                        continue
                    try:
                        if resume and tif_save_path.exists():
                            img = None
                            if need_down_sampling:
                                img = imread_tif_raw_png(tif_save_path)
                        else:
                            if is_ims:
                                img = images[idx]  # wrapper already corrected orientation (black box)
                            else:
                                start_time = time()
                                if is_tsv:
                                    # TSV: crop at read-time if requested
                                    if (crop_x0 is not None) and (crop_x1 is not None) and (crop_y0 is not None) and (crop_y1 is not None):
                                        vx0 = x0 + crop_x0
                                        vx1 = x0 + crop_x1
                                        vy0 = y0 + crop_y0
                                        vy1 = y0 + crop_y1
                                        future = pool.submit(
                                            imread_tsv, images, VExtent(vx0, vx1, vy0, vy1, idx, idx + 1), d_type
                                        )
                                    else:
                                        future = pool.submit(
                                            imread_tsv, images, VExtent(x0, x1, y0, y1, idx, idx + 1), d_type
                                        )
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

                            # Apply crop post-read (IMS uses wrapper orientation; TSV already cropped at read)
                            if (crop_x0 is not None) and (crop_x1 is not None) and (crop_y0 is not None) and (crop_y1 is not None):
                                img = crop_2d_image_px(img, crop_x0, crop_x1, crop_y0, crop_y1)

                            # rotations
                            if rotation == 90:
                                img = rot90(img, 1)
                            elif rotation == 180:
                                img = rot90(img, 2)
                            elif rotation == 270:
                                img = rot90(img, 3)

                            # optional flips for non-IMS sources (IMS already flipped in wrapper)
                            if flip_x:
                                img = flip(img, axis=1)
                            if flip_y:
                                img = flip(img, axis=0)

                            # save
                            if save_images and (is_tsv or is_ims or function is not None or rotation in (90, 180, 270) or self.crop_bbox is not None):
                                self.imsave_tif(tif_save_path, img, compression=compression)
                            if img.dtype != post_processed_d_type:
                                post_processed_d_type = img.dtype

                            # update post-processed shape, recalc target if needed
                            if rotation in (90, 270) or img.shape != post_processed_shape:
                                post_processed_shape = img.shape
                                if need_down_sampling:
                                    self.calculate_down_sampling_target(
                                        post_processed_shape, rotation in (90, 270),
                                        self.alternating_downsampling_method
                                    )
                                    z_stack = zeros((len(indices),) + self.target_shape, dtype=float32)

                        # XY downsampling into z_stack
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

                # approximate Z downsampling (coarse)
                if need_down_sampling and down_sampling_method_z is not None and z_stack is not None:
                    if is_uniform_3d(z_stack):
                        self.imsave_tif(down_sampled_tif_path, zeros(self.target_shape, dtype=float32),
                                        compression=compression)
                    else:
                        for z_method in down_sampling_method_z:
                            if z_method is not None and z_stack.shape[0] > 1:
                                z_stack = block_reduce(z_stack, block_size=(2, 1, 1), func=z_method)
                        assert z_stack.shape[0] == 1
                        img = z_stack[0]
                        if self.down_sampled_dtype not in (float32, "float32"):
                            if self.down_sampled_dtype in (uint16, "uint16"):
                                img = convert_to_16bit_fun(img)
                            elif self.down_sampled_dtype in (uint8, "uint8"):
                                if post_processed_d_type in (uint8, "uint8"):
                                    img = img.astype(uint8)
                                else:
                                    img = convert_to_8bit_fun(img)
                            else:
                                print(f"{PrintColors.FAIL}requested downsampled format is not supported{PrintColors.ENDC}")
                                raise RuntimeError
                        self.imsave_tif(down_sampled_tif_path, img, compression=compression)

            except (Empty, TimeoutError):
                self.die = True

        if is_ims and isinstance(images, ImarisZWrapper):
            images.close()
        if isinstance(pool, ProcessPoolExecutor):
            pool.shutdown()
        self.progress_queue.put(not running_next)


def calculate_downsampling_z_ranges(start: int, end: int, steps: int):
    z_list_list = []
    for idx in range(start, end, steps):
        z_range = list(range(idx, idx + steps))
        if z_range[-1] > end:
            while z_range and z_range[-1] >= end:
                del z_range[-1]
        z_list_list += [z_range]
    return z_list_list


def generate_voxel_spacing(
        shape: Tuple[int, int, int],
        source_voxel: Tuple[float, float, float],
        target_shape: Tuple[int, int, int],
        target_voxel: float):
    voxel_locations = [arange(axis_shape) * axis_v_size - (axis_shape - 1) /
                       2.0 * axis_v_size for axis_shape, axis_v_size in zip(shape, source_voxel)]
    axis_spacing = []
    for i, axis_vals in enumerate(voxel_locations):
        start = np_round(resize_local_mean(axis_vals, (int(target_shape[i]),)))[0]
        axis_spacing.append(array([start + target_voxel * val for val in range(target_shape[i])]))
    return axis_spacing


def jumpy_step_range(start: int, end: int):
    distance = end - start
    steps = [1]
    while distance / steps[-1] > 0:
        steps.append(steps[-1] * 10)
    steps.reverse()
    top_list = []
    for step in steps:
        for idx in range(start, end, step):
            if idx not in top_list:
                top_list.append(idx)
    return top_list


def parallel_image_processor(
        source: Union[TSVVolume, Path, str],
        destination: Union[Path, str],
        fun: Optional[Callable[..., ndarray]] = None,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
        rename: bool = False,
        tif_prefix: str = "img",
        channel: int = 0,
        source_voxel: Optional[Tuple[float, float, float]] = None,
        target_voxel: Optional[Union[int, float]] = None,
        downsampled_path: Optional[Path] = None,
        down_sampled_dtype: Union[str, type] = "float32",
        alternating_downsampling_method: Union[bool, Literal["contrast"]] = True,
        rotation: int = 0,
        timeout: Optional[float] = None,
        max_processors: int = cpu_count(logical=False),
        progress_bar_name: str = " ImgProc",
        compression: Tuple[str, int] = ("ADOBE_DEFLATE", 1),
        resume: bool = True,
        needed_memory: Optional[int] = None,
        save_images: bool = True,
        return_downsampled_path: bool = False,
        crop_bbox: Optional[tuple] = None
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
    downsampled_path: Path
        path to save the downsampled image. If None destination path will be used.
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
    crop_bbox: (x_start, x_end, y_start, y_end, z_start, z_end)
    """
    if isinstance(source, str):
        source = Path(source)
    if isinstance(destination, str):
        destination = Path(destination)
    if destination is not None:
        destination.mkdir(exist_ok=True)
        print(f"Modifying destination: {destination}")
    if isinstance(downsampled_path, str):
        downsampled_path = Path(downsampled_path)
    downsampled_path = destination if downsampled_path is None else downsampled_path

    # Initialize bbox vars (avoid “maybe unassigned”)
    x_start = x_end = y_start = y_end = z_start = z_end = None

    down_sampling_z_steps: int = 1
    need_down_sampling: bool = False
    if source_voxel is not None and target_voxel is not None:
        need_down_sampling = True
        down_sampling_z_steps = max(1, floor(float(target_voxel) / float(source_voxel[0])))

    args_queue = Queue()
    dtype: Union[str, Any] = "float32"  # default dtype; overwritten once we probe data

    if isinstance(source, TSVVolume):
        images = source
        full_shape = images.volume.shape[1:3]  # (Y, X)
        dtype = images.dtype
        z0, z1 = images.volume.z0, images.volume.z1

        # Defaults: full vol
        z_start, z_end = z0, z1
        shape = full_shape

        if crop_bbox is not None:
            x_start, x_end, y_start, y_end, z_s, z_e = crop_bbox
            z_start = max(z0, z_s)
            z_end = min(z1, z_e)
            shape = (y_end - y_start, x_end - x_start)

        num_images = int(z_end - z_start)

        if need_down_sampling and down_sampling_z_steps > 1:
            for ds_z_idx, z_range in enumerate(
                    calculate_downsampling_z_ranges(int(z_start), int(z_end), int(down_sampling_z_steps))):
                args_queue.put((ds_z_idx, z_range))
        else:
            for idx in jumpy_step_range(int(z_start), int(z_end)):
                args_queue.put((idx, [idx]))

    elif isinstance(source, Path) and source.is_file() and source.suffix.lower() == ".ims":
        print(f"ims file detected. using imaris_ims_file_reader!")
        with ImarisZWrapper(source, timepoint=0, channel=channel) as ims_wrapper:
            num_images_total = len(ims_wrapper)
            img0 = ims_wrapper[0]
            base_shape = img0.shape
            dtype = img0.dtype

        z_start, z_end = 0, num_images_total
        shape = base_shape

        if crop_bbox is not None:
            x_start, x_end, y_start, y_end, z_s, z_e = crop_bbox
            z_start = max(0, z_s)
            z_end = min(num_images_total, z_e)
            shape = (y_end - y_start, x_end - x_start)

        num_images = int(z_end - z_start)

        if need_down_sampling and down_sampling_z_steps > 1:
            for ds_z_idx, z_range in enumerate(calculate_downsampling_z_ranges(int(z_start), int(z_end), int(down_sampling_z_steps))):
                args_queue.put((ds_z_idx, z_range))
        else:
            for idx in range(int(z_start), int(z_end)):
                args_queue.put((idx, [idx]))
        images = str(source)

    elif isinstance(source, Path) and source.is_dir():
        images_list = natural_sorted([str(f) for f in source.iterdir() if f.is_file() and f.suffix.lower() in (
            ".tif", ".tiff", ".raw", ".png")])
        if not images_list:
            raise RuntimeError("No supported images found in the source directory.")
        num_images_total = len(images_list)
        img = imread_tif_raw_png(Path(images_list[0]))
        shape = img.shape
        dtype = img.dtype
        if crop_bbox is not None:
            x_start, x_end, y_start, y_end, z_start, z_end = crop_bbox
            images_list = images_list[z_start: z_end]
            print(f"Processing {len(images_list)} of {num_images_total} files (Z-range: {z_start}-{z_end})")
            num_images = len(images_list)
            shape = (y_end - y_start, x_end - x_start)
        else:
            num_images = num_images_total
        assert num_images > 0

        if need_down_sampling and down_sampling_z_steps > 1:
            for ds_z_idx, z_range in enumerate(calculate_downsampling_z_ranges(0, num_images, int(down_sampling_z_steps))):
                args_queue.put((ds_z_idx, z_range))
        else:
            for idx in range(num_images):
                args_queue.put((idx, [idx]))

        manager = Manager()
        images = manager.list(images_list)
        del img
    else:
        print("source can be either a tsv volume, an ims file path, or a 2D tiff series folder")
        raise RuntimeError

    if need_down_sampling:
        shape_3d = array((num_images,) + shape)
        new_source_voxel = source_voxel
        if rotation in (90, 270):
            shape_3d = array((num_images, shape[1], shape[0]))
            new_source_voxel = (source_voxel[0], source_voxel[2], source_voxel[1])  # type: ignore[index]

        reduction_times = float(target_voxel) / array(new_source_voxel)  # type: ignore[arg-type]
        target_shape_arr = shape_3d / reduction_times
        target_shape_remainder = target_shape_arr - target_shape_arr.round()
        target_voxel_actual = maximum(float(target_voxel) + target_shape_remainder / target_shape_arr.round(), new_source_voxel)  # type: ignore[arg-type]
        print(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
              f"{PrintColors.BLUE}down-sampling: {PrintColors.ENDC}\n"
              f"\tpost-processed shape zyx: {' '.join(target_shape_arr.round(0).astype(str))}\n"
              f"\tactual voxel sizes zyx: {' '.join(array(target_voxel_actual).round(3).astype(str))}")

        downsampled_path /= (f"{destination.stem}_z{down_sampling_z_steps * new_source_voxel[0]:.1f}_yx{float(target_voxel):.1f}um")  # type: ignore[index]
        downsampled_path.mkdir(exist_ok=True)
        print(f"Modifying downsampled_path: {downsampled_path}")
        if os.name == 'nt':
            os.chmod(downsampled_path, 0o666)
        else:
            os.chmod(downsampled_path, 0o777)

    initial = 0
    if resume:
        initial = sum(1 for _ in destination.glob("*.tif"))

    progress_queue = Queue()
    semaphore = Queue()
    semaphore.put(1)

    # avoid negative worker count
    total_tasks = args_queue.qsize()
    remaining = max(0, total_tasks - initial)
    workers = min(max_processors, remaining) if remaining > 0 else 0
    print(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}starting workers ...")
    for worker in tqdm(range(workers), desc=' workers'):
        if progress_queue.qsize() + worker < total_tasks:
            MultiProcess(
                progress_queue, args_queue, semaphore, fun, images, destination, args, kwargs, shape, dtype,
                rename=rename, tif_prefix=tif_prefix,
                source_voxel=source_voxel, target_voxel=target_voxel, down_sampled_path=downsampled_path,
                rotation=rotation, channel=channel, timeout=timeout, compression=compression, resume=resume,
                needed_memory=needed_memory, save_images=save_images,
                alternating_downsampling_method=alternating_downsampling_method,
                down_sampled_dtype=down_sampled_dtype,
                crop_bbox=(y_start, y_end, x_start, x_end) if crop_bbox is not None else None,
            ).start()
        else:
            print('\n the existing workers can finish the job! no more workers are needed.')
            workers = worker
            break

    # Use num_images for the total expected output planes to keep progress consistent
    return_code = progress_manager(progress_queue, workers, num_images, desc=progress_bar_name, initial=initial)
    args_queue.cancel_join_thread()
    args_queue.close()
    progress_queue.cancel_join_thread()
    progress_queue.close()

    # Accurate Z downsampling after XY
    if return_code == 0 and need_down_sampling:
        npz_file = downsampled_path.parent / f"{destination.stem}_zyx{float(target_voxel):.1f}um.npz"
        if resume and npz_file.exists():
            if return_downsampled_path:
                return return_code, downsampled_path
            return return_code

        print(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
              f"{PrintColors.BLUE}down-sampling: {PrintColors.ENDC}resizing on the z-axis accurately ...")

        target_shape_3d: Tuple[int, int, int] = (
            int(round(num_images / (float(target_voxel) / float(source_voxel[0])))),               # type: ignore[index]
            int(round(shape[0] / (float(target_voxel) / float(source_voxel[1])))),               # type: ignore[index]
            int(round(shape[1] / (float(target_voxel) / float(source_voxel[2]))))                # type: ignore[index]
        )
        if rotation in (90, 270):
            target_shape_3d = (target_shape_3d[0], target_shape_3d[2], target_shape_3d[1])

        files = sorted(downsampled_path.glob("*.tif"))
        print(f"Debug: Number of files loaded = {len(files)}")
        print(f"Debug: path used: {downsampled_path}")

        with ThreadPoolExecutor(max_workers=max_processors) as pool:
            img_stack = list(pool.map(imread_tif_raw_png, tqdm(files, desc="loading", unit="images")))
            img_stack = dstack(img_stack)               # y x z (yxz)
            img_stack = rollaxis(img_stack, -1)         # z y x

            print(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
                  f"{PrintColors.BLUE}down-sampling: {PrintColors.ENDC}resizing the z-axis ...")
            img_stack = resize(img_stack, target_shape_3d, preserve_range=True, anti_aliasing=True)
            axes_spacing = generate_voxel_spacing(
                (num_images, shape[0], shape[1]),
                source_voxel,
                target_shape_3d,
                float(target_voxel)
            )
            print(f"{PrintColors.GREEN}{date_time_now()}:{PrintColors.ENDC}"
                  f"{PrintColors.BLUE} down-sampling: {PrintColors.ENDC}saving as npz.")
            if npz_file.exists():
                stat_info = os.stat(npz_file)
                permissions = oct(stat_info.st_mode)[-3:]
                if permissions != '666':
                    print(f"Permissions for '{npz_file}' are {permissions}. Must update permissions...")
                    print(f"Modifying npz file: {npz_file}")
                    if os.name == 'nt':
                        os.chmod(npz_file, 0o666)
                    else:
                        os.chmod(npz_file, 0o777)
                else:
                    print(f"Permissions for '{npz_file}' are correctly set to 777.")
            else:
                print("Permission edit skipped")

            savez_compressed(
                npz_file,
                I=img_stack,
                xI=array(axes_spacing, dtype='object')  # avoid ragged warning
            )

    if return_downsampled_path:
        return return_code, downsampled_path
    return return_code


if __name__ == '__main__':
    freeze_support()
