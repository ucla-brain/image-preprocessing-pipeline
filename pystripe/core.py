import os
# Generic BLAS/OpenMP stacks used by NumPy/ SciPy / scikit-image / OpenCV builds
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1") # Apple Accelerate
os.environ.setdefault("BLIS_NUM_THREADS", "1")

# NumExpr (picked up at import)
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Optional: reduce busy-wait CPU burn on Intel OpenMP/MKL
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

# OpenCV: prefer single-thread and disable OpenCL via env when possible
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")
os.environ.setdefault("OPENCV_FOR_THREADS_NUM", "1")

import sys
import shutil
import subprocess
from argparse import RawDescriptionHelpFormatter, ArgumentParser
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, TimeoutError, as_completed
from concurrent.futures.process import BrokenProcessPool
from functools import reduce
from itertools import cycle, count
from math import ceil, log, sqrt
from multiprocessing import Process, Queue
from operator import iconcat
from os import scandir, DirEntry
from pathlib import Path
from queue import Empty, Full, SimpleQueue
from re import compile, IGNORECASE
from time import sleep, time
from types import GeneratorType
from typing import Tuple, Iterator, List, Union, Any, Literal, Dict
from warnings import filterwarnings

from cv2 import morphologyEx, MORPH_CLOSE, MORPH_OPEN, floodFill, GaussianBlur
from dcimg import DCIMGFile
from imageio.v3 import imread as iio_imread
from numba import jit
from numexpr import evaluate
from numpy import (uint8, uint16, float32, float64, iinfo, ndarray, generic, broadcast_to, exp, expm1, log1p, tanh,
                   zeros, ones, arange, pad, clip, where, rot90, flipud, dot, reshape, nonzero, ascontiguousarray,
                   logical_not, rint, array, asarray, min as np_min, median as np_median, mean as np_mean,
                   max as np_max, dtype as np_d_type)
from psutil import cpu_count
from ptwt import wavedec2 as pt_wavedec2, waverec2 as pt_waverec2
from pywt import wavedec2, waverec2, Wavelet, dwt_max_level
from numpy.fft import rfft, irfft
from scipy.signal import butter, sosfiltfilt
from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage.measure import block_reduce
from skimage.transform import resize
from tifffile import imwrite
from tifffile.tifffile import TiffFileError
from torch import (Tensor, as_tensor, set_grad_enabled, arange as pt_arange, broadcast_to as pt_broadcast_to,
                   exp as pt_exp, float32 as pt_float32, reshape as pt_reshape)
from torch.cuda import (is_available as cuda_is_available_for_pt, device_count as cuda_device_count,
                        device as cuda_device, set_device as cuda_set_device, get_device_name as cuda_get_device_name,
                        mem_get_info as cuda_mem_get_info, Stream, stream as cuda_stream)
from torch.fft import irfft as pt_irfft, rfft as pt_rfft
from tqdm import tqdm
from PIL import Image
from dataclasses import dataclass
Image.MAX_IMAGE_PIXELS = None

from pystripe.lightsheet_correct import correct_lightsheet, prctl
from pystripe.raw import raw_imread
from supplements.cli_interface import PrintColors, date_time_now

filterwarnings("ignore")
SUPPORTED_EXTENSIONS = ('.png', '.tif', '.tiff', '.raw', '.dcimg')
NUM_RETRIES: int = 40
USE_NUMEXPR: bool = True
USE_PYTORCH = True
CUDA_IS_AVAILABLE_FOR_PT = cuda_is_available_for_pt()
STREAMS_PER_GPU = 2   # tune: 2–4 is usually enough
_STREAM_CYCLES: Dict[str, cycle] = {}
set_grad_enabled(False)
# if sys.platform.lower() == "linux":
#     USE_PYTORCH = False
#     CUDA_IS_AVAILABLE_FOR_PT = False
PADDING_MODE: str = None
DWT_MODE: str = None


@dataclass(frozen=True)
class DeviceInfo:
    id: int
    name: str
    total: int   # bytes
    free: int    # bytes


def discover_cuda_devices() -> List[DeviceInfo]:
    if not cuda_is_available_for_pt():
        return []
    devices: List[DeviceInfo] = []
    for i in range(cuda_device_count()):
        with cuda_device(i):
            free, total = cuda_mem_get_info()
        name = cuda_get_device_name(i)
        devices.append(DeviceInfo(id=i, name=name, total=total, free=free))
    # Sort by free memory desc so we schedule heavier work first
    devices.sort(key=lambda d: d.free, reverse=True)
    return devices


def _next_stream(device_str: str) -> Stream:
    """Round-robin a small pool of streams per device."""
    cyc = _STREAM_CYCLES.get(device_str)
    if cyc is None:
        cuda_set_device(device_str)
        streams = [Stream() for _ in range(STREAMS_PER_GPU)]
        cyc = cycle(streams)
        _STREAM_CYCLES[device_str] = cyc
    return next(cyc)


@jit(nopython=True)
def is_uniform_1d(arr: ndarray) -> Union[bool, None]:
    n: int = len(arr)
    if n <= 0:
        return None
    y = arr[0]
    for x in arr:
        if x != y:
            return False
    return True


@jit(nopython=True)
def is_uniform_2d(arr: ndarray) -> Union[bool, None]:
    n: int = len(arr)
    if n <= 0:
        return None
    is_uniform: bool = is_uniform_1d(arr[0])
    if not is_uniform:
        return False
    val = arr[0, 0]
    for i in range(1, n):
        is_uniform = is_uniform_1d(arr[i])
        if not is_uniform:
            return False
        elif val != arr[i, 0]:
            return False
    return is_uniform


@jit(nopython=True)
def is_uniform_3d(arr: ndarray) -> Union[bool, None]:
    n: int = len(arr)
    if n <= 0:
        return None
    is_uniform: bool = is_uniform_2d(arr[0])
    if not is_uniform:
        return False
    val = arr[0, 0, 0]
    for i in range(1, n):
        is_uniform = is_uniform_2d(arr[i])
        if not is_uniform:
            return False
        elif val != arr[i, 0, 0]:
            return False
    return is_uniform


@jit(nopython=True)
def min_max_1d(arr: ndarray) -> Tuple[Any, Any]:
    n: int = len(arr)
    if n <= 0:
        return None, None
    max_val = min_val = arr[0]
    odd = n % 2
    if not odd:
        n -= 1
    i = 1
    while i < n:
        x = arr[i]
        y = arr[i + 1]
        if x > y:
            x, y = y, x
        min_val = min(x, min_val)
        max_val = max(y, max_val)
        i += 2
    if not odd:
        x = arr[n]
        min_val = min(x, min_val)
        max_val = max(x, max_val)
    return min_val, max_val


@jit(nopython=True)
def min_max_2d(arr: ndarray) -> Tuple[Any, Any]:
    n: int = len(arr)
    if n <= 0:
        return None, None
    min_val, max_val = min_max_1d(arr[0])
    for i in range(1, n):
        x, y = min_max_1d(arr[i])
        min_val = min(x, min_val)
        max_val = max(y, max_val)
    return min_val, max_val


def expm1_jit(img: Union[ndarray, float, int], dtype=float32) -> ndarray:
    if USE_NUMEXPR and (img.flags.c_contiguous or img.flags.f_contiguous):
        if img.dtype != dtype:
            img = img.astype(dtype)
        evaluate("expm1(img)", out=img, casting="unsafe")
    else:
        img = expm1(img).astype(dtype)
    return img


def log1p_jit(img: ndarray, dtype=float32):
    if USE_NUMEXPR and (img.flags.c_contiguous or img.flags.f_contiguous):
        if img.dtype != dtype:
            img = img.astype(dtype)
        evaluate("log1p(img)", out=img, casting="unsafe")
    else:
        img = log1p(img, dtype=dtype)
    return img


def max_level(min_len, wavelet):
    w = Wavelet(wavelet)
    return dwt_max_level(min_len, w.dec_len)


def slice_non_zero_box(img_axis: ndarray, noise: int, filter_frequency: int = 1 / 1000) -> (int, int):
    return min_max_1d(nonzero(butter_lowpass_filter(img_axis, filter_frequency).astype(uint16) > noise)[0])


def get_img_mask(img: ndarray, threshold: Union[int, float],
                 close_steps: int = 50, open_steps: int = 500, flood_fill_flag: int = 4) -> ndarray:
    # start_time = time()
    shape = list(img.shape)
    mask = (img > threshold).astype(uint8)
    mask = morphologyEx(mask, MORPH_CLOSE, ones((close_steps, close_steps), dtype=uint8))
    mask = morphologyEx(mask, MORPH_OPEN, ones((open_steps, open_steps), dtype=uint8)).astype(bool)
    inverted_mask = logical_not(mask).astype(uint8)
    floodFill(inverted_mask, None, (0, 0), 0, flags=flood_fill_flag)
    floodFill(inverted_mask, None, (0, shape[0] - 1), 0, flags=flood_fill_flag)
    floodFill(inverted_mask, None, (shape[1] - 1, 0), 0, flags=flood_fill_flag)
    floodFill(inverted_mask, None, (shape[1] - 1, shape[0] - 1), 0, flags=flood_fill_flag)
    mask |= inverted_mask.astype(bool)
    # print("mask time:", time() - start_time)
    return mask


def butter_lowpass_filter(img: ndarray, cutoff_frequency: float, order: int = 1) -> ndarray:
    d_type = img.dtype
    sos = butter(order, cutoff_frequency, output='sos')
    img = sosfiltfilt(sos, img)  # returns float64
    img = img.astype(d_type)

    return img


def correct_bleaching(
        img: ndarray,
        frequency: float,
        clip_min: float,
        clip_med: float,
        clip_max: float,
        max_method: bool = False,
) -> ndarray:
    """
    Parameters
    ----------
    img: log1p filtered image
    frequency: low pass fileter frequency. Usually 1/min(img.shape).
    clip_min: background vs foreground threshold
    clip_med: foreground intermediate
    clip_max: foreground max
    max_method: use max value on x and y axes to create the filter. Max method is faster and smoother but less accurate

    Returns
    -------
    max normalized filter values.
    """

    assert isinstance(frequency, (float, float32, float64)) and frequency > 0
    assert isinstance(clip_min, (float, float32, float64)) and clip_min >= 0
    assert isinstance(clip_med, (float, float32, float64)) and clip_med > clip_min
    assert isinstance(clip_max, (float, float32, float64)) and clip_max > clip_min
    assert clip_max > clip_med
    clip_min_lb = log1p(1)
    if clip_min < clip_min_lb:
        clip_min = clip_min_lb

    # creat the filter
    if max_method:
        img_filter_y = np_max(img, axis=1)
        img_filter_x = np_max(img, axis=0)
        img_filter_y[img_filter_y == 0] = clip_med
        img_filter_x[img_filter_x == 0] = clip_med
        img_filter_y = clip(img_filter_y, clip_min, clip_max)
        img_filter_x = clip(img_filter_x, clip_min, clip_max)
        img_filter_y = butter_lowpass_filter(img_filter_y, frequency)
        img_filter_x = butter_lowpass_filter(img_filter_x, frequency)
        img_filter_y = reshape(img_filter_y, (len(img_filter_y), 1))
        img_filter_x = reshape(img_filter_x, (1, len(img_filter_x)))
        img_filter = dot(img_filter_y, img_filter_x)
    else:
        img_filter = img.copy()
        img_filter[img_filter == 0] = clip_med
        clip(img_filter, clip_min, clip_max, out=img_filter)
        img_filter = butter_lowpass_filter(img_filter, frequency)

    # apply the filter
    img_filter_max = np_max(img_filter)
    if USE_NUMEXPR and (img.flags.c_contiguous or img.flags.f_contiguous):
        evaluate("img / img_filter * img_filter_max", out=img, casting="unsafe")
    else:
        img /= img_filter
        img *= img_filter_max
    return img


def otsu_threshold(img: ndarray) -> float:
    try:
        return threshold_otsu(img)
    except ValueError:
        return 2


def sigmoid(img: ndarray) -> ndarray:
    if img.dtype != float32:
        img = img.astype(float32)
    half = float32(.5)
    one = float32(1)
    if USE_NUMEXPR and (img.flags.c_contiguous or img.flags.f_contiguous):
        evaluate("half * (tanh(half * img) + one)", out=img, casting="unsafe")
    else:
        # img = .5 * (tanh(.5 * img) + 1)
        img *= half
        img = tanh(img)
        img += one
        img *= half
    return img


def foreground_fraction(img: ndarray, threshold: float, crossover: float, sigma: int) -> ndarray:
    if img.dtype != float32:
        ff = img.astype(float32)
    else:
        ff = img.copy()
    threshold = float32(threshold)
    crossover = float32(crossover)
    if USE_NUMEXPR and (ff.flags.c_contiguous or ff.flags.f_contiguous):
        evaluate("(ff - threshold) / crossover", out=ff, casting="unsafe")
    else:
        ff -= threshold
        ff /= crossover
    ff = sigmoid(ff)
    ksize = (sigma * 2 + 1,) * 2
    GaussianBlur(ff, ksize=ksize, sigmaX=sigma, sigmaY=sigma)
    return ff


def pt_notch(length: int, sigma: float, device: str = "cpu") -> Tensor:
    """Generates a 1D gaussian notch filter `n` pixels long

    Parameters
    ----------
    length : int
        length of the gaussian notch filter
    sigma : float
        notch width
    device : str, optional

    Returns
    -------
    g : Tensor
        (length,) array containing the gaussian notch filter

    """
    if length <= 0:
        raise ValueError('pt_notch: length must be positive')
    if sigma <= 0:
        raise ValueError('pt_notch: sigma must be positive')
    g = pt_arange(0, length, 1, dtype=pt_float32).to(device)
    g **= 2
    g /= -2 * sigma ** 2
    g = pt_exp(g)
    g = 1 - g
    return g


def np_notch(length: int, sigma: float) -> ndarray:
    """Generates a 1D gaussian notch filter `n` pixels long

    Parameters
    ----------
    length : int
        length of the gaussian notch filter
    sigma : float
        notch width

    Returns
    -------
    g : ndarray
        (length,) array containing the gaussian notch filter

    """
    if length <= 0:
        raise ValueError('np_notch: length must be positive')
    if sigma <= 0:
        raise ValueError('np_notch: sigma must be positive')
    g = arange(length, dtype=float32)
    one = float32(1)
    two = float32(2)
    if USE_NUMEXPR:
        evaluate("one - exp(-g ** 2 / (two * sigma ** 2))", out=g, casting='unsafe')
    else:
        g **= 2
        g /= -two * sigma ** 2
        g = exp(g)
        g = one - g
    return g


def notch_rise_point(sigma: int, rise: float):
    """ Compute length at which gaussian notch reaches the given rise point
    :param sigma: sigma of notch function
    :param rise: a vlue between 0 and 1.

    ----
    :return: length at which notch reaches the rise
    """
    return int(sqrt(-2 * sigma ** 2 * log(1 - rise)) + .5) // 2 * 2


def calculate_pad_size(shape: tuple, sigma: int, rise: float = 0.5):
    """ Calculate image padding based on padding size but make sure padded image fits into gpu memory
    :param shape: shape of image
    :param sigma: sigma of notch function
    :param rise: requested rise point of gaussian notch function.
        Since rfft output is symmetric 0.5 is optimum to avoid generating artifacts on the image.
        at 0.4 pad size will be equal to the sigma itself.
    ---
    :return: pad size
    """
    if (sigma == 0):
        return 0
    x = shape[1] + 1
    y = shape[0] + 1
    c = 5e14  # 2e15 for 8GB float32 image which needs ~40 GB of vRAM in pt_wavedec2
    sqrt_xyc = sqrt(x ** 2 - 2 * x * y + y ** 2 + 4 * c)
    rise = min(round(1 - exp((x + y - sqrt_xyc) / (4 * sigma ** 2)), 2) - 0.01, rise)
    return notch_rise_point(sigma, rise)


def np_gaussian_filter(shape: tuple, sigma: float, axis: int) -> ndarray:
    """Create a gaussian notch filter

    Parameters
    ----------
    shape: tuple
        shape of the output filter
    sigma: float
        filter bandwidth
    axis: int
        axis of the filter. options: -1 or -2

    Returns
    -------
    g : ndarray
        the impulse response of the gaussian notch filter

    """
    g = np_notch(length=shape[axis], sigma=sigma)
    if axis == -2:
        g = reshape(g, newshape=(shape[axis], 1))
    g = broadcast_to(g, shape)
    return g


def pt_gaussian_filter(shape: tuple, sigma: float, axis: int, device: cuda_device = "cpu") -> Tensor:
    """Create a gaussian notch filter
    Parameters
    ----------
    shape : tuple
        shape of the output filter
    sigma : float
        filter bandwidth
    axis: int
        axis of the filter. options: -1 or -2
    device : int, optional

    Returns
    ----------
    g : Tensor
        the impulse response of the gaussian notch filter
    """
    g = pt_notch(length=shape[axis], sigma=sigma, device=device)
    if axis == -2:
        g = pt_reshape(g, shape=(shape[axis], 1))
    g = pt_broadcast_to(g, shape)
    return g


def np_filter_coefficient(coef: ndarray, width_frac: float, axis=-1) -> ndarray:
    coef = ascontiguousarray(coef)
    coef_f = rfft(coef, axis=axis)
    n_freq = coef_f.shape[axis]
    sigma = n_freq * width_frac
    H, W = coef_f.shape[-2], coef_f.shape[-1]
    coef_f *= np_gaussian_filter(shape=(H, W), sigma=sigma, axis=axis)
    return irfft(coef_f, n=coef.shape[axis], axis=axis)


def pt_filter_coefficient(coef: Tensor, width_frac: float, axis: int = -1) -> Tensor:
    if axis not in (-1, -2):
        raise ValueError('axis must be -1 or -2')
    coef = coef.contiguous()
    coef_f = pt_rfft(coef, dim=axis)
    n_freq = coef_f.shape[axis]
    sigma = n_freq * width_frac
    H, W = coef_f.shape[-2], coef_f.shape[-1]
    filt = pt_gaussian_filter(shape=(H, W), sigma=sigma, axis=axis, device=coef.device)
    if coef_f.ndim == 3:
        filt = filt.unsqueeze(0)  # broadcast over batch
    coef_f *= filt
    return pt_irfft(coef_f, dim=axis, n=coef.shape[axis])  # preserve original length


def padding_mode_for_wavelet(wavelet_name: str) -> str:
    """
    Suggest a NumPy padding mode that matches a typical DWT extension mode
    for the given wavelet family.

    Parameters
    ----------
    wavelet_name : str
        e.g. 'db9', 'coif5', 'sym8', 'haar'

    Returns
    -------
    str
        NumPy padding mode: 'reflect', 'wrap', or 'symmetric'
    """
    wl = wavelet_name.lower().strip()

    # Families that often use periodization for exact reconstruction
    if wl.startswith(('coif', 'bior', 'rbio')):
        return 'wrap'  # matches 'periodization' in PyWavelets

    # Haar (db1) and most orthogonal wavelets default well to symmetric
    if wl.startswith(('haar', 'db', 'sym', 'fk', 'dmey')):
        return 'reflect'  # matches 'symmetric' mode in PyWavelets

    # Fallback
    return 'reflect'

PTWTMode = Literal["constant", "zero", "reflect", "periodic", "symmetric"]
PYWTMode = str  # PyWavelets allows more, but you could be stricter if you want
def dwt_mode_from_pad(pad_mode: str, for_ptwt: bool) -> Union[PTWTMode, PYWTMode]:
    pad_mode = pad_mode.lower()
    if pad_mode == "wrap":
        return "periodic" if for_ptwt else "periodization"
    return "symmetric"


def filter_subband(img: ndarray, sigma: float, level: int, wavelet: str,
                   gpu_semaphore: SimpleQueue, axes=-1) -> ndarray:
    level = None if level == 0 else level
    d_type = img.dtype
    img_shape = tuple(img.shape)

    global PADDING_MODE, DWT_MODE
    if DWT_MODE is None:
        DWT_MODE = dwt_mode_from_pad(PADDING_MODE, USE_PYTORCH)

    if isinstance(axes, int):
        axes = (axes,)

    device = "cpu"
    if USE_PYTORCH and CUDA_IS_AVAILABLE_FOR_PT and gpu_semaphore is not None:
        device = gpu_semaphore.get(block=True, timeout=None)

    if device != "cpu":
        stream = _next_stream(device)
        img = as_tensor(img, dtype=pt_float32)
        img = img.contiguous()
        img = img.pin_memory()
        with cuda_stream(stream):  # << use the proper context
            img = img.to(device, non_blocking=True)
            coefficients = list(pt_wavedec2(img, wavelet, mode=DWT_MODE, level=level, axes=(-2, -1)))
            for idx, c in enumerate(coefficients):
                if idx == 0:
                    continue
                coefficients[idx] = (
                    pt_filter_coefficient(c[0], sigma / img_shape[0], axis=-1) if -1 in axes else c[0],
                    pt_filter_coefficient(c[1], sigma / img_shape[1], axis=-2) if -2 in axes else c[1],
                    c[2]
                )
            img = pt_waverec2(coefficients, wavelet, axes=(-2, -1))
            img = img.contiguous()
            stream.synchronize()
            img = img.to("cpu", non_blocking=True).detach().numpy()
        if gpu_semaphore is not None:
            gpu_semaphore.put(device)
        img = img.astype(d_type, copy=False)
    else:
        coefficients = wavedec2(img, wavelet, mode=DWT_MODE, level=level, axes=(-2, -1))
        for idx, c in enumerate(coefficients):
            if idx == 0:
                continue
            coefficients[idx] = (
                np_filter_coefficient(c[0], sigma / img_shape[0], axis=-1) if -1 in axes else c[0],
                np_filter_coefficient(c[1], sigma / img_shape[1], axis=-2) if -2 in axes else c[1],
                c[2]
            )
        img = waverec2(coefficients, wavelet, mode=DWT_MODE, axes=(-2, -1)).astype(d_type)
    return img


def filter_streak_dual_band(img, sigma1, sigma2, level, wavelet, crossover, threshold, gpu_semaphore,
                            axes: Union[tuple, int] = -1, use_thresholding: bool = False):
    if (sigma1 > 0 and sigma1 == sigma2) or (threshold is not None and threshold <= 0):
        img = filter_subband(img, sigma1, level, wavelet, gpu_semaphore, axes=axes)
    elif use_thresholding:
        # this method is not compatible with log1p normalization
        smoothing: int = 1
        if threshold is None:
            threshold = otsu_threshold(img)

        foreground = img
        if sigma1 > 0:
            foreground = img.copy()
            clip(foreground, threshold, None, out=foreground)
            foreground = filter_subband(foreground, sigma1, level, wavelet, gpu_semaphore, axes=axes)

        background = img
        if sigma2 > 0:
            background = img.copy()
            clip(background, None, threshold, out=background)
            background = filter_subband(background, sigma2, level, wavelet, gpu_semaphore, axes=axes)

        fraction = foreground_fraction(img, threshold, crossover, smoothing)
        one = float32(1.0)
        if USE_NUMEXPR and (foreground.flags.c_contiguous or foreground.flags.f_contiguous) and (background.flags.c_contiguous or background.flags.f_contiguous):
            evaluate("(foreground * fraction + background * (one - fraction)) * threshold", out=img, casting="unsafe")
        else:
            foreground *= fraction
            fraction = one - fraction
            background *= fraction
            del fraction
            img = foreground + background
            img *= threshold
    else:
        img = filter_subband(img, sigma1, level, wavelet, gpu_semaphore, axes=axes)
        img = filter_subband(img, sigma2, level, wavelet, gpu_semaphore, axes=axes)
    return img


def filter_streaks(
        img: ndarray,
        sigma: Tuple[int, int] = (250, 250),
        level: int = 0,
        wavelet: str = 'db9',
        crossover: float = 10,
        threshold: float = None,
        padding_mode: str = None,
        bidirectional: bool = False,
        gpu_semaphore: SimpleQueue = None,
        bleach_correction_frequency: float = None,
        bleach_correction_max_method: bool = False,
        bleach_correction_clip_min: Union[float, int] = None,
        bleach_correction_clip_med: Union[float, int] = None,
        bleach_correction_clip_max: Union[float, int] = None,
        log1p_normalization_needed: bool = True,
        enable_masking: bool = False,
        close_steps: int = 50,
        open_steps: int = 500,
        verbose: bool = False
) -> ndarray:
    """Filter horizontal streaks using wavelet-FFT filter and apply bleach correction

    Parameters
    ----------
    img : ndarray
        input image array to filter
    sigma : tuple
        filter bandwidth(s) in pixels (larger gives more filtering)
    level : int
        number of wavelet levels to use. 0 means the maximum possible decimation.
    wavelet : str
        name of the mother wavelet
    crossover : float
        intensity range to switch between filtered background and unfiltered foreground
    threshold : float
        intensity value to separate background from foreground. Default is Otsu.
    padding_mode : str
        Padding method affects the edge artifacts. In some cases wrap method works better than reflect method.
    bidirectional : bool
        by default (False) only stripes elongated along horizontal axis will be corrected.
    gpu_semaphore: SimpleQueue
        Needed for multi-GPU processing and prevents overflowing GPUs vRAM
    bleach_correction_frequency : float
        2D bleach correction frequency in Hz. For stitched tiled images 1/tile_size is a suggested value.
    bleach_correction_max_method : bool
        use max value on x and y axes to create the filter. Max method is faster and smoother but less accurate
        for large images.
    bleach_correction_clip_min: float, int
        background vs foreground threshold.
    bleach_correction_clip_med: float, int
        foreground intermediate value
    bleach_correction_clip_max: float, int
        foreground max value
    log1p_normalization_needed: bool
        smooth the image using log plus 1 function.
        It should be true in most cases except for dual-band method combined with thresholding.
    enable_masking: bool
        Masking can clear the background for cases in which a large sample (for example a whole brain image) is
        in the middle and surrounded by a dark background. It is very hard to automate it in my testing.
    close_steps:
        morphological operation to close the holes (like ventricles) in the image.
    open_steps:
        morphological operation to clear the noise in the background.
    verbose:
        if true explain to user what's going on

    Returns
    -------
    img : ndarray
        filtered image
    """
    if not isinstance(sigma, (tuple, list)):
        sigma = (sigma, ) * 2
    sigma1 = sigma[0]  # foreground
    sigma2 = sigma[1]  # background
    if sigma1 == sigma2 == 0 and bleach_correction_frequency is None:
        return img

    global PADDING_MODE
    if PADDING_MODE is None:
        PADDING_MODE = padding_mode_for_wavelet(wavelet) if padding_mode is None else padding_mode

    # smooth the image using log plus 1 function
    d_type = img.dtype
    img_min, img_max = None, None
    if not np_d_type(d_type).kind in ("u", "i"):
        img_min, img_max = min_max_2d(img)
    if log1p_normalization_needed:
        img = log1p_jit(img, dtype=float32)

    if (bleach_correction_frequency is not None and
        (bleach_correction_clip_min is None or
         bleach_correction_clip_med is None or
         bleach_correction_clip_max is None)) or (enable_masking and bleach_correction_clip_med is None):

        lb, mb, ub = threshold_multiotsu(img, classes=4)
        if bleach_correction_clip_min is None:
            bleach_correction_clip_min = lb
        if bleach_correction_clip_med is None:
            bleach_correction_clip_med = mb
        if bleach_correction_clip_max is None:
            bleach_correction_clip_max = ub

    if enable_masking and close_steps is not None and open_steps is not None:
        img *= get_img_mask(img, bleach_correction_clip_med, close_steps=close_steps, open_steps=open_steps)

    if not sigma1 == sigma2 == 0:
        # Need to pad image to multiple of 2. It is needed even for bleach correction non-max method
        img_shape = img.shape
        pad_y, pad_x = [_ % 2 for _ in img_shape]
        if isinstance(PADDING_MODE, str):
            PADDING_MODE = PADDING_MODE.lower()
        if PADDING_MODE in ('constant', 'edge', 'linear_ramp', 'maximum', 'mean', 'median', 'minimum', 'reflect',
                            'symmetric', 'wrap', 'empty'):
            # base_pad = int(max(sigma1, sigma2) // 2) * 2
            base_pad = calculate_pad_size(shape=img_shape, sigma=max(sigma))
            min_image_length = 34  # tested for db9 to 37
            if (img_shape[0] + 2 * base_pad + pad_y) < min_image_length:
                pad_y = min_image_length - (img_shape[0] + 2 * base_pad)
            if (img_shape[1] + 2 * base_pad + pad_x) < min_image_length:
                pad_x = min_image_length - (img_shape[1] + 2 * base_pad)
        else:
            print(f"{PrintColors.FAIL}Unsupported padding mode: {PADDING_MODE}{PrintColors.ENDC}")
            raise RuntimeError
        if pad_y > 0 or pad_x > 0 or base_pad > 0:
            if PADDING_MODE == 'constant' and bleach_correction_clip_min is not None:
                img = pad(
                    img, ((base_pad, base_pad + pad_y), (base_pad, base_pad + pad_x)),
                    mode='constant', constant_values=log1p(bleach_correction_clip_min)
                )
            else:
                img = pad(
                    img, ((base_pad, base_pad + pad_y), (base_pad, base_pad + pad_x)),
                    mode=PADDING_MODE if PADDING_MODE else 'reflect'
                )

        if bidirectional:
            img = filter_streak_dual_band(
                img, sigma1, sigma2, level, wavelet, crossover, threshold, gpu_semaphore, axes=(-1, -2))
        else:
            img = filter_streak_dual_band(
                img, sigma1, sigma2, level, wavelet, crossover, threshold, gpu_semaphore, axes=-1)

        if verbose:
            print(f"de-striping applied: sigma={sigma}, level={level}, wavelet={wavelet}, crossover={crossover}, "
                  f"threshold={threshold}, bidirectional={bidirectional}.")

        # undo padding
        if pad_y > 0 or pad_x > 0 or base_pad > 0:
            img = img[
                  base_pad: img.shape[0] - (base_pad + pad_y),
                  base_pad: img.shape[1] - (base_pad + pad_x)]
            assert img.shape == img_shape

    if bleach_correction_frequency is not None:
        img = correct_bleaching(
            img,
            bleach_correction_frequency,
            bleach_correction_clip_min,
            bleach_correction_clip_med,
            bleach_correction_clip_max,
            max_method=bleach_correction_max_method
        )

        if verbose:
            print(
                f"bleach correction is applied: frequency={bleach_correction_frequency}, "
                f"max_method={bleach_correction_max_method},\n"
                f"clip_min={expm1(bleach_correction_clip_min)}, "
                f"clip_med={expm1(bleach_correction_clip_med)}, "
                f"clip_max={expm1(bleach_correction_clip_max)},\n"
            )

    # undo log plus 1
    if log1p_normalization_needed:
        img = expm1_jit(img)

    if np_d_type(d_type).kind in ("u", "i"):
        img = rint(img)
        d_type_info = iinfo(d_type)
        clip(img, d_type_info.min, d_type_info.max, out=img)
    else:
        img += img_min - np_min(img)
        clip(img, img_min, img_max, out=img)
    if img.dtype != d_type:
        img = img.astype(d_type)
    return img


def filter_streaks_3D(
        img: ndarray,
        sigma: int = 70,
        level: int = 0,
        wavelet: str = 'db9',
        crossover: float = 10,
        threshold: float = None,
        padding_mode: str = None,
        bidirectional: bool = False,
        bleach_correction_frequency: float = None,
        bleach_correction_max_method: bool = False,
        bleach_correction_clip_min: Union[float, int] = None,
        bleach_correction_clip_med: Union[float, int] = None,
        bleach_correction_clip_max: Union[float, int] = None,
        log1p_normalization_needed: bool = True,
        enable_masking: bool = False,
        close_steps: int = 50,
        open_steps: int = 500,
        verbose: bool = False,
        max_workers=None,
        threads_per_gpu: int = STREAMS_PER_GPU
):
    img = asarray(img)  # ensure ndarray / writeable
    assert img.ndim == 3, "img must be a 3D array"
    Z, Y, X = img.shape

    workers = max_workers or cpu_count(logical=False)
    gpu_semaphore = None
    if USE_PYTORCH:
        devices = discover_cuda_devices()
        if not devices:
            gpu_semaphore = None  # forces CPU path
        else:
            gpu_semaphore = SimpleQueue()
            for device in devices:
                for _ in range(threads_per_gpu):
                    gpu_semaphore.put(f"cuda:{device.id}")
        # for _ in range(max(0, workers - threads_per_gpu * len(devices))):
        #     gpu_semaphore.put("cpu")

    # per-slice op
    func = lambda slice2d: slice2d if is_uniform_2d(slice2d) else filter_streaks(
        slice2d,
        sigma=(sigma, sigma),
        level=level,
        wavelet=wavelet,
        crossover=crossover,
        threshold=threshold,
        padding_mode=padding_mode,
        bidirectional=bidirectional,
        gpu_semaphore=gpu_semaphore,
        bleach_correction_frequency=bleach_correction_frequency,
        bleach_correction_max_method=bleach_correction_max_method,
        bleach_correction_clip_min=bleach_correction_clip_min,
        bleach_correction_clip_med=bleach_correction_clip_med,
        bleach_correction_clip_max=bleach_correction_clip_max,
        log1p_normalization_needed=log1p_normalization_needed,
        enable_masking=enable_masking,
        close_steps=close_steps,
        open_steps=open_steps,
        verbose=verbose
    )

    _worker_counter = count()

    def _set_affinity_cross_platform(worker_index: int) -> None:
        """Best-effort: pin current worker across sockets/groups. Never raise."""
        try:
            if sys.platform == "win32":
                import ctypes as ct
                WORD = ct.c_ushort
                DWORD = ct.c_uint32
                KAFFINITY = ct.c_uint64  # 64-bit mask per processor group

                class GROUP_AFFINITY(ct.Structure):
                    _fields_ = [
                        ("Mask", KAFFINITY),
                        ("Group", WORD),
                        ("Reserved", WORD * 3),
                    ]

                k32 = ct.WinDLL("kernel32", use_last_error=True)

                GetCurrentThread = k32.GetCurrentThread
                GetCurrentThread.restype = ct.c_void_p

                GetActiveProcessorGroupCount = k32.GetActiveProcessorGroupCount
                GetActiveProcessorGroupCount.restype = WORD

                GetActiveProcessorCount = k32.GetActiveProcessorCount
                GetActiveProcessorCount.argtypes = [WORD]
                GetActiveProcessorCount.restype = DWORD

                SetThreadGroupAffinity = k32.SetThreadGroupAffinity
                SetThreadGroupAffinity.argtypes = [
                    ct.c_void_p,
                    ct.POINTER(GROUP_AFFINITY),
                    ct.POINTER(GROUP_AFFINITY),
                ]
                SetThreadGroupAffinity.restype = ct.c_bool

                group_count = int(GetActiveProcessorGroupCount() or 0)
                if group_count <= 0:
                    return

                group = worker_index % group_count
                cpus_in_group = int(GetActiveProcessorCount(WORD(group)) or 0)
                if cpus_in_group <= 0:
                    return

                cpu_in_group = (worker_index // max(1, group_count)) % cpus_in_group

                ga = GROUP_AFFINITY()
                ga.Group = WORD(group)
                ga.Mask = KAFFINITY(1 << cpu_in_group)
                ga.Reserved = (WORD * 3)(0, 0, 0)

                ok = SetThreadGroupAffinity(GetCurrentThread(), ct.byref(ga), None)
                if not ok:
                    # debug (optional): print(GetLastError) then fall back
                    # err = ct.get_last_error(); print(f"SetThreadGroupAffinity failed: {err}")
                    return

            elif sys.platform.startswith("linux"):
                try:
                    import psutil
                    psutil.Process(os.getpid()).cpu_affinity(list(range(psutil.cpu_count())))
                except Exception:
                    pass  # best effort
            else:
                # macOS/others: no-op
                pass
        except Exception:
            # Absolutely never let the initializer raise → avoids BrokenThreadPool
            return

    def _affinity_initializer():
        # assign a unique index to each thread at start-up
        idx = next(_worker_counter)
        _set_affinity_cross_platform(idx)

    def _batches(n_items, batch_size):
        for start in range(0, n_items, batch_size):
            stop = min(start + batch_size, n_items)
            yield range(start, stop)

    def _choose_batch_size(axis_len, workers):
        # Bigger batches → less overhead; tune the multiplier to taste
        target_tasks_per_axis = max(2 * workers, 8)
        return max(1, int(axis_len / target_tasks_per_axis + 0.5))

    def _run_axis_batched_inplace(executor, func, axis_len, slicer, writer, workers):
        batch_size = _choose_batch_size(axis_len, workers)
        futures = []
        for batch_idx in _batches(axis_len, batch_size):
            # Capture the indices so they’re not re-evaluated later
            idxs = list(batch_idx)

            # Worker processes a *batch* and returns nothing; writes in place to reduce copies
            def job(idxs=idxs):
                for i in idxs:
                    plane_in = slicer(i)
                    plane_in = ascontiguousarray(plane_in)
                    plane_out = func(plane_in)
                    writer(i, plane_out)

            futures.append(executor.submit(job))

        for f in as_completed(futures):
            f.result()  # raise if a worker failed

    with ThreadPoolExecutor(max_workers=workers, initializer=_affinity_initializer) as executor:
        # X-planes: img[:, :, x]
        _run_axis_batched_inplace(
            executor, func, X,
            slicer=lambda x: img[:, :, x],
            writer=lambda x, r: img.__setitem__((slice(None), slice(None), x), r),
            workers=workers
        )

        # Y-planes: img[:, y, :]
        _run_axis_batched_inplace(
            executor, func, Y,
            slicer=lambda y: img[:, y, :],
            writer=lambda y, r: img.__setitem__((slice(None), y, slice(None)), r),
            workers=workers
        )

        # Z-planes: img[z, :, :]
        _run_axis_batched_inplace(
            executor, func, Z,
            slicer=lambda z: img[z, :, :],
            writer=lambda z, r: img.__setitem__((z, slice(None), slice(None)), r),
            workers=workers
        )

    return img


def open_3d(img: ndarray, open_steps: int = 30, max_workers=None):
    from concurrent.futures import ThreadPoolExecutor

    shape = img.shape
    mask = img.astype(uint8)

    # Define morph function
    morph = lambda arr: morphologyEx(arr, MORPH_OPEN, ones((open_steps, open_steps), dtype=uint8))

    # Parallel Z
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        chunksize = max(1, shape[0] // (max_workers or 8))
        results = list(executor.map(morph, [mask[z] for z in range(shape[0])], chunksize=chunksize))
    for z, out in enumerate(results):
        mask[z] = out

    # Parallel Y
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        chunksize = max(1, shape[1] // (max_workers or 8))
        results = list(executor.map(morph, [mask[:, y, :] for y in range(shape[1])], chunksize=chunksize))
    for y, out in enumerate(results):
        mask[:, y, :] = out

    # Parallel X
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        chunksize = max(1, shape[2] // (max_workers or 8))
        results = list(executor.map(lambda arr: morph(arr).astype(bool), [mask[:, :, x] for x in range(shape[2])], chunksize=chunksize))
    for x, out in enumerate(results):
        mask[:, :, x] = out

    return mask


def calculate_down_sampled_size(tile_size, down_sample):
    if isinstance(down_sample, (int, float)):
        tile_size = (ceil(size / down_sample) for size in tile_size)
    elif isinstance(down_sample, (tuple, list)):
        tile_size = list(tile_size)
        for idx, down_sample_factor in enumerate(down_sample):
            if down_sample_factor is not None:
                tile_size[idx] = ceil(tile_size[idx] / down_sample_factor)
    return tile_size


def correct_slice_value(user_value: [int, None], auto_estimated_min: [int, None], auto_estimated_max: [int, None]):
    if user_value is None:
        return None
    else:
        correction = 0
        if auto_estimated_min is not None:
            if user_value > auto_estimated_min:
                correction = auto_estimated_min
            else:
                # user_value is min and slicing amount requested by user is already taken care of
                return None

        if auto_estimated_max is not None and user_value > auto_estimated_max:
            correction += user_value - auto_estimated_max
        return user_value - correction


def imread_tif_raw_png(path: Path, dtype: str = None, shape: Tuple[int, int] = None):
    extension = path.suffix.lower()
    img = None
    attempt = 0

    for attempt in range(NUM_RETRIES):
        try:
            if extension == '.raw':
                img = raw_imread(path, dtype=dtype, shape=shape)

            elif extension == '.png':
                img = iio_imread(path, extension='.png', plugin='PNG-FI')

            elif extension in ('.tif', '.tiff'):
                try:
                    img = iio_imread(path, plugin="tifffile")
                except (TiffFileError, RuntimeError) as e:
                    if attempt == 0:
                        if 'imcd_lzw_decode' in str(e):
                            print(f"{PrintColors.WARNING}[imagecodecs]: file: {path.name} LZW decode error: {e}{PrintColors.ENDC}")
                        else:
                            print(f"{PrintColors.WARNING}[tifffile]: file: {path.name} failed to read: {type(e).__name__} - {e}{PrintColors.ENDC}")
                    try:
                        with Image.open(path) as im:
                            im.load()
                            img = array(im)
                    except (OSError, ValueError) as e2:
                        if attempt == 0:
                            print(f"{PrintColors.WARNING}[pillow]: file: {path.name} also failed to read: {type(e2).__name__} - {e2}{PrintColors.ENDC}")
                        # 🛠 Try bfconvert as last resort
                        bfconvert = shutil.which("bfconvert")
                        if bfconvert:
                            fixed_path = path.with_suffix('.bf.tif')
                            if fixed_path.exists():
                                fixed_path.unlink()
                            try:
                                subprocess.run(
                                    [bfconvert, str(path), str(fixed_path)],
                                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                                )
                                if fixed_path.exists():
                                    if path.exists():
                                        path.unlink()
                                    subprocess.run(
                                        [bfconvert, str(fixed_path), "-compression", "LZW", str(path)],
                                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                                    )
                                    print(f"{PrintColors.BLUE}[bfconvert]: file: {path.name} fixed and recompressed.{PrintColors.ENDC}")
                                    continue  # Retry reading the repaired file
                            except subprocess.CalledProcessError as e3:
                                print(f"{PrintColors.FAIL}[bfconvert]: file: {path.name} failed to fix: {e3}{PrintColors.ENDC}")
                        raise

            else:
                print(f"{PrintColors.WARNING}Unsupported file format: {extension}{PrintColors.ENDC}")
        except (OSError, TypeError, PermissionError) as e:
            print(f"[Attempt {attempt+1}]: file: {path.name} Read error: {type(e).__name__} - {e}")
            sleep(0.1)
            continue

        if img is not None:
            break

    if img is None:
        print(f"{PrintColors.FAIL}Failed to load image after {attempt + 1} attempts:\n{path.name}{PrintColors.ENDC}")
    return img


def assert_file_permissions(file_path, expected_permissions):
    # Windows Permission Check
    if os.name == 'nt':
        current_permissions = os.stat(file_path).st_mode & 0o666
    else:
        current_permissions = os.stat(file_path).st_mode & 0o777
    assert current_permissions == expected_permissions, f"File permissions for {file_path} must be {oct(expected_permissions)}, but are {oct(current_permissions)}."


def imsave_tif(path: Path, img: ndarray, compression: Union[Tuple[str, int], None] = ('ADOBE_DEFLATE', 1)) -> bool:
    """Save an array as a tiff or raw image

    The file format will be inferred from the file extension in `path`

    Parameters
    ----------
    path : Path
        path to tiff or raw image
    img : ndarray
        image as a numpy array
    compression : Tuple[str, int]
        The 1st argument is compression method the 2nd compression level for tiff files
        For example, ('ZSTD', 1) or ('ADOBE_DEFLATE', 1).

    Returns
    ----------
    True if the user interrupted the program, else False even if failed to save.
    """
    # compression_method = enumarg(TIFF.COMPRESSION, "None")
    # compression_level: int = 0
    # if compression and isinstance(compression, tuple) and len(compression) >= 2 and \
    #         isinstance(compression[1], int) and compression[1] <= 0:
    #     compression = False
    for attempt in range(1, NUM_RETRIES):
        try:
            # imwrite(path, data=img, compression=compression_method, compressionargs={'level': compression_level})
            # tmp_path = path.with_suffix(".tmp")
            tmp_path = path.with_suffix(".tif")
            imwrite(tmp_path, data=img, compression=compression)
            # Windows Permission Check
            if os.name == 'nt':
                os.chmod(tmp_path, 0o666)
                expected_permissions = 0o666
            else:
                os.chmod(tmp_path, 0o777)
                expected_permissions = 0o777
            assert_file_permissions(tmp_path, expected_permissions)
            tmp_path.rename(path)
            assert path.exists()
            return False  # do not die
        except KeyboardInterrupt:
            print(f"{PrintColors.WARNING}\ndying from imsave_tif{PrintColors.ENDC}")
            imwrite(path, data=img, compression=compression)
            return True  # die
        except (OSError, TypeError, PermissionError) as inst:
            if attempt == NUM_RETRIES:
                # f"Data size={img.size * img.itemsize} should be equal to the saved file's byte_count={byte_count}?"
                # f"\nThe file_size={path.stat().st_size} should be at least larger than tif header={offset} bytes\n"
                print(
                    f"After {NUM_RETRIES} attempts failed to save the file:\n"
                    f"{path}\n"
                    f"\n{type(inst)}\n"
                    f"{inst.args}\n"
                    f"{inst}\n")
                return False  # do not die
            else:
                sleep(0.1)
            continue


def imread_dcimg(path: Path, z: int):
    """Load a slice from a DCIMG file

    Parameters
    ------------
    path : Path
        path to DCIMG file
    z : int
        z slice index to load

    Returns
    --------
    img : ndarray
        image as numpy array

    """
    with DCIMGFile(path) as arr:
        img = arr[z]
    return img


def check_dcimg_shape(path: Path):
    """Returns the image shape of a DCIMG file

    Parameters
    ------------
    path : Path
        path to DCIMG file

    Returns
    --------
    shape : tuple
        image shape

    """
    with DCIMGFile(path) as arr:
        shape = arr.shape
    return shape


def check_dcimg_start(path: Path):
    """Returns the starting z position of a DCIMG substack.

    This function assumes a zero-padded 6 digit filename in tenths of micron.
    For example, `0015250.dicom` would indicate a substack starting at z = 1525 um.

    Parameters
    ------------
    path : Path
        path to DCIMG file

    Returns
    --------
    start : int
        starting z position in tenths of micron

    """
    return int(path.name.split('.')[0])


def convert_to_16bit_fun(img: ndarray):
    clip(img, 0, 65535, out=img)  # Clip to 16-bit [0, 2 ** 16 - 1] unsigned range
    return img.astype(uint16)


def convert_to_8bit_fun(img: ndarray, bit_shift_to_right: int = 8):
    if img is None or img.dtype in ('uint8', uint8):
        return img
    elif img.dtype not in ('uint16', uint16):
        img = convert_to_16bit_fun(img)

    # bit shift then change the type to avoid floating point operations
    # img >> 8 is equivalent to img / 256
    if bit_shift_to_right is None:
        bit_shift_to_right = 8
    if 0 <= bit_shift_to_right < 9:
        lower_bound = 2 ** bit_shift_to_right
        if USE_NUMEXPR and (img.flags.c_contiguous or img.flags.f_contiguous):
            evaluate("where((0 < img) & (img < lower_bound), 1, img >> bit_shift_to_right)", out=img, casting="unsafe")
        else:
            img = where((0 < img) & (img < lower_bound), 1, img >> bit_shift_to_right)
    else:
        print("right shift should be between 0 and 8")
        raise RuntimeError
    clip(img, 0, 255, out=img)  # Clip to 8-bit [0, 2 ** 8 - 1] unsigned range
    img = img.astype(uint8)
    return img


def process_img(
        img: ndarray,
        flat: ndarray = None,
        gaussian_filter_2d: bool = False,
        down_sample: Tuple[int, int] = None,  # (2, 2),
        down_sample_method: str = 'max',
        tile_size: Tuple[int, int] = None,
        new_size: Tuple[int, int] = None,
        exclude_dark_edges_set_them_to_zero: bool = False,
        sigma: Tuple[int, int] = (0, 0),
        level: int = 0,
        wavelet: str = 'db9',
        crossover: float = 10,
        threshold: float = None,
        padding_mode: str = None,
        bidirectional: bool = False,
        gpu_semaphore: Queue = None,
        bleach_correction_frequency: float = None,
        bleach_correction_clip_min: Union[float, int] = None,
        bleach_correction_clip_med: Union[float, int] = None,
        bleach_correction_clip_max: Union[float, int] = None,
        bleach_correction_max_method: bool = False,
        log1p_normalization_needed: bool = True,
        dark: float = 0,
        lightsheet: bool = False,
        artifact_length: int = 150,
        background_window_size: int = 200,
        percentile: float = 0.25,
        lightsheet_vs_background: float = 2.0,
        rotate: int = 0,
        flip_upside_down: bool = False,
        convert_to_16bit: bool = False,
        convert_to_8bit: bool = False,
        bit_shift_to_right: int = 8,
        d_type: str = None,
        verbose: bool = False
) -> ndarray:
    if tile_size is None:
        tile_size = img.shape
    if d_type is None:
        d_type = img.dtype

    if is_uniform_2d(img):
        if new_size is not None:
            tile_size = new_size
        elif down_sample is not None:
            tile_size = calculate_down_sampled_size(tile_size, down_sample)

        if rotate in (90, 270):
            tile_size = (tile_size[1], tile_size[0])

        if convert_to_16bit:
            d_type = uint16
        elif convert_to_8bit:
            d_type = uint8

        img = zeros(shape=tile_size, dtype=d_type)
    else:
        if flat is not None:
            if tile_size == flat.shape:
                img /= flat
            else:
                print(f"{PrintColors.WARNING}"
                      f"warning: image and flat arrays had different shapes"
                      f"{PrintColors.ENDC}")

        y_slice_min = y_slice_max = x_slice_min = x_slice_max = None
        if exclude_dark_edges_set_them_to_zero:
            img_x_max = np_max(img, axis=0)
            img_y_max = np_max(img, axis=1)
            img_x_max_min, img_x_max_max = min_max_1d(img_x_max)
            img_y_max_min, img_y_max_max = min_max_1d(img_y_max)
            img_max = max(img_x_max_max, img_y_max_max)
            img_min = np_min(img)
            noise_percentile = 5
            img_x_noise = prctl(img_x_max, noise_percentile)
            img_y_noise = prctl(img_y_max, noise_percentile)
            img_noise = min(img_x_noise, img_y_noise)
            y_slice_min, y_slice_max = slice_non_zero_box(img_y_max, noise=img_x_noise)
            x_slice_min, x_slice_max = slice_non_zero_box(img_x_max, noise=img_y_noise)
            img = img[y_slice_min:y_slice_max, x_slice_min:x_slice_max]
            if verbose:
                print(f"min={img_min}, x_max_min={img_x_max_min}, y_max_min={img_y_max_min},\n"
                      f"noise={img_noise}, x_noise={img_x_noise}, y_noise={img_y_noise},\n"
                      f"max={img_max}, x_max_max={img_x_max_max}, y_max_max={img_y_max_max}.")
                speedup = 100 - (x_slice_max - x_slice_min) * (y_slice_max - y_slice_min) / (
                        img.shape[0] * img.shape[1]) * 100
                print(f"slicing: y: {y_slice_min} to {y_slice_max} and x: {x_slice_min} to {x_slice_max}, "
                      f"performance enhancement: {speedup:.1f}%")

        if gaussian_filter_2d:
            # if img.dtype != float32:
            #     img = img.astype(float32)
            # gaussian(img, sigma=1, preserve_range=True, truncate=2, output=img)
            GaussianBlur(img, ksize=(5, 5), sigmaX=1, sigmaY=1)

        if down_sample is not None:
            down_sample_method = down_sample_method.lower()
            if down_sample_method == 'min':
                down_sample_method = np_min
            elif down_sample_method == 'max':
                down_sample_method = np_max
            elif down_sample_method == 'mean':
                down_sample_method = np_mean
            elif down_sample_method == 'median':
                down_sample_method = np_median
            else:
                print(f"{PrintColors.FAIL}unsupported down-sampling method: {down_sample_method}{PrintColors.ENDC}")
                raise RuntimeError
            img = block_reduce(img, block_size=down_sample, func=down_sample_method)
            tile_size = calculate_down_sampled_size(tile_size, down_sample)

        if bleach_correction_frequency is not None or sigma > (0, 0):
            img = filter_streaks(
                img,
                sigma=sigma,
                level=level,
                wavelet=wavelet,
                crossover=crossover,
                threshold=threshold,
                padding_mode=padding_mode,
                bidirectional=bidirectional,
                gpu_semaphore=gpu_semaphore,
                bleach_correction_frequency=bleach_correction_frequency,
                bleach_correction_max_method=bleach_correction_max_method,
                bleach_correction_clip_min=bleach_correction_clip_min,
                bleach_correction_clip_med=bleach_correction_clip_med,
                bleach_correction_clip_max=bleach_correction_clip_max,
                log1p_normalization_needed=log1p_normalization_needed,
                verbose=verbose,
            )

        # Subtract the dark offset
        # dark subtraction is like baseline subtraction in Imaris
        if dark is not None and dark > 0:
            if USE_NUMEXPR and (img.flags.c_contiguous or img.flags.f_contiguous):
                evaluate("where(img > dark, img - dark, 0)", out=img, casting="unsafe")
            else:
                img = where(img > dark, img - dark, 0)
            if verbose:
                print(f"dark value of {dark} is subtracted.")

        # lightsheet method is like background subtraction in Imaris
        if lightsheet:
            img = correct_lightsheet(
                img,
                percentile=percentile,
                lightsheet=dict(
                    selem=(1, artifact_length, 1),
                    dtype=d_type,
                ),
                background=dict(
                    selem=(background_window_size, background_window_size, 1),
                    spacing=(25, 25, 1),
                    interpolate=1,
                    dtype=d_type,
                    step=(2, 2, 1)),
                lightsheet_vs_background=lightsheet_vs_background
            )

        if exclude_dark_edges_set_them_to_zero:
            img_zero = zeros(shape=tile_size, dtype=d_type)
            img_zero[y_slice_min:y_slice_max, x_slice_min:x_slice_max] = img.astype(d_type)
            img = img_zero
            del img_zero

        if new_size is not None and tile_size < new_size:
            img = resize(img, new_size, preserve_range=True, anti_aliasing=True)
        elif new_size is not None and tile_size > new_size:
            img = resize(img, new_size, preserve_range=True, anti_aliasing=False)

        if convert_to_16bit and img.dtype not in (uint16, 'uint16'):
            img = convert_to_16bit_fun(img)
        elif convert_to_8bit and img.dtype not in (uint8, 'uint8'):
            img = convert_to_8bit_fun(img, bit_shift_to_right=bit_shift_to_right)
        elif np_d_type(d_type).kind in ("u", "i"):
            clip(img, iinfo(d_type).min, iinfo(d_type).max, out=img)
            img = img.astype(d_type)
        else:
            img = img.astype(d_type)

        if flip_upside_down:
            img = flipud(img)

        if rotate == 90:
            img = rot90(img, 1)
        elif rotate == 180:
            img = rot90(img, 2)
        elif rotate == 270:
            img = rot90(img, 3)

    return img


def read_filter_save(
        input_file: Path = None,
        output_file: Path = None,
        z_idx: int = None,
        continue_process: bool = False,
        d_type: str = None,
        tile_size: Tuple[int, int] = None,
        print_input_file_names: bool = False,
        compression: Tuple[str, int] = ('ADOBE_DEFLATE', 1),
        flat: ndarray = None,
        gaussian_filter_2d: bool = False,
        sigma: Tuple[int, int] = (0, 0),
        level: int = 0,
        wavelet: str = 'db9',
        crossover: float = 10,
        threshold: float = None,
        padding_mode: str = None,
        bidirectional: bool = False,
        gpu_semaphore: Queue = None,
        bleach_correction_frequency: float = None,
        bleach_correction_max_method: bool = True,
        bleach_correction_clip_min: Union[float, int] = None,
        bleach_correction_clip_med: Union[float, int] = None,
        bleach_correction_clip_max: Union[float, int] = None,
        dark: float = 0,
        lightsheet: bool = False,
        artifact_length: int = 150,
        background_window_size: int = 200,
        percentile: float = 0.25,
        lightsheet_vs_background: float = 2.0,
        convert_to_16bit: bool = False,
        convert_to_8bit: bool = True,
        bit_shift_to_right: int = 8,
        down_sample: Tuple[int, int] = None,  # (2, 2),
        down_sample_method: str = 'max',
        new_size: Tuple[int, int] = None,
        rotate: int = 0,
        flip_upside_down: bool = False,
):
    """Convenience wrapper around filter streaks. Takes in a path to an image rather than an image array

    Note that the directory being written to must already exist before calling this function

    Parameters
    ----------
    input_file : Path
        file path to the input image
    output_file : Path
        file path to the processed image
    z_idx : int
        z index of DCIMG slice. Only applicable to DCIMG files.
    continue_process: bool
        If true do not process images if the output file is already exist
    d_type: str or None,
        optional. data type of the input file. If given will reduce the raw to tif conversion time.
    tile_size : tuple (int, int) or None
        optional. If given will reduce the raw to tif conversion time.
    print_input_file_names : bool
        to find the corrupted files causing crash print the file names
    compression : tuple (str, int)
        The 1st argument is compression method the 2nd compression level for tiff files
        For example, ('ZSTD', 1) or ('ADOBE_DEFLATE', 1).
    flat : ndarray
        reference image for illumination correction. Must be same shape as input images. Default is None
    gaussian_filter_2d : bool
        If true the image will be denoised using a 5x5 gaussian filer.
    sigma : tuple
        bandwidth of the stripe filter
    level : int
        number of wavelet levels to use. 0 means the maximum possible decimation.
    wavelet : str
        name of the mother wavelet
    crossover : float
        intensity range to switch between filtered background and unfiltered foreground
    threshold : float
        intensity value to separate background from foreground.
        Default (-1) means automatic detection using Otsu method, otherwise uses the given positive value.
    padding_mode : str
        Padding method affects the edge artifacts. In some cases wrap method works better than reflect method.
    bidirectional : bool
        by default (False) only stripes elongated along horizontal axis will be corrected.
    gpu_semaphore: Queue
        Needed for multi-GPU processing and prevents overflowing GPUs vRAM
    bleach_correction_frequency : float
        frequency of a low-pass filter that describes bleaching
    bleach_correction_max_method: bool
        use max value on x and y axes to create the filter. Max method is faster and smoother but less accurate
        for large images.
    bleach_correction_clip_min: float, int
        foreground vs background threshold
    bleach_correction_clip_med: float, int
        foreground intermediate
    bleach_correction_clip_max: float, int
        foreground max
    dark : float
        Intensity to subtract from the images for dark offset. Default is 0.
    lightsheet : bool
        if False, use wavelet method, if true use correct_lightsheet
    artifact_length : int
        # of pixels to look at in the lightsheet direction
    background_window_size : int
        Look at this size window around the pixel in x and y
    percentile : float
        Take this percentile as background with lightsheet
    lightsheet_vs_background : float
        weighting factor to use background or lightsheet background
    convert_to_16bit : bool
        Flag for converting to 16-bit
    convert_to_8bit : bool
        Save the output as an 8-bit image
    bit_shift_to_right : int [0 to 8]
        It works when converting to 8-bit. Correct 8 bit conversion needs 8 bit shift.
        Bit shifts smaller than 8 bit, enhances the signal brightness.
    down_sample : tuple (int, int)
        Sets down sample factor. Down_sample (3, 2) means 3 pixels in y-axis, and 2 pixels in x-axis merges into 1.
    down_sample_method: str
        down-sampling method. options are max, min, mean, median. Default is max.
    new_size : tuple (int, int) or None
        resize the image after down-sampling
    rotate : int
        Rotate the image. One of 0, 90, 180 or 270 degree values are accepted. Default is 0 (no rotation).
    flip_upside_down : bool
        flip the image parallel to y-axis. Default is false.
    """
    try:
        # 1150 is 1850x1850 zeros image saved as compressed tif
        # 272 is header offset size
        if continue_process and output_file.exists():  # and output_file.stat().st_size > 272
            return
        if print_input_file_names:
            print(f"\n{input_file}")
        if z_idx is None:
            img = imread_tif_raw_png(input_file, dtype=d_type, shape=tile_size)  # file must be TIFF or RAW
        else:
            img = imread_dcimg(input_file, z_idx)  # file must be DCIMG
        if img is None and d_type is not None and tile_size is not None:
            print(
                f"{PrintColors.WARNING}"
                f"\nimread function returned None. Possible damaged input file:"
                f"\n\t{input_file}."
                f"\n\toutput file is set to a dummy zeros tile of shape {tile_size} and type {d_type}, instead:"
                f"\n\t{output_file}"
                f"{PrintColors.ENDC}"
            )
            img = zeros(dtype=d_type, shape=tile_size)
        elif img is None:
            print(
                f"{PrintColors.WARNING}"
                f"\nimread function returned None. Possible damaged input file:"
                f"\n\t{input_file}."
                f"\n\toutput file could be replaced with a dummy tile of zeros if shape and d_type were provided."
                f"{PrintColors.ENDC}"
            )
            return

        if tile_size is not None and img.shape != tile_size:
            print(
                f"{PrintColors.WARNING}"
                f"\nwarning: input tile had a different shape. resizing:\n"
                f"\tinput_file: {input_file} -> \n"
                f"\t\tinput shape = {img.shape}\n"
                f"\t\tnew shape   = {tile_size}\n"
                f"{PrintColors.ENDC}")
            img = resize(img, tile_size, preserve_range=True, anti_aliasing=True)
        else:
            tile_size = img.shape

        if d_type is None:
            d_type = img.dtype

        if not output_file.parent.exists():
            output_file.parent.mkdir(parents=True, exist_ok=True)

        img = process_img(
            img,
            flat=flat,
            gaussian_filter_2d=gaussian_filter_2d,
            down_sample=down_sample,
            down_sample_method=down_sample_method,
            tile_size=tile_size,
            new_size=new_size,
            sigma=sigma,
            level=level,
            wavelet=wavelet,
            crossover=crossover,
            threshold=threshold,
            padding_mode=padding_mode,
            bidirectional=bidirectional,
            gpu_semaphore=gpu_semaphore,
            bleach_correction_frequency=bleach_correction_frequency,
            bleach_correction_max_method=bleach_correction_max_method,
            bleach_correction_clip_min=bleach_correction_clip_min,
            bleach_correction_clip_med=bleach_correction_clip_med,
            bleach_correction_clip_max=bleach_correction_clip_max,
            dark=dark,
            lightsheet=lightsheet,
            artifact_length=artifact_length,
            background_window_size=background_window_size,
            percentile=percentile,
            lightsheet_vs_background=lightsheet_vs_background,
            rotate=rotate,
            flip_upside_down=flip_upside_down,
            convert_to_16bit=convert_to_16bit,
            convert_to_8bit=convert_to_8bit,
            bit_shift_to_right=bit_shift_to_right,
            d_type=d_type,
        )

        imsave_tif(output_file, img, compression=compression)

    except (OSError, IndexError, TypeError, RuntimeError, TiffFileError) as inst:
        print(f"{PrintColors.WARNING}warning: read_filter_save function failed:"
              f"\n{type(inst)}"  # the exception instance
              f"\n{inst.args}"  # arguments stored in .args
              f"\n{inst}"
              f"\nPossible damaged input file: {input_file}"
              f"{PrintColors.ENDC}")


def glob_re(pattern: str, path: Path):
    """Recursively find all files having a specific name
        path: Path
            Search path
        pattern: str
            regular expression to search the file name.
    """
    regexp = compile(pattern, IGNORECASE)
    paths: Iterator[DirEntry] = scandir(path)
    for p in paths:
        if p.is_file() and regexp.search(p.name):
            yield Path(p.path)
        elif p.is_dir(follow_symlinks=False):
            yield from glob_re(pattern, Path(p.path))


def process_tif_raw_png_images(input_file: Path, input_path: Path, output_path: Path, args_dict_template: dict):
    """Find all images with a supported file extension within a directory and all its subdirectories

    Parameters
    ----------
    input_file: Path
        tif, tiff, or raw file
    input_path : path-like
        root directory of input images
    output_path: path-like
        root directory of out_put images
    args_dict_template: Dict
        common arguments of the read_filter_save function

    Returns
    -------
    img_paths : dict
        all arguments of the read_filter_save function including input_path and output_path
    """
    output_file = output_path / input_file.relative_to(input_path)
    output_file = output_file.parent / (output_file.name[0:-len(output_file.suffix)] + '.tif')
    if args_dict_template['continue_process'] and output_file.exists():
        return None
    args_dict_template.update({
        'input_file': input_file,
        'output_file': output_file
    })
    return args_dict_template.copy()


def process_dc_images(input_file: Path, input_path: Path, output_path: Path, args_dict_template: dict, z_step: float):
    """Find all images with a supported file extension within a directory and all its subdirectories

        Parameters
        ----------
        input_file: Path
            tif, tiff, or raw file
        input_path : path-like
            root directory of input images
        output_path: path-like
            root directory of out_put images
        args_dict_template: Dict
            common arguments of the read_filter_save function
        z_step : float
            step-size for DCIMG stacks in tenths of micron

        Returns
        -------
        img_paths : dict
            all arguments of the read_filter_save function including input_path and output_path
    """
    shape = check_dcimg_shape(input_file)
    start = check_dcimg_start(input_file)
    sub_stack = []
    for i in range(shape[0]):
        output_file = output_path / input_file.relative_to(input_path).parent / f'z{start + i * z_step:08.1f}.tif'
        args_dict_template.update({
            'input_file': input_file,
            'output_file': output_file,
            'z_idx': i
        })
        if args_dict_template['continue_process'] and output_file.exists():
            sub_stack += [None]
        else:
            sub_stack += [args_dict_template.copy()]
    return sub_stack


class MultiProcessQueueRunner(Process):
    def __init__(self, progress_queue: Queue, args_queue: Queue,
                 gpu_semaphore: SimpleQueue = None,
                 gpu: int = None,
                 fun=None,
                 timeout: float = None,
                 replace_timeout_with_dummy: bool = True):
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"
        super().__init__()
        self.daemon = False
        self.progress_queue = progress_queue
        self.args_queue = args_queue
        self.gpu_semaphore = gpu_semaphore
        self.timeout = timeout
        self.die = False
        self.function = fun
        self.replace_timeout_with_dummy = replace_timeout_with_dummy

    def safe_put(self, queue, obj, block=True, timeout=None):
        try:
            queue.put(obj, block=block, timeout=timeout)
        except (OSError, ValueError, Full):
            # Queue handle closed or already full (unlikely, but safe), or unreachable
            pass

    def run(self):
        running_next = True
        timeout = self.timeout
        gpu_semaphore = self.gpu_semaphore
        if self.timeout:
            pool = ProcessPoolExecutor(max_workers=1)
        else:
            pool = ThreadPoolExecutor(max_workers=1)
        function = self.function
        queue_timeout = None
        while not self.die and not self.args_queue.qsize() == 0:
            try:
                queue_start_time = time()
                try:
                    args: dict = self.args_queue.get(block=True, timeout=queue_timeout)
                except (OSError, ValueError, Empty):
                    # Handle closed, finished, or empty
                    self.die = True
                    break

                if gpu_semaphore is not None:
                    args.update({"gpu_semaphore": gpu_semaphore})
                if queue_timeout is not None:
                    queue_timeout = max(queue_timeout, 0.9 * queue_timeout + 0.3 * (time() - queue_start_time))
                try:
                    start_time = time()
                    future = pool.submit(function, **args)
                    future.result(timeout=timeout)
                    if timeout is not None:
                        timeout = max(timeout, 0.9 * timeout + 0.3 * (time() - start_time))
                except (BrokenProcessPool, TimeoutError, ValueError) as inst:
                    if self.replace_timeout_with_dummy:
                        output_file: Path = args["output_file"]
                        print(f"\nwarning: timeout reached for processing input file:\n\t{args['input_file']}\n\t"
                              f"a dummy (zeros) image is saved as output instead:\n\t{output_file}"
                              f"\nexception instance: {type(inst)}")
                        if not output_file.exists():
                            die = imsave_tif(
                                output_file,
                                zeros(
                                    shape=args["new_size"] if args.get("new_size") else args["tile_size"],
                                    dtype=uint8 if args.get("convert_to_8bit") else uint16
                                )
                            )
                            if die:
                                self.die = True
                    else:
                        print(f"\nwarning: timeout reached for processing input file:\n\t{args.get('file_name')}\n\t"
                              f"\nexception instance: {type(inst)}")
                    # Properly close and recreate the pool (guarded)
                    try:
                        pool.shutdown(wait=True)
                    except Exception:
                        pass
                    pool = ProcessPoolExecutor(max_workers=1) if self.timeout else ThreadPoolExecutor(max_workers=1)
                except KeyboardInterrupt:
                    self.die = True
                except Exception as inst:
                    print(
                        f"\nwarning: process unexpectedly failed for {args}."
                        f"\nexception instance: {type(inst)}"
                        f"\nexception arguments: {inst.args}"
                        f"\nexception: {inst}")
                self.safe_put(self.progress_queue, running_next)
            except Empty:
                self.die = True
            except (OSError, ValueError):
                self.die = True

        # General safeguard for pool and queue on finalization
        try:
            if isinstance(pool, ProcessPoolExecutor):
                pool.shutdown(wait=True)
            elif isinstance(pool, ThreadPoolExecutor):
                pool.shutdown(wait=True)
        except Exception:
            pass

        # Final status update; safe put to avoid error if pipe closed
        self.safe_put(self.progress_queue, not running_next)


def progress_manager(progress_queue: Queue, workers: int, total: int,
                     desc="PyStripe", unit=" images", initial=0):
    return_code = 0
    list_of_outputs = []
    print(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
          f"using {workers} workers. {total} images need to be processed.", flush=True)
    sleep(1)
    progress_bar = tqdm(total=total, ascii=True, smoothing=0.01, mininterval=1.0, unit=unit, desc=desc, initial=initial)
    progress_bar.refresh()

    while workers > 0:
        try:
            still_running = progress_queue.get(block=False)
            if isinstance(still_running, bool) and still_running:
                progress_bar.update(1)
            else:
                workers -= 1
                if not isinstance(still_running, bool):
                    list_of_outputs += [still_running]
        except Empty:
            try:
                sleep(1)
            except KeyboardInterrupt:
                print(f"\n{PrintColors.WARNING}Terminating processes with dignity!{PrintColors.ENDC}")
                return_code = 1
        except KeyboardInterrupt:
            print(f"\n{PrintColors.WARNING}Terminating processes with dignity!{PrintColors.ENDC}")
            return_code = 1
    progress_bar.close()
    return list_of_outputs if list_of_outputs else return_code


def batch_filter(
        input_path: Path,
        output_path: Path,
        files_list: List[Path] = None,
        workers: int = cpu_count(logical=False),
        threads_per_gpu: int = 8,
        flat: ndarray = None,
        gaussian_filter_2d: bool = False,
        sigma: Tuple[int, int] = (0, 0),
        level=0,
        wavelet: str = 'db9',
        crossover: int = 10,
        threshold: int = None,
        padding_mode: str = None,
        bidirectional: bool = False,
        bleach_correction_frequency: float = None,
        bleach_correction_max_method: bool = True,
        bleach_correction_clip_min: Union[float, int] = None,
        bleach_correction_clip_med: Union[float, int] = None,
        bleach_correction_clip_max: Union[float, int] = None,
        dark: int = 0,
        z_step: float = None,
        rotate: int = 0,
        flip_upside_down: bool = False,
        lightsheet: bool = False,
        artifact_length: int = 150,
        background_window_size: int = 200,
        percentile: float = .25,
        lightsheet_vs_background: float = 2.0,
        convert_to_16bit: bool = False,
        convert_to_8bit: bool = False,
        bit_shift_to_right: int = 8,
        continue_process: bool = False,
        d_type: str = None,
        tile_size: Tuple[int, int] = None,
        down_sample: Tuple[int, int] = None,  # (2, 2)
        new_size: Tuple[int, int] = None,
        print_input_file_names: bool = False,
        timeout: float = None,
        compression: Tuple[str, int] = ('ADOBE_DEFLATE', 1)
):
    """Applies `streak_filter` to all images in `input_path` and write the results to `output_path`.

    Parameters
    ----------
    input_path : Path
        root directory to search for images to filter.
        Supported formats:
            root path for a deep hierarchy of tif or raw 2D series
            dcimg files (untested)
            one raw, or tif file
    output_path : Path
        root directory for writing results
    files_list : List[Path]
        list of files to process. If given, pystripe will not spend time searching for files in input path
    workers : int
        number of CPU workers to use
    threads_per_gpu: int
        number of images processed simultaneously with each GPU.
        Depends on the vRAM capacity, size of the image, and CUDA core usage.
    sigma : tuple
        bandwidth of the stripe filter in pixels
        sigma=(foreground, background) Default is (0, 0), indicating no de-striping.
    level : int
        number of wavelet levels to use. 0 is the default and means the maximum possible decimation.
    wavelet : str
        name of the mother wavelet
    crossover : float
        intensity range to switch between filtered background and unfiltered foreground. Default: 10 a.u.
    threshold : float
        intensity value to separate background from foreground.
        Default (-1) means automatic detection using Otsu method, otherwise uses the given positive value.
    padding_mode : str
        Padding method affects the edge artifacts. In some cases wrap method works better than reflect method.
    bidirectional : bool
        by default (False) only stripes elongated along horizontal axis will be corrected.
    bleach_correction_frequency : float
        frequency of a low-pass filter that describes bleaching
    bleach_correction_max_method: bool
        use max value on x and y axes to create the filter. Max method is faster and smoother but less accurate
        for large images.
    bleach_correction_clip_min: float, int
        foreground vs background threshold
    bleach_correction_clip_med: float, int
        foreground intermediate
    bleach_correction_clip_max: float, int
        foreground max
    compression : tuple (str, int)
        The 1st argument is compression method the 2nd compression level for tiff files
        For example, ('ZSTD', 1) or ('ADOBE_DEFLATE', 1).
    flat : ndarray
        reference image for illumination correction. Must be same shape as input images. Default is None
    dark : float
        Intensity to subtract from the images for dark offset. Default is 0.
    z_step : int
        z-step in tenths of micron. only used for DCIMG files.
    rotate : int
        Rotate the image. One of 0, 90, 180 or 270 degree values are accepted. Default is 0.
    flip_upside_down : bool
        flip the image parallel to y-axis. Default is false.
    lightsheet : bool
        Specific destriping for light sheet images. Default to False.
    artifact_length : int
    background_window_size : int
    percentile : float
    lightsheet_vs_background : float
    gaussian_filter_2d : bool
        If true the image will be denoised using a 5x5 gaussian filer.
    convert_to_16bit : bool
        Flag for converting to 16-bit
    convert_to_8bit : bool
        Save the output as an 8-bit image
    bit_shift_to_right : int [0 to 8]
        It works when converting to 8-bit. Correct 8 bit conversion needs 8 bit shift.
        Bit shifts smaller than 8 bit, enhances the signal brightness.
    continue_process : bool
        True means only process the remaining images.
    d_type: str or None,
        optional. data type of input files (uint8, uint16, or etc.). If given reduces the raw to tif conversion time.
    tile_size : tuple (int, int) or None
        optional. If given will reduce the raw to tif conversion time.
    down_sample : tuple (int, int) or None
        Sets down sample factor. Down_sample (3, 2) means 3 pixels in y-axis, and 2 pixels in x-axis merges into 1.
    new_size : tuple (y: int, x: int) or None
        resize the image after down-sampling
    print_input_file_names : bool
        to find the corrupted files causing crash print the file names
    timeout: float | None
        if file processing took more than timeout seconds, terminate the process.
        It is used whenever some tiles could be corrupt and in raw format and processing halts without raising an error.
    """
    input_path = Path(input_path)
    assert input_path.is_dir()
    if convert_to_16bit is True and convert_to_8bit is True:
        print(f"{PrintColors.FAIL}Select 8-bit or 16-bit output format.{PrintColors.ENDC}")
        raise TypeError
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if sigma is None:
        sigma = (0, 0)
    if workers <= 0:
        workers = cpu_count()
    if isinstance(flat, (ndarray, generic)):
        flat = normalize_flat(flat)
    elif isinstance(flat, Path):
        flat = normalize_flat(imread_tif_raw_png(flat))
    elif isinstance(flat, str):
        flat = normalize_flat(imread_tif_raw_png(Path(flat)))
    elif flat is not None:
        print(f"{PrintColors.FAIL}flat argument should be a numpy array or a path to a flat.tif file{PrintColors.ENDC}")
        raise TypeError

    arg_dict_template = {
        'flat': flat,
        'gaussian_filter_2d': gaussian_filter_2d,
        'sigma': sigma,
        'level': level,
        'wavelet': wavelet,
        'crossover': crossover,
        'threshold': threshold,
        'padding_mode': padding_mode,
        'bidirectional': bidirectional,
        'bleach_correction_frequency': bleach_correction_frequency,
        'bleach_correction_max_method': bleach_correction_max_method,
        'bleach_correction_clip_min': bleach_correction_clip_min,
        'bleach_correction_clip_med': bleach_correction_clip_med,
        'bleach_correction_clip_max': bleach_correction_clip_max,
        'dark': dark,
        'lightsheet': lightsheet,
        'artifact_length': artifact_length,
        'background_window_size': background_window_size,
        'percentile': percentile,
        'lightsheet_vs_background': lightsheet_vs_background,
        'z_idx': None,
        'rotate': rotate,
        'flip_upside_down': flip_upside_down,
        'convert_to_16bit': convert_to_16bit,
        'convert_to_8bit': convert_to_8bit,
        'bit_shift_to_right': bit_shift_to_right,
        'continue_process': continue_process,
        'd_type': d_type,
        'tile_size': tile_size,
        'down_sample': None if isinstance(down_sample, tuple) and down_sample == (1, 1) else down_sample,
        'new_size': new_size,
        'print_input_file_names': print_input_file_names,
        'compression': compression
    }

    print(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
          f"Scheduling jobs for images in \n\t{input_path}")

    if z_step is None:
        files = glob_re(r"\.(?:tiff?|raw|png)$", input_path) if files_list is None else files_list
        args_list = list(tqdm(
            map(lambda file: process_tif_raw_png_images(file, input_path, output_path, arg_dict_template), files),
            total=None if isinstance(files, GeneratorType) else len(files),
            ascii=True, smoothing=0.05, mininterval=1.0, unit=" tif|raw|png images", desc="found",
        ))
    else:
        files = glob_re(r"\.(?:dcimg)$", input_path) if files_list is None else files_list
        args_list = tqdm(
            map(lambda file: process_dc_images(file, input_path, output_path, arg_dict_template, z_step), files),
            total=None if isinstance(files, GeneratorType) else len(files),
            ascii=True, smoothing=0.05, mininterval=1.0, unit=" dcimg images", desc="found",
        )
        args_list = reduce(iconcat, args_list, [])  # unravel the list of list the fastest way possible
    del files, files_list

    args_list = [arg for arg in args_list if arg is not None]
    num_images = len(args_list)
    args_queue = Queue(maxsize=num_images)
    for args in args_list:
        args_queue.put(args)
    del args_list

    gpu_semaphore = None
    if USE_PYTORCH and CUDA_IS_AVAILABLE_FOR_PT:
        gpu_semaphore = Queue()
        for i in range(cuda_device_count()):
            for _ in range(threads_per_gpu):
                gpu_semaphore.put(f"cuda:{i}")

    workers = min(workers, num_images)
    progress_queue = Queue()
    for worker in range(workers):
        MultiProcessQueueRunner(progress_queue, args_queue, gpu_semaphore,
                                fun=read_filter_save, timeout=timeout).start()

    return_code = progress_manager(progress_queue, workers, num_images)
    args_queue.cancel_join_thread()
    args_queue.close()
    progress_queue.cancel_join_thread()
    progress_queue.close()
    return return_code


def normalize_flat(flat):
    flat_float = flat.astype(float32)
    return flat_float / flat_float.max()


def _parse_args():
    parser = ArgumentParser(
        description="Pystripe (version 0.4.0)\n\n"
                    "If only sigma1 is specified, only foreground of the images will be filtered.\n"
                    "If sigma2 is specified and sigma1 = 0, only the background of the images will be filtered.\n"
                    "If sigma1 == sigma2 > 0, input images will not be split before filtering.\n"
                    "If sigma1 != sigma2, foreground and backgrounds will be filtered separately.\n"
                    "The crossover parameter defines the width of the transition "
                    "between the filtered foreground and background",
        formatter_class=RawDescriptionHelpFormatter,
        epilog='Developed 2018 by Justin Swaney, Kwanghun Chung Lab             at MIT\n'
               'Updated   2021 by Keivan Moradi, Hongwei Dong   Lab (B.R.A.I.N) at UCLA\n'
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to input image or path")
    parser.add_argument("--output", "-o", type=str, default='',
                        help="Path to output image or path (Default: x_destriped)")
    parser.add_argument("--sigma1", "-s1", type=float, default=0,
                        help="Foreground bandwidth [pixels], larger = more filtering")
    parser.add_argument("--sigma2", "-s2", type=float, default=0,
                        help="Background bandwidth [pixels] (Default: 0, off)")
    parser.add_argument("--level", "-l", type=int, default=0,
                        help="Number of decomposition levels (Default: max possible)")
    parser.add_argument("--wavelet", "-w", type=str, default='db3',
                        help="Name of the mother wavelet (Default: Daubechies 3 tap)")
    parser.add_argument("--threshold", "-t", type=float, default=None,
                        help="Global threshold value (Default: None --> Otsu)")
    parser.add_argument("--padding_mode", "-w", type=str, default='reflect',
                        help="Padding method affects the edge artifacts. "
                             "In some cases wrap method works better than reflect method.")
    parser.add_argument("--bidirectional", "-dr", type=bool, default=False,
                        help="destriping direction: "
                             "\n\tFalse means top-down (default), ")
    parser.add_argument("--crossover", "-x", type=float, default=10,
                        help="Intensity range to switch between foreground and background (Default: 10)")
    parser.add_argument("--workers", "-n", type=int, default=8,
                        help="Number of workers for batch processing (Default: # CPU cores)")
    parser.add_argument("--chunks", type=int, default=4,
                        help="Chunk size for parallel processing (Default: 1)")
    parser.add_argument("--compression_method", "-cm", type=str, default='ADOBE_DEFLATE',
                        help="Compression method for written tiffs (Default: ADOBE_DEFLATE)")
    parser.add_argument("--compression_level", "-cl", type=int, default=1,
                        help="Compression level for written tiffs (Default: 1)")
    parser.add_argument("--flat", "-f", type=str, default=None,
                        help="Flat reference TIFF image of illumination pattern used for correction")
    parser.add_argument("--dark", "-d", type=float, default=0,
                        help="Intensity of dark offset in flat-field correction")
    parser.add_argument("--zstep", "-z", type=float, default=None,
                        help="Z-step in micron. Only used for DCIMG files.")
    parser.add_argument("--rotate", "-r", type=int, default=0,
                        help="Rotate the image. One of 0, 90, 180 or 270 degree values are accepted. Default is 0.")
    parser.add_argument("--flip_upside_down", "-flup", action='store_false',
                        help="Flip the image upside down along the y-axis. Default is false")
    parser.add_argument("--lightsheet", action="store_true",
                        help="Use the lightsheet method")
    parser.add_argument("--artifact-length", default=150, type=int,
                        help="Look for minimum in lightsheet direction over this length")
    parser.add_argument("--background-window-size", default=200, type=int,
                        help="Size of window in x and y for background estimation")
    parser.add_argument("--percentile", type=float, default=.25,
                        help="The percentile at which to measure the background")
    parser.add_argument("--lightsheet-vs-background", type=float, default=2.0,
                        help="The background is multiplied by this weight when comparing lightsheet against background")
    parser.add_argument("--convert_to_16bit", action="store_false",
                        help="If convert the input to 16-bit or not")
    parser.add_argument("--convert_to_8bit", action="store_false",
                        help="If convert the output to 8-bit or not")
    parser.add_argument("--bit_shift_to_right", "-bsh", type=int, default=8,
                        help="Right bit shift with max thresholding. Bit shift smaller than 8 increases the image "
                             "brightness of the output 8-bit image. (Default: 8)")
    parser.add_argument("--down_sample", "-ds", type=int, default=None,
                        help="Image down-sampling factor using max function.")
    parser.add_argument("--size_x", "-sx", type=int, default=None,
                        help="New image size in X axis.")
    parser.add_argument("--size_y", "-sy", type=int, default=None,
                        help="New image size in Y axis.")
    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    sigma = (args.sigma1, args.sigma2)
    input_path = Path(args.input)
    new_size = (args.size_y, args.size_x) if args.size_y is not None and args.size_x is not None else None

    flat = None
    if args.flat is not None:
        flat = normalize_flat(imread_tif_raw_png(Path(args.flat)))

    if args.dark < 0:
        raise ValueError('Only positive values for dark offset are allowed')

    if input_path.is_file():  # single image
        if input_path.suffix not in SUPPORTED_EXTENSIONS:
            print('Input file was found but is not supported. Exiting...')
            return
        if args.output == '':
            output_path = Path(input_path.parent).joinpath(input_path.stem + '_destriped' + input_path.suffix)
        else:
            output_path = Path(args.output)
            assert output_path.suffix in SUPPORTED_EXTENSIONS

    elif input_path.is_dir():  # batch processing
        if args.output == '':
            output_path = Path(input_path.parent).joinpath(str(input_path) + '_destriped')
        else:
            output_path = Path(args.output)
            assert output_path.suffix == ''
    else:
        raise RuntimeError('Cannot find input file or directory.')
    read_filter_save(
        input_path,
        output_path,
        z_idx=int(args.zstep * 10) if args.zstep is not None else None,
        flat=flat,
        down_sample=args.down_sample,
        new_size=new_size,
        sigma=sigma,
        level=args.level,
        wavelet=args.wavelet,
        crossover=args.crossover,
        threshold=args.threshold,
        padding_mode=args.padding_mode,
        bidirectional=args.bidirectional,
        dark=args.dark,
        lightsheet=args.lightsheet,
        artifact_length=args.artifact_length,
        background_window_size=args.background_window_size,
        percentile=args.percentile,
        lightsheet_vs_background=args.lightsheet_vs_background,
        convert_to_16bit=args.convert_to_16bit,
        convert_to_8bit=args.convert_to_8bit,
        bit_shift_to_right=args.bit_shift_to_right,
        rotate=args.rotate,  # Does not work on DCIMG files
        flip_upside_down=args.flip_upside_down,
        compression=(args.compression_method, args.compression_level),
    )


if __name__ == "__main__":
    main()
