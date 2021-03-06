import os
import re
import pywt
import warnings
from argparse import RawDescriptionHelpFormatter, ArgumentParser
from numpy import max as np_max
from numpy import uint8, uint16, float32, float64, ndarray, generic, real, zeros, broadcast_to, sqrt, exp, log, \
    cumsum, imag, arange, unique, interp, pad, clip, where, rot90
from scipy import fftpack, ndimage
from tqdm import tqdm
from time import sleep
from datetime import datetime
from pathlib import Path
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import block_reduce
from skimage.transform import resize
from tifffile import imread, imwrite
from tifffile.tifffile import TiffFileError
from dcimg import DCIMGFile
from typing import Tuple, Iterator, List
from types import GeneratorType
from pystripe.raw import raw_imread
from .lightsheet_correct import correct_lightsheet
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import Process, Queue, cpu_count
from queue import Empty
from operator import iconcat
from functools import reduce
from supplements.cli_interface import PrintColors

warnings.filterwarnings("ignore")
supported_extensions = ['.tif', '.tiff', '.raw', '.dcimg']
num_retries = 40
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'


def imread_tif_raw(path: Path, dtype: str = None, shape: Tuple[int, int] = None):
    """Load a tiff or raw image

    Parameters
    ----------
    path : Path
        path to tiff or raw image
    dtype: str or None,
        optional. If given will reduce the raw to tif conversion time.
    shape : tuple (int, int) or None
        optional. If given will reduce the raw to tif conversion time.

    Returns
    -------
    img : ndarray
        image as a numpy array

    """
    img = None
    # for NAS
    attempt = 0
    for attempt in range(num_retries):
        try:
            extension = path.suffix.lower()
            if extension == '.raw':
                img = raw_imread(path, dtype=dtype, shape=shape)
            elif extension in ['.tif', '.tiff']:
                img = imread(path)
        except (OSError, TypeError, PermissionError):
            sleep(0.1)
            continue
        break
    if img is None:
        print(f"after {attempt + 1} attempts failed to read file:\n{path}")
    return img


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


def check_dcimg_shape(path):
    """Returns the image shape of a DCIMG file

    Parameters
    ------------
    path : str
        path to DCIMG file

    Returns
    --------
    shape : tuple
        image shape

    """
    with DCIMGFile(path) as arr:
        shape = arr.shape
    return shape


def check_dcimg_start(path):
    """Returns the starting z position of a DCIMG substack.

    This function assumes a zero-padded 6 digit filename in tenths of micron.
    For example, `0015250.dicom` would indicate a substack starting at z = 1525 um.

    Parameters
    ------------
    path : str
        path to DCIMG file

    Returns
    --------
    start : int
        starting z position in tenths of micron

    """
    return int(os.path.basename(path).split('.')[0])


def convert_to_16bit_fun(img: ndarray):
    clip(img, 0, 65535, out=img)  # Clip to 16-bit [0, 2 ** 16 - 1] unsigned range
    return img.astype(uint16)


def convert_to_8bit_fun(img: ndarray, bit_shift_to_right: int = 8):
    if img.dtype == 'uint8':
        return img
    else:
        img = img.astype(uint16)
    # bit shift then change the type to avoid floating point operations
    # img >> 8 is equivalent to img / 256
    if 0 <= bit_shift_to_right < 9:
        img = (img >> bit_shift_to_right)
    elif bit_shift_to_right is None:
        img = (img >> 8)
    else:
        print("right shift should be between 0 and 8")
        raise RuntimeError
    clip(img, 0, 255, out=img)  # Clip to 8-bit [0, 2 ** 8 - 1] unsigned range
    return img.astype(uint8)


def imsave_tif(path: Path, img: ndarray, compression: Tuple[str, int] = ('ZLIB', 1)):
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
        For example, ('ZSTD', 1) or ('ZLIB', 1).
    """
    for attempt in range(1, num_retries):
        try:
            imwrite(path, img, compression=compression)
            return
        except (KeyboardInterrupt, SystemExit):
            print(f"{PrintColors.WARNING}dying from imsave_tif{PrintColors.ENDC}")
            imwrite(path, img, compression=compression)
            # raise KeyboardInterrupt
        except (OSError, TypeError, PermissionError) as inst:
            if attempt == num_retries:
                # f"Data size={img.size * img.itemsize} should be equal to the saved file's byte_count={byte_count}?"
                # f"\nThe file_size={path.stat().st_size} should be at least larger than tif header={offset} bytes\n"
                print(
                    f"After {num_retries} attempts failed to save the file:\n"
                    f"{path}\n"
                    f"\n{type(inst)}\n"
                    f"{inst.args}\n"
                    f"{inst}\n")
            else:
                sleep(0.1)
            continue


def wavedec(img: ndarray, wavelet: str, level: int = None):
    """Decompose `img` using discrete (decimated) wavelet transform using `wavelet`

    Parameters
    ----------
    img : ndarray
        image to be decomposed into wavelet coefficients
    wavelet : str
        name of the mother wavelet
    level : int (optional)
        number of wavelet levels to use. Default is the maximum possible decimation

    Returns
    -------
    coeffs : list
        the approximation coefficients followed by detail coefficient tuple for each level

    """
    return pywt.wavedec2(img, wavelet, mode='symmetric', level=level, axes=(-2, -1))


def waverec(coeffs, wavelet):
    """Reconstruct an image using a multilevel 2D inverse discrete wavelet transform

    Parameters
    ----------
    coeffs : list
        the approximation coefficients followed by detail coefficient tuple for each level
    wavelet : str
        name of the mother wavelet

    Returns
    -------
    img : ndarray
        reconstructed image

    """
    return pywt.waverec2(coeffs, wavelet, mode='symmetric', axes=(-2, -1))


def fft(data, axis=-1, shift=True):
    """Computes the 1D Fast Fourier Transform of an input array

    Parameters
    ----------
    data : ndarray
        input array to transform
    axis : int (optional)
        axis to perform the 1D FFT over
    shift : bool
        indicator for centering the DC component

    Returns
    -------
    fdata : ndarray
        transformed data

    """
    fdata = fftpack.rfft(data, axis=axis)
    # fdata = fftpack.rfft(fdata, axis=0)
    if shift:
        fdata = fftpack.fftshift(fdata)
    return fdata


def ifft(fdata, axis=-1):
    # fdata = fftpack.irfft(fdata, axis=0)
    return fftpack.irfft(fdata, axis=axis)


def fft2(data, shift=True):
    """Computes the 2D Fast Fourier Transform of an input array

    Parameters
    ----------
    data : ndarray
        data to transform
    shift : bool
        indicator for center the DC component

    Returns
    -------
    fdata : ndarray
        transformed data

    """
    fdata = fftpack.fft2(data)
    if shift:
        fdata = fftpack.fftshift(fdata)
    return fdata


def ifft2(fdata):
    return fftpack.ifft2(fdata)


def magnitude(fdata):
    return sqrt(real(fdata) ** 2 + imag(fdata) ** 2)


def notch(n, sigma):
    """Generates a 1D gaussian notch filter `n` pixels long

    Parameters
    ----------
    n : int
        length of the gaussian notch filter
    sigma : float
        notch width

    Returns
    -------
    g : ndarray
        (n,) array containing the gaussian notch filter

    """
    if n <= 0:
        raise ValueError('n must be positive')
    else:
        n = int(n)
    if sigma <= 0:
        raise ValueError('sigma must be positive')
    x = arange(n)
    g = 1 - exp(-x ** 2 / (2 * sigma ** 2))
    return g


def gaussian_filter(shape, sigma):
    """Create a gaussian notch filter

    Parameters
    ----------
    shape : tuple
        shape of the output filter
    sigma : float
        filter bandwidth

    Returns
    -------
    g : ndarray
        the impulse response of the gaussian notch filter

    """
    g = notch(n=shape[-1], sigma=sigma)
    g_mask = broadcast_to(g, shape).copy()
    return g_mask


def hist_match(source, template):
    """Adjust the pixel values of a grayscale image such that its histogram matches that of a target image

    Parameters
    ----------
    source: ndarray
        Image to transform; the histogram is computed over the flattened array
    template: ndarray
        Template image; can have different dimensions to source
    Returns
    -------
    matched: ndarray
        The transformed output image

    """

    old_shape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = cumsum(s_counts).astype(float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = cumsum(t_counts).astype(float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(old_shape)


def max_level(min_len, wavelet):
    w = pywt.Wavelet(wavelet)
    return pywt.dwt_max_level(min_len, w.dec_len)


def sigmoid(x):
    return 1 / (1 + exp(-x))


def foreground_fraction(img, center, crossover, smoothing):
    z = (img - center) / crossover
    f = sigmoid(z)
    return ndimage.gaussian_filter(f, sigma=smoothing)


def filter_subband(img, sigma, level, wavelet):
    img_log = log(1 + img)

    if level == 0:
        coeffs = wavedec(img_log, wavelet)
    else:
        coeffs = wavedec(img_log, wavelet, level)
    approx = coeffs[0]
    detail = coeffs[1:]

    width_frac = sigma / img.shape[0]
    coeffs_filt = [approx]
    for ch, cv, cd in detail:
        s = ch.shape[0] * width_frac
        fch = fft(ch, shift=False)
        g = gaussian_filter(shape=fch.shape, sigma=s)
        fch_filt = fch * g
        ch_filt = ifft(fch_filt)
        coeffs_filt.append((ch_filt, cv, cd))

    img_log_filtered = waverec(coeffs_filt, wavelet)
    return exp(img_log_filtered) - 1


def apply_flat(img, flat):
    if img.shape == flat.shape:
        return (img / flat).astype(img.dtype)
    else:
        print("\033[93m"
              "warning: image and flat arrays had different shapes"
              "\033[0m")
        return img


def filter_streaks(img, sigma, level=0, wavelet='db3', crossover=10, threshold=-1):
    """Filter horizontal streaks using wavelet-FFT filter

    Parameters
    ----------
    img : ndarray
        input image array to filter
    sigma : tuple
        filter bandwidth(s) in pixels (larger gives more filtering)
    level : int
        number of wavelet levels to use
    wavelet : str
        name of the mother wavelet
    crossover : float
        intensity range to switch between filtered background and unfiltered foreground
    threshold : float
        intensity value to separate background from foreground. Default is O tsu

    Returns
    -------
    f_img : ndarray
        filtered image

    """
    sigma1 = sigma[0]  # foreground
    sigma2 = sigma[1]  # background
    if sigma1 == sigma2 == 0:
        return img
    dtype = img.dtype
    smoothing = 1
    if threshold == -1 and sigma1 != sigma2:
        try:
            threshold = threshold_otsu(img)
        except ValueError:
            threshold = 1

    img = img.astype(float32)
    #
    # Need to pad image to multiple of 2
    #
    pad_y, pad_x = [_ % 2 for _ in img.shape]
    if pad_y == 1 or pad_x == 1:
        img = pad(img, ((0, pad_y), (0, pad_x)), mode="edge")

    # TODO: Clean up this logic with some dual-band CLI alternative
    if sigma1 > 0:
        if sigma2 > 0:
            if sigma1 == sigma2:  # Single band
                f_img = filter_subband(img, sigma1, level, wavelet)
            else:  # Dual-band
                background = clip(img, None, threshold)
                foreground = clip(img, threshold, None)
                background_filtered = filter_subband(background, sigma[1], level, wavelet)
                foreground_filtered = filter_subband(foreground, sigma[0], level, wavelet)
                # Smoothed homotopy
                f = foreground_fraction(img, threshold, crossover, smoothing=smoothing)
                f_img = foreground_filtered * f + background_filtered * (1 - f)
        else:  # Foreground filter only
            foreground = clip(img, threshold, None)
            foreground_filtered = filter_subband(foreground, sigma[0], level, wavelet)
            # Smoothed homotopy
            f = foreground_fraction(img, threshold, crossover, smoothing=smoothing)
            f_img = foreground_filtered * f + img * (1 - f)
    else:
        if sigma2 > 0:  # Background filter only
            background = clip(img, None, threshold)
            background_filtered = filter_subband(background, sigma[1], level, wavelet)
            # Smoothed homotopy
            f = foreground_fraction(img, threshold, crossover, smoothing=smoothing)
            f_img = img * f + background_filtered * (1 - f)
        else:
            # sigma1 and sigma2 are both 0, so skip the destriping
            f_img = img

    # TODO: Fix code to clip back to original bit depth
    # scaled_fimg = hist_match(f_img, img)
    # clip(scaled_fimg, np.iinfo(img.dtype).min, np.iinfo(img.dtype).max, out=scaled_fimg)

    # Convert to 16 bit image
    if dtype == uint16 or dtype in ["uint16", "float64"]:
        f_img = convert_to_16bit_fun(f_img)
    elif dtype == uint8 or dtype == "uint8":
        f_img = convert_to_8bit_fun(f_img, bit_shift_to_right=0)
    else:
        print(f"unexpected image dtype {f_img.dtype}")
        raise RuntimeError

    if pad_x > 0:
        f_img = f_img[:, :-pad_x]
    if pad_y > 0:
        f_img = f_img[:-pad_y]
    return f_img


def read_filter_save(
        input_file: Path = None,
        output_file: Path = None,
        sigma: Tuple[int, int] = (0, 0),
        level: int = 0,
        wavelet: str = 'db3',
        crossover: float = 10,
        threshold: float = -1,
        compression: Tuple[str, int] = ('ZLIB', 1),
        flat: ndarray = None,
        dark: float = 0,
        z_idx: int = None,
        rotate: bool = False,
        lightsheet: bool = False,
        artifact_length: int = 150,
        background_window_size: int = 200,
        percentile: float = 0.25,
        lightsheet_vs_background: float = 2.0,
        gaussian_filter_2d: bool = False,
        convert_to_16bit: bool = False,
        convert_to_8bit: bool = True,
        bit_shift_to_right: int = 8,
        continue_process: bool = False,
        dtype: str = None,
        tile_size: Tuple[int, int] = None,
        down_sample: Tuple[int, int] = None,  # (2, 2),
        new_size: Tuple[int, int] = None,
        print_input_file_names: bool = True
):
    """Convenience wrapper around filter streaks. Takes in a path to an image rather than an image array

    Note that the directory being written to must already exist before calling this function

    Parameters
    ----------
    input_file : Path
        path to the image to filter
    output_file : Path
        path to write the result
    sigma : tuple
        bandwidth of the stripe filter
    level : int
        number of wavelet levels to use
    wavelet : str
        name of the mother wavelet
    crossover : float
        intensity range to switch between filtered background and unfiltered foreground
    threshold : float
        intensity value to separate background from foreground. Default is Otsu
    compression : tuple (str, int)
        The 1st argument is compression method the 2nd compression level for tiff files
        For example, ('ZSTD', 1) or ('ZLIB', 1).
    flat : ndarray
        reference image for illumination correction. Must be same shape as input images. Default is None
    dark : float
        Intensity to subtract from the images for dark offset. Default is 0.
    z_idx : int
        z index of DCIMG slice. Only applicable to DCIMG files.
    rotate : bool
        rotate x and y if true
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
    gaussian_filter_2d : bool
        If true the image will be denoised using a 5x5 gaussian filer.
    convert_to_16bit : bool
        Flag for converting to 16-bit
    convert_to_8bit : bool
        Save the output as an 8-bit image
    bit_shift_to_right : int [0 to 8]
        It works when converting to 8-bit. Correct 8 bit conversion needs 8 bit shift.
        Bit shifts smaller than 8 bit, enhances the signal brightness.
    continue_process: bool
        If true do not process images if the output file is already exist
    dtype: str or None,
        optional. data type of the input file. If given will reduce the raw to tif conversion time.
    tile_size : tuple (int, int) or None
        optional. If given will reduce the raw to tif conversion time.
    down_sample : tuple (int, int)
        Sets down sample factor. Down_sample (3, 2) means 3 pixels in y axis, and 2 pixels in x-axis merges into 1.
    new_size : tuple (int, int) or None
        resize the image after down-sampling
    print_input_file_names : bool
        to find the corrupted files causing crash print the file names
    """
    try:
        # 1150 is 1850x1850 zeros image saved as compressed tif
        # 272 is header offset size
        if continue_process and output_file.exists():  # and output_file.stat().st_size > 272
            return
        if print_input_file_names:
            print(f"\n{input_file}")
        if z_idx is None:
            img = imread_tif_raw(input_file, dtype=dtype, shape=tile_size)  # file must be TIFF or RAW
        else:
            img = imread_dcimg(input_file, z_idx)  # file must be DCIMG
        if img is None and dtype is not None and tile_size is not None:
            print(
                f"{PrintColors.WARNING}"
                f"\nimread function returned None. Possible damaged input file:"
                f"\n\t{input_file}."
                f"\n\toutput file is set to a dummy zeros tile of shape {tile_size} and type {dtype}, instead:"
                f"\n\t{output_file}"
                f"{PrintColors.ENDC}"
            )
            img = zeros(dtype=dtype, shape=tile_size)
        elif img is None:
            print(
                f"{PrintColors.WARNING}"
                f"\nimread function returned None. Possible damaged input file:"
                f"\n\t{input_file}."
                f"\n\toutput file could be replaced with a dummy tile of zeros if shape and dtype were provided."
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

        d_type = img.dtype
        if not output_file.parent.exists():
            output_file.parent.mkdir(parents=True, exist_ok=True)
        if flat is not None:
            img = apply_flat(img, flat)
        if dark is not None and dark > 0:
            img = where(img > dark, img - dark, 0)  # Subtract the dark offset
        if gaussian_filter_2d:
            img = gaussian(img, sigma=1, preserve_range=True, truncate=2)
        if down_sample is not None:
            img = block_reduce(img, block_size=down_sample, func=np_max)
        if new_size is not None and tile_size > new_size:
            img = resize(img, new_size, preserve_range=True, anti_aliasing=True).astype(d_type)
        if rotate:
            img = rot90(img)
        if lightsheet:
            img = correct_lightsheet(
                img.reshape(img.shape[0], img.shape[1], 1),
                percentile=percentile,
                lightsheet=dict(selem=(1, artifact_length, 1)),
                background=dict(
                    selem=(background_window_size, background_window_size, 1),
                    spacing=(25, 25, 1),
                    interpolate=1,
                    dtype=float32,
                    step=(2, 2, 1)),
                lightsheet_vs_background=lightsheet_vs_background
            ).reshape(img.shape[0], img.shape[1])
        if not sigma[0] == sigma[1] == 0:
            img = filter_streaks(
                img,
                sigma,
                level=level,
                wavelet=wavelet,
                crossover=crossover,
                threshold=threshold
            )

        if new_size is not None and tile_size < new_size:
            img = resize(img, new_size, preserve_range=True, anti_aliasing=False)

        if convert_to_16bit and img.dtype != uint16:
            clip(img, 0, 65535, out=img)  # Clip to 16-bit [0, 2 ** 16 - 1] unsigned range
            img = img.astype(uint16)
        elif convert_to_8bit and img.dtype != uint8:
            img = convert_to_8bit_fun(img, bit_shift_to_right=bit_shift_to_right)
        elif img.dtype != d_type:
            img = img.astype(d_type)
        imsave_tif(output_file, img, compression=compression)

    except (OSError, IndexError, TypeError, RuntimeError, TiffFileError) as inst:
        print(f"{PrintColors.WARNING}warning: read_filter_save function failed:"
              f"\n{type(inst)}"  # the exception instance
              f"\n{inst.args}"  # arguments stored in .args
              f"\n{inst}"
              f"\nPossible damaged input file: {input_file}"
              f"{PrintColors.ENDC}")


def _read_filter_save(input_dict):
    """Same as `read_filter_save' but with a single input dictionary. Used for pool.imap() in batch_filter

    Parameters
    ----------
    input_dict : dict
        input dictionary with arguments for `read_filter_save`.
    """
    read_filter_save(**input_dict)


def glob_re(pattern: str, path: Path):
    """Recursively find all files having a specific name
        path: Path
            Search path
        pattern: str
            regular expression to search the file name.
    """
    regexp = re.compile(pattern, re.IGNORECASE)
    paths: Iterator[os.DirEntry] = os.scandir(path)
    for p in paths:
        if p.is_file() and regexp.search(p.name):
            yield Path(p.path)
        elif p.is_dir(follow_symlinks=False):
            yield from glob_re(pattern, Path(p.path))


def process_tif_raw_imgs(input_file: Path, input_path: Path, output_path: Path, args_dict_template: dict):
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


def process_dc_imgs(input_file: Path, input_path: Path, output_path: Path, args_dict_template: dict, z_step: float):
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
    shape = check_dcimg_shape(str(input_file))
    start = check_dcimg_start(str(input_file))
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


class MultiProcess(Process):
    def __init__(self, progress_queue: Queue, args_queue: Queue, timeout: float = None):
        Process.__init__(self)
        self.daemon = False
        self.progress_queue = progress_queue
        self.args_queue = args_queue
        self.timeout = timeout
        self.die = False

    def run(self):
        running = True
        pool = ProcessPoolExecutor(max_workers=1)
        while not self.args_queue.empty():
            if self.die:
                break
            try:
                args = self.args_queue.get(block=True, timeout=10)
                try:
                    future = pool.submit(_read_filter_save, args)
                    future.result(timeout=self.timeout)
                except (BrokenProcessPool, TimeoutError) as inst:
                    output_file: Path = args["output_file"]
                    print(f"{PrintColors.WARNING}"
                          f"\nwarning: timeout reached for processing input file:\n\t{args['input_file']}\n\t"
                          f"a dummy (zeros) image is saved as output instead:\n\t{output_file}"
                          f"\nexception instance: {type(inst)}"
                          f"{PrintColors.ENDC}")
                    if not output_file.exists():
                        imsave_tif(
                            output_file,
                            zeros(
                                shape=args["new_size"] if args["new_size"] else args["tile_size"],
                                dtype=uint8 if args["convert_to_8bit"] else uint16
                            )
                        )
                    pool.shutdown()
                    pool = ProcessPoolExecutor(max_workers=1)
                except (KeyboardInterrupt, SystemExit):
                    self.die = True
                except Exception as inst:
                    print(
                        f"{PrintColors.WARNING}"
                        f"\nwarning: process unexpectedly failed for {args}."
                        f"\nexception instance: {type(inst)}"
                        f"\nexception arguments: {inst.args}"
                        f"\nexception: {inst}"
                        f"{PrintColors.ENDC}")
                self.progress_queue.put(running)
            except Empty:
                self.die = True
        pool.shutdown()
        running = False
        self.progress_queue.put(running)


def progress_manager(process_list: List[Process], progress_queue: Queue, workers: int, total: int, desc="PyStripe"):
    return_code = 0
    print(f'{datetime.now()}: using {workers} workers. {total} images need preprocessing.')
    progress_bar = tqdm(total=total, ascii=True, smoothing=0.05, mininterval=1.0, unit="images", desc=desc)

    def kill_processes(list_of_processes):
        print(f"\n{PrintColors.WARNING}Terminating processes with dignity!{PrintColors.ENDC}")
        for process in list_of_processes:
            process.die = True
        return 1

    while workers > 0:
        try:
            still_running = progress_queue.get(block=False)
            if still_running:
                progress_bar.update(1)
            else:
                workers -= 1
        except Empty:
            try:
                sleep(1)
            except (KeyboardInterrupt, SystemExit):
                return_code = kill_processes(process_list)
        except (KeyboardInterrupt, SystemExit):
            return_code = kill_processes(process_list)
    progress_bar.close()
    progress_queue.close()
    return return_code


def batch_filter(
        input_path: Path,
        output_path: Path,
        files_list: List[Path] = None,
        workers: int = cpu_count(),
        sigma: Tuple[int, int] = (0, 0),
        level=0,
        wavelet: str = 'db10',
        crossover: int = 10,
        threshold: int = -1,
        compression: Tuple[str, int] = ('ZLIB', 1),
        flat: ndarray = None,
        dark: int = 0,
        z_step: float = None,
        rotate: bool = False,
        lightsheet: bool = False,
        artifact_length: int = 150,
        background_window_size: int = 200,
        percentile: float = .25,
        lightsheet_vs_background: float = 2.0,
        gaussian_filter_2d: bool = False,
        convert_to_16bit: bool = False,
        convert_to_8bit: bool = False,
        bit_shift_to_right: int = 8,
        continue_process: bool = False,
        dtype: str = None,
        tile_size: Tuple[int, int] = None,
        down_sample: Tuple[int, int] = None,  # (2, 2)
        new_size: Tuple[int, int] = None,
        print_input_file_names: bool = False,
        timeout: float = None
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
    sigma : tuple
        bandwidth of the stripe filter in pixels
        sigma=(foreground, background) Default is (0, 0), indicating no de-striping.
    level : int
        number of wavelet levels to use
    wavelet : str
        name of the mother wavelet
    crossover : float
        intensity range to switch between filtered background and unfiltered foreground. Default: 100 a.u.
    threshold : float
        intensity value to separate background from foreground. Default is O tsu
    compression : tuple (str, int)
        The 1st argument is compression method the 2nd compression level for tiff files
        For example, ('ZSTD', 1) or ('ZLIB', 1).
    flat : ndarray
        reference image for illumination correction. Must be same shape as input images. Default is None
    dark : float
        Intensity to subtract from the images for dark offset. Default is 0.
    z_step : int
        z-step in tenths of micron. only used for DCIMG files.
    rotate : bool
        Flag for 90-degree rotation.
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
    dtype: str or None,
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
        print('Select 8-bit or 16-bit output format.')
        raise RuntimeError
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if sigma is None:
        sigma = [0, 0]
    if workers <= 0:
        workers = cpu_count()
    if isinstance(flat, (ndarray, generic)):
        flat = normalize_flat(flat)
    elif isinstance(flat, Path):
        flat = normalize_flat(imread_tif_raw(flat))
    elif isinstance(flat, str):
        flat = normalize_flat(imread_tif_raw(Path(flat)))
    elif flat is not None:
        print('flat argument should be a numpy array or a path to a flat.tif file')
        raise RuntimeError

    arg_dict_template = {
        'sigma': sigma,
        'level': level,
        'wavelet': wavelet,
        'crossover': crossover,
        'threshold': threshold,
        'compression': compression,
        'flat': flat,
        'dark': dark,
        'z_idx': None,
        'rotate': rotate,
        'lightsheet': lightsheet,
        'artifact_length': artifact_length,
        'background_window_size': background_window_size,
        'percentile': percentile,
        'lightsheet_vs_background': lightsheet_vs_background,
        'gaussian_filter_2d': gaussian_filter_2d,
        'convert_to_16bit': convert_to_16bit,
        'convert_to_8bit': convert_to_8bit,
        'bit_shift_to_right': bit_shift_to_right,
        'continue_process': continue_process,
        'dtype': dtype,
        'tile_size': tile_size,
        'down_sample': down_sample,
        'new_size': new_size,
        'print_input_file_names': print_input_file_names
    }

    print(f'{datetime.now()}: Scheduling jobs for images in \n\t{input_path}')

    if z_step is None:
        files = glob_re(r"\.(?:tiff?|raw)$", input_path) if files_list is None else files_list
        args_list = list(tqdm(
            map(lambda file: process_tif_raw_imgs(file, input_path, output_path, arg_dict_template), files),
            total=None if isinstance(files, GeneratorType) else len(files),
            ascii=True, smoothing=0.05, mininterval=1.0, unit=" tif|raw images", desc="found",
        ))
    else:
        files = glob_re(r"\.(?:dcimg)$", input_path) if files_list is None else files_list
        args_list = tqdm(
            map(lambda file: process_dc_imgs(file, input_path, output_path, arg_dict_template, z_step), files),
            total=None if isinstance(files, GeneratorType) else len(files),
            ascii=True, smoothing=0.05, mininterval=1.0, unit=" dcimg images", desc="found",
        )
        args_list = reduce(iconcat, args_list, [])  # unravel the list of list the fastest way possible
    del files, files_list

    num_images = len(args_list)
    args_queue = Queue(maxsize=num_images)
    for arg in args_list:
        if arg is not None:
            args_queue.put(arg)
    del args_list

    workers = min(workers, args_queue.qsize())
    progress_queue = Queue()
    process_list: List[MultiProcess, ...] = []
    for worker in range(workers):
        process_list += [MultiProcess(progress_queue, args_queue, timeout)]
        process_list[-1].start()

    return_code = progress_manager(process_list, progress_queue, workers, args_queue.qsize())
    args_queue.close()
    progress_queue.close()
    return return_code


def normalize_flat(flat):
    flat_float = flat.astype(float32)
    return flat_float / flat_float.max()


def _parse_args():
    parser = ArgumentParser(
        description="Pystripe (version 0.3.0)\n\n"
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
    parser.add_argument("--threshold", "-t", type=float, default=-1,
                        help="Global threshold value (Default: -1, Otsu)")
    parser.add_argument("--crossover", "-x", type=float, default=10,
                        help="Intensity range to switch between foreground and background (Default: 10)")
    parser.add_argument("--workers", "-n", type=int, default=8,
                        help="Number of workers for batch processing (Default: # CPU cores)")
    parser.add_argument("--chunks", type=int, default=4,
                        help="Chunk size for parallel processing (Default: 1)")
    parser.add_argument("--compression_method", "-cm", type=str, default='ZLIB',
                        help="Compression method for written tiffs (Default: ZLIB)")
    parser.add_argument("--compression_level", "-cl", type=int, default=1,
                        help="Compression level for written tiffs (Default: 1)")
    parser.add_argument("--flat", "-f", type=str, default=None,
                        help="Flat reference TIFF image of illumination pattern used for correction")
    parser.add_argument("--dark", "-d", type=float, default=0,
                        help="Intensity of dark offset in flat-field correction")
    parser.add_argument("--zstep", "-z", type=float, default=None,
                        help="Z-step in micron. Only used for DCIMG files.")
    parser.add_argument("--rotate", "-r", action='store_true',
                        help="Rotate output images 90 degrees counter-clockwise")
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
        flat = normalize_flat(imread_tif_raw(Path(args.flat)))

    zstep = None
    if args.zstep is not None:
        zstep = int(args.zstep * 10)

    if args.dark < 0:
        raise ValueError('Only positive values for dark offset are allowed')

    if input_path.is_file():  # single image
        if input_path.suffix not in supported_extensions:
            print('Input file was found but is not supported. Exiting...')
            return
        if args.output == '':
            output_path = Path(input_path.parent).joinpath(input_path.stem + '_destriped' + input_path.suffix)
        else:
            output_path = Path(args.output)
            assert output_path.suffix in supported_extensions
        read_filter_save(
            input_path,
            output_path,
            sigma=sigma,
            level=args.level,
            wavelet=args.wavelet,
            crossover=args.crossover,
            threshold=args.threshold,
            compression=(args.compression_method, args.compression_level),
            flat=flat,
            dark=args.dark,
            rotate=args.rotate,  # Does not work on DCIMG files
            lightsheet=args.lightsheet,
            artifact_length=args.artifact_length,
            background_window_size=args.background_window_size,
            percentile=args.percentile,
            lightsheet_vs_background=args.lightsheet_vs_background,
            convert_to_16bit=args.convert_to_16bit,
            convert_to_8bit=args.convert_to_8bit,
            bit_shift_to_right=args.bit_shift_to_right,
            down_sample=args.down_sample,
            new_size=new_size
        )
    elif input_path.is_dir():  # batch processing
        if args.output == '':
            output_path = Path(input_path.parent).joinpath(str(input_path) + '_destriped')
        else:
            output_path = Path(args.output)
            assert output_path.suffix == ''
        batch_filter(
            input_path,
            output_path,
            workers=args.workers,
            # chunks=args.chunks,
            sigma=sigma,
            level=args.level,
            wavelet=args.wavelet,
            crossover=args.crossover,
            threshold=args.threshold,
            compression=(args.compression_method, args.compression_level),
            flat=flat,
            dark=args.dark,
            z_step=zstep,
            rotate=args.rotate,
            lightsheet=args.lightsheet,
            artifact_length=args.artifact_length,
            background_window_size=args.background_window_size,
            percentile=args.percentile,
            lightsheet_vs_background=args.lightsheet_vs_background,
            convert_to_16bit=args.convert_to_16bit,
            convert_to_8bit=args.convert_to_8bit,
            bit_shift_to_right=args.bit_shift_to_right,
            down_sample=args.down_sample,
            new_size=new_size
        )
    else:
        print('Cannot find input file or directory. Exiting...')


if __name__ == "__main__":
    main()
