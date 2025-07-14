import os
import sys
import contextlib
from argparse import RawDescriptionHelpFormatter, ArgumentParser, BooleanOptionalAction
from multiprocessing import freeze_support, Queue, set_start_method
from pathlib import Path
from platform import uname
from itertools import cycle

import psutil
from nrrd import read, write
from numpy import dtype as np_d_type
from numpy import rot90, float32, iinfo, clip, ndarray, pad
from psutil import cpu_count
from pycudadecon import make_otf, decon
from scipy.ndimage import gaussian_filter as gaussian
from tifffile import imwrite, TiffFile
from tqdm import tqdm
from shutil import copy

from LsDeconvolveMultiGPU.psf_generator import generate_psf
from pystripe.core import filter_streaks, is_uniform_2d, is_uniform_3d, MultiProcessQueueRunner, progress_manager
from subprocess import check_output

from align_images import trim_to_shape

# Good dimension is defined as one that can be factorized int 2s, 3s, 5s, and 7s.
# According to CUFFT manual, such dimension would warrant fast FFT
# see https://github.com/scopetools/cudadecon/blob/main/src/RL-Biggs-Andrews.cpp
def get_next_good_dim(n: int):
    while True:
        temp = n
        for prime in [2, 3, 5, 7]:
            while temp % prime == 0:
                temp //= prime
        if temp != 1:
            n += 1
        else:
            break
    return n


def pad_to_good_dim(arr: ndarray, otf_shape: tuple):
    shape = arr.shape
    output_shape = []
    # must add 1 for the edge case where n is a good dim, and therefore will have pixels cut off from the convolution

    for n, dim in enumerate(shape):
        output_shape.append(get_next_good_dim(dim + max(4, otf_shape[n])))

    assert(len(output_shape) == len(shape))

    pad_amount = tuple(((s[0] - s[1]) // 2, (s[0] - s[1] + 1) // 2) for s in zip(output_shape, shape))
    return pad(arr, pad_amount, 'reflect')


# pass in a target and a ndarray, and this function attempts to pad/trim to make the target shape.
def crop_to_shape(target_shape: tuple, arr: ndarray):
    assert len(target_shape) == len(arr.shape)
    pad_amount = tuple(map(lambda s, a: (0, s - a if s > a else 0), target_shape, arr.shape))
    # pad_amount = tuple([(0, s[0] - s[1] if s[0] > s[1] else 0) for s in zip(target_shape, arr)])
    arr = pad(arr, pad_width=pad_amount, mode='reflect')
    arr = arr[0:target_shape[0], 0: target_shape[1], 0: target_shape[2]]
    return arr

def make_a_list_of_input_output_paths(args):
    input_folder = Path(args.input)
    output_folder = Path(args.output)

    deconvolution_args = None
    if args.deconvolution:
        psf, dxy_psf = generate_psf(
            lambda_em=args.wavelength_em,
            lambda_ex=args.wavelength_ex,
            numerical_aperture=args.na,
            dxy=args.dxy * 1000.0,
            dz=args.dz * 1000.0,
            refractive_index=args.nimm,
            f_cylinder_lens=args.f_cylinder_lens,
            slit_width=args.slit_width,
            gaussian_sgima=float(args.gaussian[0]) if isinstance(args.gaussian, tuple) else args.gaussian,
            doubled_psf=args.doubled_psf
        )
        psf: ndarray = rot90(psf, k=1, axes=(0, 2))
        psf_filepath = (output_folder / 'temp' / 'psf.tif').__str__()
        (output_folder / r"temp").mkdir(parents=True, exist_ok=True)
        imwrite(psf_filepath, psf)

        otf_file = make_otf(
            psf_filepath,
            outpath=(output_folder / 'temp' / 'otf.tif').__str__(),
            dzpsf=args.dz,
            dxpsf=dxy_psf / 1000.0,
            wavelength=args.wavelength_em,
            na=args.na,
            nimm=args.nimm,
            fixorigin=0,
        )

        deconvolution_args = make_deconvolution_args(
            otf_file,
            n_iters=args.n_iters,
            dz_data=args.dz,
            dx_data=args.dxy,
            dz_psf=args.dz,
            dxy_psf=dxy_psf / 1000.0,
            background=args.background,
            wavelength_em=args.wavelength_em,
            na=args.na,
            nimm=args.nimm,
            dg_interation=args.dg_interation,
            contrast_enhancement_factor=args.contrast_enhancement_factor,
            doubled_psf=args.doubled_psf
        )

    def get_args(input_file: Path):
        output_file = output_folder / input_file.relative_to(input_folder)
        if output_file.exists():
            return None
        else:
            return {
                "input_file": input_file,
                "output_file": output_file,
                "need_destripe": args.destripe,
                "gaussian_sigma": args.gaussian,
                "gaussian_mode": args.gaussian_mode,
                "need_deconvolution": args.deconvolution,
                "deconvolution_args": deconvolution_args,
            }

    return tuple(c for c in tqdm(map(get_args, input_folder.rglob("*.nrrd")),
                                 desc="Finding cubes", unit=" Paths") if c is not None)


def make_deconvolution_args(
        # psf: ndarray,
        otf: str,
        n_iters: int = 9,
        dz_data: float = 0.7,
        dx_data: float = 0.7,
        dz_psf: float = 1.8,
        dxy_psf: float = 0.2,
        background: int = 0,
        wavelength_em: int = 525,
        na: float = 0.4,
        nimm: float = 1.42,
        contrast_enhancement_factor: float = 1.0,
        dg_interation: int = 4,
        doubled_psf: bool = False,
) -> dict:
    # the 0 is for the z-axis since otf is actually 2D
    otf_shape = (0, ) + TiffFile(otf).asarray().shape
    print("otf shape: ", otf_shape)
    return {
        # 'psf': psf,
        'otf': otf,
        'otf_shape': otf_shape,
        'n_iters': n_iters,
        'dz_data': dz_data,
        'dx_data': dx_data,
        'dz_psf': dz_psf,
        'dxy_psf': dxy_psf,
        'background': background,
        'wavelength_em': wavelength_em,
        'na': na,
        'nimm': nimm,
        'contrast_enhancement_factor': contrast_enhancement_factor,
        'dg_interation': dg_interation,
        'doubled_psf': doubled_psf
    }


@contextlib.contextmanager
def suppress_output():
    # Save the original file descriptors
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()

    # Duplicate the original file descriptors
    saved_stdout_fd = os.dup(original_stdout_fd)
    saved_stderr_fd = os.dup(original_stderr_fd)

    # Open /dev/null and redirect stdout and stderr to it
    with open(os.devnull, 'w') as devnull:
        os.dup2(devnull.fileno(), original_stdout_fd)
        os.dup2(devnull.fileno(), original_stderr_fd)

    try:
        yield
    finally:
        # Restore the original file descriptors
        os.dup2(saved_stdout_fd, original_stdout_fd)
        os.dup2(saved_stderr_fd, original_stderr_fd)

        # Close the duplicated file descriptors
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def apply_deconvolution(img, deconvolution_args, gpu_semaphore, num_gaussian_decons):
    gpu = 0
    if gpu_semaphore is not None:
        gpu = gpu_semaphore.get(block=True)
    with suppress_output():
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"
        img = decon(
            img,
            deconvolution_args['otf'],
            # deconvolution_args['psf'],
            fpattern=None,  # str: used to filter files in a directory, by default "*.tif"
            n_iters=deconvolution_args['n_iters'] // num_gaussian_decons,  # int: Number of iterations, by default 10
            dzdata=deconvolution_args['dz_data'],  # float: Z-step size of data, by default 0.5 um
            dxdata=deconvolution_args['dx_data'],  # float: XY pixel size of data, by default 0.1 um
            dzpsf=deconvolution_args['dz_psf'],  # float: Z-step size of the OTF, by default 0.1 um
            dxpsf=deconvolution_args['dxy_psf'],  # float: XY pixel size of the OTF, by default 0.1 um
            background=deconvolution_args['background'] / num_gaussian_decons,  # int or 'auto': background to subtract.
            wavelength=deconvolution_args['wavelength_em'],  # int: Emission wavelength in nm
            na=deconvolution_args['na'],  # float: Numerical Aperture (default: {1.5})
            nimm=deconvolution_args['nimm'],  # float: Refractive index of immersion medium (default: {1.3})
        )
    if gpu_semaphore is not None:
        gpu_semaphore.put(gpu)
    return img


def process_cube(
        input_file: Path,
        output_file: Path,
        need_destripe: bool = False,
        gaussian_sigma=None,
        gaussian_mode: str = "single",
        need_deconvolution: bool = False,
        deconvolution_args: dict = None,
        gpu_semaphore: Queue = None,
):
    if gaussian_sigma is None:
        gaussian_sigma = [0]
    if gaussian_mode not in ("single", "triple"):
        raise ValueError(f"Invalid gaussian_mode: {gaussian_mode}")

    return_code = 0
    output_file.parent.mkdir(exist_ok=True, parents=True)
    # Check if any Gaussian is requested at all
    gauss_requested = (gaussian_mode == "single" and gaussian_sigma[0] > 0) or \
                      (gaussian_mode == "triple" and any(x > 0 for x in gaussian_sigma))
    if need_destripe or gauss_requested or need_deconvolution:
        img, header = read(input_file.__str__())
        dtype = img.dtype
        if is_uniform_3d(img):
            copy(input_file, output_file)
        else:
            if img.dtype != float32:
                img = img.astype(float32)

            if need_destripe:
                img = rot90(img, k=1, axes=(1, 2))
                for idx in range(0, img.shape[0], 1):
                    if not is_uniform_2d(img[idx]):
                        img[idx] = filter_streaks(img[idx], sigma=(1, 1), wavelet="db9", bidirectional=True)
                img = rot90(img, k=-1, axes=(1, 2))

            if need_deconvolution and deconvolution_args is not None:
                if deconvolution_args['contrast_enhancement_factor'] > 1:
                    img /= deconvolution_args['contrast_enhancement_factor']

                img_decon = pad_to_good_dim(img, deconvolution_args['otf_shape'])
                num_gaussian_decons = max(1, deconvolution_args['n_iters'] // deconvolution_args['dg_interation'])

                for i in range(num_gaussian_decons):
                    # SINGLE GAUSSIAN: use your special Z logic for first iteration, then isotropic
                    if gaussian_mode == 'single' and gaussian_sigma[0] > 0:
                        if i == 0:
                            sigma = (
                                gaussian_sigma[0],
                                gaussian_sigma[0],
                                (round(gaussian_sigma[0], 0) + (2.0 if deconvolution_args['doubled_psf'] else 1.5))
                            )
                        else:
                            sigma = (gaussian_sigma[0],) * 3
                        gaussian(img_decon, sigma=sigma, output=img_decon)
                    # TRIPLE GAUSSIAN: use user values for all axes, all iterations
                    elif gaussian_mode == 'triple' and any(x > 0 for x in gaussian_sigma):
                        gaussian(img_decon, sigma=tuple(gaussian_sigma) if i == 0 else 0.5, output=img_decon)

                    img_decon = apply_deconvolution(img_decon, deconvolution_args, gpu_semaphore, num_gaussian_decons)

                    # from PIL import Image
                    # img = trim_to_shape(img.shape, img_decon.copy())
                    # if img.dtype != dtype and np_d_type(dtype).kind in ("u", "i"):
                    #     clip(img, iinfo(dtype).min, iinfo(dtype).max, out=img)
                    #     img = img.astype(dtype)
                    # image_stack = [Image.fromarray(_) for _ in img]
                    # image_stack[0].save(output_file.parent / (output_file.stem + f"_{i}.tif"),
                    #                     save_all=True,
                    #                     append_images=image_stack[1:],
                    #                     compression='tiff_lzw')

                # resize image to match original
                img = trim_to_shape(img.shape, img_decon)
            elif gauss_requested:
                if gaussian_mode == 'single':
                    gaussian(img, sigma=gaussian_sigma[0], output=img)
                elif gaussian_mode == 'triple':
                    gaussian(img, sigma=tuple(gaussian_sigma), output=img)

        if img.dtype != dtype and np_d_type(dtype).kind in ("u", "i"):
            clip(img, iinfo(dtype).min, iinfo(dtype).max, out=img)
            img = img.astype(dtype)
        tmp_file = output_file.parent / (output_file.name + ".tmp")
        write(file=tmp_file.__str__(), data=img, header=header, compression_level=9)
        tmp_file.rename(output_file)
    return return_code


def main(args):
    return_code = 0

    # gpu_semaphore = None
    # print(f"args.threads_per_gpu: {args.threads_per_gpu}, cuda_device_count: {num_gpus}, args.exclude_gpus: {args.exclude_gpus}")
    gpus = tuple(gpu for gpu in range(num_gpus) if str(gpu) not in args.exclude_gpus)
    cycling_gpus = cycle(gpus)
    gpu_semaphore = Queue(maxsize=len(gpus) * args.threads_per_gpu)
    for _ in range(args.threads_per_gpu):
        for gpu in gpus:
            gpu_semaphore.put(gpu)

    args_list = make_a_list_of_input_output_paths(args)
    num_images = len(args_list)
    if args.num_processes == 1:
        # could be used for debugging purpose
        list(tqdm(map(lambda _: process_cube(**_), args_list),
                  total=num_images, desc="FNT Cube Processor", unit=" cubes"))
    else:
        args_queue = Queue(maxsize=num_images)
        for item in args_list:
            args_queue.put(item)
        del args_list
        workers = min(args.num_processes, num_images)
        progress_queue = Queue()
        for _ in range(workers):
            MultiProcessQueueRunner(progress_queue, args_queue,
                                    gpu_semaphore=gpu_semaphore, gpu=next(cycling_gpus), fun=process_cube, timeout=None,
                                    replace_timeout_with_dummy=False).start()
        return_code = progress_manager(progress_queue, workers, num_images, desc="FNT Cube Processor", unit=" cubes")
        args_queue.cancel_join_thread()
        args_queue.close()
        progress_queue.cancel_join_thread()
        progress_queue.close()
    return return_code


if __name__ == '__main__':
    set_start_method("spawn")
    freeze_support()
    if sys.platform == "win32":
        psutil.Process().nice(getattr(psutil, "IDLE_PRIORITY_CLASS"))
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        nvidia_smi = "nvidia-smi.exe"
    elif sys.platform == 'linux' and 'microsoft' not in uname().release.lower():
        psutil.Process().nice(value=19)
        os.environ["NUMPY_MADVISE_HUGEPAGE"] = "1"
        os.environ["TERM"] = "xterm"
        nvidia_smi = "nvidia-smi"
    else:
        print("yet unsupported OS")
        raise RuntimeError

    num_gpus = str(check_output([nvidia_smi, "-L"])).count('UUID')
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{','.join(map(str, range(num_gpus)))}"
    print(f"{','.join(map(str, range(num_gpus)))}")
    parser = ArgumentParser(
        description="Process FNT cubes in parallel\n\n",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="Developed 2023 by Keivan Moradi at UCLA, Hongwei Dong Lab (B.R.A.I.N.) \n"
    )
    num_processes = cpu_count(logical=False) + 4
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path folder containing all nrrd files.")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output folder for converted files.")
    parser.add_argument("--num_processes", "-n", type=int, required=False,
                        default=num_processes,
                        help="Number of CPU cores.")
    parser.add_argument("--gaussian", "-g", type=float, nargs='+', required=False,
                        default=[0.5],
                        help=(
            "Sigma(s) of a 3D Gaussian filter. "
            "You can provide either 1 value (e.g. -g 0.5) or 3 values (e.g. -g 0.5 0.5 1.5) "
            "corresponding to sigma_x, sigma_y, sigma_z. For the first iteration in deconvolution, "
            "If 1 value is given, the filter uses:\n"
            "    sigma = (v, v, round(v)+2.0 if doubled_psf else round(v)+1.5) "
            "If 3 values are provided, they are used directly for (sigma_x, sigma_y, sigma_z) with no modification."
        )
    )

    parser.add_argument("--destripe", "-ds", default=False, action=BooleanOptionalAction,
                        help="Enables axial image destriping. Default is --no-destripe.")
    parser.add_argument("--deconvolution", "-d", default=True, action=BooleanOptionalAction,
                        help="Apply a deconvolution after destriping.  Default is --deconvolution.")
    parser.add_argument("--exclude_gpus", nargs='*',
                        default=[],
                        help="Exclude GPUs during deconvolution.")
    parser.add_argument("--threads_per_gpu", type=int, required=False,
                        default=num_processes // num_gpus,
                        help=f"Number of thread per GPU. Default is {num_processes // num_gpus}")
    parser.add_argument("--dxy", "-dxy", type=float, required=False,
                        default=0.7,
                        help="Voxel size of x and y dimensions in micrometers: "
                             "0.4 for 15x and 0.7 (default) for 9x lenses.")
    parser.add_argument("--dz", "-dz", type=float, required=False,
                        default=1.4,
                        help="Voxel size of z dimension in micrometers.")
    parser.add_argument("--f_cylinder_lens", "-fc", type=float, required=False,
                        default=240.0,
                        help="f cylinder lens dimension. Default value is 240.")
    parser.add_argument("--slit_width", "-dw", type=float, required=False,
                        default=12.0,
                        help="Slit width.  Default value is 12.")
    parser.add_argument("--wavelength_ex", "-ex", type=float, required=False,
                        default=488,
                        help="Excitation wavelength in nm. For example 488, 561, and 647. by default 488.")
    parser.add_argument("--wavelength_em", "-em", type=float, required=False,
                        default=525,
                        help="Emission wavelength in nm. For example 525, 600, and 690. by default 525.")
    parser.add_argument("--na", "-na", type=float, required=False,
                        default=0.40,
                        help="Numerical Aperture, by default 0.4.")
    parser.add_argument("--nimm", "-im", type=float, required=False,
                        default=1.42,
                        help="Refractive index of immersion medium, by default 1.42.")
    parser.add_argument("--background", "-b", type=float, required=False,
                        default=0,
                        help="int or 'auto': User-supplied background to subtract.")
    parser.add_argument("--n_iters", "-it", type=int, required=False,
                        default=6,
                        help="int: Number of iterations, by default 12")
    parser.add_argument("--dg_interation", "-dgi", type=int, required=False,
                        default=3,
                        help="int: Apply the 3D Gaussian filter after every dg_interation of deconvolution, "
                             "by default 3")
    parser.add_argument("--contrast_enhancement_factor", "-cef", type=float, required=False,
                        default=1,
                        help="float: Divide the image by this value before deconvolution to reduce background intensity. "
                             "This helps control contrast enhancement, as deconvolution amplifies the foreground signal. "
                             "by default 1. Suggested values are 2 to 4.")
    parser.add_argument("--doubled_psf", "-dpsf", default=False, action=BooleanOptionalAction,
                        help="Use a specific psf that can better eliminate the doubling. default is --no-doubled_psf.")

    args_ = parser.parse_args()

    # --- Gaussian Argument Normalization ---
    if len(args_.gaussian) == 1:
        gaussian_mode = 'single'
        gaussian_sigma = float(args_.gaussian[0])
    elif len(args_.gaussian) == 3:
        gaussian_mode = 'triple'
        gaussian_sigma = tuple(float(x) for x in args_.gaussian)
    else:
        raise ValueError(
            "The --gaussian/-g argument must be either a single value or exactly three values (sigma_x sigma_y sigma_z)."
        )

    args_.gaussian_mode = gaussian_mode
    args_.gaussian_sigma = gaussian_sigma

    main(args_)
