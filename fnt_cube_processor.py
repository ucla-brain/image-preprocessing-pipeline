import os
import sys
import contextlib
from argparse import RawDescriptionHelpFormatter, ArgumentParser, BooleanOptionalAction
from multiprocessing import freeze_support, Queue
from pathlib import Path
from platform import uname

import psutil
from nrrd import read, write
from numpy import dtype as np_d_type
from numpy import rot90, float32, iinfo, clip
from psutil import cpu_count
from pycudadecon import make_otf, decon
from skimage.filters import gaussian
from tifffile import imwrite
from tqdm import tqdm
from shutil import copy

from LsDeconvolveMultiGPU.psf_generation import generate_psf
from pystripe.core import filter_streaks, is_uniform_2d, is_uniform_3d, MultiProcessQueueRunner, progress_manager


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
        )
        psf = rot90(psf, k=1, axes=(0, 2))
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
            nimm=args.nimm
        )
        # otf = imread(otf_file)

        print(f"dxy_psf: {dxy_psf}")

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
            nimm=args.nimm
        )

    def output_file(input_file: Path):
        return {
            "input_file": input_file,
            "output_file": (output_folder / input_file.relative_to(input_folder)),
            "need_destripe": args.destripe,
            "need_gaussian": args.gaussian,
            "need_deconvolution": args.deconvolution,
            "deconvolution_args": deconvolution_args
        }

    print("Deconvolution args:")
    print(deconvolution_args)

    return tuple(map(output_file, input_folder.rglob("*.nrrd")))


def make_deconvolution_args(
        otf: str,
        n_iters: int = 9,
        dz_data: float = 1.0,
        dx_data: float = 1.0,
        dz_psf: float = 1.0,
        dxy_psf: float = 1.0,
        background: int = 0,
        wavelength_em: int = 642,
        na: float = 0.4,
        nimm: float = 1.52
) -> dict:
    return {
        'otf': otf,
        'n_iters': n_iters,
        'dz_data': dz_data,
        'dx_data': dx_data,
        'dz_psf': dz_psf,
        'dxy_psf': dxy_psf,
        'background': background,
        'wavelength_em': wavelength_em,
        'na': na,
        'nimm': nimm
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


def process_cube(
        input_file: Path,
        output_file: Path,
        need_destripe: bool = False,
        need_gaussian: bool = False,
        need_deconvolution: bool = False,
        deconvolution_args: dict = None,
):
    return_code = 0
    if not output_file.exists():
        output_file.parent.mkdir(exist_ok=True, parents=True)
        if need_destripe or need_gaussian or need_deconvolution:
            img, header = read(input_file.__str__())
            dtype = img.dtype

            if is_uniform_3d(img):
                copy(input_file, output_file)
            else:
                if need_gaussian:
                    if img.dtype != float32:
                        img = img.astype(float32)
                    gaussian(img, .5, output=img)

                if need_destripe:
                    img = rot90(img, k=1, axes=(1, 2))
                    for idx in range(0, img.shape[0], 1):
                        if not is_uniform_2d(img[idx]):
                            img[idx] = filter_streaks(img[idx], sigma=(1, 1), wavelet="db9", bidirectional=True)
                    img = rot90(img, k=-1, axes=(1, 2))

                if need_deconvolution and deconvolution_args is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
                    if img.dtype != float32:
                        img = img.astype(float32)
                    with suppress_output():
                        img = decon(
                            img,
                            deconvolution_args['otf'],  # '/mnt/md0/psf_ex642_em680.tif',  #
                            fpattern=None,  # str: used to filter files in a directory, by default "*.tif"
                            n_iters=deconvolution_args['n_iters'],  # int: Number of iterations, by default 10
                            dzdata=deconvolution_args['dz_data'],  # float: Z-step size of data, by default 0.5 um
                            dxdata=deconvolution_args['dx_data'],  # float: XY pixel size of data, by default 0.1 um
                            dzpsf=deconvolution_args['dz_psf'],  # float: Z-step size of the OTF, by default 0.1 um
                            dxpsf=deconvolution_args['dxy_psf'],  # float: XY pixel size of the OTF, by default 0.1 um
                            background=deconvolution_args['background'], # int or 'auto': User-supplied background to subtract.
                            # If 'auto', the median value of the last Z plane will be used as background. by default 80
                            # rotate=0.0,
                            # # float: Rotation angle; if not 0.0 then rotation will be performed around Y axis after deconvolution, by default 0
                            # deskew=0.0,
                            # # float: Deskew angle. If not 0.0 then deskewing will be performed before deconvolution, by default 0
                            # width=0,  # int: If deskewed, the output image's width, by default 0 (do not crop)
                            # shift=0,  # int: If deskewed, the output image's extra shift in X (positive->left), by default 0
                            # pad_val=0.0,  # float: Value with which to pad image when deskewing, by default 0.0
                            # save_deskewed=False,  # bool: Save deskewed raw data as well as deconvolution result, by default False
                            napodize=8,  # int: Number of pixels to soften edge with, by default 15
                            # nz_blend=27,  # int: Number of top and bottom sections to blend in to reduce axial ringing, by default 0
                            # dup_rev_z=True,  # bool: Duplicate reversed stack prior to decon to reduce axial ringing, by default False

                            wavelength=deconvolution_args['wavelength_em'],  # 642 int: Emission wavelength in nm (default: {520})
                            na=deconvolution_args['na'],  # float: Numerical Aperture (default: {1.5})
                            nimm=deconvolution_args['nimm'],  # float: Refractive index of immersion medium (default: {1.3})
                            # otf_bgrd=None,  # int, None: Background to subtract. "None" = autodetect. (default: {None})
                            # krmax=0,  # int: Pixels outside this limit will be zeroed (overwriting estimated value from NA and NIMM) (default: {0})
                            # fixorigin=8,
                            # # int: for all kz, extrapolate using pixels kr=1 to this pixel to get value for kr=0 (default: {10})
                            # cleanup_otf=True,  # bool: Clean-up outside OTF support (default: {False})
                            # max_otf_size=60000,  # int: Make sure OTF is smaller than this many bytes.
                            # # Deconvolution may fail if the OTF is larger than 60KB (default: 60000)
                        )

                if img.dtype != dtype and np_d_type(dtype).kind in ("u", "i"):
                    clip(img, iinfo(dtype).min, iinfo(dtype).max, out=img)
                    img = img.astype(dtype)
                write(file=output_file.__str__(), data=img, header=header, compression_level=1)
    return return_code


def main(args):
    return_code = 0
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
        gpu_semaphore = None
        # if args.use_gpu:
        #     gpu_semaphore = Queue()
        #     for i in range(cuda_device_count()):
        #         for _ in range(args.threads_per_gpu):
        #             gpu_semaphore.put((f"cuda:{i}", cuda_get_device_properties(i).total_memory))
        for _ in range(workers):
            MultiProcessQueueRunner(progress_queue, args_queue,
                                    gpu_semaphore=gpu_semaphore, fun=process_cube, timeout=None).start()
        return_code = progress_manager(progress_queue, workers, num_images, desc="FNT Cube Processor", unit=" cubes")
        args_queue.cancel_join_thread()
        args_queue.close()
        progress_queue.cancel_join_thread()
        progress_queue.close()
    return return_code


if __name__ == '__main__':
    freeze_support()
    if sys.platform == "win32":
        psutil.Process().nice(getattr(psutil, "IDLE_PRIORITY_CLASS"))
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
    elif sys.platform == 'linux' and 'microsoft' not in uname().release.lower():
        psutil.Process().nice(value=19)
        os.environ["NUMPY_MADVISE_HUGEPAGE"] = "1"
        os.environ["TERM"] = "xterm"
    else:
        print("yet unsupported OS")
        raise RuntimeError
    parser = ArgumentParser(
        description="Process FNT cubes in parallel\n\n",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="Developed 2023 by Keivan Moradi at UCLA, Hongwei Dong Lab (B.R.A.I.N.) \n"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path folder containing all nrrd files.")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output folder for converted files.")
    parser.add_argument("--num_processes", "-n", type=int, required=False,
                        default=cpu_count(logical=False) + 4,
                        help="Number of CPU cores.")
    parser.add_argument("--destripe", default=True, action=BooleanOptionalAction,
                        help="Destripe the image by default. Disable by --no-destripe.")
    parser.add_argument("--gaussian", "-g", default=False, action=BooleanOptionalAction,
                        help="Apply a 3D gaussian filter after destriping. Default is --no-gaussian.")
    parser.add_argument("--deconvolution", "-d", default=False, action=BooleanOptionalAction,
                        help="Apply a deconvolution after destriping.  Default is --no-deconvolution.")
    parser.add_argument("--use_gpu", default=False, action=BooleanOptionalAction,
                        help="Use gpu acceleration for destriping. Default is --no-use-gpu.")
    parser.add_argument("--exclude_gpus", default=False,  # action=List[int],
                        help="Exclude GPUs during deconvolution.")
    parser.add_argument("--threads_per_gpu", type=int, required=False,
                        default=8,
                        help="Number of thread per GPU.")
    parser.add_argument("--dxy", type=float, required=False,
                        default=0.4,
                        help="Voxel size of x and y dimensions in micrometers")
    parser.add_argument("--dz", type=float, required=False,
                        default=0.8,
                        help="Voxel size of z dimension in micrometers")
    parser.add_argument("--f_cylinder_lens", type=float, required=False,
                        default=240.0,
                        help="f cylinder lens dimension")
    parser.add_argument("--slit_width", type=float, required=False,
                        default=12.0,
                        help="Slit width.")
    parser.add_argument("--wavelength_ex", type=float, required=False,
                        default=488,
                        help="Excitation wavelength in nm, by default 488")
    parser.add_argument("--wavelength_em", type=float, required=False,
                        default=525,
                        help="Emission wavelength in nm, by default 525")
    parser.add_argument("--na", type=float, required=False,
                        default=0.40,
                        help="Numerical Aperture, by default 0.4")
    parser.add_argument("--nimm", type=float, required=False,
                        default=1.42,
                        help="Refractive index of immersion medium, by default 1.42")
    parser.add_argument("--background", type=float, required=False,
                        default=0,
                        help="int or 'auto': User-supplied background to subtract")
    parser.add_argument("--n_iters", type=float, required=False,
                        default=10,
                        help="int: Number of iterations, by default 10")
    main(parser.parse_args())
