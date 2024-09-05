import os
import sys
from argparse import RawDescriptionHelpFormatter, ArgumentParser, BooleanOptionalAction
from multiprocessing import freeze_support, Queue
from pathlib import Path
from platform import uname
from subprocess import call

from nrrd import read, write
import psutil
from numpy import rot90, float32
from psutil import cpu_count
from skimage.filters import gaussian
from tqdm import tqdm

from pystripe.core import (filter_streaks, is_uniform_2d, MultiProcessQueueRunner, progress_manager, cuda_device_count,
                           cuda_get_device_properties, USE_PYTORCH)

from LsDeconvolveMultiGPU.deconvolution import new_deconvolution, generate_psf



def make_a_list_of_input_output_paths(args):
    input_folder = Path(args.input)
    output_folder = Path(args.output)

    def output_file(input_file: Path):
        return {
            "input_file": input_file,
            "output_file": (output_folder / input_file.relative_to(input_folder)),
            "need_destripe": args.destripe,
            "need_gaussian": args.gaussian,
            "need_video": args.video
        }

    return tuple(map(output_file, input_folder.rglob("*.nrrd")))


def process_cube(
        input_file: Path,
        output_file: Path,
        need_destripe: bool = False,
        need_gaussian: bool = False,
        need_video: bool = False,
        need_deconvolution: bool = False
):
    return_code = 0
    if not output_file.exists():
        output_file.parent.mkdir(exist_ok=True, parents=True)
        if need_destripe or need_gaussian:
            img, header = read(input_file.__str__())
            dtype = img.dtype
            if need_destripe:
                img = rot90(img, k=1, axes=(1, 2))
                for idx in range(0, img.shape[0], 1):
                    if not is_uniform_2d(img[idx]):
                        img[idx] = filter_streaks(img[idx], sigma=(1, 1), wavelet="db9", bidirectional=True)
                img = rot90(img, k=-1, axes=(1, 2))
            if need_gaussian:
                if img.dtype != float32:
                    img = img.astype(float32)
                gaussian(img, 1, output=img)
                img = img.astype(dtype)
            write(filename=output_file.__str__(), data=img, header=header, compression_level=1)
            if need_deconvolution:
                psf, dxy_psf, full_half_with_maxima_xy, full_half_with_maxima_z = generate_psf(
                     dxy=422.0,
                     f_cylinder_lens=240.0,
                     slit_width=12.0,
                 )
                # TODO: FIX THIS
                new_deconvolution(
                    input_path=None,
                    output_path=None,
                    psf=psf,
                    dxy_psf=dxy_psf,
                    convert_ims=False
                )
            if need_video:
                input_file = output_file
        if need_video:
            return_code = call(f"{fnt_cube2video} {input_file} {output_file}", shell=True)
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
        fnt_cube2video = Path(r".") / "fnt" / "Windows" / "fnt-cube2video.exe"
        psutil.Process().nice(getattr(psutil, "IDLE_PRIORITY_CLASS"))
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
    elif sys.platform == 'linux' and 'microsoft' not in uname().release.lower():
        fnt_cube2video = Path(r".") / "fnt" / "Linux" / "fnt-cube2video"
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
                        help="Destripe the image by default. Disable by --no_destripe.")
    parser.add_argument("--gaussian", "-g", default=False, action=BooleanOptionalAction,
                        help="Apply a 3D gaussian filter after destriping. Default is --no_gaussian.")
    parser.add_argument("--video", "-v", default=False, action=BooleanOptionalAction,
                        help="Convert cubes to video format. Default is --no_video.")
    parser.add_argument("--use_gpu", default=False, action=BooleanOptionalAction,
                        help="Use gpu acceleration for destriping. Default is --no_use_gpu.")
    parser.add_argument("--threads_per_gpu", type=int, required=False,
                        default=8,
                        help="Number of thread per GPU.")
    main(parser.parse_args())
