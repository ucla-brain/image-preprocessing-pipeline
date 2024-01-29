import os
import sys
import nrrd
import psutil
from pystripe.core import *
from multiprocessing import freeze_support
from subprocess import call
from platform import uname
from numpy import rot90
from pystripe.core import filter_streaks
from scipy.ndimage import gaussian_filter
from argparse import RawDescriptionHelpFormatter, ArgumentParser, BooleanOptionalAction


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
        need_video: bool = False
):
    return_code = 0
    if not output_file.exists():
        output_file.parent.mkdir(exist_ok=True, parents=True)
        if need_destripe or need_gaussian:
            img, header = nrrd.read(input_file.__str__())
            dtype = img.dtype
            if need_destripe:
                img = rot90(img, k=1, axes=(1, 2))
                for idx in range(0, img.shape[0], 1):
                    if not is_uniform_2d(img[idx]):
                        img[idx] = filter_streaks(img[idx], sigma=(1, 1), bidirectional=True)
                img = rot90(img, k=-1, axes=(1, 2))
            if need_gaussian:
                img = gaussian_filter(img, sigma=1)
                img = img.astype(dtype)
            nrrd.write(filename=output_file.__str__(), data=img, header=header, compression_level=1)
            if need_video:
                input_file = output_file
        if need_video:
            return_code = call(f"{fnt_cube2video} {input_file} {output_file}", shell=True)
    return return_code


def main(args):
    args_list = make_a_list_of_input_output_paths(args)
    num_images = len(args_list)
    args_queue = Queue(maxsize=num_images)
    for item in args_list:
        args_queue.put(item)
    del args_list
    workers = min(args.num_processes, num_images)
    progress_queue = Queue()
    for _ in range(workers):
        MultiProcessQueueRunner(progress_queue, args_queue, fun=process_cube, timeout=None).start()
    return_code = progress_manager(progress_queue, workers, num_images, desc="FNT Cube Processor")
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
                        help="Destripe the image by default. In order to disable use --no_destripe.")
    parser.add_argument("--gaussian", "-g", default=False, action=BooleanOptionalAction,
                        help="Apply a 3D gaussian filter after destriping. Default is --no_gaussian.")
    parser.add_argument("--video", "-v", default=False, action=BooleanOptionalAction,
                        help="Convert cubes to video format. Default is --no_video.")
    main(parser.parse_args())

