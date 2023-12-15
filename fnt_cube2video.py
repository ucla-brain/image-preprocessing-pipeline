from pystripe.core import *
from multiprocessing import freeze_support
from subprocess import call


def make_a_list_of_input_output_paths(input_folder: Path, output_folder: Path):
    def output_file(input_file: Path):
        return {
            "input_file": input_file,
            "output_file": (output_folder / input_file.relative_to(input_folder))
        }
    return tuple(map(output_file, input_folder.rglob("*.nrrd")))


def convert_cube_to_video(input_file: Path, output_file: Path):
    return_code = 0
    if not output_file.exists():
        output_file.parent.mkdir(exist_ok=True, parents=True)
        return_code = call(f"fnt-cube2video {input_file} {output_file}", shell=True)
    return return_code


def main(args):
    args_list = make_a_list_of_input_output_paths(Path(args.input), Path(args.output))
    num_images = len(args_list)
    args_queue = Queue(maxsize=num_images)
    for item in args_list:
        args_queue.put(item)
    del args_list
    workers = min(args.num_processes, num_images)
    progress_queue = Queue()
    for _ in range(workers):
        MultiProcessQueueRunner(progress_queue, args_queue, fun=convert_cube_to_video, timeout=None).start()

    return_code = progress_manager(progress_queue, workers, num_images)
    args_queue.cancel_join_thread()
    args_queue.close()
    progress_queue.cancel_join_thread()
    progress_queue.close()
    return return_code


if __name__ == '__main__':
    freeze_support()
    parser = ArgumentParser(
        description="Convert cube to video in parallel\n\n",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="Developed 2023 by Keivan Moradi at UCLA, Hongwei Dong Lab (B.R.A.I.N.) \n"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path folder containing all nrrd files.")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output folder for converted files.")
    parser.add_argument("--num_processes", "-n", type=int, required=False,
                        default=cpu_count(logical=False),
                        help="Number of CPU cores.")
    main(parser.parse_args())

