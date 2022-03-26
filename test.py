from multiprocessing import Queue, Process, Manager, freeze_support, cpu_count
from queue import Empty
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, TimeoutError
from concurrent.futures.process import BrokenProcessPool
from typing import List, Iterator, Tuple
from numpy import uint8, uint16, zeros, ndarray
from tqdm import tqdm
from time import sleep, time
from pathlib import Path
from supplements.cli_interface import PrintColors
from pystripe.core import imread_tif_raw, convert_to_8bit_fun, calculate_cores_and_chunk_size
import h5py
from tifffile import imwrite
from numpy import where


def image_processing_pool(
        fun,
        args_list: List[dict, ...],
        max_processors: int,
        chunks: int,
        progress_bar_name: str,
        input_file_key: str,
        output_file_key: str,
        default_image: ndarray,
):
    class MultiProcess(Process):
        def __init__(self, queue: Queue, function, shared_list: List[dict], indices: Iterator[int], timeout: float):
            Process.__init__(self)
            self.daemon = False
            self.queue = queue
            self.function = function
            self.shared_list = shared_list
            self.ims_file = None
            self.indices = indices
            self.die = False
            self.timeout = timeout

        def run(self):
            running = True
            pool = ProcessPoolExecutor(max_workers=1)
            function = self.function
            timeout = self.timeout
            for idx in self.indices:
                if self.die:
                    break
                args = self.shared_list[idx]
                try:
                    future = pool.submit(function, args)
                    future.result(timeout=timeout)
                except (BrokenProcessPool, TimeoutError):
                    output_file: Path = args[output_file_key]
                    print(f"\033[93m"
                          f"\nwarning: timeout reached for processing input file:\n\t{args[input_file_key]}\n\t"
                          f"a dummy (zeros) image is saved as output instead:\n\t{output_file}"
                          f"\033[0m")
                    if not output_file.exists() and default_image is not None:
                        imwrite(output_file, default_image)
                    pool.shutdown()
                    pool = ProcessPoolExecutor(max_workers=1)
                except KeyboardInterrupt:
                    self.die = True
                except Exception as inst:
                    print(
                        f"\033[93m"
                        f"\nwarning: process failed for {args}."
                        f"exception instance: {type(inst)}"
                        f"exception arguments: {inst.args}"
                        f"exception: {inst}"
                        f"\033[0m")
                self.queue.put(running)
            pool.shutdown()
            running = False
            self.queue.put(running)

    progress_queue = Queue()
    manager = Manager()
    args_list = manager.list(args_list)
    num_images = len(args_list)
    progress_bar = tqdm(total=num_images, ascii=True, smoothing=0.01, mininterval=1.0, unit="img",
                        desc=progress_bar_name)
    process_list: List[MultiProcess, ...] = []
    for worker in range(max_processors - 1):
        process_list += [MultiProcess(progress_queue, fun, args_list, range(worker * chunks, (worker + 1) * chunks))]
        process_list[-1].start()
    process_list += [MultiProcess(
        progress_queue, fun, args_list, range(num_images - num_images % (max_processors - 1), num_images))]
    process_list[-1].start()
    running_processes = max_processors
    while running_processes > 0:
        try:
            still_running = progress_queue.get()
            if still_running:
                progress_bar.update(1)
            else:
                running_processes -= 1
        except Empty:
            sleep(1)
        except KeyboardInterrupt:
            print("Please be patient. Terminating processes with dignity! :)")
            for process in process_list:
                process.die = True
    progress_bar.close()


# ----------------------------------------------------------------------------------------------------------------------
def convert_ims_to_tif(
        ims_path: Path, tif_path: Path, tif_file_prefix: str, indices: range, channel: int, queue: Queue,
):
    dark = 120
    bit_shift = 0
    with h5py.File(ims_path, "r") as file:
        images = file[f"DataSet/ResolutionLevel 0/TimePoint 0/Channel {channel}/Data"]
        for idx in indices:
            image = images[idx]
            if dark > 0:
                image = where(image > dark, image - dark, 0)
            image = convert_to_8bit_fun(image, bit_shift_to_right=bit_shift)
            imwrite(tif_path / f"{tif_file_prefix}_{idx:04}.tif", image)
            queue.put(False)
    queue.put(True)


if __name__ == "__main__":
    freeze_support()
    source = Path(r"X:\Mitchell\Etv1-MORF\M19-1\10x\MS19-1_10x_c01\MS19-1_10x_c01_Stitched.ims")
    destination = Path(r"X:\Mitchell\Etv1-MORF\M19-1\10x\MS19-1_10x_c01\tif")
    destination.mkdir(exist_ok=True)
    channel = 0
    with h5py.File(source, "r") as f:
        shape = f[f"DataSet/ResolutionLevel 0/TimePoint 0/Channel {channel}/Data"].shape

    args_list = [{
        "ims_path": source,
        "tif_path": destination,
        "tif_file_prefix": "img",
        "idx": idx,
        "channel": channel,

    } for idx in range(shape[0])]
    cores, chunks = calculate_cores_and_chunk_size(shape[0], pool_can_handle_more_than_61_cores=True)
    image_processing_pool(convert_ims_to_tif, cores, chunks, "ims2tif")
