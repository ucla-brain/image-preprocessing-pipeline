"""
Read tif files and if the file was damaged print its name.
"""
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import freeze_support
from pathlib import Path
from typing import List

from psutil import cpu_count
from nrrd import read as read_nrrd

from pystripe.core import imread_tif_raw_png, glob_re
from supplements.cli_interface import PrintColors
from tsv.convert import calculate_cores_and_chunk_size


def test_image(file: Path):
    try:
        if not isinstance(file, Path):
            file = Path(file)
        with ThreadPoolExecutor(1) as tp:
            if file.suffix.lower() in (".tif", ".tiff", ".raw", ".png"):
                future = tp.submit(imread_tif_raw_png, file)
            if file.suffix.lower() == ".nrrd":
                future = tp.submit(read_nrrd, file)
            future.result(timeout=200)
    except Exception as e:
        print(f"{file} {type(e)} {e.args} {e}\n")
        file.unlink()


if __name__ == "__main__":
    freeze_support()
    if len(sys.argv) == 2:
        source = Path(sys.argv[1]).absolute()
    else:
        print(f"{PrintColors.FAIL}Only one argument is allowed!{PrintColors.ENDC}")
        raise RuntimeError
    files: List[Path] = list(glob_re(r"\.(?:tiff?|raw|png)$", source))
    num_threads, chunks = calculate_cores_and_chunk_size(
        len(files),
        cores=cpu_count(logical=True),
        pool_can_handle_more_than_61_cores=not sys.platform.lower() == 'win32')
    with ProcessPoolExecutor(num_threads) as pool:
        list(pool.map(test_image, files, chunksize=chunks))
