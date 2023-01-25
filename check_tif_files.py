"""
Read tif files and if the file was damaged print its name.
"""
import sys
from pathlib import Path
from multiprocessing import freeze_support
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pystripe.core import imread_tif_raw_png, glob_re
from tsv.convert import calculate_cores_and_chunk_size
from supplements.cli_interface import PrintColors


def test_image(file: Path):
    try:
        # if not isinstance(file, Path):
        #     file = Path(file)
        if file.is_file() and file.suffix == ".tif":
            with ThreadPoolExecutor(1) as tp:
                future = tp.submit(imread_tif_raw_png, file)
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
    with ProcessPoolExecutor(60) as pool:
        files = list(glob_re(r"\.(?:tiff?|raw)$", source))
        cores, chunks = calculate_cores_and_chunk_size(len(files), cores=60, pool_can_handle_more_than_61_cores=False)
        list(pool.map(test_image, files, chunksize=chunks))
