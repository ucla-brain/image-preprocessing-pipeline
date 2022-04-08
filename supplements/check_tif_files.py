"""
Read tif files and if the file was damaged print its name.
"""

from pathlib import Path
from multiprocessing import freeze_support
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pystripe.core import imread_tif_raw, calculate_cores_and_chunk_size, glob_re

source = Path(
    r"D:\20220331_15_18_23_SW220203_04_LS_15x_1000z_lightsheet_cleaned_tif_bitshift.g4.r0_downsampled")


def test_image(file: Path):
    try:
        # if not isinstance(file, Path):
        #     file = Path(file)
        if file.is_file() and file.suffix == ".tif":
            with ThreadPoolExecutor(1) as tp:
                future = tp.submit(imread_tif_raw, file)
                future.result(timeout=200)
    except Exception as e:
        print(f"{file} {type(e)} {e.args} {e}\n")
        file.unlink()


if __name__ == "__main__":
    freeze_support()
    with ProcessPoolExecutor(61) as pool:
        files = list(glob_re(r"\.(?:tiff?|raw)$", source))
        cores, chunks = calculate_cores_and_chunk_size(
            len(files), cores=61, pool_can_handle_more_than_61_cores=False)
        list(pool.map(test_image, files, chunksize=chunks))
