"""
Read tif files and if the file was damaged print its name.
"""

from pathlib import Path
from multiprocessing import freeze_support
from concurrent.futures import ProcessPoolExecutor
from pystripe.core import imread_tif_raw


source = Path(r"Y:\SmartSPIM_Data\2022_03_15\20220315_14_15_20_SW220203_03_LS_15x_1000z_lightsheet_cleaned_tif_bitshift.b3.r0_downsampled\Ex_642_Em_680")


def test_image(file: Path):
    try:
        return imread_tif_raw(file) if file.is_file() and file.suffix == ".tif" else None
    except Exception:
        print(file)


if __name__ == "__main__":
    freeze_support()
    with ProcessPoolExecutor(max_workers=61) as pool:
        for x_folder in source.iterdir():
            if x_folder.is_dir():
                for y_folder in x_folder.iterdir():
                    if y_folder.is_dir():
                        # y_folder_source2 = source2.joinpath(y_folder.relative_to(source))
                        # list(pool.map(test_image, map(lambda f: y_folder_source2/f.name, y_folder.iterdir())))
                        list(pool.map(test_image, y_folder.iterdir(), chunksize=16))