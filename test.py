from pystripe.core import batch_filter, MultiProcessQueueRunner
from pathlib import Path
from multiprocessing import freeze_support
from supplements.cli_interface import PrintColors
from process_images import merge_all_channels
import os


def main():
    # batch_filter(
    #     Path(r"E:\sm20220717_01_40X_ZStep2um_20220813_175015_17_F50\Ex_642_Em_680"),
    #     Path(r"D:\test_interruption"),
    #     lightsheet=True,
    #     compression=("ZLIB", 0)
    # )
    merge_all_channels(
        [
            Path(r"/data/20220818_12_24_18_SW220405_05_LS_6x_1000z_stitched/Ex_488_Em_525_tif"),
            Path(r"/data/20220818_12_24_18_SW220405_05_LS_6x_1000z_stitched/Ex_642_Em_680_tif")
        ],
        Path(r"/data/20220818_12_24_18_SW220405_05_LS_6x_1000z_stitched/test"),
        workers=128,
        compression=("ZLIB", 0)
    )


if __name__ == '__main__':
    # os.environ['MKL_NUM_THREADS'] = '1'
    # os.environ['NUMEXPR_NUM_THREADS'] = '1'
    # os.environ['OMP_NUM_THREADS'] = '1'
    freeze_support()
    main()



