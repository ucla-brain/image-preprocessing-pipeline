from pystripe.core import batch_filter
from pathlib import Path
from multiprocessing import freeze_support
from supplements.cli_interface import PrintColors


def main():
    batch_filter(
        Path(r"E:\sm20220717_01_40X_ZStep2um_20220813_175015_17_F50\Ex_642_Em_680"),
        Path(r"D:\test_interruption"),
        lightsheet=True,
        compression=("ZLIB", 0)
    )


if __name__ == '__main__':
    freeze_support()
    main()
