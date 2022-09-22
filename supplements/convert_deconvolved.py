# this is a script to convert the 15x deconvouted 2d-series to 6x/12x 2d-series

from pystripe.core import batch_filter
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import freeze_support
from tifffile import imread


def convert_deconvolved():
    img = imread(str(list(input_path.glob("*.tif"))[0]))

    # 15x to 6x
    if magnification == '6x':
        batch_filter(
            input_path,
            output_path,
            compression=('ZLIB', 1),
            dtype="uint8",
            tile_size=img.shape,
            down_sample=(2, 2),
            new_size=(img.shape[0] * 0.42 / 1, img.shape[1] * 0.42 / 1)
        )
    # 15x to 12x
    elif magnification == '12x':
        batch_filter(
            input_path,
            output_path,
            compression=('ZLIB', 1),
            dtype="uint8",
            tile_size=img.shape,
            # down_sample=(2, 2),
            new_size=(img.shape[0] * 0.42 / 0.5, img.shape[1] * 0.42 / 0.5)
        )


if __name__ == '__main__':
    freeze_support()
    parser = ArgumentParser(
        description="Convert 15x deconvoluted image to 6x and 12x"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to input TIF image")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Path to output TIF image")
    parser.add_argument("--magnification", "-m", type=str, required=True,
                        help="12x or 6x")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    magnification = args.magnification

    convert_deconvolved()
