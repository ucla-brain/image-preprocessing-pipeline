from pathlib import Path
from typing import Union
from tifffile import natural_sorted

from pystripe.core import imread_tif_raw_png


# build a tif class with similar interface
class TifStack:
    """
    We need a tif stack with an interface that will load a slice one at a time
    We assume each tif has the same size
    """

    def __init__(self, input_directory: Union[Path, str]):
        if isinstance(input_directory, str):
            input_directory = Path(input_directory)
        self.input_directory = input_directory

        self.files = []
        for file in input_directory.iterdir():
            if file.is_file() and file.suffix.lower() in (".tif", ".tiff"):
                self.files += [file.__str__()]
        self.files = list(map(Path, natural_sorted(self.files)))
        self.suffix = self.files[0].suffix
        img = imread_tif_raw_png(self.files[0])
        self.dtype = img.dtype
        self.nyx = img.shape
        self.nz = len(self.files)
        self.shape = (self.nz, self.nyx[0], self.nyx[1])

    def __getitem__(self, i):
        if i < 0 or i >= self.nz:
            return None
        return imread_tif_raw_png(self.files[i])

    def close(self):
        pass
