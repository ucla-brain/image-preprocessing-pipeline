from tifffile import imread
from typing import Union
from pathlib import Path


# build a tif class with similar interface
class TifStack:
    """
    We need a tif stack with an interface that will load a slice one at a time
    We assume each tif has the same size
    We assume 16-bit images
    """

    def __init__(self, input_directory: Union[Path, str], pattern='*.tif'):
        if isinstance(input_directory, str):
            input_directory = Path(input_directory)
        self.input_directory = input_directory
        self.pattern = pattern
        self.files = sorted(input_directory.glob(pattern))
        if not self.files and pattern.lower() == '*.tif':
            self.files = sorted(input_directory.glob(pattern.lower()+'f'))
        elif not self.files and pattern.lower() == '*.tiff':
            self.files = sorted(input_directory.glob(pattern[:-1]))
        self.files.sort()
        self.suffix = self.files[0].suffix
        img = imread(self.files[0])
        self.dtype = img.dtype
        self.nyx = img.shape
        self.nz = len(self.files)
        self.shape = (self.nz, self.nyx[0], self.nyx[1])

    def __getitem__(self, i):
        return imread(self.files[i]) / (2 ** 16 - 1)

    def __len__(self):
        return len(self.files)

    def get_file(self, i):
        return self.files[i]

    def close(self):
        pass
