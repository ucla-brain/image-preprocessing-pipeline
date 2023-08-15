from typing import Union
from pathlib import Path


class FileStack:
    """
        Stores a list of file paths
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
        self.current = 0  # this should be 0.

    def get_count(self):
        return len(self.files)

    def get_next(self):
        if self.current == len(self.files) - 1:
            return None
        self.current += 1
        return self.files[self.current - 1]

    def get_index(self, index):
        if index < 0 or index >= len(self.files):
            return None
            # raise IndexError("Invalid index accessed in FileStack")
        return self.files[index]
