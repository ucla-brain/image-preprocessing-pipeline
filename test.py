from pathlib import Path
import os
import re
def glob_re(pattern: str, path: Path):
    """Recursively find all files having a specific name
        path: Path
            Search path
        pattern: str
            regular expression to search the file name.
    """
    regexp = re.compile(pattern, re.IGNORECASE)
    for p in os.scandir(path):
        # p = Path(p)
        if p.is_file() and regexp.search(p.name):
            yield Path(p.path)
        elif p.is_dir(follow_symlinks=False):  # 
            yield from glob_re(pattern, p.path)

for p in glob_re(r"\.(?:tiff?|raw)$", Path(r"C:\test-25G")):
    print(p)