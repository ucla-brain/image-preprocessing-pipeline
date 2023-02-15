from supplements.downsampling import TifStack
from scipy.ndimage import zoom
from pathlib import Path
from numpy import ndarray, zeros
from tifffile import imwrite


source = Path(r"D:\tmp_tif")
destination = Path(r"D:\tmp_tif2x")
destination.mkdir(exist_ok=True)
img: TifStack = TifStack(source)
img3d: ndarray = zeros(img.shape, dtype=img.dtype)
# img3d = ndarray(list(map(lambda idx: img[idx], range(img.nz))))
for idx in range(img.nz):
    img3d[idx] = img[idx]
img3d = zoom(img3d, 2.0)
for idx in range(img3d.shape[0]):
    imwrite(destination/f"img_{idx:06}.tif", img3d[idx])
