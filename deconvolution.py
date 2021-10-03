from pycudadecon import decon
from tifffile import imwrite
result = decon('/mnt/md0/hpc_ex642_em680_04_04.tif', '/mnt/md0/psf_ex642_em680.tif')
imwrite('/mnt/md0/hpc_ex642_em680_04_04_deconvolved.tif', result)
