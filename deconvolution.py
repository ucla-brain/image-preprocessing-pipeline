from pycudadecon import decon
from tifffile import imwrite
import subprocess
result = decon(
    '/mnt/md0/hpc_ex642_em680_04_04.tif',
    '/mnt/md0/psf_ex642_em680.tif',
    fpattern='*.tif',  # str: used to filter files in a directory, by default "*.tif"
    dzdata=1,  # float: Z-step size of data, by default 0.5 um
    dxdata=0.422,  # float: XY pixel size of data, by default 0.1 um
    dzpsf=1.0,  # float: Z-step size of the OTF, by default 0.1 um
    dxpsf=1.0,  # float: XY pixel size of the OTF, by default 0.1 um
    deskew=0.0,  # float: Deskew angle. If not 0.0 then deskewing will be performed before deconvolution, by default 0
    rotate=0.0,  # float: Rotation angle; if not 0.0 then rotation will be performed around Y axis after deconvolution, by default 0
    width=0,  # int: If deskewed, the output image's width, by default 0 (do not crop)
    background=115,  # int or 'auto': User-supplied background to subtract.
                     # If 'auto', the median value of the last Z plane will be used as background. by default 80
    n_iters=9,  # int: Number of iterations, by default 10
    shift=0,  # int: If deskewed, the output image's extra shift in X (positive->left), by default 0
    save_deskewed=False,  # bool: Save deskewed raw data as well as deconvolution result, by default False
    napodize=15,  # int: Number of pixels to soften edge with, by default 15
    nz_blend=0,  # int: Number of top and bottom sections to blend in to reduce axial ringing, by default 0
    pad_val=0.0,  # float: Value with which to pad image when deskewing, by default 0.0
    dup_rev_z=False,  # bool: Duplicate reversed stack prior to decon to reduce axial ringing, by default False

    wavelength=642,  # int: Emission wavelength in nm (default: {520})
    na=0.4,  # float: Numerical Aperture (default: {1.25})
    nimm=1.52,  # float: Refractive index of immersion medium (default: {1.3})
    otf_bgrd=None,  # int, None: Background to subtract. "None" = autodetect. (default: {None})
    krmax=0,  # int: Pixels outside this limit will be zeroed (overwriting estimated value from NA and NIMM) (default: {0})
    fixorigin=10,  # int: for all kz, extrapolate using pixels kr=1 to this pixel to get value for kr=0 (default: {10})
    cleanup_otf=False,  # bool: Clean-up outside OTF support (default: {False})
    max_otf_size=60000,  # int: Make sure OTF is smaller than this many bytes.
                         # Deconvolution may fail if the OTF is larger than 60KB (default: 60000)
)
imwrite('/mnt/md0/hpc_ex642_em680_04_04_deconvolved.tif', result)
subprocess.call(
    r"wine "
    r"./imaris/ImarisConvertiv.exe "
    r"-i /mnt/md0/hpc_ex642_em680_04_04_deconvolved.tif "
    r"-o /mnt/md0/hpc_ex642_em680_04_04_deconvolved.ims",
    shell=True
)
