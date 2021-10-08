from pycudadecon import decon
from tifffile import imwrite
from scipy.optimize import fsolve
import scipy.special as sp
from scipy.integrate import quad
import numpy as np
import subprocess
import math
import cmath
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return np.real(func(x))

    def imag_func(x):
        return np.imag(func(x))

    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)

    return real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:]


def f1(p, x, y, z, lambda_val, numerical_aperture, refractive_index):
    return sp.jv(  # Bessel function of the first kind
        0,
        2.0 * math.pi * numerical_aperture * math.sqrt(x ** 2 + y ** 2) * p / (lambda_val * refractive_index)
    ) * cmath.exp(1j * (-cmath.pi * p ** 2 * z * numerical_aperture ** 2) / (lambda_val * refractive_index ** 2)) * p


def psf_eq(x, y, z, numerical_aperture, refractive_index, lambda_val):

    def f2(p):
        return f1(p, x, y, z, lambda_val, numerical_aperture, refractive_index)

    integral = complex_quadrature(f2, 0.0, 1.0)  # 'AbsTol', 1e-3

    return 4 * abs(integral[0])**2


def ls_psf_eq(x, y, z, numerical_aperture_obj, refractive_index, lambda_ex, lambda_em, numerical_aperture_ls):
    """calculates PSF at point (x,y,z)
    """
    psf_ls = psf_eq(z, 0, x, numerical_aperture_ls, refractive_index, lambda_ex)
    psf_obj = psf_eq(x, y, z, numerical_aperture_obj, refractive_index, lambda_em)
    return psf_ls * psf_obj


def generate_psf(
        lambda_em=642.0,
        lambda_ex=680.0,
        numerical_aperture=0.4,
        dxy=422.0,
        dz=1000.0,
        refractive_index=1.52,  # 1.45 of the lens
        f_cylinder_lens=240.0,  # 240
        slit_width=12.0,
):
    """generate point spread function matrix

    Parameters
    ----------
    lambda_em : float
        the wavelength of the emitted light in nm
    lambda_ex : float
        the wavelength of the emitted light in nm
    numerical_aperture : float
        numerical aperture objective of the lens
    dxy : float
        voxel size in x and y dimensions in nm (of the camera pixels)
    dz : float
        voxel size in z dimensions in nm (or the z-steps)
    refractive_index : float
        the refractive index of immersion medium
    f_cylinder_lens : float
        F cylinder lens in mm
    slit_width:
        slit aperture width in mm

    Returns
    -------
    psf : ndarray
        point spread function matrix
    dxy_psf : float
        corrected voxel size in x and y dimensions in um
    """

    resolution_xy = 0.61 * lambda_em / numerical_aperture  # the minimum distance of two distinguishable objects
    resolution_z = 2.0 * lambda_ex * refractive_index / numerical_aperture ** 2
    print(f"Two objects are distinguishable in xy-plane if they are {resolution_xy:.0f} nm apart. "
          f"The camera pixel size is {dxy:.0f} nm.\n"
          f"Two objects are distinguishable in z-axis if they are {resolution_z:.0f} nm apart. "
          f"The z-step is {dz:.0f} nm.")

    dxy_psf = min(dxy, resolution_xy / 3)  # the voxel size of PSF should be smaller than the diffraction-limited

    nxy, nz, full_half_with_maxima_xy, full_half_with_maxima_z = determine_psf_size(
        dxy_psf, dz, numerical_aperture, refractive_index, lambda_ex, lambda_em, f_cylinder_lens, slit_width,
        resolution_xy, resolution_z
    )

    numerical_aperture_ls = math.sin(math.atan(slit_width / (2.0 * f_cylinder_lens)))

    psf = sample_psf(dxy_psf, dz, nxy, nz, numerical_aperture, refractive_index, lambda_ex, lambda_em,
                     numerical_aperture_ls)

    print(f"full width half maxima of xy-plane is {full_half_with_maxima_xy:.1f} nm.\n"
          f"full width half maxima of z-axis is {full_half_with_maxima_z:.1f} nm.")

    return psf, dxy_psf, full_half_with_maxima_xy, full_half_with_maxima_z


def determine_psf_size(
        dxy_psf, dz, numerical_aperture, refractive_index, lambda_ex, lambda_em, f_cylinder_lens, slit_width,
        resolution_xy, resolution_z
):

    grid_size_xy = 2
    grid_size_z = 2

    numerical_aperture_ls = math.sin(math.atan(0.5 * slit_width / f_cylinder_lens))
    half_max = 0.5 * ls_psf_eq(
        0, 0, 0, numerical_aperture, refractive_index, lambda_ex, lambda_em, numerical_aperture_ls)

    # find zero crossings
    def fxy(x):
        return ls_psf_eq(
            x, 0, 0, numerical_aperture, refractive_index, lambda_ex, lambda_em, numerical_aperture_ls) - half_max

    def fz(x):
        return ls_psf_eq(
            0, 0, x, numerical_aperture, refractive_index, lambda_ex, lambda_em, numerical_aperture_ls) - half_max

    full_half_with_maxima_xy = 2 * abs(fsolve(fxy, np.array([resolution_xy/2], dtype='single'))[0])
    full_half_with_maxima_z = 2 * abs(fsolve(fz, np.array([resolution_z/2], dtype='single'))[0])

    nxy = math.ceil(grid_size_xy * full_half_with_maxima_xy / dxy_psf)
    nz = math.ceil(grid_size_z * full_half_with_maxima_z / dz)

    # ensure that the grid dimensions are odd
    if nxy % 2 == 0:
        nxy = nxy + 1
    if nz % 2 == 0:
        nz = nz + 1
    return nxy, nz, full_half_with_maxima_xy, full_half_with_maxima_z


def sample_psf(
        dxy=1.0, dz=1.0, nxy=5, nz=7,
        numerical_aperture_obj=1.0, rf=1.0, lambda_ex=1.0, lambda_em=1.0, numerical_aperture_ls=1.0):

    if nxy % 2 == 0 or nz % 2 == 0:
        print(f'function sample_psf: nxy is {nxy} and nz is {nz}, but must be odd!')
        raise RuntimeError

    psf = np.zeros(
        [(nz - 1) // 2 + 1, (nxy - 1) // 2 + 1, (nxy - 1) // 2 + 1],
        dtype='single')

    for z in range(0, (nz - 1) // 2 + 1):
        for y in range(0, (nxy - 1) // 2 + 1):
            for x in range(0, (nxy - 1) // 2 + 1):
                psf[z, y, x] = ls_psf_eq(
                    x * dxy, y * dxy, z * dz, numerical_aperture_obj, rf, lambda_ex, lambda_em, numerical_aperture_ls)

    # Since the PSF is symmetrical around all axes only the first Octant is calculated for computation efficiency.
    # The other 7 Octanes are obtained by mirroring around the respective axes
    psf = mirror8(psf)

    # normalize psf to integral one
    psf = psf / np.sum(psf)
    return psf


def mirror8(psf_quadrant):
    """mirrors the content of the first quadrant to all other quadrants to obtain the complete PSF
    """
    sx, sy, sz = np.array(psf_quadrant.shape, dtype='int') * 2 - 1
    cx, cy, cz = np.array([sx, sy, sz]) // 2

    result = np.zeros([sx, sy, sz], dtype='single')
    result[cx:sx, cy:sy, cz:sz] = psf_quadrant
    result[cx:sx, 0:cy + 1, cz:sz] = flip_3d(psf_quadrant, 0, 1, 0)
    result[0:cx + 1, 0:cy + 1, cz:sz] = flip_3d(psf_quadrant, 1, 1, 0)
    result[0:cx + 1, cy:sy, cz:sz] = flip_3d(psf_quadrant, 1, 0, 0)
    result[cx:sx, cy:sy, 0:cz + 1] = flip_3d(psf_quadrant, 0, 0, 1)
    result[cx:sx, 0:cy + 1, 0:cz + 1] = flip_3d(psf_quadrant, 0, 1, 1)
    result[0:cx + 1, 0:cy + 1, 0:cz + 1] = flip_3d(psf_quadrant, 1, 1, 1)
    result[0:cx + 1, cy:sy, 0:cz + 1] = flip_3d(psf_quadrant, 1, 0, 1)
    return result


def flip_3d(data, x, y, z):
    result = data
    if x:
        result = np.flip(result, axis=0)
    if y:
        result = np.flip(result, axis=1)
    if z:
        result = np.flip(result, axis=2)
    return result


def deconvolution():
    psf, dxy_psf, full_half_with_maxima_xy, full_half_with_maxima_z = generate_psf(
        dxy=422.0,
        f_cylinder_lens=240.0,
        slit_width=12.0,
    )
    result = decon(
        '/mnt/md0/hpc_ex642_em680_04_04_1.tif',
        psf,  # '/mnt/md0/psf_ex642_em680.tif',  #
        n_iters=9,  # int: Number of iterations, by default 10
        # fpattern='*.tif',  # str: used to filter files in a directory, by default "*.tif"
        dzdata=1000.0/1000.0,  # float: Z-step size of data, by default 0.5 um
        dxdata=422.0/1000.0,  # float: XY pixel size of data, by default 0.1 um
        dzpsf=1000.0/1000.0,  # float: Z-step size of the OTF, by default 0.1 um
        dxpsf=dxy_psf/1000.0,  # float: XY pixel size of the OTF, by default 0.1 um
        background=115,  # int or 'auto': User-supplied background to subtract.
                         # If 'auto', the median value of the last Z plane will be used as background. by default 80
        # rotate=0.0,
        # # float: Rotation angle; if not 0.0 then rotation will be performed around Y axis after deconvolution, by default 0
        # deskew=0.0,
        # # float: Deskew angle. If not 0.0 then deskewing will be performed before deconvolution, by default 0
        # width=0,  # int: If deskewed, the output image's width, by default 0 (do not crop)
        # shift=0,  # int: If deskewed, the output image's extra shift in X (positive->left), by default 0
        # pad_val=0.0,  # float: Value with which to pad image when deskewing, by default 0.0
        # save_deskewed=False,  # bool: Save deskewed raw data as well as deconvolution result, by default False
        napodize=8,  # int: Number of pixels to soften edge with, by default 15
        # nz_blend=27,  # int: Number of top and bottom sections to blend in to reduce axial ringing, by default 0
        # dup_rev_z=True,  # bool: Duplicate reversed stack prior to decon to reduce axial ringing, by default False

        wavelength=642,  # int: Emission wavelength in nm (default: {520})
        na=0.4,  # float: Numerical Aperture (default: {1.25})
        nimm=1.52,  # float: Refractive index of immersion medium (default: {1.3})
        # otf_bgrd=None,  # int, None: Background to subtract. "None" = autodetect. (default: {None})
        # krmax=0,
        # # int: Pixels outside this limit will be zeroed (overwriting estimated value from NA and NIMM) (default: {0})
        # fixorigin=8,
        # # int: for all kz, extrapolate using pixels kr=1 to this pixel to get value for kr=0 (default: {10})
        # cleanup_otf=True,  # bool: Clean-up outside OTF support (default: {False})
        # max_otf_size=60000,  # int: Make sure OTF is smaller than this many bytes.
        # # Deconvolution may fail if the OTF is larger than 60KB (default: 60000)
    )
    imwrite('/mnt/md0/hpc_ex642_em680_04_04_1_deconvolved.tif', result)
    subprocess.call(
        r"wine "
        r"./imaris/ImarisConvertiv.exe "
        r"-i /mnt/md0/hpc_ex642_em680_04_04_1_deconvolved.tif "
        r"-o /nafs/dong/kmoradi/hpc_ex642_em680_04_04_1_deconvolved.ims",
        shell=True
    )


if __name__ == "__main__":
    # PSF,  = generate_psf()
    # PSF = sample_psf()
    # from tifffile import imread
    # psf = imread('/mnt/md0/psf_ex642_em680.tif')
    # img = imread('/mnt/md0/hpc_ex642_em680_04_04.tif')
    # print(PSF.shape, psf.shape)
    # PSF = PSF.flatten()
    # psf = psf.flatten()
    # for elem in zip(PSF, psf):
    #     print(f"{elem[0]:.4f}, {elem[1]:.4f}")
    deconvolution()
