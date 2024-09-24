from pycudadecon import decon
from tifffile import imwrite, imread
from scipy.optimize import fsolve
import scipy.special as sp
from scipy.integrate import quad
from numpy import real, imag, array, ndarray, zeros, flip, sum

import subprocess
import math
import cmath
import os

from argparse import ArgumentParser
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return real(func(x))

    def imag_func(x):
        return imag(func(x))

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

    nxy, nz, full_half_width_maxima_xy, full_half_width_maxima_z = determine_psf_size(
        dxy_psf, dz, numerical_aperture, refractive_index, lambda_ex, lambda_em, f_cylinder_lens, slit_width,
        resolution_xy, resolution_z
    )

    numerical_aperture_ls = math.sin(math.atan(slit_width / (2.0 * f_cylinder_lens)))

    psf = sample_psf(dxy_psf, dz, nxy, nz, numerical_aperture, refractive_index, lambda_ex, lambda_em,
                     numerical_aperture_ls)

    print(f"full width half maxima of xy-plane is {full_half_width_maxima_xy:.1f} nm.\n"
          f"full width half maxima of z-axis is {full_half_width_maxima_z:.1f} nm.")

    return psf, dxy_psf


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

    full_half_with_maxima_xy = 2 * abs(fsolve(fxy, array([resolution_xy/2], dtype='single'))[0])
    full_half_with_maxima_z = 2 * abs(fsolve(fz, array([resolution_z/2], dtype='single'))[0])

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

    psf = zeros(
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
    psf = psf / sum(psf)
    return psf


def mirror8(psf_quadrant):
    """mirrors the content of the first quadrant to all other quadrants to obtain the complete PSF
    """
    sx, sy, sz = array(psf_quadrant.shape, dtype='int') * 2 - 1
    cx, cy, cz = array([sx, sy, sz]) // 2

    result = zeros([sx, sy, sz], dtype='single')
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
        result = flip(result, axis=0)
    if y:
        result = flip(result, axis=1)
    if z:
        result = flip(result, axis=2)
    return result


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='output directory psf tif will be written to')
    parser.add_argument('--lambda_em', '-em', default=642.0, type=float,
                        help='the wavelength of the emitted light in nm')
    parser.add_argument('--lambda_ex', '-ex', default=680.0, type=float,
                        help='the wavelength of the emitted light in nm')
    parser.add_argument('--numerical_aperture', '-n', default=0.4, type=float,
                        help='numerical aperture objective of the lens')
    parser.add_argument('--dxy', '-dx', default=422.0, type=float,
                        help='voxel size in x and y dimensions in nm (of the camera pixels)')
    parser.add_argument('--dz', '-dz', default=1000.0, type=float,
                        help='voxel size in z dimensions in nm (or the z-steps)')
    parser.add_argument('--refractive_index', '-ri', default=1.52, type=float,
                        help='the refractive index of immersion medium')
    parser.add_argument('--f_cylinder_lens', '-f', default=240, type=float,
                        help='F cylinder lens in mm')
    parser.add_argument('--slit_width', '-s', default=12.0, type=float,
                        help='slit aperture width in mm')
    args = parser.parse_args()

    psf, dxy_psf = generate_psf(
        lambda_em=args.lambda_em,
        lambda_ex=args.lambda_ex,
        numerical_aperture=args.numerical_aperture,
        dxy=args.dxy,
        dz=args.dz,
        refractive_index=args.refractive_index,  # 1.45 of the lens
        f_cylinder_lens=args.f_cylinder_lens,  # 240
        slit_width=args.slit_width,
    )

    imwrite((args.output / 'psf.tif').__str__(), psf)
    print(f"psf written to {args.output}")
    print(f"dxy_psf: {dxy_psf}")