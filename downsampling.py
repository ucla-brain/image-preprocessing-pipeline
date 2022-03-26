import numpy as np
import matplotlib.pyplot as plt
import time
from numpy import ndarray
from pathlib import Path
from tifffile import imread


def draw_slices(image: ndarray, n=5, channel_ax=0, fig=None, vmin=None, vmax=None, **kwargs):
    """
    Draw slices in 3 orientations

    Parameters
    ----------
    image : ndarray.
        Assumed to be multichannel of the form C x row x col lx slice
    channel_ax : int or None
        Which axis stores the channels, if any. Defaults to 0, which is the pytorch convention


    Returns
    -------
    fig : matplotlib figure that object drawn into

    ax : array of matplotlib axes as a 3xn array.

    """

    # Identify channel axes and move it to end
    if channel_ax is None:
        image = image[..., None]
    else:
        # swap it with a dummy axis at the end, then squeeze out the dummy axis
        image = np.swapaxes(image[..., None], channel_ax, -1).squeeze()
        # check the number of channels and make sure it is 3
        nc = image.shape[-1]

    # check normalization to do it consistently
    if vmin is None:
        vmin = np.quantile(image, 0.001)
    if vmax is None:
        vmax = np.quantile(image, 0.999)
    image -= vmin
    image /= (vmax - vmin)

    # set up a figure
    if fig is None:
        fig = plt.figure()

    # now we'll set up slices along each axis
    ax = []
    inds = np.round(np.linspace(0, image.shape[0], n + 2)[1:-1]).astype(int)
    ax_ = []
    for i in range(len(inds)):
        ax__ = fig.add_subplot(3, n, i + 1)
        ax__.imshow(image[inds[i]], vmin=0, vmax=1)
        ax_.append(ax__)
    ax.append(ax_)
    # second axis
    inds = np.round(np.linspace(0, image.shape[1], n + 2)[1:-1]).astype(int)
    ax_ = []
    for i in range(len(inds)):
        ax__ = fig.add_subplot(3, n, i + 1 + n)
        ax__.imshow(image[:, inds[i]], vmin=0, vmax=1)
        ax_.append(ax__)
    ax.append(ax_)
    # third axis
    inds = np.round(np.linspace(0, image.shape[2], n + 2)[1:-1]).astype(int)
    ax_ = []
    for i in range(len(inds)):
        ax__ = fig.add_subplot(3, n, i + 1 + n * 2)
        ax__.imshow(image[:, :, inds[i]], vmin=0, vmax=1)
        ax_.append(ax__)
    ax.append(ax_)

    return fig, ax


def downsample_ax(image: ndarray, downsampling_factor: int, ax: int):
    """
    Downsample along a given axis

    Use np take so that ax can be an argument
    and no fancy indexing is needed

    Note downsampling here leaves the end off.

    Parameters
    ----------
    image : ndarray
        Image to be downsampled along one axis
    downsampling_factor : int
        Downsampling factor
    ax : int
        Axis to be downsampled on


    Returns
    -------
    downsampled_image : ndarray
        Image downsampled by averaging on given axis.

    """
    if downsampling_factor == 1:
        return np.array(image)  # note this is making a new array, just like below

    downsampled_shape = np.array(image.shape)
    downsampled_shape[ax] //= downsampling_factor
    downsampled_image = np.zeros(downsampled_shape, dtype=image.dtype)
    for i in range(downsampling_factor):
        downsampled_image += np.take(
            image,
            np.arange(i, downsampled_shape[ax] * downsampling_factor, downsampling_factor),
            axis=ax) / downsampling_factor
    return downsampled_image


def is_prime(d):
    """
    Determine if an integer is prime by looping over all its possible factors.
    This is not a high performance algorithm and should be done with small numbers

    Parameters
    ----------
    d : int
        An integer to test if it is prime

    Returns
    -------
    is_prime : bool
        Whether the number is prime
    """
    for i in range(2, d):  # between 2 and d-2
        if d // i == d / i:
            return False
    return True


def prime_factor(d):
    """
    Input an integer and return a list of prime factors.

    Parameters
    ----------
    d : int
        An integer to factorize

    Returns
    -------
    factors : list of int
        A list of prime factors sorted from lowest to highest

    """
    if is_prime(d):
        return [d]
    # otherwise I'll recurse and get a left factor and a right factor
    output = []
    for i in range(2, d):
        if d // i == d / i:
            # this is a factor
            left = i
            right = d // i
            output = []
            output += prime_factor(left)
            output += prime_factor(right)
    output.sort()
    return output


def downsample(I, d):
    '''
    Downsample along each axis

    If it can be factored it would be really good

    In factors always put the small number first

    Not pixels are left off the end.

    Parameters
    ----------
    I : numpy array
        Imaging data to downsample
    d : list of ints
        Downsampling factors along each axis



    Returns
    -------
    Id : numpy array
        Downsampled image

    '''
    Id = I
    for i, di in enumerate(d):
        for dii in prime_factor(di):
            Id = downsample_ax(Id, dii, i)
    return Id


def downsample_dataset(data, down, xI=None):
    # okay now I have to iterate over the dataset
    fig, ax = plt.subplots(1, 2)
    working = []
    output = []
    start = time.time()
    for i in range(data.shape[0]):
        starti = time.time()

        s = data[i]
        sd = downsample(s.astype(float), down[1:])
        ax[0].cla()
        ax[0].imshow(sd)
        working.append(sd)
        if len(working) == down[0]:
            out = downsample(np.stack(working), [down[0], 1, 1])
            ax[1].cla()
            ax[1].imshow(out[0])
            output.append(out)
            working = []
        fig.canvas.draw()
        print(f'Finished loading slice {i} of {data.shape[0]}, time {time.time() - starti} s')
    output = np.concatenate(output)
    Id = output

    output = Id
    if xI is not None:
        xId = [downsample(x, [d]) for x, d in zip(xI, down)]
        output = (Id, xId)

    print(f'Finished downsampling, time {time.time() - start}')

    return output


# build a tif class with similar interface
class TifStack:
    """
    We need a tif stack with an interface that will load a slice one at a time
    We assume each tif has the same size
    We assume 16-bit images
    """

    def __init__(self, input_directory: Path, pattern='*.tif'):
        self.input_directory = input_directory
        self.pattern = pattern
        self.files = list(input_directory.glob(pattern))
        self.files.sort()
        self.nyx = imread(self.files[0]).shape
        self.nz = len(self.files)
        self.shape = (self.nz, self.nyx[0], self.nyx[1])

    def __getitem__(self, i):
        return imread(self.files[i]) / (2 ** 16 - 1)

    def __len__(self):
        return len(self.files)

    def close(self):
        pass  # nothing necessary


def downsample_2d_tif_series(
        input_directory: Path, output_filename: Path, voxel_sizes: tuple, target_voxel_size: float
):
    """
    input_directory : Path
        input path containing 2d tif series
    output_filename : Path
        the name of output npz file name
    voxel_sizes : tuple
        a tuple of (z, y, x) form containing voxel sizes of the input files
    """

    # we need a temporary output directory for intermediate results (each slice)
    output_path = input_directory.parent.joinpath("downsampling_tmp")
    output_path.mkdir(parents=True, exist_ok=True)

    dI = np.array(voxel_sizes)  # we need to input the voxel size, slice thickness first, in microns
    res = target_voxel_size

    if output_filename is None:
        output_filename = input_directory.parent.joinpath(input_directory.name + "_down.npz")

    print(f'Input directory is {input_directory}')
    print(f'Output filename is {output_filename}')
    print(f'Resolution is {dI}')
    print(f'Desired resolution is {res}')

    down = np.floor(res / dI).astype(int)
    print(f'Downsampling factors are {down}')
    print(f'Downsampled res {dI * down}')

    # load the data
    data = TifStack(input_directory)
    print(f'Dataset shape {data.shape}')

    nI = np.array(data.shape)
    xI = [np.arange(n) * d - (n - 1) / 2.0 * d for n, d in zip(nI, dI)]

    xId = [downsample(x, [d]) for x, d in zip(xI, down)]
    dId = [x[1] - x[0] for x in xId]
    print(f"output voxel sizes {dId:.2f}")

    # iterate over the dataset
    # we need to save intermediate outputs (each slice) in case of errors
    working, output = [], []
    start = time.time()
    for i in range(data.shape[0]):
        start_time = time.time()
        downsampled_output_file = output_path / f"{i:04d}.npy"
        if downsampled_output_file.exists():
            sd = np.load(str(downsampled_output_file))
        else:
            s = data[i] ** 0.25  # test reduce dynamic range before downsampling with this power
            sd = downsample(s.astype(float), down[1:])
            np.save(str(downsampled_output_file), sd)

        working.append(sd)
        if len(working) == down[0]:
            out = downsample(np.stack(working), [down[0], 1, 1])
            output += [out]
            working = []
        print(f'Finished loading slice {i} of {data.shape[0]}, time {time.time() - start_time:.1f} s')
    output = np.concatenate(output)
    Id = output
    print(f'Finished downsampling, time {time.time() - start:.1f}')
    np.savez(output_filename, I=Id, xI=np.array(xId, dtype='object'))  # note specify object to avoid "ragged" warning
    assert Id.shape == (len(xId[0]), len(xId[1]), len(xId[2]))

