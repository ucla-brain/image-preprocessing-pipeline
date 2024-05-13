from numpy import zeros, pad, copy, stack, min, max, ndarray, uint8, uint16, uint32, float32, float64
from multiprocessing import Pool
from process_images import get_gradient, get_transformation_matrix

from pathlib import Path

from copy import deepcopy

from skimage.filters import sobel
from skimage import feature

from tifffile import imwrite
from os.path import exists, isdir, join
from os import listdir, system

from supplements.tifstack import TifStack

# # for floodfill attempt
# from pystripe.core import get_img_mask
# from skimage.filters.thresholding import threshold_multiotsu
def write_to_file(images: list[ndarray], filepath: Path, data_type, save_RGB=False, verbose=False):
    filepath.mkdir(parents=True, exist_ok=True)
    match data_type:
        case 'uint8': dtype = uint8
        case 'uint16': dtype = uint16
        case 'uint32': dtype = uint32
        case 'float32': dtype = float32
        case 'float64': dtype = float64
        case _:
            print("Invalid data type provided!  Writing to file with uint8.")
            dtype = uint8

    for n, image in enumerate(images):
        local = filepath / f'cha{n}'
        local.mkdir(parents=True, exist_ok=True)
        for layer in range(image.shape[0]):
            path = local.absolute() / (str(layer) + ".tif")
            imwrite(path, get_layer(layer, image, "yx").astype(dtype), dtype=dtype)

    # save RGB file
    if save_RGB and len(images) <= 3:
        local = filepath / 'RGB'
        local.mkdir(parents=True, exist_ok=True)
        for layer in range(image.shape[0]):
            composite = stack([get_layer(layer, image, "yx") for image in images], axis=-1)
            path = local.absolute() / (str(layer) + ".tif")
            imwrite(path, composite.astype(dtype), dtype=dtype)

    if verbose: print("wrote to file")


# written by ChatGPT, modifies parameters
def normalize_array_inplace(arr: ndarray):
    min_val = min(arr)
    max_val = max(arr)

    arr -= min_val
    arr /= (max_val - min_val)

    # Scale the values to be between 0 and 255
    arr *= 255
    arr.astype(uint8, copy=False)


# # floodfill attempt
# def get_borders(img: ndarray, copy=False):
#     # print("get_borders")
#     mask = zeros_like(img)
#     for ind in range(img.shape[0]):
#         # print(f'layer {ind}')
#         try:
#             threshold = threshold_multiotsu(img[ind], classes=4)[2]
#             print("threshold", threshold)
#             mask[ind] = get_img_mask(img[ind], threshold, close_steps=3, open_steps=5, flood_fill_flag=4)
#         except ValueError:
#             # assume blank image since there will only be one pixel color.  keep mask as all zeros.
#             continue
#     img *= mask


# pads arr with zeroes evenly (as possible) on all sides to match pad_shape
def pad_to_shape(pad_shape: tuple, arr: ndarray):
    assert len(pad_shape) == len(arr.shape)
    if pad_shape == arr.shape: return arr
    pad_dim = [pad_shape[i] - arr.shape[i] for i in range(len(pad_shape))]
    pad0 = list(map(lambda x: (x // 2, (x + 1) // 2), pad_dim))
    return pad(arr, pad_width=pad0, mode='constant')


def resize_arrays(arrays: list[ndarray]):
    shapes = [a.shape for a in arrays]
    pad_size = tuple((max(i) for i in zip(*shapes)))
    for i in range(len(arrays)):
        arrays[i] = pad_to_shape(pad_size, arrays[i])
    return arrays


# similar to numpy.roll(), but numbers moved to the other side are lost and replaced with zeroes.
# MODIFIES PARAMETERS
def roll_pad(arr: ndarray, move: int, axis: int = 0):
    t = arr.dtype
    if axis > len(arr.shape) - 1 or axis < 0: raise Exception
    if move == 0: return
    if axis != 0:
        for s in range(arr.shape[0]): roll_pad(arr[s], move, axis - 1)
        return
    if move > 0:
        arr[move:] = arr[:-move]
        arr[:move] = zeros((move,) + arr[0].shape, dtype=t)
    else:
        arr[:move] = arr[-move:]
        arr[move:] = zeros((-move,) + arr[0].shape, dtype=t)
    return


# gets a plane from a 3d image
def get_layer(
    index: int,          # layer of image requested
    image: ndarray,      # 3-D image (use TifStack.as_3d_numpy())
    plane="xy",          # must be "xy", "yx", "xz", "zx", "yz", "zy"
    img_format="zyx",    # xyz in some order
):
    # guards
    if plane not in {"xy", "yx", "xz", "zx", "yz", "zy"} or img_format not in {"zyx", "zxy", "yxz", "yzx", "xyz", "xzy"}:
        print(f"Invalid plane selected in get_layer().  Plane: {plane}, Layer: {index}, Img_format: {img_format}\nReturning to caller...")
        return None

    # get the layer
    if 'x' not in plane:   sub = img_format.index('x')
    elif 'y' not in plane: sub = img_format.index('y')
    elif 'z' not in plane: sub = img_format.index('z')

    if sub == 0:   layer_image = image[index, :, :]
    elif sub == 1: layer_image = image[:, index, :]
    elif sub == 2: layer_image = image[:, :, index]

    # if plane is flipped compared to image format, return the transpose.
    if plane not in (img_format[:sub] + img_format[sub + 1:]):
        return layer_image.transpose()
    return layer_image


# returns alignment matrices for aligning 3d images based on a 2d plane
def get_offsets(
        images: list[ndarray],
        plane: str,
        verbose=False,
):
    assert(len(images) > 1)
    assert(plane in {'xy', 'xz', 'yz'})

    if plane == 'xy':   img_reference_idx = images[0].shape[0] // 2
    elif plane == 'xz': img_reference_idx = images[0].shape[1] // 2
    elif plane == 'yz': img_reference_idx = images[0].shape[2] // 2
    # print(f"img_reference_idx: {img_reference_idx}")

    img_samples = []

    for image in images:
        img_samples.append(get_layer(img_reference_idx, image, plane))

    with Pool(len(images)) as pool:
        img_samples = list(pool.map(get_gradient, img_samples))

    assert all([img is not None for img in img_samples])

    # I added a verbose argument to get_transformation_matrix. If that is causing errors, just delete the argument.
    transformation_matrices = [get_transformation_matrix(img_samples[0], img, verbose=verbose) for img in img_samples[1:]]
    del img_samples
    return transformation_matrices


# applies canny edge detection algorithm on a 3d image (modifies image in parameter)
def apply_canny(image: ndarray,
                sigma=1.0,
                low_threshold=None,
                high_threshold=None,
                mask=None,
                use_quantiles=False,
                *,
                mode='constant',
                cval=0.0):
    for layer in range(image.shape[0]):
        image[layer] = feature.canny(image[layer], sigma=sigma, low_threshold=low_threshold,
                                     high_threshold=high_threshold, mask=mask, use_quantiles=use_quantiles,
                                     mode=mode, cval=cval)


def write_alignments(channels: list[list], residuals: list[list], reference: int, filepath: str):
    try:
        f = open(filepath / 'alignments.txt', "x")
        output_file = filepath / 'alignments.txt'
        f.close()
    except FileExistsError:
        i = 1
        while True:
            try:
                f = open(filepath / f"alignments ({i}).txt", "x")
                output_file = filepath / f"alignments ({i}).txt"
                f.close()
                break
            except FileExistsError:
                i += 1

    f = open(output_file, "a")
    f.write(f"Number of channels: {len(channels) + 1}\n")
    f.write(f"Reference channel: {reference}\n")

    index = 0
    for n in range(len(channels) + 1):
        if n == reference:
            continue
        f.write(f'Channel {n}:\n')
        f.write(f'\tx-alignment: {channels[index][0]}\t\t Residuals: {residuals[index][0]}\n')
        f.write(f'\ty-alignment: {channels[index][1]}\t\t Residuals: {residuals[index][1]}\n')
        f.write(f'\tz-alignment: {channels[index][2]}\t\t Residuals: {residuals[index][2]}\n\n')
        index += 1

    print(f"Alignments saved in file: {output_file}")


# aligns images in 3d, using 2d alignment algorithm as a blackbox
def align_images(img1_: ndarray, img2_: ndarray, max_iter: int = 50, make_copy: bool = False, verbose=False):
    # copy images if copy flag is true
    if not make_copy:
        img1 = img1_
        img2 = img2_
    else:
        img1 = copy(img1_)
        img2 = copy(img2_)

    # make images the same size
    resize_arrays([img1, img2])

    if verbose:
        print("Loaded images.")
        print("Resized shapes: " + str(img1.shape))

    # instantiate variables
    iteration = 0
    residual = None

    x_moves = []
    y_moves = []
    z_moves = []

    prev_matr = []
    found = False

    # iterate until converge
    while iteration < max_iter:
        if verbose: print(f"Iteration {iteration}")
        xy_matrix = get_offsets([img1, img2], "xy", verbose=verbose)
        xz_matrix = get_offsets([img1, img2], "xz", verbose=verbose)
        yz_matrix = get_offsets([img1, img2], "yz", verbose=verbose)

        x_moves.append(int(round(xy_matrix[0][1][2] + xz_matrix[0][1][2]) / 2))
        y_moves.append(int(round(xy_matrix[0][0][2] + yz_matrix[0][1][2]) / 2))
        z_moves.append(int(round(xz_matrix[0][0][2] + yz_matrix[0][0][2]) / 2))

        if verbose:
            print(x_moves[-1])
            print(y_moves[-1])
            print(z_moves[-1])

        roll_pad(img2, x_moves[-1], axis=2)
        roll_pad(img2, y_moves[-1], axis=1)
        roll_pad(img2, z_moves[-1], axis=0)

        matr = [(int(xy_matrix[0][0][2]), int(xy_matrix[0][1][2])),
                (int(xz_matrix[0][0][2]), int(xz_matrix[0][1][2])),
                (int(yz_matrix[0][0][2]), int(yz_matrix[0][1][2]))
                ]

        # check if in cycle
        for i in prev_matr:
            if i == matr:
                found = True

        if found:
            residual = ((xy_matrix[0][0][2] + xz_matrix[0][0][2]) / 2, (xy_matrix[0][1][2] + yz_matrix[0][1][2]) / 2, (xz_matrix[0][1][2] + yz_matrix[0][0][2]) / 2)
            if verbose:
                print("No absolute convergence found; cycle detected.")
                print("Residual: " + str(residual))
            break

        if x_moves[-1] == 0 and y_moves[-1] == 0 and z_moves[-1] == 0:
            residual = ((xy_matrix[0][0][2] + xz_matrix[0][0][2]) / 2, (xy_matrix[0][1][2] + yz_matrix[0][1][2]) / 2, (xz_matrix[0][1][2] + yz_matrix[0][0][2]) / 2)
            if verbose:
                print("Images converged.")
                print("Residual: " + str(residual))
            break

        prev_matr.append(matr)
        iteration += 1

    return x_moves, y_moves, z_moves, residual


# aligns all images
def align_all_images(
        images: list[ndarray],  # list of 3d images, padded to the same size for best results.
                                # load with TifStack.as_3d_numpy, pad by calling resize_arrays
        reference: int = 0,
        max_iter: int = 50,
        make_copy: bool = True,
        verbose: bool = False
):
    moves = []
    residuals = []
    for i in range(len(images)):
        if i == reference:
            continue
        img_x_moves, img_y_moves, img_z_moves, img_residual = align_images(images[reference], images[i], max_iter, make_copy=make_copy, verbose=verbose)
        moves.append([sum(img_x_moves), sum(img_y_moves), sum(img_z_moves)])
        residuals.append(img_residual)

    return moves, residuals


# entrance if run in terminal
def main():
    from argparse import ArgumentParser
    parser = ArgumentParser("Align images")
    parser.add_argument('--input', '-i', required=True, type=str, help="Absolute file path of tiff stacks representing images to be aligned.  This directory must contain exactly three subfolders with the tiff files.")
    parser.add_argument('--output', '-o', required=True, type=str, help="Absolute file path of output")
    parser.add_argument('--num_channels', default=3, type=int, help="Number of channels to align")
    parser.add_argument('--edge_detection', type=str, help="Selects which edge detection algorithm to use.  Options: 'sobel', 'canny'")
    parser.add_argument('--pad_only', action='store_true', help="If present, only pad images to same shape without aligning")
    parser.add_argument('--write_alignments', action='store_true', help="If present, write alignments to a .txt file.")
    parser.add_argument('--generate_ims', action='store_true', help="If present, generate .ims files along with output.")
    parser.add_argument('--max_iterations', type=int, default=10, help="Maximum iterations allowed for image alignment.")
    parser.add_argument('--reference', type=int, default=0, help="The channel to use as the reference image.  Default 0.")
    parser.add_argument('--save_rgb', action='store_true', help="If present, saves an RGB channel along with the single channel images.  (Must have <= 3 channels)")
    parser.add_argument('--dtype', type=str, default='uint8', help="Data type of output tifs.  Options include 'uint8', 'uint16', 'uint32', 'float32', 'float64'")
    parser.add_argument('--dx', type=int, default=10, help="dx for .ims file (if generated)")
    parser.add_argument('--dy', type=int, default=10, help="dy for .ims file (if generated)")
    parser.add_argument('--dz', type=int, default=10, help="dz for .ims file (if generated)")

    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    max_iterations = args.max_iterations
    edge_detection = args.edge_detection
    write_alignments_bool = args.write_alignments
    reference = args.reference
    num_channels = args.num_channels
    generate_ims = args.generate_ims
    pad_only = args.pad_only
    save_rgb = args.save_rgb
    data_type = args.dtype
    dx = args.dx
    dy = args.dy
    dz = args.dz

    # Directory checking ------------------------------------------------------------------
    if not exists(input_file) or not isdir(input_file):
        print(f"Error: Input directory '{input_file}' is invalid.")
        exit(1)

    filepaths = []
    num_dirs = 0

    for item in listdir(input_file):
        item_path = join(input_file, item)
        if isdir(item_path):
            num_dirs += 1
            filepaths.append(item_path)

    if num_dirs != num_channels:
        print(f"Error: Input directory '{input_file}' contains a different number of channels than the number specified.  Found: {num_dirs}.  Expected: {num_channels}")
        exit(1)

    # Image Processing --------------------------------------------------------------------
    print("Loading images...")
    count = 0
    raw_channels = []
    try:
        while count < num_channels:
            raw_channels.append(TifStack(filepaths[count]).as_3d_numpy())
            count += 1
        print("Images loaded")
    except Exception:
        print(f"Error: Invalid TifStack found at {filepaths[count]}")
        exit(1)

    output_path = Path(output_file)
    output_path.mkdir(parents=True, exist_ok=True)

    print(filepaths)

    print("Resizing images...")

    channels = resize_arrays(raw_channels)
    print("Images resized")

    copy_channels = [deepcopy(img) for img in channels]

    if not pad_only:
        print("Aligning images... (this may take a while)")
        for i in range(len(channels)):
            if edge_detection:
                if edge_detection.lower() == 'sobel':
                    if i == 0: print("Running Sobel Operator")
                    copy_channels[i] = sobel(copy_channels[i])
                elif edge_detection.lower() == 'canny':
                    if i == 0: print("Running Canny Operator")
                    apply_canny(copy_channels[i])

        # align images
        alignments, residuals = align_all_images(copy_channels, max_iter=max_iterations, reference=reference, verbose=True, make_copy=False)
        print("Images aligned")

        # apply transformations to actual images
        index = 0
        for n, img in enumerate(channels):
            if n == reference:
                continue
            roll_pad(img, alignments[index][0], axis=2)
            roll_pad(img, alignments[index][1], axis=1)
            roll_pad(img, alignments[index][2], axis=0)
            index += 1

    print("Writing to file")
    write_to_file(channels, output_path, data_type, save_RGB=save_rgb)

    if write_alignments_bool:
        write_alignments(alignments, residuals, reference, output_path)

    print("Wrote to file")

    if generate_ims:
        print("Generating .ims files")
        for i in range(len(channels)):
            system(f'python convert.py -i "{output_path}/cha{i}" -o "{output_path}/cha{i}.ims" -dx {dx} -dy {dy} -dz {dz}')
        if save_rgb:
            system(f'python convert.py -i "{output_path}/RGB" -o "{output_path}/RGB.ims" -dx {dx} -dy {dy} -dz {dz}')
        print(".ims files created")

    if not pad_only:
        print("Alignments:")
        print(alignments)
    print("\n\nOperation completed.")


if __name__ == '__main__':
    main()
