from numpy import zeros, zeros_like, pad, copy, stack, min, max, ndarray, uint8, uint16, uint32, float32, float64
from multiprocessing import Pool
from functools import partial

from pathlib import Path

from copy import deepcopy

from skimage.filters import sobel
from skimage import feature

from tifffile import natural_sorted, imread, imwrite
from os.path import exists, isdir, join
from os import listdir, system

from supplements.tifstack import TifStack
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter

from tqdm import tqdm

# # for floodfill attempt
# from pystripe.core import get_img_mask
# from skimage.filters.thresholding import threshold_multiotsu


def write_to_file(images: list[ndarray], input_files: list[str], reference: int, filepath: Path, data_type, save_singles=False, verbose=False):
    filepath.mkdir(parents=True, exist_ok=True)
    print("Images:", len(images))
    print("input_files:", input_files)

    match data_type:
        case 'uint8': dtype = uint8
        case 'uint16': dtype = uint16
        case 'uint32': dtype = uint32
        case 'float32': dtype = float32
        case 'float64': dtype = float64
        case _:
            print("Invalid data type provided!  Writing to file with uint8.")
            dtype = uint8

    # save singles
    if save_singles:
        for n, f in enumerate(input_files):
            if f is None: continue
            local = filepath / f.split('/')[-1]
            local.mkdir(parents=True, exist_ok=True)
            for layer in range(images[n].shape[0]):
                path = local.absolute() / (str(layer) + ".tif")
                imwrite(path, get_layer(layer, images[n], "yx").astype(dtype), dtype=dtype)

    # save RGB file
    local = filepath / 'RGB'
    local.mkdir(parents=True, exist_ok=True)
    for layer in range(images[reference].shape[0]):
        layers = []
        for n, f in enumerate(input_files):
            if f is None:
                layers.append(zeros_like(images[reference][0]))
            else:
                layers.append(get_layer(layer, images[n], "yx"))
        composite = stack(layers, axis=-1)
        path = local.absolute() / (str(layer) + ".tif")
        imwrite(path, composite.astype(dtype), dtype=dtype)

        # composite = stack([get_layer(layer, image, "yx") for image in images], axis=-1)
        # path = local.absolute() / (str(layer) + ".tif")
        # imwrite(path, composite.astype(dtype), dtype=dtype)

    if verbose:
        print("wrote to file")


# modifies parameters
# def normalize_array_inplace(arr: ndarray):
#     min_val = min(arr)
#     max_val = max(arr)
#
#     arr -= min_val
#     arr /= (max_val - min_val)
#
#     # Scale the values to be between 0 and 255
#     arr *= 255
#     arr.astype(uint8, copy=False)


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


def trim_to_shape(output_shape: tuple, arr: ndarray):
    if arr is None: return None
    assert len(output_shape) == len(arr.shape)
    if output_shape == arr.shape: return arr
    trim_dim = [arr.shape[i] - output_shape[i] for i in range(len(output_shape))]
    trim0 = list(map(lambda x: (x // 2, (x + 1) // 2), trim_dim))
    slices = [slice(trim0[i][0], arr.shape[i] - trim0[i][1]) for i in range(len(output_shape))]

    return arr[tuple(slices)]


def resize_arrays(arrays: list[ndarray]):
    shapes = []
    for a in arrays:
        if a is None: continue
        shapes.append(a.shape)
    # shapes = [a.shape for a in arrays if a is not None]
    pad_size = tuple((max(i) for i in zip(*shapes)))
    for i in range(len(arrays)):
        if arrays[i] is not None:
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
    if image is None:
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
    from process_images import get_gradient, get_transformation_matrix
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


def write_alignments(channels: list[list], input_files: list[str], residuals: list[list], reference: int, filepath: str):
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
    f.write(f"Number of channels: {len(channels)}\n")
    for i in range(len(channels)):
        f.write(f"\t Channel {i}: {input_files[i]}\n")

    f.write(f"Reference channel: {reference}\n")

    index = 0
    for n in range(len(channels) + 1):
        if n == reference:
            continue
        f.write(f'Channel {n}:\n')
        if residuals[index] is not None:
            f.write(f'\tx-alignment: {channels[index][0]}\t\t Residuals: {residuals[index][0]}\n')
            f.write(f'\ty-alignment: {channels[index][1]}\t\t Residuals: {residuals[index][1]}\n')
            f.write(f'\tz-alignment: {channels[index][2]}\t\t Residuals: {residuals[index][2]}\n\n')
        else:
            f.write(f'\tx-alignment: {channels[index][0]}\n')
            f.write(f'\ty-alignment: {channels[index][1]}\n')
            f.write(f'\tz-alignment: {channels[index][2]}\n\n')
        index += 1

    print(f"Alignments saved in file: {output_file}")


# helper function for process_big_images
def process_single_big_image(n_ref: int,  # the only non-constant var between iterations
                             file_paths: list[str],
                             reference_index: int,
                             pad_to_max: list[list[list[int]]],
                             offsets: list[list[int]],
                             image_shapes: list[list[int]],
                             operation_shape: list[int],
                             file_path_output: Path,
                             data_type,  # dtype
                             save_singles: bool,
                             file_path_inputs: list[Path],
                             missing_channel: str):
    n_orig = []
    for i in range(len(file_paths)):
        if file_paths[i] is None:
            n_orig.append(None)
            continue
        if i == reference_index:
            n_orig.append(n_ref)
            continue
        n_orig.append(n_ref + pad_to_max[reference_index][0][0] - pad_to_max[i][0][0] - offsets[i][0])

    # print("n_ref: ", n_ref)
    # print("n_orig: ", n_orig)
    combined_image = []
    for count, n_img in enumerate(n_orig):
        if n_img is None:
            combined_image.append(zeros(image_shapes[reference_index][1:]))
        elif 0 <= n_img < image_shapes[count][0]:
            file = imread(file_paths[count][n_img])
            # pad image to operation dimensions
            file = pad_to_shape(operation_shape[1:], file)

            # shift image in x and y directions
            roll_pad(file, offsets[count][1], axis=0)
            roll_pad(file, offsets[count][2], axis=1)

            # reshape image to reference dimensions
            file = trim_to_shape(image_shapes[reference_index][1:], file)
            combined_image.append(file)
        else:
            # save zeroes for that layer if out of bounds
            combined_image.append(zeros(image_shapes[reference_index][1:]))

    # if missing_channel == 'r':
    #     combined_image.insert(0, zeros(image_shapes[reference_index][1:]))
    # elif missing_channel == 'g':
    #     combined_image.insert(1, zeros(image_shapes[reference_index][1:]))
    # elif missing_channel == 'b':
    #     combined_image.insert(2, zeros(image_shapes[reference_index][1:]))

    # save RGB image
    local = file_path_output / 'RGB'
    local.mkdir(parents=True, exist_ok=True)
    composite = stack(combined_image, axis=-1)

    path = local.absolute() / Path(file_paths[reference_index][n_ref]).name
    imwrite(path, composite.astype(data_type), dtype=data_type)

    # save individual images
    if save_singles:
        for n, channel_img in enumerate(combined_image):
            # print(n)
            local = file_path_output / file_path_inputs[n].name
            local.mkdir(parents=True, exist_ok=True)
            path = local.absolute() / Path(file_paths[reference_index][n_ref]).name
            # print(path)
            imwrite(path, channel_img.astype(data_type), dtype=data_type)


# offsets must have shape (D, 3), where D is the number of file path inputs
# offsets[reference_index] = [0, 0, 0], x-y-z order (NOT z-y-x)
def process_big_images(file_path_inputs: list[Path], file_path_output: Path, reference_index: int,
                       offsets: list[list[int]], num_threads=8, save_singles=False, missing_channel=None):
    # load image paths
    file_paths = []
    for input_path in file_path_inputs:
        if input_path:
            temp = natural_sorted([file.__str__() for file in input_path.iterdir() if
                                   file.is_file() and file.suffix.lower() in (".tif", ".tiff")])
            file_paths.append(temp)
        else:
            file_paths.append(None)

    image_shapes = []  # list[tuple]
    data_type = imread(file_paths[reference_index][0]).dtype
    for image in file_paths:
        if image is None:
            image_shapes.append(None)
        else:
            temp_shape = (len(image), *imread(image[0]).shape)
            image_shapes.append(temp_shape)

    # calculate operation shape
    operation_shape = [max(dim) for dim in zip(*filter(lambda x: x is not None, image_shapes))]

    # calculate pad amounts
    pad_to_max = []
    for image in range(len(file_paths)):
        if image_shapes[image] is None:
            pad_to_max.append(None)
        else:
            pad_dim = [operation_shape[i] - image_shapes[image][i] for i in range(len(image_shapes[image]))]
            pad_to_max.append(list(map(lambda x: (x // 2, (x + 1) // 2), pad_dim)))

    # process layers
    print("Aligning large images...")

    # print("file_paths ", len(file_paths))
    # print("reference_index ", reference_index)
    # print("pad_to_max ", pad_to_max)
    # print("offsets ", offsets)
    # print("image_shapes ", image_shapes)

    # fill in constant variables for partial function
    partial_func = partial(process_single_big_image,
                           file_paths=file_paths,
                           reference_index=reference_index,
                           pad_to_max=pad_to_max,
                           offsets=offsets,
                           image_shapes=image_shapes,
                           operation_shape=operation_shape,
                           file_path_output=file_path_output,
                           data_type=data_type,
                           save_singles=save_singles,
                           file_path_inputs=file_path_inputs,
                           missing_channel=missing_channel)

    if num_threads <= 1:
        for i in tqdm(range(len(file_paths[reference_index]))):
            partial_func(i)
    else:
        pool = Pool(processes=num_threads)
        try:
            # need to convert to list so the tqdm iterator is consumed; otherwise progress bar doesn't update.
            list(tqdm(
                pool.imap_unordered(partial_func, range(len(file_paths[reference_index]))),
                total=len(file_paths[reference_index])))
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected, terminating thread pool...")
            pool.terminate()
            pool.join()
        except Exception as e:  # this one ideally should never occur...
            print("Thread pool for aligning large images encountered an error, terminating...")
            print(e)
            pool.terminate()
            pool.join()
        else:
            pool.close()
            pool.join()


# aligns images in 3d, using 2d alignment algorithm as a blackbox
def align_images(img1_: ndarray, img2_: ndarray, max_iter: int = 50, make_copy: bool = False, verbose=False):
    if img1_ is None or img2_ is None:
        return None, None, None, None
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
        if images[i] is None or i == reference:
            moves.append([None, None, None])
            residuals.append(None)
            continue
        img_x_moves, img_y_moves, img_z_moves, img_residual = align_images(images[reference], images[i], max_iter, make_copy=make_copy, verbose=verbose)
        moves.append([sum(img_x_moves), sum(img_y_moves), sum(img_z_moves)])
        residuals.append(img_residual)

    return moves, residuals


# entrance if run in terminal
def main(args: Namespace):
    # input_file = args.input
    red_paths = args.red
    green_paths = args.green
    blue_paths = args.blue
    output_file = args.output
    max_iterations = args.max_iterations
    # edge_detection = args.edge_detection
    write_alignments_bool = args.write_alignments
    reference_str = args.reference
    num_channels = 3 #args.num_channels
    num_threads = args.num_threads
    generate_ims = args.generate_ims
    # pad_only = args.pad_only
    save_singles = args.save_singles
    data_type = args.dtype
    dx = args.dx
    dy = args.dy
    dz = args.dz

    # Ensure there are 2 or 3 channels
    if sum(map(lambda x: bool(x[0]), [red_paths, green_paths, blue_paths])) < 2:
        print("You must select at least two channels to align.")
        exit(1)

    missing_channel = None
    if red_paths[0] is None: missing_channel = 'r'
    if green_paths[0] is None: missing_channel = 'g'
    if blue_paths[0] is None: missing_channel = 'b'

    # Directory checking ------------------------------------------------------------------
    for i in red_paths + green_paths + blue_paths:
        if i is None:
            continue
        if not exists(i) or not isdir(i):
            print(f"Error: Input directory '{i}' is invalid.")
            exit(1)

    # if not exists(input_file) or not isdir(input_file):
    #     print(f"Error: Input directory '{input_file}' is invalid.")
    #     exit(1)

    original_input, downsampled_input = zip(red_paths, green_paths, blue_paths)

    match reference_str.lower().strip():
        case 'red' | 'r':
            reference = 0
        case 'green' | 'g':
            reference = 1
        case 'blue' | 'b':
            reference = 2
        case _:
            print("Error: Invalid reference image provided!")
            print(reference_str.lower().strip())
            exit(1)

    if reference == 0 and red_paths[0] is None or reference == 1 and green_paths[0] is None or reference == 2 and blue_paths[0] is None:
        print("Error: Reference channel selected does not exist.")
        exit(1)

    # filepaths = []
    # num_dirs = 0

    # for item in listdir(input_file):
    #     item_path = join(input_file, item)
    #     if isdir(item_path):
    #         num_dirs += 1
    #         filepaths.append(item_path)

    # if num_dirs != num_channels:
    #     print(f"Error: Input directory '{input_file}' contains a different number of channels than the number specified.  Found: {num_dirs}.  Expected: {num_channels}")
    #     exit(1)

    # Image Processing --------------------------------------------------------------------
    print("Loading images...")
    count = 0
    raw_channels = []
    try:
        while count < num_channels:
            if downsampled_input[count]:
                raw_channels.append(TifStack(downsampled_input[count]).as_3d_numpy())
            else:
                raw_channels.append(None)
            count += 1
        print("Images loaded")
    except Exception:
        print(f"Error: Invalid TifStack found at {downsampled_input[count]}")
        exit(1)

    output_path = Path(output_file)
    output_path.mkdir(parents=True, exist_ok=True)

    print(downsampled_input)

    print("Resizing images...")
    original_downsampled_reference_shape = raw_channels[reference].shape

    channels = resize_arrays(raw_channels)
    print("Images resized")

    copy_channels = [deepcopy(img) for img in channels]


    print("Finding alignments... (this may take a while)")
    # for i in range(len(channels)):
    #     if edge_detection:
    #         if edge_detection.lower() == 'sobel':
    #             if i == 0: print("Running Sobel Operator")
    #             copy_channels[i] = sobel(copy_channels[i])
    #         elif edge_detection.lower() == 'canny':
    #             if i == 0: print("Running Canny Operator")
    #             apply_canny(copy_channels[i])

    # align images
    alignments, residuals = align_all_images(copy_channels, max_iter=max_iterations, reference=reference, verbose=True, make_copy=False)

    # apply transformations to actual images
    print("Aligning downsampled images...")
    for n, img in tqdm(enumerate(channels)):
        if not n or n == reference:
            continue
        roll_pad(img, alignments[n][0], axis=2)
        roll_pad(img, alignments[n][1], axis=1)
        roll_pad(img, alignments[n][2], axis=0)

    # reshape downsampled to reference
        for n, img in enumerate(channels):
            channels[n] = trim_to_shape(original_downsampled_reference_shape, img)

    # write downsampled to file
    print("Writing downsampled images to file...")
    write_to_file(channels, downsampled_input, reference, output_path / "downsampled", data_type, save_singles=save_singles)

    if write_alignments_bool:
        write_alignments(alignments, downsampled_input, residuals, reference, output_path)

    # process big images
    print("Preparing to process large images...")
    ratios = [float(o) / d for o, d in [dx, dy, dz]]
    scaled_alignments = []
    for n in range(len(original_input)):
        if n == reference or alignments[n][0] is None:
            scaled_alignments.append([0 for i in range(len(alignments[0]))])
        else:
            # alignments and ratios in x-y-z order, we want it in z-y-x order.  iterate backwards.
            scaled_alignments.append([int(alignments[n][i] / ratios[i]) for i in range(len(alignments[0]) - 1, -1, -1)])

    print("alignments: ", alignments)
    print("scaled alignments: ", scaled_alignments)
    original_paths = []
    for o in original_input:
        if o:
            original_paths.append(Path(o))
        else:
            original_paths.append(None)
    original_output_path = output_path / "original"

    # print("Original paths: ", original_paths)
    # print("Original output path: ", original_output_path)
    # print("Reference index: ", reference)
    # print("Scaled Alignments: ", scaled_alignments)
    # print("Save singles: ", save_singles)

    process_big_images(original_paths, original_output_path, reference, scaled_alignments, num_threads=num_threads, save_singles=save_singles, missing_channel=missing_channel)

    if generate_ims:
        print("Generating .ims files")
        system(f'python convert.py -i "{output_path}/downsampled/RGB" -o "{output_path}/downsampled/RGB.ims" -dx {dx[1]} -dy {dy[1]} -dz {dz[1]}')
        system(f'python convert.py -i "{output_path}/original/RGB" -o "{output_path}/original/RGB.ims" -dx {dx[0]} -dy {dy[0]} -dz {dz[0]}')
        if save_singles:
            for i in range(len(channels)):
                if not downsampled_input[i]: continue
                temp = Path(downsampled_input[i]).name
                system(f'python convert.py -i "{output_path}/downsampled/{temp}" -o "{output_path}/downsampled/{temp}.ims" -dx {dx[1]} -dy {dy[1]} -dz {dz[1]}')
            for i in range(len(channels)):
                if not original_input[i]: continue
                temp = Path(original_input[i]).name
                system(f'python convert.py -i "{output_path}/original/{temp}" -o "{output_path}/original/{temp}.ims" -dx {dx[0]} -dy {dy[0]} -dz {dz[0]}')

        print(".ims files created")

    # if not pad_only:
    #     print("Alignments:")
    #     print(alignments)
    print(f"Alignments: {alignments}")
    print("\n\nOperation completed.")


if __name__ == '__main__':
    parser = ArgumentParser(description="Align 3D images using iterative search with Python-OpenCV\n\n",
                            formatter_class=RawDescriptionHelpFormatter,
                            epilog="Developed by Aidan Jan, 2024, Hongwei Dong Lab (B.R.A.I.N) at UCLA\n")
    parser.add_help = True
    parser.add_argument('--red', '-r', nargs=2, default=[None, None],
                        help='Input file paths for the red original and downsampled images (in that order).')
    parser.add_argument('--green', '-g', nargs=2, default=[None, None],
                        help='Input file paths for the green original and downsampled images (in that order).')
    parser.add_argument('--blue', '-b', nargs=2, default=[None, None],
                        help='Input file paths for the blue original and downsampled images (in that order).')
    # parser.add_argument('--input', '-i', required=True, type=str, help="Absolute file path of tiff stacks representing images to be aligned.  This directory must contain exactly three subfolders with the tiff files."
    parser.add_argument('--output', '-o', required=True, type=str,
                        help="Absolute file path of output.  [REQUIRED]")
    # parser.add_argument('--num_channels', default=3, type=int, help="Number of channels to align")
    # parser.add_argument('--edge_detection', type=str,
    #                     help="Selects which edge detection algorithm to use.  Options: 'sobel', 'canny'.  [REQUIRED]")
    # parser.add_argument('--pad_only', action='store_true', help="If present, only pad images to same shape without aligning")
    parser.add_argument('--write_alignments', action='store_true',
                        help="If present, write alignments to a .txt file.")
    parser.add_argument('--generate_ims', action='store_true',
                        help="If present, generate .ims files along with output.")
    parser.add_argument('--max_iterations', type=int, default=10,
                        help="Maximum iterations allowed for image alignment.")
    parser.add_argument('--reference', type=str, default='red',
                        help="The channel to use as the reference image.  Default red.")
    parser.add_argument('--num_threads', type=int, default=8,
                        help="Number of threads to use for processing large images.  Default 8.")
    parser.add_argument('--save_singles', action='store_true',
                        help="If present, saves single channels with the RGB channel.")
    parser.add_argument('--dtype', type=str, default='uint8',
                        help="Data type of output tifs.  Options include 'uint8', 'uint16', 'uint32', 'float32', 'float64'")
    parser.add_argument('--dx', required=True, nargs=2, type=int,
                        help="micrometers per x-dimension of voxel in original and downsampled images, respectively.  [REQUIRED]")
    parser.add_argument('--dy', required=True, nargs=2, type=int,
                        help="micrometers per y-dimension of voxel in original and downsampled images, respectively.  [REQUIRED]")
    parser.add_argument('--dz', required=True, nargs=2, type=int,
                        help="micrometers per z-dimension of voxel in original and downsampled images, respectively.  [REQUIRED]")

    main(parser.parse_args())
