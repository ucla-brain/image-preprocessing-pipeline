from numpy import ndarray, zeros, pad, copy
from multiprocessing import Pool
from process_images import get_gradient, get_transformation_matrix


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
    plane = "xy",        # must be "xy", "yx", "xz", "zx", "yz", "zy"
    img_format = "zyx",  # xyz in some order
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
    # pad_size = tuple((max(i) for i in zip(img1.shape, img2.shape)))
    #
    # img1 = pad_to_shape(pad_size, img1)
    # img2 = pad_to_shape(pad_size, img2)

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

        x_moves.append(-int((xy_matrix[0][0][2] + xz_matrix[0][0][2]) / 2))
        y_moves.append(int((xy_matrix[0][1][2] + yz_matrix[0][1][2]) / 2))
        z_moves.append(int((xz_matrix[0][1][2] + yz_matrix[0][0][2]) / 2))

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
