from argparse import ArgumentParser
from pathlib import Path
from tifffile import imread, imwrite, natural_sorted
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


# axis = [y-axis (rows), x-axis (cols)]
def flip_single_image(path_to_image: Path, output_path: Path, axis=[False, False]):
    img = imread(path_to_image.absolute())
    if axis[0]:
        img = img[::-1]
    if axis[1]:
        img = img[:, ::-1]
    imwrite(output_path.absolute(), img)
    return -1


# used for the specific case where input/output are the same directory and there is a flip over z-axis
# thus, both images must be loaded and swapped to avoid one overwriting the other
def flip_paired_image(path_to_image1: Path, path_to_image2: Path, axis=[False, False]):
    # check to make sure the two paths aren't equal to avoid extra operations
    if path_to_image1.samefile(path_to_image2):
        return flip_single_image(path_to_image1, path_to_image2, axis=axis)

    img1 = imread(path_to_image1.absolute())
    img2 = imread(path_to_image2.absolute())
    if axis[0]:
        img1 = img1[::-1]
        img2 = img2[::-1]
    if axis[1]:
        img1 = img1[:, ::-1]
        img2 = img2[:, ::-1]
    imwrite(path_to_image2.absolute(), img1)
    imwrite(path_to_image1.absolute(), img2)
    return 0


def pair(lst):
    n = len(lst)
    paired_list = [(lst[i], lst[n - 1 - i]) for i in range((n + 1) // 2)]
    return paired_list


# this is to allow starmap to call the functions while still having progress bar
    # dirs is a tuple (input_path, output_path)
    # for some reason this breaks if I make it two separate parameters, so this will have to work.
def flip_helper(dirs, axis, func):
    return 2 + func(dirs[0], dirs[1], axis)


def main():
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='path to input tiff stack', type=str)
    parser.add_argument('--output', '-o', default=None, type=str,
                        help='path to output tiff stack.  Defaults to input if not included.')
    parser.add_argument('--x', '-x', action='store_true', help='include to flip image over the x-axis')
    parser.add_argument('--y', '-y', action='store_true', help='include to flip image over the y-axis')
    parser.add_argument('--z', '-z', action='store_true', help='include to flip image over the z-axis')
    parser.add_argument('--num_threads', '-n', default=4, type=int, help='number of threads to use.  default 4')

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    flip_x = args.x
    flip_y = args.y
    flip_z = args.z
    num_threads = args.num_threads
    execute_pair = False

    if not (flip_x or flip_y or flip_z):
        print('No axis to flip over.  Nothing has been changed.')
        exit(1)

    # Directory checking ------------------------------------------------------------------
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: Input directory '{input_path.absolute()}' is invalid.")
        exit(1)

    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path
        execute_pair = True

    # prepare to flip images --------------------------------------------------------------
    input_tiff_files = list(map(Path, natural_sorted([file.__str__() for file in input_path.iterdir() if file.is_file()
                                and file.suffix.lower() in (".tif", ".tiff")])))

    pool = Pool(processes=num_threads)

    if execute_pair:
        pairs = pair(input_tiff_files)
        partial_func = partial(flip_helper, axis=[flip_y, flip_x], func=flip_paired_image)
    else:
        output_tiff_files = [output_path / path.name for path in input_tiff_files]

        if flip_z:
            output_tiff_files = output_tiff_files[::-1]

        pairs = list(zip(input_tiff_files, output_tiff_files))
        partial_func = partial(flip_helper, axis=[flip_y, flip_x], func=flip_single_image)

    with tqdm(total=len(input_tiff_files)) as progress_bar:
        for a in pool.imap_unordered(partial_func, pairs):
            progress_bar.update(a)

    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
