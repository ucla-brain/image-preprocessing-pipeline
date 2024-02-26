# For Stitching Light Sheet data
# Version 2 by Keivan Moradi on July 2022
# Please read the readme file for more information:
# https://github.com/ucla-brain/image-preprocessing-pipeline/blob/main/README.md
import logging as log
import os
import platform
import sys
from datetime import timedelta
from math import floor
from multiprocessing import freeze_support, Queue, Process, Pool, set_start_method
from pathlib import Path
from platform import uname
from queue import Empty
from re import compile, match, findall, IGNORECASE, MULTILINE
from subprocess import check_output, call, Popen, PIPE, CalledProcessError
from time import time, sleep
from typing import List, Tuple, Union, LiteralString
from argparse import RawDescriptionHelpFormatter, ArgumentParser, BooleanOptionalAction

import mpi4py
import psutil
from cpufeature.extension import CPUFeature
from cv2 import MOTION_TRANSLATION, findTransformECC, TERM_CRITERIA_COUNT, TERM_CRITERIA_EPS
from numpy import ndarray, zeros, uint8, float32, eye, dstack, append, array, absolute, expm1
from numpy import round as np_round
from numpy.linalg import inv
from psutil import cpu_count, virtual_memory
from tqdm import tqdm
from skimage.measure import block_reduce
from skimage.transform import warp
from skimage.filters import sobel
from skimage.filters.thresholding import threshold_multiotsu

from flat import create_flat_img
from parallel_image_processor import parallel_image_processor, jumpy_step_range
from pystripe.core import (batch_filter, imread_tif_raw_png, imsave_tif, MultiProcessQueueRunner, progress_manager,
                           process_img, convert_to_8bit_fun, log1p_jit, prctl, np_max, np_mean, is_uniform_2d,
                           calculate_pad_size, cuda_get_device_properties, cuda_device_count,
                           CUDA_IS_AVAILABLE_FOR_PT, USE_PYTORCH, USE_JAX)
from supplements.cli_interface import (ask_for_a_number_in_range, date_time_now, PrintColors)
from supplements.tifstack import TifStack, imread_tif_stck
from tsv.volume import TSVVolume, VExtent

# experiment setup: user needs to set them right
# AllChannels = [(channel folder name, rgb color)]
AllChannels: List[Tuple[str, str]] = [
    ("Ex_488_Em_525", "b"), ("Ex_561_Em_600", "g"), ("Ex_647_Em_690", "r"), ("Ex_642_Em_690", "r"),
    ("Ex_488_Em_1", "b"), ("Ex_561_Em_1", "g"), ("Ex_642_Em_1", "r"),
    ("Ex_488_Ch0", "b"), ("Ex_561_Ch1", "g"), ("Ex_642_Ch2", "r"),
    ("Ex_488_Em_2", "b"), ("Ex_561_Em_2", "g"), ("Ex_642_Em_2", "r"), ("Ex_642_Em_680", "r")
]
VoxelSizeX_4x, VoxelSizeY_4x = (1.809,) * 2  # old stage --> 1.835
VoxelSizeX_8x, VoxelSizeY_8x = (0.82,) * 2
VoxelSizeX_9x, VoxelSizeY_9x = (0.72,) * 2
VoxelSizeX_10x, VoxelSizeY_10x = (0.62,) * 2
VoxelSizeX_15x, VoxelSizeY_15x = (0.41,) * 2
VoxelSizeX_40x, VoxelSizeY_40x = (0.14, 0.14)  # 0.143, 0.12
SUPPORTED_EXTENSIONS = ('.png', '.tif', '.tiff', '.raw')


def p_log(txt: Union[str, list]):
    txt: str = str(txt)
    print(txt)
    for _ in range(40):
        try:
            txt = txt.replace(
                PrintColors.ENDC, '').replace(
                PrintColors.WARNING, '').replace(
                PrintColors.BLUE, '').replace(
                PrintColors.GREEN, '').replace(
                PrintColors.FAIL, '').replace(
                PrintColors.BOLD, '').replace(
                PrintColors.CYAN, '').replace(
                PrintColors.HEADER, '').replace(
                PrintColors.UNDERLINE, '')
            log.info(txt)
        except (OSError, PermissionError):
            sleep(0.1)
            continue
        break


def get_voxel_sizes(objective: str, path: Path, is_mip: bool):
    tile_size = (2000, 2000)
    if objective == "4x":
        voxel_size_x = VoxelSizeX_4x
        voxel_size_y = VoxelSizeY_4x
        tile_size = (1600, 2000)  # y, x = tile_size
    elif objective == "8x":
        voxel_size_x = VoxelSizeX_8x
        voxel_size_y = VoxelSizeY_8x
    elif objective == "9x":
        voxel_size_x = VoxelSizeX_9x
        voxel_size_y = VoxelSizeY_9x
    elif objective == "10x":
        voxel_size_x = VoxelSizeX_10x
        voxel_size_y = VoxelSizeY_10x
    elif objective == "15x":
        voxel_size_x = VoxelSizeX_15x
        voxel_size_y = VoxelSizeY_15x
    elif objective == "40x":
        voxel_size_x = VoxelSizeX_40x
        voxel_size_y = VoxelSizeY_40x
        tile_size = (2048, 2048)
    # elif objective == "6":
    #     objective = ""
    #     tile_size_x = ask_for_a_number_in_range("what is the tile size on x axis in pixels?", (1, 2049), int)
    #     tile_size_y = ask_for_a_number_in_range("what is the tile size on y axis in pixels?", (1, 2049), int)
    #     voxel_size_x = ask_for_a_number_in_range("what is the x voxel size in µm?", (0.001, 1000), float)
    #     voxel_size_y = ask_for_a_number_in_range("what is the y voxel size in µm?", (0.001, 1000), float)
    #     tile_size = (tile_size_y, tile_size_x)
    else:
        print("Error: unsupported objective")
        log.error("Error: unsupported objective")
        raise RuntimeError

    voxel_size_z = None
    for y_folder in path.iterdir():
        if y_folder.is_dir() and voxel_size_z is None:
            for x_folder in y_folder.iterdir():
                if x_folder.is_dir():
                    files = sorted([f for f in x_folder.iterdir() if
                                    f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file()])
                    if len(files) > 1:
                        try:
                            voxel_size_z = (int(files[1].stem) - int(files[0].stem)) / 10
                            break
                        except ValueError as e:
                            print(e)
                            pass

    if voxel_size_z is None:
        voxel_size_z = ask_for_a_number_in_range(
            "what is the z-step size in µm?\n"
            f"{PrintColors.BLUE}hint: z-step is {'400' if is_mip else 'typically 0.8'} µm for the "
            f"{'MIP' if is_mip else 'main'} images generated by SmartSPIM{PrintColors.ENDC}",
            (0.001, 1000), float)
    return voxel_size_x, voxel_size_y, voxel_size_z, tile_size


def get_list_of_files(y_folder: Path, extensions=SUPPORTED_EXTENSIONS) -> List[Path]:
    extensions: Tuple[LiteralString, ...] = tuple(ext.lower() for ext in extensions)
    files_list = []
    for file in y_folder.iterdir():
        if file.suffix.lower() in extensions:
            files_list += [file]
    return files_list


def inspect_for_missing_tiles_get_files_list(channel_path: Path):
    p_log(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
          f"inspecting channel {channel_path.name} for missing files.")
    folders_list = [y for x in channel_path.iterdir() if x.is_dir() for y in x.iterdir() if y.is_dir()]
    file_list = list(tqdm(
        map(get_list_of_files, folders_list),
        total=len(folders_list),
        desc="inspection",
        mininterval=1.0,
        unit=" tiles",
        ascii=True,
        smoothing=0.05
    ))

    path_dict = {}
    unraveled_file_list = []
    for y_folder, files in zip(folders_list, file_list):
        count = len(files)
        y_folders_list: list = path_dict.get(count, [])
        y_folders_list += [str(y_folder)]
        path_dict.update({count: y_folders_list})
        unraveled_file_list += files

    counts_list = sorted([int(count) for count, folder_list in path_dict.items()])
    if (len(counts_list) > 1 and counts_list[0] != 1) or (len(counts_list) > 2 and counts_list[0] == 1):
        p_log(counts_list)
        p_log(f"{PrintColors.WARNING}warning: following folders might have missing tiles:{PrintColors.ENDC}")
        for count in counts_list[:-1]:
            if count != 1:
                folders = "\n\t\t".join(path_dict[count])
                p_log(f"{PrintColors.WARNING}\tfolders having {count} tiles: \n"
                      f"\t\t{folders}{PrintColors.ENDC}")

    return unraveled_file_list, counts_list[-1]


def correct_path_for_cmd(filepath):
    if sys.platform == "win32":
        return f"\"{filepath}\""
    else:
        return str(filepath).replace(" ", r"\ ").replace("(", r"\(").replace(")", r"\)")


def correct_path_for_wsl(filepath):
    p = compile(r"/mnt/(.)/")
    new_path = p.sub(r'\1:\\\\', str(filepath))
    new_path = new_path.replace(" ", r"\ ").replace("(", r"\(").replace(")", r"\)").replace("/", "\\\\")
    return new_path


def worker(command: str):
    return_code = call(command, shell=True)
    p_log(f"\nfinished:\n\t{command}\n\treturn code: {return_code}\n")
    return return_code


class MultiProcessCommandRunner(Process):
    def __init__(self, queue, command, pattern: str = "", position: int = None, percent_conversion: float = 100,
                 check_stderr: bool = False):
        Process.__init__(self)
        super().__init__()
        self.daemon = True
        self.queue = queue
        self.command = command
        self.position = position
        self.pattern = pattern
        self.percent_conversion = percent_conversion
        self.check_stderr = check_stderr

    def run(self):
        return_code = None  # 0 == success and any other number is an error code
        previous_percent, percent = 0, 0
        try:
            if self.position is None:
                return_code = worker(self.command)
            else:
                process = Popen(
                    self.command,
                    stdout=PIPE,
                    stderr=PIPE,
                    shell=True,
                    text=True,
                    start_new_session=False,
                    bufsize=0
                )
                pattern = compile(self.pattern, IGNORECASE | MULTILINE)
                while return_code is None:
                    return_code = process.poll()
                    if self.check_stderr:
                        output = process.stderr.readline()
                        if not isinstance(output, str):
                            output = str(output)
                    else:
                        output = process.stdout.readline()
                    matches = match(pattern, output)
                    if matches:
                        percent = round(float(matches[1]) * self.percent_conversion, 1)
                        self.queue.put([percent - previous_percent, self.position, return_code, self.command])
                        previous_percent = percent
                if not self.check_stderr:
                    error = process.stderr.read()
                    if error:
                        print(f"{PrintColors.FAIL}Errors:\n{error}{PrintColors.ENDC}")
        except Exception as inst:
            p_log(f"{PrintColors.FAIL}"
                  f"Process failed for command:\n"
                  f"\t{self.command}.\n"
                  f"Error:\n"
                  f"\ttype: {type(inst)}\n"
                  f"\targs: {inst.args}\n"
                  f"\t{inst}"
                  f"{PrintColors.ENDC}")

        final_percent = 100 - previous_percent
        if return_code is not None and int(return_code) != 0:
            final_percent = 0
        self.queue.put([final_percent, self.position, return_code, self.command])


def reorder_list(a, b):
    for c in b:
        a.remove(c)
    return b + a


def execute(command):
    popen = Popen(command, stdout=PIPE, shell=True, text=True, universal_newlines=True, bufsize=1)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        p_log(f"\n{PrintColors.FAIL}Process failed for command:\n\t{command}{PrintColors.ENDC}\n")
        raise CalledProcessError(return_code, command)


def run_command(command, need_progress_dot=True):
    pattern = compile(r"error|warning|fail", IGNORECASE | MULTILINE)
    for stdout in execute(command):
        if need_progress_dot:
            important_messages = findall(pattern, stdout)
            if important_messages:
                p_log(f"\n{PrintColors.WARNING}{stdout}{PrintColors.ENDC}\n")
            else:
                print(".", end="", flush=True)
        else:
            print(stdout)
    p_log("")


def get_gradient(img: ndarray) -> ndarray:
    # threshold: float = 98.5
    # img_percentile = prctl(img, threshold)
    img = img.astype(float32)
    img = sobel(img)
    # img_percentile = prctl(img, threshold)
    # img = where(img >= img_percentile, 255, img / img_percentile * 255)  # scale images
    return img


def estimate_bit_shift(img, threshold: float, percentile=99.9):
    try:
        upper_bound = prctl(img[img > threshold], percentile)
    except (ValueError, AssertionError):
        upper_bound = np_max(img)
    upper_bound = int(np_round(expm1(upper_bound)))
    right_bit_shift: int = 8
    for b in range(0, 9):
        if 256 * 2 ** b >= upper_bound:
            right_bit_shift = b
            break
    return right_bit_shift


def process_channel(
        source_path: Path,
        channel: str,
        channel_for_alignment: str,
        preprocessed_path: Path,
        stitched_path: Path,
        voxel_size_x: float,
        voxel_size_y: float,
        voxel_size_z: float,
        objective: str,
        queue: Queue,
        stitch_mip: bool,
        isotropic_downsampl_downsampled_path: Path,
        isotropic_downsampling_resolution: float = 10,
        files_list: List[Path] = None,
        need_flat_image_application: bool = False,
        image_classes_training_data_path=None,
        need_gaussian_filter_2d: bool = True,
        dark: int = 0,
        down_sampling_factor: Tuple[int, int] = None,
        tile_size: Tuple[int, int] = None,
        new_tile_size: Tuple[int, int] = None,
        need_destriping: bool = False,
        need_raw_png_to_tiff_conversion: bool = False,
        compression_level: int = 1,
        compression_method: str = "ADOBE_DEFLATE",
        need_bleach_correction: bool = False,
        padding_mode: str = "wrap",
        need_lightsheet_cleaning: bool = True,
        need_rotation_stitched_tif: bool = False,
        need_16bit_to_8bit_conversion: bool = False,
        continue_process_pystripe: bool = True,
        continue_process_terastitcher: bool = True,
        need_tera_fly_conversion: bool = False,
        terafly_path: Path = None,
        subvolume_depth: int = 1,
        print_input_file_names: bool = False,
        timeout: float = None,
        nthreads: int = cpu_count(logical=False)
):
    # preprocess each tile as needed using PyStripe --------------------------------------------------------------------

    assert source_path.joinpath(channel).exists()
    assert isotropic_downsampl_downsampled_path.exists()
    if need_gaussian_filter_2d or need_destriping or need_flat_image_application or \
            need_raw_png_to_tiff_conversion or \
            down_sampling_factor not in (None, (1, 1)) or new_tile_size is not None:
        img_flat = None
        if need_flat_image_application:
            flat_img_created_already = source_path / f'{channel}_flat.tif'
            if flat_img_created_already.exists():
                img_flat = imread_tif_raw_png(flat_img_created_already)
                # with open(source_path / f'{channel}_dark.txt', "r") as f:
                #     dark = int(f.read())
                p_log(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
                      f"{channel}: using the existing flat image:\n"
                      f"\t{flat_img_created_already.absolute()}.")
            else:
                p_log(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
                      f"{channel}: creating a new flat image.")
                img_flat, dark = create_flat_img(
                    source_path / channel,
                    image_classes_training_data_path,
                    tile_size,
                    max_images=1024,  # the number of flat images averaged
                    batch_size=nthreads,
                    patience_before_skipping=nthreads - 1,
                    # the number of non-flat images found successively before skipping
                    skips=256,  # the number of images should be skipped before testing again
                    sigma_spatial=1,  # the de-noising parameter
                    save_as_tiff=True
                )

        tile_destriping_sigma = (0, 0)  # sigma=(foreground, background) Default is (0, 0), indicating no de-striping.
        if need_destriping:
            if objective == "4x":
                tile_destriping_sigma = (100, 100)
            elif objective == "40x":
                tile_destriping_sigma = (128, 256)
            else:
                tile_destriping_sigma = (250, 250)
        p_log(
            f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
            f"{channel}: started preprocessing tile images and converting them to tif.\n"
            f"\tsource: {source_path / channel}\n"
            f"\tdestination: {preprocessed_path / channel}\n"
            f"\tcompression: ({compression_method}, {compression_level})\n"
            f"\tflat application: {img_flat is not None}\n"
            f"\tgaussian: {need_gaussian_filter_2d}\n"
            f"\tbaseline subtraction value: {dark}\n"
            f"\ttile de-striping sigma: {tile_destriping_sigma}\n"
            f"\ttile size: {tile_size}\n"
            f"\tdown sampling factor on xy-plane: {down_sampling_factor}\n"
            f"\tresizing target on xy-plane: {new_tile_size}"
        )

        return_code = batch_filter(
            source_path / channel,
            preprocessed_path / channel,
            files_list=files_list,
            workers=nthreads + get_cpu_sockets() * 2,
            continue_process=continue_process_pystripe,
            print_input_file_names=print_input_file_names,
            timeout=timeout,  # 600.0,
            flat=img_flat,
            gaussian_filter_2d=need_gaussian_filter_2d,
            bleach_correction_frequency=None,  # 0.0005
            bleach_correction_max_method=False,
            sigma=tile_destriping_sigma,
            level=0,
            wavelet="db9",
            crossover=10,
            threshold=None,
            padding_mode="reflect",
            bidirectional=True,
            dark=dark,
            lightsheet=False,
            down_sample=down_sampling_factor,
            tile_size=tile_size,
            new_size=new_tile_size,
            d_type="uint16",
            convert_to_8bit=False,  # need_16bit_to_8bit_conversion
            bit_shift_to_right=8,
            compression=(compression_method, compression_level) if compression_level > 0 else None,
            threads_per_gpu=8  # if sys.platform.lower() == "win32" else 1
        )

        if return_code != 0:
            exit(return_code)

    inspect_for_missing_tiles_get_files_list(preprocessed_path / channel)

    # stitching: align the tiles GPU accelerated & parallel ------------------------------------------------------------
    if (not stitched_path.joinpath(f"{channel_for_alignment}_xml_import_step_5.xml").exists() or
            not continue_process_terastitcher):
        p_log(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
              f"{channel_for_alignment}: aligning tiles using parastitcher ...")

        proj_out = stitched_path / f'{channel_for_alignment}_xml_import_step_1.xml'
        command = [
            f"{terastitcher}",
            "-1",
            f"--ref1={'V' if objective == '40x' else 'H'}",  # x horizontal
            f"--ref2={'H' if objective == '40x' else 'V'}",  # y vertical
            "--ref3=D",  # z depth?
            f"--vxl1={voxel_size_y if objective == '40x' else voxel_size_x:.3f}",
            f"--vxl2={voxel_size_x if objective == '40x' else voxel_size_y:.3f}",
            f"--vxl3={voxel_size_z}",
            "--sparse_data",
            f"--volin={preprocessed_path / channel_for_alignment}",
            f"--projout={proj_out}",
            "--noprogressbar"
        ]
        p_log(f"\t{PrintColors.BLUE}import command:{PrintColors.ENDC}\n\t\t" + " ".join(command))
        run_command(" ".join(command))
        if not proj_out.exists():
            p_log(f"{PrintColors.FAIL}{channel_for_alignment}: importing tif files failed.{PrintColors.ENDC}")
            raise RuntimeError

        def calculate_subvol_and_threads(alignment_depth: int):
            # just a scope to clear unneeded variables
            max_subvolume_depth = 100
            alignment_depth = int(10 if objective == '40x' else min(alignment_depth, max_subvolume_depth))
            alignment_cores: int = 1
            memory_needed_per_thread = 34 * alignment_depth  # 48 or 32
            if isinstance(new_tile_size, tuple):
                for resolution in new_tile_size:
                    memory_needed_per_thread *= resolution
            elif isinstance(tile_size, tuple):
                if isinstance(down_sampling_factor, tuple):
                    for resolution, factor in zip(tile_size, down_sampling_factor):
                        memory_needed_per_thread *= resolution / factor
                else:
                    for resolution in tile_size:
                        memory_needed_per_thread *= resolution
            else:
                memory_needed_per_thread *= 2048 * 2048
            # if TifStack(glob_re(r"\.tiff?$", preprocessed_path.joinpath(channel)).__next__().parent).dtype == uint8:
            #     memory_needed_per_thread /= 2
            memory_ram = virtual_memory().available // 1024 ** 3  # in GB
            memory_needed_per_thread //= 1024 ** 3

            if memory_needed_per_thread <= memory_ram:
                alignment_cores = nthreads
                if memory_needed_per_thread > 0:
                    alignment_cores = min(floor(memory_ram / memory_needed_per_thread), alignment_cores)
                if num_gpus > 0 and sys.platform.lower() == 'linux':
                    while alignment_cores < 6 * num_gpus and alignment_depth > max_subvolume_depth:
                        alignment_depth //= 2
                        alignment_cores *= 2
                else:
                    while alignment_cores < nthreads and alignment_depth > max_subvolume_depth:
                        alignment_depth //= 2
                        alignment_cores *= 2
            else:
                memory_needed_per_thread //= alignment_depth
                while memory_needed_per_thread * alignment_depth > memory_ram and alignment_depth > max_subvolume_depth:
                    alignment_depth //= 2
                memory_needed_per_thread *= alignment_depth

            if num_gpus > 0 and sys.platform.lower() == 'linux':
                alignment_cores = min(alignment_cores, num_gpus * 16)

            return memory_ram, memory_needed_per_thread, alignment_cores, alignment_depth

        free_ram, ram_needed_per_thread, n_cores, subvolume_depth = calculate_subvol_and_threads(subvolume_depth)

        steps_str = ["alignment", "z-displacement", "threshold-displacement", "optimal tiles placement"]
        for step in [2, 3, 4, 5]:
            p_log(
                f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
                f"{channel_for_alignment}: starting step {step} of stitching ..." + ((
                    f"\n\tmemory needed per thread = {ram_needed_per_thread} GB"
                    f"\n\ttotal needed ram {n_cores * ram_needed_per_thread} GB"
                    f"\n\tavailable ram = {free_ram} GB"
                    f"\n\tsubvolume depth = {subvolume_depth} z-steps") if step == 2 else ""))
            proj_in = stitched_path / f"{channel_for_alignment}_xml_import_step_{step - 1}.xml"
            proj_out = stitched_path / f"{channel_for_alignment}_xml_import_step_{step}.xml"

            assert proj_in.exists()
            if step == 2 and n_cores > 1:
                os.environ["slots"] = f"{cpu_count(logical=True)}"
                command = [
                    f"mpiexec {'--bind-to none ' if sys.platform.lower() == 'linux' else ''}"
                    f"-np {int(n_cores + 1)} "  # one extra thread is needed for management
                    f"python -m mpi4py {parastitcher}"
                ]
            else:
                command = [f"{terastitcher}"]
            command += [
                f"-{step}",
                # Overlap (in pixels) between two adjacent tiles along H. Providing oH is harmful sometimes.
                # f"--oH={tile_overlap_x}",
                # Overlap (in pixels) between two adjacent tiles along V. Providing oV is harmful sometimes.
                # f"--oV={0 if objective == '40x' else tile_overlap_y}",
                # Displacements search radius along H (in pixels). Default value is 25!
                # f"--sH=0",
                # Displacements search radius along V (in pixels). Default value is 25!
                # f"--sV=0",
                # Displacements search radius along D (in pixels).
                f"--sD={0 if (objective == '40x' or stitch_mip) else 10}",
                # Number of slices per subvolume partition
                f"--subvoldim={1 if stitch_mip else subvolume_depth}",
                # used in the pairwise displacements computation step.
                # dimension of layers obtained by dividing the volume along D
                "--threshold=0.65",  # threshold between 0.55 and 0.7 is good. Higher values block alignment.
                f"--projin={proj_in}",
                f"--projout={proj_out}",
                # "--restoreSPIM",
            ]
            command = " ".join(command)
            p_log(f"\t{PrintColors.BLUE}{steps_str[step - 2]} command:{PrintColors.ENDC}\n\t\t" + command)
            run_command(command)
            assert proj_out.exists()
            proj_in.unlink(missing_ok=False)

    # stitching: merge tiles to generate stitched 2D tiff series -------------------------------------------------------

    # mpiexec -np 12 python -m mpi4py %Parastitcher% -6 --projin=.\xml_merging.xml --volout="..\%OUTPUTDIR%"
    # --volout_plugin="TiledXY|2Dseries" --slicewidth=100000 --sliceheight=150000

    stitched_tif_path = stitched_path / f"{channel}_tif"
    stitched_tif_path.mkdir(exist_ok=True)

    cosine_blending = False  # True if need_bleach_correction else False
    tsv_volume = TSVVolume(
        stitched_path / f'{channel_for_alignment}_xml_import_step_5.xml',
        alt_stack_dir=preprocessed_path.joinpath(channel).__str__(),
        cosine_blending=cosine_blending
    )
    shape: Tuple[int, int, int] = tsv_volume.volume.shape  # shape is in z y x format

    def estimate_img_related_params():
        # just a scope to clear unneeded variables
        sig = 0
        frequency = None
        background, bit_shift, clip_min, clip_med, clip_max = 0, 8, None, None, None
        if need_16bit_to_8bit_conversion or need_bleach_correction:
            found_threshold = False
            z1 = shape[0] // 2
            while not found_threshold:
                try:
                    p_log(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
                          f"calculating thresholding, and bit shift for 8-bit conversion using img_{z1:06n}.tif ...")
                    img = tsv_volume.imread(
                        VExtent(
                            tsv_volume.volume.x0, tsv_volume.volume.x1,
                            tsv_volume.volume.y0, tsv_volume.volume.y1,
                            tsv_volume.volume.z0 + z1, tsv_volume.volume.z0 + z1 + 1),
                        tsv_volume.dtype)[0]
                    assert not is_uniform_2d(img)
                    assert isinstance(img, ndarray)
                    img = log1p_jit(img, dtype=float32)
                    clip_min, clip_med, clip_max = threshold_multiotsu(img, classes=4)
                    bit_shift = estimate_bit_shift(img, threshold=clip_max, percentile=99.9)
                    assert isinstance(clip_min, float32)
                    assert isinstance(clip_med, float32)
                    assert isinstance(clip_max, float32)
                    assert isinstance(bit_shift, int)
                    assert 0 <= bit_shift <= 8
                    found_threshold = True
                except (ValueError, AssertionError):
                    z1 += 1
            if need_bleach_correction:
                background = int(np_round(expm1(clip_min)))
                if new_tile_size is not None:
                    sig = min(new_tile_size)
                elif down_sampling_factor is not None:
                    sig = min(new_tile_size) // min(down_sampling_factor)
                else:
                    sig = min(tile_size)
                # frequency = 1 / sig

        sigma = (int(sig * 2), ) * 2
        memory_needed_per_thread = 17 if need_bleach_correction else 16
        memory_needed_per_thread *= shape[1] + 2 * calculate_pad_size(shape=shape[1:3], sigma=max(sigma)) + shape[1] % 2
        memory_needed_per_thread *= shape[2] + 2 * calculate_pad_size(shape=shape[1:3], sigma=max(sigma)) + shape[2] % 2
        memory_needed_per_thread /= 1024 ** 3
        if tsv_volume.dtype in (uint8, "uint8"):
            memory_needed_per_thread /= 2
        memory_ram = virtual_memory().available / 1024 ** 3  # in GB
        merge_step_cores = max(1, min(floor(memory_ram / memory_needed_per_thread), nthreads))
        return (
            background, bit_shift,
            sigma, clip_min, clip_med, clip_max, frequency,
            memory_ram, memory_needed_per_thread, merge_step_cores)

    (dark, right_bit_shift,
     bleach_correction_sigma, bleach_correction_clip_min, bleach_correction_clip_med, bleach_correction_clip_max,
     bleach_correction_frequency, free_ram, ram_needed_per_thread, n_cores) = estimate_img_related_params()
    down_sampled_destriping_sigma = (2000, 2000) if need_bleach_correction else (0, 0)  # for 10 um target

    def expm1_int(x: float):
        return x if x is None else int(np_round(expm1(x)))
    p_log(
        f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
        f"{channel}: starting step 6 of stitching, merging tiles into 2D tif series and "
        f"postprocessing the stitched images, using TSV ...\n"
        f"\tsource: {stitched_path / f'{channel}_xml_import_step_5.xml'}\n"
        f"\tdestination: {stitched_tif_path}\n"
        f"\tmemory needed per thread: \t{ram_needed_per_thread:.1f} GB\n"
        f"\tmemory needed total: \t\t{ram_needed_per_thread * min(n_cores, shape[0]):.1f} GB\n"
        f"\tavailable ram: \t\t\t{free_ram:.1f} GB\n"
        f"\tmax threads based on ram: \t{n_cores}\n"
        f"\ttsv volume shape (zyx): \t{shape}\n"
        f"\ttsv volume data type: \t\t{tsv_volume.dtype}\n"
        f"\tbleach correction sigma main: \t{bleach_correction_sigma}\n"
        f"\tbleach correction sigma down sampled: {down_sampled_destriping_sigma}\n"
        f"\tfg vs bg threshold: \t\t{expm1_int(bleach_correction_clip_med)}\n"
        f"\tbidirectional axes (-1, -2): \t{True if need_bleach_correction else False}\n"
        f"\tpadding mode: \t\t\t{padding_mode}\n"
        f"\tpadding size: \t\t\t{calculate_pad_size(shape=shape[1:3], sigma=max(bleach_correction_sigma))}\n"
        f"\tbleach correction frequency: \t{bleach_correction_frequency}\n"
        f"\tbleach correction clip min: \t{expm1_int(bleach_correction_clip_min)}\n"
        f"\tbleach correction clip med: \t{expm1_int(bleach_correction_clip_med)}\n"
        f"\tbleach correction clip max: \t{expm1_int(bleach_correction_clip_max)}\n"
        f"\tbaseline subtraction: \t\t{dark}\n"
        f"\tbackground subtraction: \t{need_lightsheet_cleaning}\n"
        f"\t8-bit conversion: \t\t{need_16bit_to_8bit_conversion}\n"
        f"\tbit-shift to right: \t\t{right_bit_shift}\n"
        f"\trotate: \t\t\t{90 if need_rotation_stitched_tif else 0}"
    )

    gpu_semaphore = None
    if CUDA_IS_AVAILABLE_FOR_PT:
        gpu_semaphore = Queue()
        for i in range(cuda_device_count()):
            gpu_semaphore.put((f"cuda:{i}", cuda_get_device_properties(i).total_memory))
        if sys.platform.lower() != "win32" or (sys.platform.lower() == "win32" and get_cpu_sockets() == 1):
            gpu_semaphore.put(("cpu", virtual_memory().available))

    return_code = parallel_image_processor(
        source=tsv_volume,
        destination=stitched_tif_path,
        fun=process_img,
        kwargs={
            "sigma": bleach_correction_sigma,
            "wavelet": "coif15",  # db37
            "padding_mode": padding_mode,  # wrap reflect
            "gpu_semaphore": gpu_semaphore,
            "bidirectional": True if need_bleach_correction else False,
            "threshold": bleach_correction_clip_med,  # for dual-band sigma
            "bleach_correction_frequency": bleach_correction_frequency,
            "bleach_correction_clip_min": bleach_correction_clip_min,
            "bleach_correction_clip_med": bleach_correction_clip_med,
            "bleach_correction_clip_max": bleach_correction_clip_max,
            "bleach_correction_max_method": False,
            "dark": dark,
            "lightsheet": need_lightsheet_cleaning,
            "percentile": 0.25,
            "rotate": 0,
            "convert_to_8bit": need_16bit_to_8bit_conversion,
            "bit_shift_to_right": right_bit_shift,
            "tile_size": shape[1:3],
            "d_type": tsv_volume.dtype
        },
        source_voxel=(voxel_size_z, voxel_size_y, voxel_size_x),
        target_voxel=None if stitch_mip else isotropic_downsampling_resolution,
        downsampled_path=isotropic_downsampl_downsampled_path,
        down_sampled_destriping_sigma=down_sampled_destriping_sigma,
        rotation=90 if need_rotation_stitched_tif else 0,
        timeout=timeout,
        max_processors=n_cores,
        progress_bar_name="TSV",
        compression=(compression_method, compression_level) if compression_level > 0 else None,
        resume=continue_process_terastitcher,
        needed_memory=ram_needed_per_thread * 1024 ** 3 * 2
    )
    if need_rotation_stitched_tif:
        shape = (shape[0], shape[2], shape[1])
    if return_code != 0:
        exit(return_code)

    # TeraFly ----------------------------------------------------------------------------------------------------------

    # TODO: Paraconverter: Support converting with more cores
    # TODO: Paraconverter: add a progress bar
    running_processes: int = 0
    if need_tera_fly_conversion:
        if terafly_path is None:
            terafly_path = stitched_path
        terafly_path /= f'{channel}_TeraFly'
        terafly_path.mkdir(exist_ok=True)
        command = " ".join([
            f"mpiexec {'--bind-to none ' if sys.platform == 'linux' else ''}"
            f"-np {min(11, nthreads)} "
            f"python -m mpi4py {paraconverter}",
            # f"{teraconverter}",
            "--sfmt=\"TIFF (series, 2D)\"",
            "--dfmt=\"TIFF (tiled, 3D)\"",
            "--resolutions=\"012345\"",
            "--clist=0",
            "--halve=mean",
            # "--noprogressbar",
            # "--sparse_data",
            # "--fixed_tiling",
            # "--height=256",
            # "--width=256",
            # "--depth=256",
            f"-s={stitched_tif_path}",
            f"-d={terafly_path}",
        ])
        p_log(
            f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
            f"{channel}: starting to convert to TeraFly format ...\n"
            f"\tsource: {stitched_tif_path}\n"
            f"\tdestination: {terafly_path}\n"
            f"\t{PrintColors.BLUE}TeraFly conversion command:{PrintColors.ENDC}\n\t\t{command}\n"
        )
        MultiProcessCommandRunner(queue, command).start()
        running_processes += 1

    return stitched_tif_path, shape, running_processes


def get_transformation_matrix(reference: ndarray, subject: ndarray,
                              iterations: int = 10000, termination: float = 1e-10) -> ndarray:
    warp_matrix = eye(2, 3, dtype=float32)  # i.e. no transformation
    if reference is not None and subject is not None:
        downsampling_factors = [1, 1]  # y, x
        for idx in range(len(downsampling_factors)):
            max_size = max(reference.shape[idx], subject.shape[idx])
            while max_size > 32767:
                downsampling_factors[idx] *= 2
                max_size //= 2
        downsampling_factors = max(downsampling_factors)
        print(f"downsampling factor for transformation_matrix {downsampling_factors}")
        reference = block_reduce(reference, block_size=downsampling_factors, func=np_mean)
        subject = block_reduce(subject, block_size=downsampling_factors, func=np_mean)

        # generate matrices
        cc, warp_matrix = findTransformECC(
            reference,
            subject,
            warp_matrix,
            MOTION_TRANSLATION,  # MOTION_AFFINE MOTION_TRANSLATION
            (TERM_CRITERIA_COUNT | TERM_CRITERIA_EPS, iterations, termination),
            inputMask=None,
            gaussFiltSize=5  # default value is 5
        )
        warp_matrix[0, 2] *= downsampling_factors  # x
        warp_matrix[1, 2] *= downsampling_factors  # y

    warp_matrix = inv(append(warp_matrix, array([[0, 0, 1]], dtype=float32), axis=0))
    print(np_round(warp_matrix, 2))
    return warp_matrix


def correct_shape(img: ndarray, shape: Tuple[int, int], zero_is_origin: bool = True):
    if img.shape == shape:
        return img
    elif zero_is_origin:
        if img.shape[0] >= shape[0] and img.shape[1] >= shape[1]:
            return img[0:shape[0], 0:shape[1]]
        else:
            img_new = zeros(shape=shape, dtype=img.dtype)
            if img.shape[0] < shape[0] and img.shape[1] >= shape[1]:
                img_new[0:img.shape[0], :] = img[:, 0:shape[1]]
            elif img.shape[0] >= shape[0] and img.shape[1] < shape[1]:
                img_new[:, 0:img.shape[1]] = img[0:shape[0], :]
            else:
                img_new[0:img.shape[0], 0:img.shape[1]] = img
            return img_new
    else:
        if img.shape[0] >= shape[0] and img.shape[1] >= shape[1]:
            return img[img.shape[0]-shape[0]:, img.shape[1]-shape[1]:]
        else:
            img_new = zeros(shape=shape, dtype=img.dtype)
            if img.shape[0] < shape[0] and img.shape[1] >= shape[1]:
                img_new[shape[0] - img.shape[0]:, :] = img[:, img.shape[1] - shape[1]:]
            elif img.shape[0] >= shape[0] and img.shape[1] < shape[1]:
                img_new[:, shape[1] - img.shape[1]:] = img[img.shape[0] - shape[0]:, :]
            else:
                img_new[shape[0] - img.shape[0]:, shape[1] - img.shape[1]:] = img
            return img_new


def transformation_is_needed(matrix: ndarray):
    matrix = absolute(np_round(matrix, 2))
    if matrix[0, 2] >= 1 or matrix[1, 2] >= 1:  # translation is needed
        return True
    elif matrix[0, 1] > 0.01 or matrix[1, 0] > 0.01:  # shearing is needed
        return True
    else:
        return False


def generate_composite_image(
        img_idx: int,
        tif_stacks: List[TifStack],
        transformation_matrices: List[ndarray],
        order_of_colors: str,
        merged_tif_path: Path,
        resume: bool,
        compression: Union[Tuple[str, int], None] = ("ADOBE_DEFLATE", 1),
        right_bit_shifts: Union[Tuple[int, ...], None] = None
):
    # there should always be 1 more image than number of matrices.  Example: 3 images, 2 matrices.
    assert len(transformation_matrices) + 1 == len(tif_stacks)
    save_path = merged_tif_path / f"img_{img_idx:06n}.tif"
    if resume and save_path.exists():
        return
    images = [tif_stack[img_idx] for tif_stack in tif_stacks]
    assert images[0] is not None
    if right_bit_shifts is not None:
        images = [convert_to_8bit_fun(img, bit_shift_to_right=bsh) for img, bsh in zip(images, right_bit_shifts)]

    img_shape = images[0].shape
    img_dtype = images[0].dtype
    for idx in range(1, len(images)):
        if images[idx] is None:
            images[idx] = zeros(img_shape, dtype=img_dtype)
        elif transformation_is_needed(transformation_matrices[idx - 1]):
            images[idx] = warp(images[idx], transformation_matrices[idx - 1],
                               output_shape=img_shape, preserve_range=True)
            images[idx] = images[idx].astype(img_dtype)
            # images[idx] = correct_shape(images[idx], img_shape, zero_is_origin=True)
        else:
            images[idx] = correct_shape(images[idx], img_shape, zero_is_origin=True)
        assert images[idx].shape == img_shape

    if len(tif_stacks) == 3:
        color_idx = {color: idx for idx, color in enumerate(order_of_colors.lower())}
        images = [images[color_idx[color]] for color in "rgb"]
    elif len(tif_stacks) == 2:
        images += [zeros(img_shape, dtype=img_dtype)]
    images = dstack(images)
    imsave_tif(save_path, images, compression=compression)


def merge_all_channels(
        tif_paths: List[Path],
        z_offsets: List[int],
        merged_tif_path: Path,
        order_of_colors: str = "gbr",
        workers: int = cpu_count(logical=False),
        resume: bool = True,
        compression: Union[Tuple[str, int], None] = ("ADOBE_DEFLATE", 1),
        right_bit_shifts: Union[Tuple[int, ...], None] = None
):
    """
    Merge and align different channels to RGB color tif files.
    file names should be identical for each z-step of each channel

    stitched_tif_paths:
        list of Path objects for different channel locations. The first element is the reference channel.
    z_offsets:
        z-step offset of the channels with respect to the reference channel.
        Both negative and positive z-steps are allowed.
    merged_tif_path:
        Path for saving the results.
    order_of_colors:
        the color of each channel.
    workers:
        number of parallel threads.
    resume:
        resume the work by working on remaining files.
    compression:
        compression method and level.
    right_bit_shift:
        If not none convert the image to 8-bit with the requested bit shift.
        None, and any number between 0 and 8 is accepted. Default if None = no 8-bit conversion and bit shifting.
    """
    z_offsets = [0] + z_offsets
    assert len(tif_paths) == len(z_offsets)
    if right_bit_shifts is not None:
        assert len(right_bit_shifts) == len(tif_paths)
    with Pool(len(tif_paths)) as pool:
        tif_stacks = list(pool.starmap(TifStack, zip(tif_paths, z_offsets)))
    assert all([tif_stack.nz > 0 for tif_stack in tif_stacks])

    merged_tif_path.mkdir(exist_ok=True)
    if resume and len(list(merged_tif_path.glob("*.tif"))) >= max([tif_stack.nz for tif_stack in tif_stacks]):
        return

    img_reference_idx = tif_stacks[0].nz // 2
    if not all([tif_stack.nz > img_reference_idx for tif_stack in tif_stacks]):
        img_reference_idx = min([tif_stack.nz for tif_stack in tif_stacks])
    print(f"reference image index = {img_reference_idx}")
    assert img_reference_idx >= 0
    with Pool(len(tif_paths)) as pool:
        img_samples = list(pool.starmap(imread_tif_stck, zip(tif_stacks, (img_reference_idx,) * len(tif_paths))))
        img_samples = list(map(get_gradient, img_samples))
    assert all([img is not None for img in img_samples])
    transformation_matrices = [get_transformation_matrix(img_samples[0], img) for img in img_samples[1:]]
    del img_samples

    args_queue = Queue(maxsize=tif_stacks[0].nz)
    for idx in jumpy_step_range(0, tif_stacks[0].nz):
        args_queue.put({
            "img_idx": idx,
            "tif_stacks": tif_stacks,
            "transformation_matrices": transformation_matrices,
            "order_of_colors": order_of_colors,
            "merged_tif_path": merged_tif_path,
            "resume": resume,
            "compression": compression,
            "right_bit_shifts": right_bit_shifts
        })

    workers = min(workers, tif_stacks[0].nz)
    progress_queue = Queue()
    for worker_ in range(workers):
        MultiProcessQueueRunner(progress_queue, args_queue,
                                fun=generate_composite_image, replace_timeout_with_dummy=False).start()

    return_code = progress_manager(progress_queue, workers, tif_stacks[0].nz, desc="RGB")
    args_queue.cancel_join_thread()
    args_queue.close()
    progress_queue.cancel_join_thread()
    progress_queue.close()
    if return_code != 0:
        p_log(f"{PrintColors.FAIL}merge_all_channels function failed{PrintColors.ENDC}")
        raise RuntimeError
    # map(lambda arg: generate_composite_image(**arg), [args_queue.get() for _ in range(tif_stacks[0].nz)])


def get_imaris_command(imaris_path: Path, input_path: Path, output_path: Path = None,
                       voxel_size_x: float = 1, voxel_size_y: float = 1, voxel_size_z: float = 1,
                       workers: int = cpu_count(logical=False),
                       dtype: str = 'uint8'):
    files = sorted(list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff")))
    file = files[0]
    command = []
    if imaris_path.exists() and len(files) > 0:
        p_log(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
              f"converting {input_path.name} to ims ... ")

        ims_file_path = input_path.parent / f'{input_path.name}.ims'
        if output_path:
            ims_file_path = output_path

        command = [
            f"" if sys.platform == "win32" else f"WINEDEBUG=-all GLIBC_TUNABLES=glibc.malloc.hugetlb=2 wine",
            f"{imaris_path}",
            f"--input {file}",
            f"--output {ims_file_path}",
        ]
        if sys.platform == "linux" and 'microsoft' in uname().release.lower():
            command = [
                f'{correct_path_for_cmd(imaris_path)}',
                f'--input {correct_path_for_wsl(file)}',
                f"--output {correct_path_for_wsl(ims_file_path)}",
            ]
        if len(files) > 1:
            command += ["--inputformat TiffSeries"]

        command += [
            f"--nthreads {workers if dtype == 'uint8' or sys.platform == 'win32' else 1}",
            f"--compression 3",
            f"--voxelsize {voxel_size_x:.2f}-{voxel_size_y:.2f}-{voxel_size_z:.2f}",  # x-y-z
            "--logprogress"
        ]

    else:
        if len(files) > 0:
            p_log("\tnot found Imaris View: not converting tiff to ims ... ")
        else:
            p_log("\tno tif file found to convert to ims!")

    return " ".join(command)


def commands_progress_manger(queue: Queue, progress_bars: List[tqdm], running_processes: int):
    while running_processes > 0:
        try:
            [percent_addition, position, return_code, command] = queue.get()
            if position is not None and 0 < len(progress_bars) <= position + 1:
                progress_bars[position].update(percent_addition)
            if return_code is not None:
                if return_code > 0:
                    p_log(f"\nFollowing command failed:\n\t{command}\n\treturn code: {return_code}\n")
                else:
                    p_log(f"\nFollowing command succeeded:\n\t{command}\n")
                running_processes -= 1
        except Empty:
            sleep(1)  # waite one second before checking the queue again


def main(args):
    global AllChannels

    source_path = Path(args.input)
    # make sure input path does not have "-" char in its name
    if source_path.exists() and "-" in source_path.name:
        source_path = source_path.rename(source_path.parent / source_path.name.replace("-", "_"))
        print(f"{PrintColors.WARNING}--input path renamed to replace '-' with '_'{PrintColors.ENDC}")
    if not source_path.exists():
        print(f"{PrintColors.FAIL}--input path {source_path} does not exist!{PrintColors.ENDC}")
        raise RuntimeError
    if not source_path.is_dir():
        print(f"{PrintColors.FAIL}--input path {source_path} should be a folder!{PrintColors.ENDC}")
        raise RuntimeError
    all_channels = [c for c, _ in AllChannels if source_path.joinpath(c).exists()]
    if args.stitch_mip:
        all_channels = [
            channel + "_MIP" for channel, _ in AllChannels if source_path.joinpath(channel + "_MIP").exists()]
    if not all_channels:
        print(f"{PrintColors.FAIL}"
              f"could not find unstitched files. --input path should have at least one of these folders "
              f"{[c + '_MIP' for c, _ in AllChannels] if args.stitch_mip else [c for c, _ in AllChannels]}!"
              f"{PrintColors.ENDC}")
        raise RuntimeError

    if not Path(args.tmptif).exists():
        print(f"{PrintColors.FAIL}--tmptif path {args.tmptif} does not exist!{PrintColors.ENDC}")
        raise RuntimeError
    preprocessed_path = Path(args.tmptif) / (source_path.name + "_tif")
    preprocessed_path.mkdir(exist_ok=True)

    if not Path(args.stitched).exists():
        print(f"{PrintColors.FAIL}--stitched path {args.stitched} does not exist!{PrintColors.ENDC}")
        raise RuntimeError
    stitched_path = Path(args.stitched) / (source_path.name + "_stitched")
    stitched_path.mkdir(exist_ok=True)

    need_composite_image = False
    reference_channel = all_channels[0]
    composite_path = Path(args.tmptif)
    if len(all_channels) > 1:
        if args.composite:
            need_composite_image = True
            composite_path = Path(args.composite)
            if not composite_path.exists():
                print(f"{PrintColors.FAIL}composite path {composite_path} did not exist.{PrintColors.ENDC}")
                raise RuntimeError
            composite_path /= source_path.name + "_composite" + ("_MIP" if args.stitch_mip else "")
            composite_path.mkdir(exist_ok=True)
        if args.reference_channel:
            reference_channel = args.reference_channel
        if reference_channel not in all_channels:
            print(f"{PrintColors.FAIL}provided --reference_channel should be among: {all_channels} {PrintColors.ENDC}")
            raise RuntimeError
    channel_color_dict = {channel: color for channel, color in AllChannels} | \
                         {channel + "_MIP": color for channel, color in AllChannels}
    channel_color_dict = {channel: channel_color_dict[channel] for channel in all_channels}

    need_imaris_conversion = False
    imaris_files = None
    if args.imaris:
        need_imaris_conversion = True
        imaris_files = [Path(args.imaris)]
        if len(all_channels) > 1 and not need_composite_image:
            imaris_files = [Path(args.imaris).parent / (channel + ".ims") for channel in all_channels]

        for imaris_file in imaris_files:
            if imaris_file.exists() and imaris_file.is_file():
                print(f"{PrintColors.FAIL}"
                      f"provided --imaris file already exist. You should not overwrite an existing image!"
                      f"{PrintColors.ENDC}")
                raise RuntimeError
            elif not imaris_file.suffix.lower() == ".ims":
                print(f"{PrintColors.FAIL}provided --imaris file should have .ims extension!{PrintColors.ENDC}")
                raise RuntimeError
            elif not imaris_file.parent.exists():
                print(f"{PrintColors.FAIL}parent folder of --imaris file does not exist!{PrintColors.ENDC}")
                raise RuntimeError
            else:
                # make sure the path is writable
                imaris_file.touch(exist_ok=False)
                imaris_file.unlink()

    downsampled_path = stitched_path
    if args.imaris:
        downsampled_path = imaris_files[0].parent
    downsampled_path /= "Downsampled"
    downsampled_path.mkdir(exist_ok=True)

    log_file = downsampled_path.parent / ("log_mip.txt" if args.stitch_mip else "log.txt")
    log.basicConfig(filename=str(log_file), level=log.INFO)
    log.FileHandler(str(log_file), mode="w")  # rewrite the file instead of appending

    objective = args.objective.lower()
    voxel_size_x, voxel_size_y, voxel_size_z, tile_size = get_voxel_sizes(
        objective, source_path / all_channels[0], args.stitch_mip)
    if not args.stitch_mip:
        assert 1 <= args.voxel_size_target
        assert voxel_size_x < args.voxel_size_target
        assert voxel_size_y < args.voxel_size_target
        assert voxel_size_z < args.voxel_size_target

    down_sampling_factor = None
    new_tile_size = None
    if voxel_size_z < voxel_size_x or voxel_size_z < voxel_size_y:
        need_up_sizing = args.isotropic
        if need_up_sizing:
            new_tile_size = (
                int(round(tile_size[0] * voxel_size_y / voxel_size_z, 0)),
                int(round(tile_size[1] * voxel_size_x / voxel_size_z, 0))
            )
            voxel_size_x = voxel_size_y = voxel_size_z
    elif voxel_size_z > voxel_size_x or voxel_size_z > voxel_size_y:
        need_down_sampling = args.isotropic
        if need_down_sampling:
            new_tile_size = (
                int(round(tile_size[0] * voxel_size_y / voxel_size_z, 0)),
                int(round(tile_size[1] * voxel_size_x / voxel_size_z, 0))
            )
            voxel_size_x = voxel_size_y = voxel_size_z
            down_sampling_factor = (int(voxel_size_z // voxel_size_y), int(voxel_size_z // voxel_size_x))
            if down_sampling_factor == (1, 1):
                down_sampling_factor = None
    else:
        p_log(f'{PrintColors.WARNING}'
              f'voxel_size_x = {voxel_size_x}\n'
              f'voxel_size_y = {voxel_size_y}\n'
              f'voxel_size_z = {voxel_size_z}\n'
              f'make sure voxel sizes are really equal!'
              f'{PrintColors.ENDC}')

    def select_channels(needed: bool, channels: List[str], message: str) -> List[str]:
        selected_channels: List[str] = []
        if needed:
            if len(all_channels) == 1:
                selected_channels = all_channels.copy()
            elif channels:
                for channel in channels:
                    if channel in all_channels:
                        selected_channels += [channel]
                    else:
                        print(f"{PrintColors.FAIL}{message} channel {channel} is not among {all_channels}"
                              f"{PrintColors.ENDC}")
                        raise RuntimeError
            else:
                selected_channels = all_channels.copy()
        return selected_channels

    channels_need_bleach_correction = select_channels(
        args.bleach_correction, args.bleach_correction_channels, "bleach correction")

    channels_need_background_subtraction = select_channels(
        args.background_subtraction, args.background_subtraction_channels, "background subtraction")

    channels_need_tera_fly_conversion = select_channels(
        True if args.terafly_channels else False, args.terafly_channels, "terafly conversion")

    terafly_path = stitched_path
    if args.imaris:
        terafly_path = imaris_files[0].parent
    if channels_need_tera_fly_conversion and args.terafly_path:
        terafly_path = Path(args.terafly_path)
        if not terafly_path.exists():
            print(f"{PrintColors.FAIL}--terafly_path {args.terafly_path} does not exits!{PrintColors.ENDC}")
    terafly_path.joinpath("test").touch(exist_ok=False)
    terafly_path.joinpath("test").unlink()
    # Start ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    start_time = time()
    p_log(
        f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC} stitching started"
        f"\n\tRun on computer: {platform.node()}"
        f"\n\tFree physical memory: {virtual_memory().available // 1024 ** 3} GB"
        f"\n\tPhysical CPU core count: {cpu_count(logical=False)}"
        f"\n\tLogical CPU core count: {cpu_count(logical=True)}"
        f"\n\tRequested CPU core count: {args.nthreads}"
        f"\n\tmpi4py version: {mpi4py.__version__}"
        f"\n\tcompression: {(args.compression_method, args.compression_level)}"
        f"\n\tSource folder path:\n\t\t{source_path}"
        f"\n\t\tChannels: {all_channels}"
        f"\n\t\tObjective: {objective}"
        f"\n\t\ttile size: {tile_size}"
        f"\n\t\tVoxel size x: {voxel_size_x} µm"
        f"\n\t\tVoxel size y: {voxel_size_y} µm"
        f"\n\t\tVoxel size z: {voxel_size_z} µm"
        f"\n\tPreprocessed folder path:\n\t\t{preprocessed_path}"
        f"\n\t\tpreprocessing, gaussian: {args.gaussian}"
        f"\n\t\tpreprocessing, down sampling factors (y, x): {down_sampling_factor}"
        f"\n\t\tpreprocessing, new tile size (y, x): {new_tile_size}"
        f"\n\t\tpreprocessing, tile destriping: {args.de_stripe}"
        f"\n\t\tpreprocessing, tif conversion: {args.need_raw_png_to_tiff_conversion}"
        f"\n\tStitched folder path:\n\t\t{stitched_path}"
        f"\n\t\tpostprocessing, bleach correction: {args.bleach_correction}"
        f"\n\t\tpostprocessing, bleach correction channels: {channels_need_bleach_correction}"
        f"\n\t\tpostprocessing, bleach correction padding mode: {args.padding_mode}"
        f"\n\t\tpostprocessing, background subtraction: {args.background_subtraction}"
        f"\n\t\tpostprocessing, background subtraction channels: {channels_need_background_subtraction}"
        f"\n\t\tpostprocessing, conversion to 8-bit: {args.convert_to_8bit}"
        f"\n\t\tpostprocessing, 90 degree rotation: {args.rot90}"
        f"\n\tTerafly path:\n\t\t{terafly_path}"
        f"\n\t\tChannels need terafly conversion: {channels_need_tera_fly_conversion}"
        f"\n\tComposite folder path:\n\t\t{composite_path}"
        f"\n\t\tReference channel:{reference_channel}"
        f"\n\t\tChannel colors:{channel_color_dict}"
        f"\n\tDownsampled path:\n\t\t{downsampled_path}"
        f"\n\tImaris conversion: {need_imaris_conversion}"
        f"\n\tImaris files:\n\t\t{imaris_files}"
        f"\n\ttimeout: {args.timeout}"
        f"\n\tresume: {args.resume}"
        f"\n\tskipconf: {args.skipconf}"
    )

    if not args.skipconf:
        input(f"{PrintColors.BLUE}press enter to continue if everything is OK ... {PrintColors.ENDC}")
    
    # stitch :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # channels need reconstruction will be stitched first to start slow TeraFly conversion as soon as possible
    all_channels = reorder_list(all_channels, channels_need_tera_fly_conversion)
    if args.stitch_based_on_reference_channel_alignment:
        all_channels = reorder_list(all_channels, [reference_channel])
    files_list = [(None, 100), ] * len(all_channels)
    if not args.skip_inspection:
        files_list = list(map(
            inspect_for_missing_tiles_get_files_list,
            [source_path / channel for channel in all_channels]))
    stitched_tif_paths, channel_volume_shapes = [], []
    queue = Queue()
    running_processes: int = 0
    for channel, (file_list, subvolume_depth) in zip(all_channels, files_list):
        stitched_tif_path, shape, running_processes_addition = process_channel(
            source_path,
            channel,
            reference_channel if args.stitch_based_on_reference_channel_alignment else channel,
            preprocessed_path,
            stitched_path,
            voxel_size_x,
            voxel_size_y,
            voxel_size_z,
            args.objective,
            queue,
            args.stitch_mip,
            isotropic_downsampl_downsampled_path=downsampled_path,
            isotropic_downsampling_resolution=args.voxel_size_target,
            files_list=file_list,
            need_flat_image_application=False,
            image_classes_training_data_path=None,
            need_gaussian_filter_2d=args.gaussian,
            dark=0,
            need_destriping=args.de_stripe,
            down_sampling_factor=down_sampling_factor,
            tile_size=tile_size,
            new_tile_size=new_tile_size,
            need_raw_png_to_tiff_conversion=args.need_raw_png_to_tiff_conversion,
            compression_level=args.compression_level,
            compression_method=args.compression_method,
            need_lightsheet_cleaning=channel in channels_need_background_subtraction,
            need_bleach_correction=channel in channels_need_bleach_correction,
            padding_mode=args.padding_mode,
            need_rotation_stitched_tif=args.rot90,
            need_16bit_to_8bit_conversion=args.convert_to_8bit,
            continue_process_pystripe=args.resume,
            continue_process_terastitcher=args.resume,
            need_tera_fly_conversion=channel in channels_need_tera_fly_conversion,
            terafly_path=terafly_path,
            subvolume_depth=subvolume_depth,
            print_input_file_names=False,  # for debugging only
            timeout=args.timeout,
            nthreads=args.nthreads
        )
        stitched_tif_paths += [stitched_tif_path]
        channel_volume_shapes += [shape]
        running_processes += running_processes_addition
    del files_list, file_list

    if not channel_volume_shapes.count(channel_volume_shapes[0]) == len(channel_volume_shapes):
        p_log(
            f"{PrintColors.WARNING}warning: channels had different shapes:\n\t" + "\n\t".join(
                map(lambda p: f"channel {p[0]}: volume shape={p[1]}",
                    zip(all_channels, channel_volume_shapes))) + PrintColors.ENDC
        )

    # merge channels to RGB ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    memory_ram = virtual_memory().available // 1024 ** 3  # in GB
    memory_ram_needed_per_thread = 7 if args.convert_to_8bit else 14
    merge_channels_cores = min(args.nthreads + 2, memory_ram // memory_ram_needed_per_thread)

    composite_tif_paths = stitched_tif_paths.copy()
    if need_composite_image and len(stitched_tif_paths) > 1:
        p_log(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
              f"merging channels to composite RGB started ...\n\t"
              f"time elapsed so far {timedelta(seconds=time() - start_time)}")
        # make sure the reference channel is the first channel in stitched_tif_paths
        stitched_tif_paths = [path for path in stitched_tif_paths if reference_channel.lower() in path.name.lower()] + \
                             [path for path in stitched_tif_paths if reference_channel.lower() not in path.name.lower()]

        order_of_colors: str = channel_color_dict[reference_channel]
        for channel in all_channels:
            if channel != reference_channel:
                order_of_colors += channel_color_dict[channel]

        # print(stitched_tif_paths, order_of_colors)
        if 1 < len(stitched_tif_paths) < 4:
            merge_all_channels(
                stitched_tif_paths,
                [0, ] * (len(stitched_tif_paths) - 1),
                composite_path,
                order_of_colors=order_of_colors,
                workers=merge_channels_cores,
                resume=args.resume,
                compression=(args.compression_method, args.compression_level) if args.compression_level > 0 else None,
                right_bit_shifts=None
            )
            composite_tif_paths = [composite_path]
        elif len(stitched_tif_paths) >= 4:
            p_log(f"{PrintColors.WARNING}"
                  f"Warning: since number of channels are more than 3 merging the first 3 channels only."
                  f"{PrintColors.ENDC}")
            merge_all_channels(
                stitched_tif_paths[0:3],
                [0, ] * 3,
                composite_path,
                order_of_colors=order_of_colors,
                workers=merge_channels_cores,
                resume=args.resume,
                compression=(args.compression_method, args.compression_level) if args.compression_level > 0 else None,
                right_bit_shifts=None
            )
            composite_tif_paths = [composite_path] + stitched_tif_paths[3:]

    # Imaris File Conversion :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    progress_bars = []
    if need_imaris_conversion:
        p_log(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
              f"started ims conversion ...")
        for idx, composite_tif_path in enumerate(composite_tif_paths):
            command = get_imaris_command(
                imaris_path=imaris_converter,
                input_path=composite_tif_path,
                output_path=imaris_files[idx] if idx < len(imaris_files) else None,
                voxel_size_x=voxel_size_y if args.rot90 else voxel_size_x,
                voxel_size_y=voxel_size_x if args.rot90 else voxel_size_y,
                voxel_size_z=voxel_size_z,
                workers=args.nthreads,
                dtype='uint8' if args.convert_to_8bit else 'uint16'
            )
            p_log(f"\t{PrintColors.BLUE}tiff to ims conversion command:{PrintColors.ENDC}\n\t\t{command}\n")
            MultiProcessCommandRunner(queue, command, pattern=r"WriteProgress:\s+(\d*.\d+)\s*$", position=idx).start()
            running_processes += 1
            progress_bars += [
                tqdm(total=100, ascii=True, position=idx, unit=" %", smoothing=0.01,
                     desc=f"IMS {(idx + 1) if len(composite_tif_paths) > 1 else ''}")]

    # waite for TeraFly and Imaris conversion to finish ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    commands_progress_manger(queue, progress_bars, running_processes)

    if channels_need_tera_fly_conversion:
        p_log(
            f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
            f"waiting for TeraFly conversion to finish.\n\t"
            f"time elapsed so far {timedelta(seconds=time() - start_time)}")

    # Done :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    p_log(f"\n{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}done.\n\t"
          f"Time elapsed: {timedelta(seconds=time() - start_time)}")

    stitched_path.joinpath(log_file.name).write_bytes(log_file.read_bytes())


def get_cpu_sockets() -> int:
    try:
        # Execute a shell command to get the physical IDs
        command = "grep \"physical id\" /proc/cpuinfo"
        if sys.platform.lower() == "win32":
            command = "wmic cpu get socketdesignation"
        output: str = check_output(command, shell=True).decode()
        # Count the unique physical IDs
        sockets = len(set(output.strip().split('\n')[1:]))
        return sockets
    except Exception as inst:
        print(f"{PrintColors.FAIL}Unable to determine the number of CPU sockets."
              f"Error:\n"
              f"\ttype: {type(inst)}\n"
              f"\targs: {inst.args}\n"
              f"\t{inst}{PrintColors.ENDC}")
        return 1


if __name__ == '__main__':
    freeze_support()
    FlatNonFlatTrainingData = "image_classes.csv"
    cpu_instruction = "SSE2"
    os.environ['NUMEXPR_NUM_THREADS'] = f'{cpu_count(logical=False) // get_cpu_sockets()}'
    for item in ["SSE2", "AVX", "AVX2", "AVX512f"]:
        cpu_instruction = item if CPUFeature[item] else cpu_instruction
    PyScriptPath = Path(r".") / "TeraStitcher" / "pyscripts"
    if sys.platform.lower() == "win32":
        if get_cpu_sockets() > 1:
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['OMP_NUM_THREADS'] = '1'
        print("Windows is detected.")
        psutil.Process().nice(getattr(psutil, "IDLE_PRIORITY_CLASS"))
        TeraStitcherPath = Path(r"TeraStitcher") / "Windows" / cpu_instruction
        os.environ["PATH"] = f"{os.environ['PATH']};{TeraStitcherPath.as_posix()}"
        os.environ["PATH"] = f"{os.environ['PATH']};{PyScriptPath.as_posix()}"
        terastitcher = "terastitcher.exe"
        mergedisplacements = "mergedisplacements.exe"
        teraconverter = "teraconverter.exe"
        nvidia_smi = "nvidia-smi.exe"
    elif sys.platform.lower() == 'linux':
        if USE_PYTORCH or USE_JAX:
            set_start_method('spawn')
        if USE_JAX:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        os.environ["NUMPY_MADVISE_HUGEPAGE"] = "1"
        if 'microsoft' in uname().release.lower():
            print("Windows subsystem for Linux is detected.")
            nvidia_smi = "nvidia-smi.exe"
        else:
            print("Linux is detected.")
            nvidia_smi = "nvidia-smi"
        psutil.Process().nice(value=19)
        TeraStitcherPath = Path(r".") / "TeraStitcher" / "Linux" / cpu_instruction
        os.environ["PATH"] = f"{os.environ['PATH']}:{TeraStitcherPath.as_posix()}"
        os.environ["PATH"] = f"{os.environ['PATH']}:{PyScriptPath.as_posix()}"
        terastitcher = "terastitcher"
        mergedisplacements = "mergedisplacements"
        teraconverter = "teraconverter"
        os.environ["TERM"] = "xterm"
        os.environ["USECUDA_X_NCC"] = "1"  # set to '' to stop GPU acceleration
        if os.environ["USECUDA_X_NCC"] == "1":
            cuda_version = "11.7"
            if Path("/usr/lib/jvm/java-11-openjdk-amd64/lib/server").exists():
                os.environ["LD_LIBRARY_PATH"] = "/usr/lib/jvm/java-11-openjdk-amd64/lib/server"
            else:
                log.error("Error: JAVA path not found")
                raise RuntimeError
            try:
                cuda_version = compile(r"CUDA *Version: *(\d+(?:\.\d+)?)").findall(str(check_output([nvidia_smi])))[0]
            except (IndexError, CalledProcessError):
                pass
            if Path(f"/usr/local/cuda-{cuda_version}/").exists() and \
                    Path(f"/usr/local/cuda-{cuda_version}/bin").exists():
                os.environ["CUDA_ROOT_DIR"] = f"/usr/local/cuda-{cuda_version}/"
            elif Path(f"/usr/local/cuda/").exists() and \
                    Path(f"/usr/local/cuda/bin").exists():
                os.environ["CUDA_ROOT_DIR"] = f"/usr/local/cuda/"
            elif Path(f"/usr/lib/cuda/").exists() and \
                    Path(f"/usr/lib/cuda/bin").exists():
                os.environ["CUDA_ROOT_DIR"] = f"/usr/lib/cuda/"
            else:
                cuda_version = ask_for_a_number_in_range(
                    f"What is your cuda version (for example {cuda_version})?", (1, 20), float)
                if Path(f"/usr/local/cuda-{cuda_version}/").exists() and \
                        Path(f"/usr/local/cuda-{cuda_version}/bin").exists():
                    os.environ["CUDA_ROOT_DIR"] = f"/usr/local/cuda-{cuda_version}/"
                else:
                    log.error(f"Error: CUDA path not found in {os.environ['CUDA_ROOT_DIR']}")
                    raise RuntimeError
            os.environ["PATH"] = f"{os.environ['PATH']}:{os.environ['CUDA_ROOT_DIR']}/bin"
            os.environ["LD_LIBRARY_PATH"] = f"{os.environ['LD_LIBRARY_PATH']}:{os.environ['CUDA_ROOT_DIR']}/lib64"
            # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # to train on a specific GPU on a multi-gpu machine
    else:
        log.error("yet untested OS")
        raise RuntimeError

    num_gpus = 0
    try:
        num_gpus = str(check_output([nvidia_smi, "-L"])).count('UUID')
    except FileNotFoundError:
        pass

    terastitcher = TeraStitcherPath / terastitcher
    if not terastitcher.exists():
        log.error("Error: TeraStitcher not found")
        raise RuntimeError

    mergedisplacements = TeraStitcherPath / mergedisplacements  # parastitcher needs this program
    if not mergedisplacements.exists():
        log.error("Error: mergedisplacements not found")
        raise RuntimeError

    teraconverter = TeraStitcherPath / teraconverter
    if not terastitcher.exists():
        log.error("Error: TeraConverter not found")
        raise RuntimeError

    parastitcher = PyScriptPath / "Parastitcher.py"
    if not parastitcher.exists():
        log.error("Error: Parastitcher.py not found")
        raise RuntimeError

    paraconverter = PyScriptPath / "paraconverter.py"
    if not paraconverter.exists():
        log.error(f"Error: paraconverter.py not found\n{paraconverter}")
        raise RuntimeError

    imaris_converter = Path(r"imaris") / "ImarisConvertiv.exe"
    if not imaris_converter.exists():
        log.error("Error: ImarisConvertiv.exe not found")
        raise RuntimeError

    parser = ArgumentParser(
        description="Imaris to tif and TeraFly converter (version 0.1.0)\n\n",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="Developed 2022 by Keivan Moradi, Hongwei Dong Lab (B.R.A.I.N) at UCLA\n"
    )
    parser.add_argument("--objective", type=str, required=True,
                        choices=("15x", "9x", "8x", "4x", "40x"),
                        help="objective resolution that is used for imaging: 4x, 8x, 9x, 15x, or 40x.")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to unstitched image in terastither format.")
    parser.add_argument("--tmptif", "-t", type=str, required=True,
                        help="path to temporary preprocessed but unstitched tif files.")
    parser.add_argument("--stitched", "-s", type=str, required=True,
                        help="path to stitched tif files.")
    parser.add_argument("--composite", type=str, required=False,
                        help="path to the composite RGB tif files. "
                             "If provided the image will be converted to a composite RGB color format.")
    parser.add_argument("--reference_channel", type=str, required=False, default='',
                        help="reference channel name for alignment. Applies to the composite image at the moment.")
    parser.add_argument("--stitch_based_on_reference_channel_alignment", default=False,
                        action=BooleanOptionalAction,
                        help="Apply alignment (terastitcger xml) of the reference channel to the rest of channels. "
                             "Works if more than one channels is acquired in one acquisition. "
                             "Disable if channels are acquired in separate acquisitions. "
                             "Default is --no-stitch_based_on_reference_channel_alignment.")
    parser.add_argument("--imaris", "-o", type=str, required=False,
                        help="path to imaris output file.")
    parser.add_argument("--terafly_channels", "-f", required=False, nargs='+', default=[],
                        help="list of channels that need terafly conversion")
    parser.add_argument("--terafly_path", type=str, required=False, default='',
                        help="terafly path. Imaris file parent folder by default.")
    # parser.add_argument("--deconvolve", "-fnt", type=str, required=False, default='',
    #                     help="path to output Fast Neurite Tracer files.")
    # parser.add_argument("--fnt", "-fnt", type=str, required=False, default='',
    #                     help="path to output Fast Neurite Tracer files.")
    parser.add_argument("--nthreads", "-n", type=int, default=cpu_count(logical=False),
                        help="number of threads. default is all physical cores for tif conversion and 12 for TeraFly.")
    parser.add_argument("--stitch_mip", default=False, action=BooleanOptionalAction,
                        help="stitch the MIP image. Default is --no-stitch_mip.")
    parser.add_argument("--need_raw_png_to_tiff_conversion", default=True, action=BooleanOptionalAction,
                        help="Image pre-processing: terastitcher cannot process raw of png file therefore tif "
                             "conversion is needed. "
                             "resume processing remaining files. Disable by --no-need_raw_png_to_tiff_conversion.")
    parser.add_argument("--gaussian", "-g", default=True, action=BooleanOptionalAction,
                        help="image pre-processing: apply Gaussian filter to denoise. Disabled by --no-gaussian.")
    parser.add_argument("--de_stripe", default=True, action=BooleanOptionalAction,
                        help="image pre-processing: apply de-striping algorithm. Disabled by --no-de_stripe")
    parser.add_argument("--padding_mode", type=str, default='wrap',
                        choices=("constant", "edge", "linear_ramp", "maximum", "mean", "median", "minimum", "reflect",
                                 "symmetric", "wrap", "empty"),
                        help="Padding method affects the edge artifacts during bleach correction. "
                             "The default mode is wrap, but in some cases reflect method works better. "
                             "Options: constant, edge, linear_ramp, maximum, mean, median, minimum, reflect, "
                             "symmetric, wrap, and empty")
    parser.add_argument("--isotropic", default=False, action=BooleanOptionalAction,
                        help="image pre-processing: during png/raw to tif conversion downsize or upsize the image "
                             "on xy plane so that voxel size on xy plane become identical to the z axis. "
                             "Default is --no-isotropic")
    parser.add_argument("--bleach_correction", default=True, action=BooleanOptionalAction,
                        help="image post-processing: correct image bleaching. Disable by --no-bleach_correction.")
    parser.add_argument("--bleach_correction_channels", required=False, nargs='+', default=[],
                        help="image post-processing: list of channels that need bleach correction. "
                             "All channels by default.")
    parser.add_argument("--background_subtraction", default=False, action=BooleanOptionalAction,
                        help="image post-processing: apply lightsheet cleaning algorithm. "
                             "Default is --no-background_subtraction")
    parser.add_argument("--background_subtraction_channels", required=False, nargs='+', default=[],
                        help="image post-processing:  list of channels that need background subtraction. "
                             "All channels by default.")
    parser.add_argument("--convert_to_8bit", default=True, action=BooleanOptionalAction,
                        help="Image post-processing: convert to 8-bit. Disable by --no-convert_to_8bit")
    parser.add_argument("--rot90", default=True, action=BooleanOptionalAction,
                        help="Image post-processing: rotate the stitched image for 90 degrees. Disable by --no-rot90")
    parser.add_argument("--compression_method", "-zm", type=str, default="ADOBE_DEFLATE",
                        choices=("ADOBE_DEFLATE", "LZW", "PackBits"),
                        help="image pre/post-processing: compression method for tif files. Default is ADOBE_DEFLATE. "
                             "LZW and PackBits are also supported.")
    parser.add_argument("--compression_level", "-zl", type=int, default=1,
                        choices=range(0, 10), metavar="[0-9]",
                        help="image pre/post-processing: compression level for tif files. Default is 1.")
    parser.add_argument("--voxel_size_target", "-dt", type=float, default=10,
                        help="Image post-processing: target voxel size in µm for 3D downsampling & atlas registration.")
    parser.add_argument("--resume", default=True, action=BooleanOptionalAction,
                        help="Does not apply to FNT or IMS conversion. "
                             "resume processing remaining files. Disable by --no-resume.")
    parser.add_argument("--timeout", type=float, default=None,
                        help="corrupt files: timeout in seconds for image reading. "
                             "Applies to image series and tsv volumes (not ims). "
                             "adds up to 30 percent overhead for copying the data from one process to another "
                             "but if you have corrupt files you can find them this way.")
    parser.add_argument("--skipconf", default=False, action=BooleanOptionalAction,
                        help="Skip confirmation message before beginning processing.")
    parser.add_argument("--skip_inspection", default=False, action=BooleanOptionalAction,
                        help="Skip inspecting unstitched image folders for missing tiles.")
    main(parser.parse_args())
