# For Stitching Light Sheet data
# Version 2 by Keivan Moradi on July 2022
# Please read the readme file for more information:
# https://github.com/ucla-brain/image-preprocessing-pipeline/blob/main/README.md
import os
import sys
import mpi4py
import platform
import logging as log
import psutil
from subprocess import check_output, call, Popen, PIPE, CalledProcessError
from cpufeature.extension import CPUFeature
from re import compile, match, findall, IGNORECASE, MULTILINE
from psutil import cpu_count, virtual_memory
from tqdm import tqdm
from numpy import ndarray, zeros, uint8
from pathlib import Path
from flat import create_flat_img
from datetime import timedelta
from time import time, sleep
from platform import uname
from tsv.volume import TSVVolume
from pystripe.core import batch_filter, imread_tif_raw_png, imsave_tif, MultiProcessQueueRunner, progress_manager, \
    glob_re, process_img
from queue import Empty
from multiprocessing import freeze_support, Queue, Process
from supplements.cli_interface import select_among_multiple_options, ask_true_false_question, PrintColors
from supplements.cli_interface import ask_for_a_number_in_range, date_time_now
from typing import List, Tuple, Dict, Union
from supplements.downsampling import TifStack
from parallel_image_processor import parallel_image_processor
from math import floor

# experiment setup: user needs to set them right
# AllChannels = [(channel folder name, rgb color)]
AllChannels: List[Tuple[str, str]] = [
    ("Ex_488_Em_525", "b"), ("Ex_561_Em_600", "g"), ("Ex_647_Em_690", "r"), ("Ex_642_Em_690", "r"),
    ("Ex_488_Em_1", "b"), ("Ex_561_Em_1", "g"), ("Ex_642_Em_1", "r"),
    ("Ex_488_Em_2", "b"), ("Ex_561_Em_2", "g"), ("Ex_642_Em_2", "r"), ("Ex_642_Em_680", "r")
]
VoxelSizeX_4x, VoxelSizeY_4x = (1.809,) * 2  # old stage --> 1.835
VoxelSizeX_8x, VoxelSizeY_8x = (0.8,) * 2
VoxelSizeX_10x, VoxelSizeY_10x = (0.661,) * 2  # new stage --> 0.6, 0.6
VoxelSizeX_15x, VoxelSizeY_15x = (0.422,) * 2
VoxelSizeX_15x_new, VoxelSizeY_15x_new = (0.41,) * 2
VoxelSizeX_40x, VoxelSizeY_40x = (0.14, 0.14)  # 0.143, 0.12


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


def get_voxel_sizes(is_mip):
    objective = select_among_multiple_options(
        "What is the Objective?",
        [
            f" 4x: Voxel Size X = {VoxelSizeX_4x:.3f}, Y = {VoxelSizeY_4x:.3f}, tile_size = 1600 x 2000",
            f"10x: Voxel Size X = {VoxelSizeX_10x:.3f}, Y = {VoxelSizeY_10x:.3f}, tile_size = 1850 x 1850",
            f"15x: Voxel Size X = {VoxelSizeX_15x:.3f}, Y = {VoxelSizeY_15x:.3f}, tile_size = 1850 x 1850",
            f"15x: Voxel Size X = {VoxelSizeX_15x_new:.3f}, Y = {VoxelSizeY_15x_new:.3f}, tile_size = 2000 x 2000",
            f"40x: Voxel Size X = {VoxelSizeX_40x:.3f}, Y = {VoxelSizeY_40x:.3f}, tile_size = 2048 x 2048",
            f"other: allows entering custom voxel sizes for custom tile_size"
        ],
        return_index=True
    )

    if objective == "0":
        objective = "4x"
        voxel_size_x = VoxelSizeX_4x
        voxel_size_y = VoxelSizeY_4x
        tile_size = (1600, 2000)  # y, x = tile_size
    elif objective == "1":
        objective = "10x"
        voxel_size_x = VoxelSizeX_10x
        voxel_size_y = VoxelSizeY_10x
        tile_size = (1850, 1850)
    elif objective == "2":
        objective = "15x"
        voxel_size_x = VoxelSizeX_15x
        voxel_size_y = VoxelSizeY_15x
        tile_size = (1850, 1850)
    elif objective == "3":
        objective = "15x"
        voxel_size_x = VoxelSizeX_15x_new
        voxel_size_y = VoxelSizeY_15x_new
        tile_size = (2000, 2000)
    elif objective == "4":
        objective = "40x"
        voxel_size_x = VoxelSizeX_40x
        voxel_size_y = VoxelSizeY_40x
        tile_size = (2048, 2048)
    elif objective == "5":
        objective = ""
        tile_size_x = ask_for_a_number_in_range("what is the tile size on x axis in pixels?", (1, 2049), int)
        tile_size_y = ask_for_a_number_in_range("what is the tile size on y axis in pixels?", (1, 2049), int)
        voxel_size_x = ask_for_a_number_in_range("what is the x voxel size in µm?", (0.001, 1000), float)
        voxel_size_y = ask_for_a_number_in_range("what is the y voxel size in µm?", (0.001, 1000), float)
        tile_size = (tile_size_y, tile_size_x)
    else:
        print("Error: unsupported objective")
        log.error("Error: unsupported objective")
        raise RuntimeError
    voxel_size_z = ask_for_a_number_in_range(
        "what is the z-step size in µm?\n"
        f"{PrintColors.BLUE}hint: z-step is {'500' if is_mip else 'typically 1'} µm for the "
        f"{'MIP' if is_mip else 'main'} images generated by SmartSPIM{PrintColors.ENDC}",
        (0.001, 1000), float)
    tile_overlap_percent = ask_for_a_number_in_range(
        f"what is the tile overlap in percent?\n"
        f"{PrintColors.BLUE}hint: for SmartSPIM typically 10, for the other microscope 16.2{PrintColors.ENDC}",
        (0, 100), float)
    # For tile overlap in smartSPIM is 1 percent higher than the requested value
    if objective != "40x":
        tile_overlap_percent += 1
    p_log(
        f"Objective is {objective} so voxel sizes are:\n"
        f"\tx = {voxel_size_x} µm,\n"
        f"\ty = {voxel_size_y} µm, and\n"
        f"\tz = {voxel_size_z} µm.\n"
        f"Tile overlap is {tile_overlap_percent}%."
    )
    return objective, voxel_size_x, voxel_size_y, voxel_size_z, tile_size, tile_overlap_percent


def get_destination_path(folder_name_prefix, what_for='tif', posix='', default_path=Path('')):
    path_exists = False
    input_path = ''
    drive_path = Path(input_path)
    while not path_exists:
        input_path = input(
            f"\nEnter a valid destination path for {what_for}. "
            f"for example: {CacheDriveExample}\n"
            f"If nothing entered, {default_path.absolute()} will be used.\n").strip()
        if input_path and sys.platform.lower() == "win32" and (
                input_path.endswith(":") or not input_path.endswith("\\")):
            input_path = input_path + "\\"
        elif input_path and sys.platform.lower() == "linux" and not input_path.endswith("/"):
            input_path = input_path + "/"
        drive_path = Path(input_path)
        path_exists = drive_path.exists()
        if not path_exists and ask_true_false_question('The path did not exist! Do you want to create it?'):
            try:
                drive_path.mkdir(parents=True, exist_ok=True)
                path_exists = drive_path.exists()
            except PermissionError:
                print('did not have permission to creat the path!')
    if input_path == '':
        destination_path = default_path.absolute()
    else:
        destination_path = drive_path / (folder_name_prefix + posix)
    continue_process = False
    if destination_path.exists():
        is_ok_to_overwrite = ask_true_false_question(
            f"The path {destination_path.absolute()} exists already. \n"
            f"Do you want to put the processed images in this path?")
        if is_ok_to_overwrite:
            continue_process = ask_true_false_question(
                "If processed images already exist in the path, do you want to process the remaining images?\n"
                "Yes means process the remaining images. \n"
                "No means start over from the beginning and overwrite files.")
        else:
            i = 0
            while destination_path.exists():
                i += 1
                destination_path = destination_path.parent / (destination_path.name + '_' + str(i))
    print(f"\nDestination path for {what_for} is:\n\t{destination_path.absolute()}\n")
    try:
        destination_path.mkdir(exist_ok=True, parents=True)
    except PermissionError:
        print("No permission to create the destination folder.")
        raise RuntimeError
    except Exception as e:
        print(f"An unexpected error happened!\n{e}")
        raise RuntimeError
    return destination_path, continue_process


def get_list_of_files(y_folder: Path, extensions=(".tif", ".tiff", ".raw", ".png")) -> List[Path]:
    extensions: Tuple[str] = tuple(ext.lower() for ext in extensions)
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
        unit="tile",
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
    def __init__(self, queue, command, pattern="", position=None):
        Process.__init__(self)
        super().__init__()
        self.daemon = True
        self.queue = queue
        self.command = command
        self.position = position
        self.pattern = pattern

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
                    # stderr=PIPE,
                    shell=True,
                    text=True)
                pattern = compile(self.pattern, IGNORECASE)
                while return_code is None:
                    return_code = process.poll()
                    m = match(pattern, process.stdout.readline())
                    if m:
                        percent = int(float(m[2]) * 100)
                        self.queue.put([percent - previous_percent, self.position, return_code, self.command])
                        previous_percent = percent
        except Exception as inst:
            p_log(f'Process failed for {self.command}.')
            print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args
            print(inst)
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
            m = findall(pattern, stdout)
            if m:
                p_log(f"\n{PrintColors.WARNING}{stdout}{PrintColors.ENDC}\n")
            else:
                print(".", end="", flush=True)
        else:
            print(stdout)
    p_log("")


def process_channel(
        source_path: Path,
        channel: str,
        preprocessed_path: Path,
        stitched_path: Path,
        voxel_size_x: float,
        voxel_size_y: float,
        voxel_size_z: float,
        objective: str,
        tile_overlap_percent: float,
        queue: Queue,
        stitch_mip: bool,
        dark: int = 0,
        files_list: List[Path] = None,
        need_flat_image_application: bool = False,
        image_classes_training_data_path=None,
        need_raw_png_to_tiff_conversion: bool = False,
        need_lightsheet_cleaning: bool = True,
        need_destriping: bool = False,
        need_gaussian_filter_2d: bool = True,
        need_compression: bool = False,
        need_compression_stitched_tif: bool = True,
        need_rotation_stitched_tif: bool = False,
        need_16bit_to_8bit_conversion: bool = False,
        right_bit_shift=3,
        continue_process_pystripe: bool = True,
        continue_process_terastitcher: bool = True,
        down_sampling_factor: Tuple[int, int] = None,
        tile_size: Tuple[int, int] = None,
        new_tile_size: Tuple[int, int] = None,
        need_tera_fly_conversion: bool = False,
        print_input_file_names: bool = False,
        subvolume_depth: int = 1
):
    # preprocess each tile as needed using PyStripe --------------------------------------------------------------------

    assert source_path.joinpath(channel).exists()
    if (need_destriping and objective == "40x") or need_flat_image_application or \
            need_raw_png_to_tiff_conversion or need_compression or \
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
                    batch_size=cpu_logical_core_count,
                    patience_before_skipping=cpu_logical_core_count - 1,
                    # the number of non-flat images found successively before skipping
                    skips=256,  # the number of images should be skipped before testing again
                    sigma_spatial=1,  # the de-noising parameter
                    save_as_tiff=True
                )

        p_log(
            f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
            f"{channel}: started preprocessing images and converting them to tif.\n"
            f"\tsource: {source_path / channel}\n"
            f"\tdestination: {preprocessed_path / channel}\n"
            f"\tcompression: (ADOBE_DEFLATE, {1 if need_compression else 0})\n"
            f"\tdark = {dark}"
        )

        # sigma=(foreground, background) Default is (0, 0), indicating no de-striping.
        sigma = (0, 0)
        # if need_destriping:
        #     if objective == "4x":
        #         sigma = (32, 32)
        #     elif objective == "40x":
        #         sigma = (128, 256)
        #     else:
        #         sigma = (256, 256)
        if need_destriping and objective == "40x":
            sigma = (128, 256)

        return_code = batch_filter(
            source_path / channel,
            preprocessed_path / channel,
            files_list=files_list,
            workers=cpu_physical_core_count + 2,
            # chunks=128,  # is calculated automatically now
            flat=img_flat,
            gaussian_filter_2d=need_gaussian_filter_2d,
            dark=dark,
            sigma=sigma,
            level=0,
            wavelet="db9" if objective == "40x" else "db10",
            crossover=10,
            threshold=-1,
            directions='vh' if objective == "40x" else "v",
            # z_step=voxel_size_z,  # z-step in micron. Only used for DCIMG files.
            # rotate=False,
            lightsheet=False,  # need_lightsheet_cleaning
            # artifact_length=artifact_length,
            # percentile=0.25,
            # convert_to_16bit=False,  # defaults to False
            convert_to_8bit=False,  # need_16bit_to_8bit_conversion
            bit_shift_to_right=right_bit_shift,
            continue_process=continue_process_pystripe,
            down_sample=down_sampling_factor,
            dtype="uint16",
            tile_size=tile_size,
            new_size=new_tile_size,
            print_input_file_names=print_input_file_names,
            timeout=600.0,
            compression=('ADOBE_DEFLATE', 1) if need_compression else None,  # ('ZSTD', 1) conda install imagecodecs
        )

        if return_code != 0:
            exit(return_code)

    inspect_for_missing_tiles_get_files_list(preprocessed_path / channel)

    # stitching: align the tiles GPU accelerated & parallel ------------------------------------------------------------
    p_log(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
          f"{channel}: aligning tiles using parastitcher ...")

    if not stitched_path.joinpath(f"{channel}_xml_import_step_5.xml").exists() or not continue_process_terastitcher:
        tile_overlap_y = int(tile_size[0] * tile_overlap_percent / 100)
        tile_overlap_x = int(tile_size[1] * tile_overlap_percent / 100)
        if new_tile_size is not None:
            tile_overlap_y = int(new_tile_size[0] * tile_overlap_percent / 100)
            tile_overlap_x = int(new_tile_size[1] * tile_overlap_percent / 100)

        proj_out = stitched_path / f'{channel}_xml_import_step_1.xml'
        command = [
            f"{terastitcher}",
            "-1",
            f"--ref1={'V' if objective == '40x' else 'H'}",  # x horizontal
            f"--ref2={'H' if objective == '40x' else 'V'}",  # y vertical
            "--ref3=D",  # z depth?
            f"--vxl1={voxel_size_y if objective == '40x' else voxel_size_x}",
            f"--vxl2={voxel_size_x if objective == '40x' else voxel_size_y}",
            f"--vxl3={voxel_size_z}",
            "--sparse_data",
            f"--volin={preprocessed_path / channel}",
            f"--projout={proj_out}",
            "--noprogressbar"
        ]
        p_log(f"\t{PrintColors.BLUE}import command:{PrintColors.ENDC}\n\t\t" + " ".join(command))
        run_command(" ".join(command))
        if not proj_out.exists():
            p_log(f"{PrintColors.FAIL}{channel}: importing tif files failed.{PrintColors.ENDC}")
            raise RuntimeError

        # each alignment thread needs about 16GB of RAM in 16bit and 8GB in 8bit
        subvolume_depth = int(10 if objective == '40x' else subvolume_depth)
        memory_needed_per_thread = 64 * subvolume_depth
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
        if TifStack(glob_re(r"\.tiff?$", preprocessed_path.joinpath(channel)).__next__().parent).dtype == uint8:
            memory_needed_per_thread /= 2
        memory_ram = virtual_memory().available // 1024 ** 3  # in GB
        memory_needed_per_thread //= 1024 ** 3

        alignment_cores: int = 1
        min_subvolume_depth = 100
        if memory_needed_per_thread <= memory_ram:
            alignment_cores = cpu_physical_core_count
            if memory_needed_per_thread > 0:
                alignment_cores = min(floor(memory_ram / memory_needed_per_thread), cpu_physical_core_count)
            if num_gpus > 0 and sys.platform.lower() == 'linux':
                while alignment_cores < 6*num_gpus and subvolume_depth > min_subvolume_depth:
                    subvolume_depth //= 2
                    alignment_cores *= 2
            else:
                while alignment_cores < cpu_physical_core_count and subvolume_depth > min_subvolume_depth:
                    subvolume_depth //= 2
                    alignment_cores *= 2
        else:
            memory_needed_per_thread //= subvolume_depth
            while memory_needed_per_thread * subvolume_depth > memory_ram and subvolume_depth > min_subvolume_depth:
                subvolume_depth //= 2
            memory_needed_per_thread *= subvolume_depth

        if num_gpus > 0 and sys.platform.lower() == 'linux':
            if memory_needed_per_thread > 0:
                alignment_cores = min(alignment_cores, num_gpus * 6)
            else:
                alignment_cores = max(alignment_cores, num_gpus * 6)
        p_log(f"memory_needed_per_thread = {memory_needed_per_thread} GB\n"
              f"memory_ram = {memory_ram} GB")
        # while alignment_cores < cpu_physical_core_count and subvolume_depth > 600:
        #     subvolume_depth //= 2
        #     alignment_cores *= 2
        alignment_cores = int(alignment_cores + 1)  # one extra thread is needed for management

        steps_str = ["alignment", "z-displacement", "threshold-displacement", "optimal tiles placement"]
        for step in [2, 3, 4, 5]:
            p_log(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
                  f"{channel}: starting step {step} of stitching ...")
            proj_in = stitched_path / f"{channel}_xml_import_step_{step - 1}.xml"
            proj_out = stitched_path / f"{channel}_xml_import_step_{step}.xml"

            assert proj_in.exists()
            if step == 2 and alignment_cores > 1:
                os.environ["slots"] = f"{cpu_logical_core_count}"
                command = [
                    f"mpiexec{' --use-hwthread-cpus' if sys.platform.lower() == 'linux' else ''} -np {alignment_cores} "
                    f"python -m mpi4py {parastitcher}"]
            else:
                command = [f"{terastitcher}"]
            command += [
                f"-{step}",
                # Overlap (in pixels) between two adjacent tiles along H. Providing oH is harmful sometimes.
                # f"--oH={tile_overlap_x}",
                # Overlap (in pixels) between two adjacent tiles along V. Providing oV is harmful sometimes.
                # f"--oV={0 if objective == '40x' else tile_overlap_y}",
                # Displacements search radius along H (in pixels). Default value is 25!
                f"--sH={max(25, tile_overlap_x - 1)}",
                # Displacements search radius along V (in pixels). Default value is 25!
                f"--sV={max(25, tile_overlap_y - 1)}",
                # Displacements search radius along D (in pixels).
                f"--sD={0 if (objective == '40x' or stitch_mip) else min(min_subvolume_depth, subvolume_depth - 1)}",
                # Number of slices per subvolume partition
                f"--subvoldim={min_subvolume_depth}",
                # used in the pairwise displacements computation step.
                # dimension of layers obtained by dividing the volume along D
                "--threshold=0.6",  # threshold between 0.55 and 0.7 is good. Higher values block alignment.
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
    p_log(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
          f"{channel}: starting step 6 of stitching, merging tiles into 2D tif series, using TSV ..."
          f"\n\tsource: {stitched_path / f'{channel}_xml_import_step_5.xml'}"
          f"\n\tdestination: {stitched_tif_path}")

    tsv_volume = TSVVolume.load(stitched_path / f'{channel}_xml_import_step_5.xml')
    shape: Tuple[int, int, int] = tsv_volume.volume.shape  # shape is in z y x format

    memory_needed_per_thread = 16 * shape[1] * shape[2] / 1024 ** 3
    if tsv_volume.dtype in (uint8, "uint8"):
        memory_needed_per_thread /= 2
    memory_ram = virtual_memory().available / 1024 ** 3  # in GB
    merge_step_cores = min(floor(memory_ram / memory_needed_per_thread), cpu_physical_core_count)

    # need_lightsheet_cleaning
    return_code = parallel_image_processor(
        process_img,
        source=tsv_volume,
        destination=stitched_tif_path,
        args=(),
        kwargs={
            "correct_bleaching": False,
            "sigma": (256, 256) if need_destriping else (0, 0),
            "lightsheet": need_lightsheet_cleaning,
            "rotate": 90 if need_rotation_stitched_tif else 0,
            "convert_to_8bit": need_16bit_to_8bit_conversion,
            "bit_shift_to_right": right_bit_shift,
            "tile_size": shape[1:3],
            "d_type": tsv_volume.dtype
        },
        timeout=None,
        max_processors=merge_step_cores,
        progress_bar_name="TSV",
        compression=("ADOBE_DEFLATE", 1) if need_compression_stitched_tif else None,
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
        tera_fly_path = stitched_path / f'{channel}_TeraFly'
        tera_fly_path.mkdir(exist_ok=True)
        p_log(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
              f"{channel}: starting to convert to TeraFly format ...")
        command = " ".join([
            f"mpiexec -np {min(11, merge_step_cores)}{' --oversubscribe' if sys.platform == 'linux' else ''} "
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
            f"-d={tera_fly_path}",
        ])
        p_log(f"\t{PrintColors.BLUE}TeraFly conversion command:{PrintColors.ENDC}\n\t\t{command}\n")
        MultiProcessCommandRunner(queue, command).start()
        running_processes += 1

    return stitched_tif_path, shape, running_processes


def merge_channels_by_file_name(
        file_name: str = "",
        stitched_tif_paths: List[Path] = None,
        order_of_colors: str = "gbr",  # the order of r, g and b letters can be arbitrary here
        merged_tif_path: Path = None,
        shape: Tuple[int, int] = None,
        resume: bool = True,
        compression: Tuple[str, int] = ("ADOBE_DEFLATE", 1)
):
    rgb_file = merged_tif_path / file_name
    if resume and rgb_file.exists():
        return

    images: Dict[{str, ndarray}, {str, None}] = {}
    dtypes = []
    for idx, path in enumerate(stitched_tif_paths):
        file_path = path / file_name
        if file_path.exists():
            image = imread_tif_raw_png(file_path)
            images.update({order_of_colors[idx]: image})
            dtypes += [image.dtype]
        else:
            images.update({order_of_colors[idx]: None})
    del image, file_path
    if dtypes.count(dtypes[0]) != len(dtypes):
        paths = "\n\t".join(map(str, stitched_tif_paths))
        p_log(f"\n{PrintColors.WARNING}warning: merging channels should have identical dtypes:\n\t"
              f"{paths}{PrintColors.ENDC}")
        del paths

    if len(stitched_tif_paths) == 2:  # then the last color channel should remain all zeros
        images.update({order_of_colors[2]: None})

    # height (y), width(x), colors
    multi_channel_img: ndarray = zeros((shape[0], shape[1], 3), dtype=dtypes[0])
    for idx, color in enumerate("rgb"):  # the order of letters should be "rgb" here
        image: ndarray = images[color]
        if image is not None:
            image_shape = image.shape
            if image_shape != shape:
                if image_shape[0] <= shape[0] or image_shape[1] <= shape[1]:
                    padded_image = zeros(shape, dtype=image.dtype)
                    padded_image[:image_shape[0], :image_shape[1]] = image
                    image = padded_image
                else:
                    image = image[:shape[0], :shape[1]]
            multi_channel_img[:, :, idx] = image

    imsave_tif(rgb_file, multi_channel_img, compression=compression)


def merge_all_channels(
        stitched_tif_paths: List[Path],
        merged_tif_path: Path,
        channel_volume_shapes: list = None,
        order_of_colors: str = "gbr",
        workers: int = cpu_count(),
        resume: bool = True,
        compression: Tuple[str, int] = ("ADOBE_DEFLATE", 1)
):
    """
    file names should be identical for each z-step of each channel
    """
    num_files_in_each_path = list(map(lambda p: len(list(p.glob("*.tif"))), stitched_tif_paths))

    if channel_volume_shapes is None:
        channel_volume_shapes = [TifStack(path).shape for path in stitched_tif_paths]

    x, y = 0, 0
    for path, n, shape in zip(stitched_tif_paths, num_files_in_each_path, channel_volume_shapes):
        if n != shape[0]:
            p_log(f"{PrintColors.WARNING}warning: path {path} has {shape[0] - n} missing tiles!{PrintColors.ENDC}")

        y, x = max(y, shape[1]), max(x, shape[2])

    merged_tif_path.mkdir(exist_ok=True)

    num_images = max(num_files_in_each_path)
    args_queue = Queue(maxsize=num_images)
    for file in stitched_tif_paths[num_files_in_each_path.index(num_images)].glob("*.tif"):
        args_queue.put({
            "file_name": file.name,
            "stitched_tif_paths": stitched_tif_paths,
            "order_of_colors": order_of_colors,
            "merged_tif_path": merged_tif_path,
            "shape": (y, x),
            "resume": resume,
            "compression": compression
        })

    workers = min(workers, num_images)
    progress_queue = Queue()
    for worker_ in range(workers):
        MultiProcessQueueRunner(progress_queue, args_queue,
                                fun=merge_channels_by_file_name, replace_timeout_with_dummy=False).start()

    return_code = progress_manager(progress_queue, workers, num_images, desc="RGB")
    args_queue.cancel_join_thread()
    args_queue.close()
    progress_queue.cancel_join_thread()
    progress_queue.close()
    if return_code != 0:
        p_log(f"{PrintColors.FAIL}merge_all_channels function failed{PrintColors.ENDC}")
        raise RuntimeError


def get_imaris_command(imaris_path: Path, input_path: Path, output_path: Path = None,
                       voxel_size_x: float = 1, voxel_size_y: float = 1, voxel_size_z: float = 1,
                       workers: int = cpu_count(logical=False),
                       dtype: str = 'uint8'):
    files = sorted(input_path.rglob("*.tif"))
    file = files[0]
    command = []
    if imaris_path.exists() and len(files) > 0:
        p_log(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
              f"converting {input_path.name} to ims ... ")

        ims_file_path = input_path.parent / f'{input_path.name}.ims'
        if output_path:
            ims_file_path = output_path

        command = [
            f"{imaris_path}" if sys.platform == "win32" else f"WINEDEBUG=-all wine {imaris_path}",
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
            f"--nthreads {workers if dtype=='uint8' or sys.platform == 'win32' else 1}",
            f"--compression 1",
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
            if return_code is not None:
                if return_code > 0:
                    p_log(f"\nFollowing command failed:\n\t{command}\n\treturn code: {return_code}\n")
                else:
                    p_log(f"\nFollowing command succeeded:\n\t{command}\n")
                running_processes -= 1
            if position is not None and 0 < len(progress_bars) <= position + 1:
                progress_bars[position].update(percent_addition)
        except Empty:
            sleep(1)  # waite one second before checking the queue again


def main(source_path):
    if "-" in source_path.name and source_path.exists():
        source_path = source_path.rename(source_path.parent / source_path.name.replace("-", "_"))
        print(f"{PrintColors.WARNING}input path renamed to replace '-' with '_'{PrintColors.ENDC}")
    else:
        source_path = source_path.parent / source_path.name.replace("-", "_")
    if not (source_path.exists() and source_path.is_dir()):
        print(f"{PrintColors.FAIL}input path should be an existing folder!{PrintColors.ENDC}")
        raise RuntimeError

    log_file = source_path / "log.txt"
    log.basicConfig(filename=str(log_file), level=log.INFO)
    log.FileHandler(str(log_file), mode="w")  # rewrite the file instead of appending
    # Ask questions ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    global AllChannels
    all_channels = [
        channel + "_MIP" for channel, color in AllChannels if source_path.joinpath(channel + "_MIP").exists()]
    stitch_mip = ask_true_false_question("Do you need to stitch the MIP image first?") if all_channels else False
    if stitch_mip:
        channel_color_dict = {channel + "_MIP": color for channel, color in AllChannels}
    else:
        all_channels = [channel for channel, color in AllChannels if source_path.joinpath(channel).exists()]
        channel_color_dict = {channel: color for channel, color in AllChannels}

    objective, voxel_size_x, voxel_size_y, voxel_size_z, tile_size, tile_overlap_percent = get_voxel_sizes(stitch_mip)

    de_striped_posix, what_for = "", ""
    image_classes_training_data_path = source_path / FlatNonFlatTrainingData
    need_flat_image_application = ask_true_false_question("Do you need to apply a flat image?")
    if need_flat_image_application:
        de_striped_posix += "_flat_applied"
        flat_img_not_exist = []
        for channel in all_channels:
            flat_img_created_already = source_path.joinpath(channel + '_flat.tif')
            flat_img_not_exist.append(not flat_img_created_already.exists())
        if any(flat_img_not_exist):
            if not image_classes_training_data_path.exists():
                p_log(
                    f'Looked for flat vs not-flat training data in {image_classes_training_data_path} '
                    f'and it was missing!')
                use_default_flat_classification_data = ask_true_false_question(
                    "Do you want to use classification data that comes with this package? \n"
                    "(It might not be compatible with your microscopes.)"
                )
                if use_default_flat_classification_data:
                    image_classes_training_data_path = Path(__file__).parent / "image_classes.csv"
                    p_log(f"default classification data path is:\n"
                          f"{image_classes_training_data_path.absolute()}")
                else:
                    p_log("All images will be used for flat image generation!")
                    image_classes_training_data_path = None

    need_gaussian_filter_2d = ask_true_false_question(
        f"Do you need to apply a 5x5 Gaussian filter to remove camera artifacts and "
        f"produce up to two times smaller files?"
    )

    def destination_name(name: str):
        new_name = name
        new_resolution: int = 0
        re_match = compile(r"(\d+)[xX]_(\d+)").findall(name)
        re_match2 = compile(r"(\d+)[xX]_([zZ])(\d+)").findall(name)
        if re_match:
            resolution, z_step = re_match[0]
            if resolution:
                new_resolution = int(round(int(resolution) / (voxel_size_z / voxel_size_x), 0))
            if new_resolution:
                new_name = name.replace(
                    f"_{resolution}x_{z_step}", f"_{new_resolution}X_{z_step}").replace(
                    f"_{resolution}X_{z_step}", f"_{new_resolution}X_{z_step}")

        elif re_match2:
            resolution, z, z_step = re_match2[0]
            if resolution:
                new_resolution = int(round(int(resolution) / (voxel_size_z / voxel_size_x), 0))
            if new_resolution:
                new_name = name.replace(
                    f"_{resolution}x_{z}{z_step}", f"_{new_resolution}X_z{z_step}").replace(
                    f"_{resolution}X_{z}{z_step}", f"_{new_resolution}X_z{z_step}")
        return new_name

    new_destination_name = destination_name(source_path.name)
    down_sampling_factor = None
    need_down_sampling = False
    need_up_sizing = False
    new_tile_size = None
    if voxel_size_z < voxel_size_x or voxel_size_z < voxel_size_y:
        need_up_sizing = ask_true_false_question(
            "Do you need to upsize images for isotropic voxel generation before stitching?")
        if need_up_sizing:
            de_striped_posix += "_upsized"
            what_for += "upsizing "
            new_tile_size = (
                int(round(tile_size[0] * voxel_size_y / voxel_size_z, 0)),
                int(round(tile_size[1] * voxel_size_x / voxel_size_z, 0))
            )
            voxel_size_x = voxel_size_y = voxel_size_z
    elif voxel_size_z > voxel_size_x or voxel_size_z > voxel_size_y:
        need_down_sampling = ask_true_false_question(
            "Do you need to down-sample and resize images for isotropic voxel generation before stitching?")
        if need_down_sampling:
            de_striped_posix += "_downsampled"
            what_for += "down-sampling "
            new_tile_size = (
                int(round(tile_size[0] * voxel_size_y / voxel_size_z, 0)),
                int(round(tile_size[1] * voxel_size_x / voxel_size_z, 0))
            )
            voxel_size_x = voxel_size_y = voxel_size_z
            down_sampling_factor = (int(voxel_size_z // voxel_size_y), int(voxel_size_z // voxel_size_x))
            if down_sampling_factor == (1, 1):
                down_sampling_factor = None
    else:
        print(f'{PrintColors.WARNING}make sure voxel sizes are really equal!{PrintColors.ENDC}')

    need_destriping = ask_true_false_question("Do you need to remove stripes from images?")
    if need_destriping:
        de_striped_posix += "_destriped"
        what_for += "destriped "

    dark_threshold: Dict[str, int] = {channel: 0 for channel in all_channels}
    need_baseline_subtraction = ask_true_false_question("Do you need to remove baseline (dark) from images?")
    if need_baseline_subtraction:
        # pixel values smaller than the dark value (camera noise) will be set to 0 to increase compression and clarity
        for channel in all_channels:
            dark_threshold[channel] = ask_for_a_number_in_range(
                f"Enter foreground vs background threshold (dark uint) for channel {channel}:", (0, 2 ** 16 - 1), int)

    channels_need_lightsheet_cleaning: List[str] = []
    need_lightsheet_cleaning = ask_true_false_question("Do you need to apply lightsheet cleaning algorithm?")
    if need_lightsheet_cleaning:
        de_striped_posix += "_lightsheet_cleaned"
        what_for += "lightsheet cleaning "
        channels_need_lightsheet_cleaning = all_channels.copy()
        # if len(all_channels) == 1 or \
        #         ask_true_false_question(
        #             "Lightsheet cleaning is computationally expensive. "
        #             "For sparsely labeled channels, it can help compression algorithms "
        #             "to reduce the final file sizes up to a factor of 5. \n"
        #             "Do you want to clean all channels? (if no, then you may choose a subset of channels)"):
        #     channels_need_lightsheet_cleaning = all_channels.copy()
        # else:
        #     for channel in all_channels:
        #         if ask_true_false_question(f"Do you need to apply lightsheet cleaning to {channel} channel?"):
        #             channels_need_lightsheet_cleaning += [channel]

    need_raw_png_to_tiff_conversion = ask_true_false_question("Are images in raw or png format?")

    need_16bit_to_8bit_conversion = ask_true_false_question(
        "Do you need to convert 16-bit images to 8-bit before stitching to reduce final file size?")
    right_bit_shift: Dict[str, int] = {channel: 8 for channel in all_channels}
    if need_16bit_to_8bit_conversion:
        for channel in all_channels:
            right_bit_shift[channel] = int(select_among_multiple_options(
                f"For {channel} channel, enter right bit shift [0 to 8] for 8-bit conversion: \n"
                "\tbitshift smaller than 8 will increase the pixel brightness. "
                "The smaller the value the brighter the pixels.\n"
                "\tA small bitshift is less destructive for dim (axons) pixels.\n"
                "\tWe suggest 0-4 for 3D images and 8 for max projection. \n",
                [
                    "any value larger than   255 will be set to 255 in 8 bit, values smaller than 255 will not change",
                    "any value larger than   511 will be set to 255 in 8 bit, 0-  1 will be set to 0,   2-  3 to 1,...",
                    "any value larger than  1023 will be set to 255 in 8 bit, 0-  3 will be set to 0,   4-  7 to 1,...",
                    "any value larger than  2047 will be set to 255 in 8 bit, 0-  7 will be set to 0,   8- 15 to 1,...",
                    "any value larger than  4095 will be set to 255 in 8 bit, 0- 15 will be set to 0,  16- 31 to 1,...",
                    "any value larger than  8191 will be set to 255 in 8 bit, 0- 31 will be set to 0,  32- 63 to 1,...",
                    "any value larger than 16383 will be set to 255 in 8 bit, 0- 63 will be set to 0,  64-127 to 1,...",
                    "any value larger than 32767 will be set to 255 in 8 bit, 0-127 will be set to 0, 128-255 to 1,...",
                    "any value larger than 65535 will be set to 255 in 8 bit, 0-255 will be set to 0, 256-511 to 1,...",
                ],
                return_index=True
            ))
        de_striped_posix += "_bitshift." + ".".join(
            [f"{channel_color_dict[channel]}{right_bit_shift[channel]}" for channel in all_channels])

    need_compression = ask_true_false_question("Do you need to compress temporary tif files?")
    if need_raw_png_to_tiff_conversion or need_compression:
        de_striped_posix += "_tif"
        what_for += "tif "

    need_compression_stitched_tif = ask_true_false_question("Do you need to compress stitched tif files?")
    need_rotation_stitched_tif = ask_true_false_question("Do you need to rotate stitched tif files for 90 degrees?")

    preprocessed_path = source_path.parent / (source_path.name + de_striped_posix)
    continue_process_pystripe = False
    stitched_path = source_path.parent / (source_path.name + "_stitched")
    print_input_file_names = False

    if (need_destriping and objective == '40x') or need_flat_image_application or need_raw_png_to_tiff_conversion or \
            need_16bit_to_8bit_conversion or need_down_sampling or need_compression or need_gaussian_filter_2d or \
            need_baseline_subtraction:

        print_input_file_names = False  # ask_true_false_question(
        # "Do you need to print raw or tif file names to find corrupt files during preprocessing stage?")
        preprocessed_path, continue_process_pystripe = get_destination_path(
            new_destination_name if need_down_sampling or need_up_sizing else source_path.name,
            what_for=what_for + "files",
            posix=de_striped_posix,
            default_path=preprocessed_path)
    else:
        preprocessed_path = source_path

    stitched_path, continue_process_terastitcher = get_destination_path(
        new_destination_name if need_down_sampling or need_up_sizing else source_path.name,
        what_for="stitched files",
        posix="_MIP_stitched" if stitch_mip else "_stitched",
        default_path=stitched_path)

    need_tera_fly_conversion = ask_true_false_question("Do you need to convert a channel to TeraFly format?")
    channels_need_tera_fly_conversion: list = []
    if need_tera_fly_conversion:
        if len(all_channels) == 1:
            channels_need_tera_fly_conversion = all_channels.copy()
        elif len(channels_need_lightsheet_cleaning) > 0 and ask_true_false_question(
                "List of channels need TeraFly conversion is identical to the "
                "list of channels need lightsheet cleaning?"):
            channels_need_tera_fly_conversion = channels_need_lightsheet_cleaning.copy()
        else:
            for channel in all_channels:
                if ask_true_false_question(f"Do you need to convert {channel} channel to TeraFly format?"):
                    channels_need_tera_fly_conversion += [channel]

        p_log(f"\n\n{' and '.join(channels_need_tera_fly_conversion)} "
              f"channel{'s' if len(channels_need_tera_fly_conversion) > 1 else ''}"
              f" will be converted to TeraFly format.\n")
    need_merged_channels = False
    need_compression_merged_channels = False
    if len(all_channels) > 1:
        need_merged_channels = ask_true_false_question("Do you need to merge channels to RGB color tiff?")
        if need_merged_channels:
            need_compression_merged_channels = ask_true_false_question("Do you need to compress RGB color tif files?")
    need_imaris_conversion = ask_true_false_question("Do you need to convert to Imaris format?")
    # Start ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    start_time = time()
    memory_ram = virtual_memory().available // 1024 ** 3  # in GB
    p_log(
        f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
        f"stitching started"
        f"\n\tRun on computer: {platform.node()}"
        f"\n\tFree physical memory: {memory_ram} GB"
        f"\n\tPhysical CPU core count: {cpu_physical_core_count}"
        f"\n\tLogical CPU core count: {cpu_logical_core_count}"
        f"\n\tSource folder path:\n\t\t{source_path}"
        f"\n\tPreprocessed folder path:\n\t\t{preprocessed_path}"
        f"\n\tStitched folder path:\n\t\t{stitched_path}"
    )

    # stitch :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # channels need reconstruction will be stitched first to start slow TeraFly conversion as soon as possible
    all_channels = reorder_list(all_channels, channels_need_tera_fly_conversion)
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
            preprocessed_path,
            stitched_path,
            voxel_size_x,
            voxel_size_y,
            voxel_size_z,
            objective,
            tile_overlap_percent,
            queue,
            stitch_mip,
            dark=dark_threshold[channel],
            files_list=file_list,
            need_tera_fly_conversion=channel in channels_need_tera_fly_conversion,
            need_flat_image_application=need_flat_image_application,
            image_classes_training_data_path=image_classes_training_data_path,
            need_raw_png_to_tiff_conversion=need_raw_png_to_tiff_conversion,
            need_lightsheet_cleaning=channel in channels_need_lightsheet_cleaning,
            need_destriping=need_destriping,
            need_gaussian_filter_2d=need_gaussian_filter_2d,
            need_compression=need_compression,
            need_compression_stitched_tif=need_compression_stitched_tif,
            need_rotation_stitched_tif=need_rotation_stitched_tif,
            need_16bit_to_8bit_conversion=need_16bit_to_8bit_conversion,
            right_bit_shift=right_bit_shift[channel],
            continue_process_pystripe=continue_process_pystripe,
            continue_process_terastitcher=continue_process_terastitcher,
            down_sampling_factor=down_sampling_factor,
            tile_size=tile_size,
            new_tile_size=new_tile_size,
            print_input_file_names=print_input_file_names,
            subvolume_depth=subvolume_depth
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

    # TODO: calculate based on memory requirement
    memory_ram = virtual_memory().available // 1024 ** 3  # in GB
    merge_channels_cores = cpu_physical_core_count if cpu_physical_core_count * 14 < memory_ram else memory_ram // 14
    if need_16bit_to_8bit_conversion:
        merge_channels_cores = cpu_physical_core_count if cpu_physical_core_count * 7 < memory_ram else memory_ram // 7
    merge_channels_cores = min(merge_channels_cores, cpu_physical_core_count + 2)

    merged_tif_paths = stitched_tif_paths
    if need_merged_channels and len(stitched_tif_paths) > 1:
        p_log(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
              f"merging channels to RGB started ...\n\t"
              f"time elapsed so far {timedelta(seconds=time() - start_time)}")
        merged_tif_paths = [stitched_path / ("merged_channels_MIP_tif" if stitch_mip else "merged_channels_tif")]
        order_of_colors: str = ""
        for channel in all_channels:
            order_of_colors += channel_color_dict[channel]
        for channel, color in AllChannels:
            if color not in order_of_colors:
                order_of_colors += color

        if 1 < len(stitched_tif_paths) < 4:
            merge_all_channels(
                stitched_tif_paths, merged_tif_paths[0],
                channel_volume_shapes=channel_volume_shapes,
                order_of_colors=order_of_colors,
                workers=merge_channels_cores,
                resume=continue_process_terastitcher,
                compression=("ADOBE_DEFLATE", 1) if need_compression_merged_channels else None
            )
        elif len(stitched_tif_paths) >= 4:
            p_log("Warning: since number of channels are more than 3 merging channels is impossible.\n\t"
                  "merging the first 3 channels instead.")
            merge_all_channels(
                stitched_tif_paths[0:3], merged_tif_paths[0],
                channel_volume_shapes=channel_volume_shapes,
                order_of_colors=order_of_colors,
                workers=merge_channels_cores,
                resume=continue_process_terastitcher,
                compression=("ADOBE_DEFLATE", 1) if need_compression_merged_channels else None
            )
            merged_tif_paths += stitched_tif_paths[3:]
        else:
            merged_tif_paths = []

    # Imaris File Conversion :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    progress_bars = []
    if need_imaris_conversion:
        p_log(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
              f"started ims conversion ...")
        for idx, merged_tif_path in enumerate(merged_tif_paths):
            command = get_imaris_command(
                imaris_path=imaris_converter,
                input_path=merged_tif_path,
                voxel_size_x=voxel_size_y if need_rotation_stitched_tif else voxel_size_x,
                voxel_size_y=voxel_size_x if need_rotation_stitched_tif else voxel_size_y,
                voxel_size_z=voxel_size_z,
                workers=cpu_physical_core_count,
                dtype='uint8' if need_16bit_to_8bit_conversion else 'uint16'
            )
            p_log(f"\t{PrintColors.BLUE}tiff to ims conversion command:{PrintColors.ENDC}\n\t\t{command}\n")
            MultiProcessCommandRunner(queue, command, pattern=r"(WriteProgress:)\s+(\d*.\d+)\s*$", position=idx).start()
            running_processes += 1
            progress_bars += [
                tqdm(total=100, ascii=True, position=idx, unit=" %", smoothing=0.01,
                     desc=f"Imaris {(idx + 1) if len(merged_tif_paths) > 1 else ''}")]

    # waite for TeraFly and Imaris conversion to finish ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    commands_progress_manger(queue, progress_bars, running_processes)
    # while running_processes > 0:
    #     try:
    #         [percent_addition, position, return_code, command] = queue.get()
    #         if return_code is not None:
    #             if return_code > 0:
    #                 p_log(f"\nFollowing command failed:\n\t{command}\n\treturn code: {return_code}\n")
    #             else:
    #                 p_log(f"\nFollowing command succeeded:\n\t{command}\n")
    #             running_processes -= 1
    #         if position is not None and 0 < len(progress_bar) <= position + 1:
    #             progress_bar[position].update(percent_addition)
    #     except Empty:
    #         sleep(1)  # waite one second before checking the queue again

    if need_tera_fly_conversion:
        p_log(
            f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}"
            f"waiting for TeraFly conversion to finish.\n\t"
            f"time elapsed so far {timedelta(seconds=time() - start_time)}")

    # Done :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    p_log(f"{PrintColors.GREEN}{date_time_now()}: {PrintColors.ENDC}done.\n\t"
          f"Time elapsed: {timedelta(seconds=time() - start_time)}")


if __name__ == '__main__':
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    freeze_support()
    FlatNonFlatTrainingData = "image_classes.csv"
    cpu_physical_core_count = cpu_count(logical=False)
    cpu_logical_core_count = cpu_count(logical=True)
    # cpu_instruction = select_among_multiple_options(
    #     "Select the best CPU instruction supported by your CPU:",
    #     ["SSE2", "AVX", "AVX2", "AVX512f"]
    # )
    cpu_instruction = "SSE2"
    for item in ["SSE2", "AVX", "AVX2", "AVX512f"]:
        cpu_instruction = item if CPUFeature[item] else cpu_instruction
    PyScriptPath = Path(r".") / "TeraStitcher" / "pyscripts"
    if sys.platform.lower() == "win32":
        print("Windows is detected.")
        psutil.Process().nice(getattr(psutil, "IDLE_PRIORITY_CLASS"))
        CacheDriveExample = "D:\\"  # "W:\\3D_stitched\\"
        TeraStitcherPath = Path(r"TeraStitcher") / "Windows" / cpu_instruction
        os.environ["PATH"] = f"{os.environ['PATH']};{TeraStitcherPath.as_posix()}"
        os.environ["PATH"] = f"{os.environ['PATH']};{PyScriptPath.as_posix()}"
        terastitcher = "terastitcher.exe"
        mergedisplacements = "mergedisplacements.exe"
        teraconverter = "teraconverter.exe"
        nvidia_smi = "nvidia-smi.exe"
    elif sys.platform.lower() == 'linux':
        if 'microsoft' in uname().release.lower():
            print("Windows subsystem for Linux is detected.")
            CacheDriveExample = "/mnt/d/"
            nvidia_smi = "nvidia-smi.exe"
        else:
            print("Linux is detected.")
            CacheDriveExample = "/mnt/scratch"
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
            except IndexError:
                pass
            if Path(f"/usr/local/cuda-{cuda_version}/").exists() and \
                    Path(f"/usr/local/cuda-{cuda_version}/bin").exists():
                os.environ["CUDA_ROOT_DIR"] = f"/usr/local/cuda-{cuda_version}/"
            elif Path(f"/usr/local/cuda/").exists() and \
                    Path(f"/usr/local/cuda/bin").exists():
                os.environ["CUDA_ROOT_DIR"] = f"/usr/local/cuda/"
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
    print(f"mpi4py version is {mpi4py.__version__}")

    imaris_converter = Path(r"imaris") / "ImarisConvertiv.exe"
    if not imaris_converter.exists():
        log.error("Error: ImarisConvertiv.exe not found")
        raise RuntimeError

    if len(sys.argv) == 1:
        main(source_path=Path(__file__).parent.absolute())
    elif len(sys.argv) == 2:
        main(source_path=Path(sys.argv[1]).absolute())
    else:
        print(f"{PrintColors.FAIL}Only one argument is allowed!{PrintColors.ENDC}")
        raise RuntimeError
