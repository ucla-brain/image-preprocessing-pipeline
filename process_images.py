# ::::::::::::::::::::::: For Stitching Light Sheet data:::::::::::::
# Version 4 by Keivan Moradi on July 2021
# Please read the readme file for more information:
# https://github.com/ucla-brain/image-preprocessing-pipeline/blob/main/README.md
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
import os
import sys
import mpi4py
import platform
import subprocess
import logging as log
import psutil
from re import compile, match, IGNORECASE
from psutil import cpu_count, virtual_memory
from tqdm import tqdm
from numpy import ndarray, zeros
from pathlib import Path
from flat import create_flat_img
from datetime import datetime, timedelta
from time import time, sleep
from platform import uname
from tsv.volume import TSVVolume
from tsv.convert import convert_to_2D_tif
from pystripe.core import batch_filter, imread_tif_raw, imsave_tif, calculate_cores_and_chunk_size
from queue import Empty
from multiprocessing import freeze_support, Pool, Queue, Process
from supplements.cli_interface import select_among_multiple_options, ask_true_false_question, PrintColors
from typing import List, Tuple, Dict

# experiment setup: user needs to set them right
# AllChannels = [(channel folder name, rgb color)]
AllChannels: List[Tuple[str, str]] = [("Ex_488_Em_525", "b"), ("Ex_561_Em_600", "g"), ("Ex_642_Em_680", "r")]
VoxelSizeX_4x, VoxelSizeY_4x = 1.835, 1.835
VoxelSizeX_10x, VoxelSizeY_10x = 0.661, 0.661  # new stage --> 0.6, 0.6
VoxelSizeX_15x, VoxelSizeY_15x = 0.422, 0.422  # new stage --> 0.4, 0.4
# pixel values smaller than the dark value are camera noise and will be set to 0 to increase compression and clarity
DarkThreshold = 110


def get_voxel_sizes():
    objective = select_among_multiple_options(
        "What is the Objective?",
        [
            f"4x: Voxel Size X = {VoxelSizeX_4x}, Y = {VoxelSizeY_4x}, tile_size = 1600 x 2000",
            f"10x: Voxel Size X = {VoxelSizeX_10x}, Y = {VoxelSizeY_10x}, tile_size = 1850 x 1850",
            f"15x: Voxel Size X = {VoxelSizeX_15x}, Y = {VoxelSizeY_15x}, tile_size = 1850 x 1850",
            f"15x 1/2 sample: Voxel Size X = {VoxelSizeX_15x * 2}, Y = {VoxelSizeY_15x * 2}, tile_size = 925 x 925",
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
        voxel_size_x = VoxelSizeX_15x * 2
        voxel_size_y = VoxelSizeY_15x * 2
        tile_size = (925, 925)
    elif objective == "4":
        objective = ""
        tile_size_x = int(input("what is the tile size on x axis in pixels?\n").strip())
        tile_size_y = int(input("what is the tile size on y axis in pixels?\n").strip())
        voxel_size_x = float(input("what is the x voxel size in µm?\n").strip())
        voxel_size_y = float(input("what is the y voxel size in µm?\n").strip())
        tile_size = (tile_size_y, tile_size_x)
    else:
        print("Error: unsupported objective")
        log.error("Error: unsupported objective")
        raise RuntimeError
    voxel_size_z = float(input("what is the z-step size in µm?\n").strip())
    print(
        f"Objective is {objective} so voxel sizes are x = {voxel_size_x}, y = {voxel_size_y}, and z = {voxel_size_z}")
    log.info(
        f"Objective is {objective} so voxel sizes are x = {voxel_size_x}, y = {voxel_size_y}, and z = {voxel_size_z}")
    return objective, voxel_size_x, voxel_size_y, voxel_size_z, tile_size


def get_destination_path(folder_name_prefix, what_for='tif', posix='', default_path=Path('')):
    input_path = input(
        f"\nEnter destination path for {what_for}.\n"
        f"for example: {CacheDriveExample}\n"
        f"If nothing entered, {default_path.absolute()} will be used.\n").strip()
    drive_path = Path(input_path)
    while not drive_path.exists():
        input_path = input(
            f"\nEnter a valid destination path for {what_for}. "
            f"for example: {CacheDriveExample}\n"
            f"If nothing entered, {default_path.absolute()} will be used.\n").strip()
        drive_path = Path(input_path)
    if input_path == '':
        destination_path = default_path
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
        print("An unexpected error happened!")
        print(e)
        raise RuntimeError
    return destination_path, continue_process


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


def p_log(txt: str):
    print(txt)
    for _ in range(10):
        try:
            log.info(txt)
        except OSError or PermissionError:
            sleep(0.1)
            continue
        break


def worker(command: str):
    return_code = subprocess.call(command, shell=True)
    print(f"\nfinished:\n\t{command}\n\treturn code: {return_code}\n")
    return return_code


class MultiProcess(Process):
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
                process = subprocess.Popen(
                    self.command,
                    stdout=subprocess.PIPE,
                    # stderr=subprocess.PIPE,
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
            print(f'Process failed for {self.command}.')
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


def run_command(command):
    return_code = None
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        # stderr=subprocess.PIPE,
        shell=True,
        text=True)
    pattern = compile(r"error|warning|fail", IGNORECASE)
    while return_code is None:
        return_code = process.poll()
        stdout = process.stdout.readline()
        m = match(pattern, stdout)
        if m:
            print(f"\n{PrintColors.WARNING}{stdout}{PrintColors.ENDC}\n")
        else:
            print(".", end="")
    if return_code > 0:
        print(f"{PrintColors.FAIL}TeraStitcher failed with return code {return_code}.{PrintColors.ENDC}")
    else:
        print("")


def process_channel(
        source_path: Path,
        channel: str,
        preprocessed_path: Path,
        stitched_path: Path,
        voxel_size_x: float,
        voxel_size_y: float,
        voxel_size_z: float,
        objective: str,
        queue: Queue,
        memory_ram: int,
        need_flat_image_application=False,
        image_classes_training_data_path=None,
        need_raw_to_tiff_conversion=False,
        need_lightsheet_cleaning=True,
        artifact_length=150,
        need_destriping=False,
        need_compression=False,
        need_16bit_to_8bit_conversion=False,
        right_bit_shift=3,
        continue_process_pystripe=True,
        continue_process_terastitcher=True,
        down_sampling_factor=None,
        tile_size=None,
        new_tile_size=None,
        need_tera_fly_conversion=False,
        print_input_file_names=False
):
    # preprocess each tile as needed using PyStripe --------------------------------------------------------------------

    assert source_path.joinpath(channel).exists()
    if need_lightsheet_cleaning or need_destriping or need_flat_image_application or need_raw_to_tiff_conversion or \
            need_16bit_to_8bit_conversion or down_sampling_factor is not None or new_tile_size is not None:
        global DarkThreshold
        dark = DarkThreshold
        img_flat = None
        if need_flat_image_application:
            flat_img_created_already = source_path / f'{channel}_flat.tif'
            if flat_img_created_already.exists():
                img_flat = imread_tif_raw(flat_img_created_already)
                with open(source_path / f'{channel}_dark.txt', "r") as f:
                    dark = int(f.read())
                print(f"{datetime.now().isoformat(timespec='seconds', sep=' ')}: "
                      f"{channel}: using the existing flat image:\n"
                      f"{flat_img_created_already.absolute()}.")
            else:
                print(f"{datetime.now().isoformat(timespec='seconds', sep=' ')}: {channel}: creating a new flat image.")
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

        print(
            f"{datetime.now().isoformat(timespec='seconds', sep=' ')} - "
            f"{channel}: started preprocessing images and converting them to tif.\n"
            f"\tsource: {source_path / channel}\n"
            f"\tdestination: {preprocessed_path / channel}\n"
            f"\tcompression: (ZLIB, {1 if need_compression else 0})\n"
            f"\tdark = {dark}"
        )
        batch_filter(
            source_path / channel,
            preprocessed_path / channel,
            workers=cpu_logical_core_count * (2 if need_raw_to_tiff_conversion and not need_lightsheet_cleaning else 1),
            # chunks=128,
            # sigma=[foreground, background] Default is [0, 0], indicating no de-striping.
            sigma=((32, 32) if objective == "4x" else (256, 256)) if need_destriping else (0, 0),
            # level=0,
            wavelet="db10",
            crossover=10,
            # threshold=-1,
            compression=('ZLIB', 1 if need_compression else 0),  # ('ZSTD', 1) conda install imagecodecs
            flat=img_flat,
            dark=dark,
            # z_step=voxel_size_z,  # z-step in micron. Only used for DCIMG files.
            # rotate=False,
            lightsheet=need_lightsheet_cleaning,
            artifact_length=artifact_length,
            # percentile=0.25,
            # convert_to_16bit=False,  # defaults to False
            convert_to_8bit=need_16bit_to_8bit_conversion,
            bit_shift_to_right=right_bit_shift,
            continue_process=continue_process_pystripe,
            down_sample=down_sampling_factor,
            tile_size=tile_size,
            new_size=new_tile_size,
            print_input_file_names=print_input_file_names
        )

    # stitching: align the tiles GPU accelerated & parallel ------------------------------------------------------------
    if not stitched_path.joinpath(f"{channel}_xml_import_step_5.xml").exists() or not continue_process_terastitcher:
        print(f"{datetime.now().isoformat(timespec='seconds', sep=' ')} - "
              f"{channel}: aligning tiles using parastitcher ...")
        proj_out = stitched_path / f'{channel}_xml_import_step_1.xml'
        command = [
            f"{terastitcher}",
            "-1",
            "--ref1=H",  # x horizontal?
            "--ref2=V",  # y vertical?
            "--ref3=D",  # z depth?
            f"--vxl1={voxel_size_x}",
            f"--vxl2={voxel_size_y}",
            f"--vxl3={voxel_size_z}",
            "--sparse_data",
            f"--volin={preprocessed_path / channel}",
            f"--projout={proj_out}",
            "--noprogressbar"
        ]
        print("\timport command:\n\t\t" + " ".join(command))
        # subprocess.run(command, check=True)
        run_command(" ".join(command))
        if not proj_out.exists():
            print(f"{channel}: importing tif files failed.")
            raise RuntimeError

        # each alignment thread needs about 16GB of RAM in 16bit and 8GB in 8bit
        alignment_cores = cpu_logical_core_count if cpu_logical_core_count * 16 < memory_ram else memory_ram // 16
        if need_16bit_to_8bit_conversion:
            alignment_cores = cpu_logical_core_count if cpu_logical_core_count * 8 < memory_ram else memory_ram // 8

        for step in [2, 3, 4, 5]:
            print(f"{datetime.now().isoformat(timespec='seconds', sep=' ')} - "
                  f"{channel}: starting step {step} of stitching ...")
            proj_in = stitched_path / f"{channel}_xml_import_step_{step - 1}.xml"
            proj_out = stitched_path / f"{channel}_xml_import_step_{step}.xml"

            assert proj_in.exists()
            if step == 2:
                command = [f"mpiexec -np {alignment_cores} python -m mpi4py {parastitcher}"]
            else:
                command = [f"{terastitcher}"]
            command += [
                f"-{step}",
                "--threshold=0.7",
                f"--projin={proj_in}",
                f"--projout={proj_out}",
            ]
            command = " ".join(command)
            print("\tstitching command:\n\t\t" + command)
            # subprocess.call(command, shell=True)  # subprocess.run(command)
            run_command(command)
            assert proj_out.exists()
            proj_in.unlink(missing_ok=False)

    # stitching: merge tiles to generate stitched 2D tiff series -------------------------------------------------------

    stitched_tif_path = stitched_path / f"{channel}_tif"
    stitched_tif_path.mkdir(exist_ok=True)
    print(f"{datetime.now().isoformat(timespec='seconds', sep=' ')} - "
          f"{channel}: starting step 6 of stitching, merging tiles to tif, using TSV ..."
          f"\n\tsource: {stitched_path / f'{channel}_xml_import_step_5.xml'}"
          f"\n\tdestination: {stitched_tif_path}")
    shape: Tuple[int, int, int] = convert_to_2D_tif(
        TSVVolume.load(stitched_path / f'{channel}_xml_import_step_5.xml'),
        str(stitched_tif_path / "img_{z:06d}.tif"),
        compression=("ZLIB", 1),
        cores=cpu_logical_core_count,  # here the limit is 61 on Windows
        dtype='uint8' if need_16bit_to_8bit_conversion else 'uint16',
        resume=continue_process_terastitcher
    )  # shape is in z y x format

    # TeraFly ----------------------------------------------------------------------------------------------------------

    # TODO: Paraconverter: Support converting with more than 4 cores
    # TODO: Paraconverter: add a progress bar
    running_processes: int = 0
    if need_tera_fly_conversion:
        tera_fly_path = stitched_path / f'{channel}_TeraFly'
        tera_fly_path.mkdir(exist_ok=True)
        print(f"{datetime.now().isoformat(timespec='seconds', sep=' ')} - "
              f"{channel}: starting to convert to TeraFly format ...")
        command = " ".join([
            f"mpiexec -np {4} python -m mpi4py {paraconverter}",
            # f"{teraconverter}",
            "--sfmt=\"TIFF (series, 2D)\"",
            "--dfmt=\"TIFF (tiled, 3D)\"",
            "--resolutions=\"012345\"",
            "--clist=0",
            "--halve=max",
            # "--noprogressbar",
            # "--sparse_data",
            # "--fixed_tiling",
            # "--height=256",
            # "--width=256",
            # "--depth=256",
            f"-s={stitched_tif_path}",
            f"-d={tera_fly_path}",
        ])
        print(f"\tTeraFly conversion command:\n\t\t{command}\n")
        # subprocess.call(" ".join(command), shell=True)
        MultiProcess(queue, command).start()
        running_processes += 1

    return stitched_tif_path, shape, running_processes


def merge_channels_by_file_name(
        file_name: str = "",
        stitched_tif_paths: List[Path] = None,
        order_of_colors: str = "gbr",  # the order of r, g and b letters can be arbitrary here
        merged_tif_path: Path = None,
        shape: Tuple[int, int] = None,
        resume: bool = True
):
    rgb_file = merged_tif_path / file_name
    if resume and rgb_file.exists():
        return

    images: Dict[{str, ndarray}, {str, None}] = {}
    dtypes = []
    for idx, path in enumerate(stitched_tif_paths):
        file_path = path / file_name
        if file_path.exists():
            image = imread_tif_raw(file_path)
            images.update({order_of_colors[idx]: image})
            dtypes += [image.dtype]
        else:
            images.update({order_of_colors[idx]: None})
    del image, file_path
    if dtypes.count(dtypes[0]) != len(dtypes):
        paths = "\n\t".join(map(str, stitched_tif_paths))
        print(f"\n{PrintColors.WARNING}warning: merging channels should have identical dtypes:\n\t"
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

    imsave_tif(rgb_file, multi_channel_img, compression=("ZLIB", 1))


def merge_channels_by_file_name_worker(input_dict):
    merge_channels_by_file_name(**input_dict)


def merge_all_channels(
        stitched_tif_paths: List[Path],
        merged_tif_path: Path,
        channel_volume_shapes: list = None,
        order_of_colors: str = "gbr",
        workers: int = cpu_count(),
        resume: bool = True
):
    num_files_in_each_path = list(map(lambda p: len(list(p.rglob("*.tif"))), stitched_tif_paths))
    x, y = 0, 0
    for path, n, shape in zip(stitched_tif_paths, num_files_in_each_path, channel_volume_shapes):
        if n != shape[0]:
            print(f"{PrintColors.WARNING}warning: path {path} has {shape[0] - n} missing tiles!{PrintColors.ENDC}")

        y, x = max(y, shape[1]), max(x, shape[2])

    merged_tif_path.mkdir(exist_ok=True)
    work: List[dict] = [{
        "file_name": file.name,
        "stitched_tif_paths": stitched_tif_paths,
        "order_of_colors": order_of_colors,
        "merged_tif_path": merged_tif_path,
        "shape": (y, x),
        "resume": resume
    } for file in stitched_tif_paths[num_files_in_each_path.index(max(num_files_in_each_path))].rglob("*.tif")]
    num_images = len(work)
    workers, chunks = calculate_cores_and_chunk_size(num_images, workers, pool_can_handle_more_than_61_cores=False)
    print(f"\tusing {workers} cores and {chunks} chunks")
    # TODO: RGB: Support more than 61 cores on windows
    with Pool(processes=workers) as pool:
        list(
            tqdm(
                pool.imap_unordered(merge_channels_by_file_name_worker, work, chunksize=chunks),
                total=max(num_files_in_each_path),
                ascii=True,
                smoothing=0.05,
                unit="img",
                desc="RGB"
            ))


def get_imaris_command(path, voxel_size_x: float, voxel_size_y: float, voxel_size_z: float, workers: int = cpu_count()):
    files = list(path.rglob("*.tif"))
    file = files[0]
    command = []
    if imaris_converter.exists() and len(files) > 0:
        print(f"{datetime.now().isoformat(timespec='seconds', sep=' ')}: converting {path.name} to ims ... ")
        ims_file_path = path.parent / f'{path.name}.ims'
        command = [
            f"{imaris_converter}" if sys.platform == "win32" else f"wine {imaris_converter}",
            f"--input {file}",
            f"--output {ims_file_path}",
        ]
        if sys.platform == "linux" and 'microsoft' in uname().release.lower():
            command = [
                f'{correct_path_for_cmd(imaris_converter)}',
                f'--input {correct_path_for_wsl(file)}',
                f"--output {correct_path_for_wsl(ims_file_path)}",
            ]
        if len(files) > 1:
            command += ["--inputformat TiffSeries"]

        command += [
            f"--nthreads {workers}",
            f"--compression 1",
            f"--voxelsize {voxel_size_x}-{voxel_size_y}-{voxel_size_z}",  # x-y-z
            "--logprogress"
        ]
        print(f"\ttiff to ims conversion command:\n\t\t{' '.join(command)}\n")

    else:
        if len(files) > 0:
            print("\tnot found Imaris View: not converting tiff to ims ... ")
        else:
            print("\tno tif file found to convert to ims!")

    return " ".join(command)


def main(source_path):
    log_file = source_path / "log.txt"
    log.basicConfig(filename=str(log_file), level=log.INFO)
    log.FileHandler(str(log_file), mode="w")  # rewrite the file instead of appending
    # Ask questions ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    objective, voxel_size_x, voxel_size_y, voxel_size_z, tile_size = get_voxel_sizes()
    global AllChannels
    all_channels = [channel for channel, color in AllChannels if source_path.joinpath(channel).exists()]
    channel_color_dict = {channel: color for channel, color in AllChannels}
    de_striped_posix, what_for = "", ""
    image_classes_training_data_path = source_path / FlatNonFlatTrainingData
    need_lightsheet_cleaning = ask_true_false_question("Do you need to apply lightsheet cleaning algorithm?")
    need_destriping = False  # ask_true_false_question("Do you need to remove stripes from images?")
    channels_need_lightsheet_cleaning: List[str] = []
    if need_destriping:
        de_striped_posix += "_destriped"
        what_for += "destriped "
    elif need_lightsheet_cleaning:
        de_striped_posix += "_lightsheet_cleaned"
        what_for += "lightsheet cleaning "
        if len(all_channels) == 1 or \
                ask_true_false_question(
                    "Lightsheet cleaning is computationally expensive. "
                    "For sparsely labeled channels, it can help compression algorithms "
                    "to reduce the final file sizes up to a factor of 5. \n"
                    "Do you want to clean all channels? (if no, then you may choose a subset of channels)"):
            channels_need_lightsheet_cleaning: List[str] = all_channels.copy()
        else:
            channels_need_lightsheet_cleaning: List[str] = []
            for channel in all_channels:
                if ask_true_false_question(f"Do you need to apply lightsheet cleaning to {channel} channel?"):
                    channels_need_lightsheet_cleaning += [channel]

    need_flat_image_application = False  # ask_true_false_question("Do you need to apply a flat image?")
    if need_flat_image_application:
        de_striped_posix += "_flat_applied"
        flat_img_not_exist = []
        for channel in all_channels:
            flat_img_created_already = source_path.joinpath(channel + '_flat.tif')
            flat_img_not_exist.append(not flat_img_created_already.exists())
        if any(flat_img_not_exist):
            if not image_classes_training_data_path.exists():
                print(
                    f'Looked for flat vs not-flat training data in {image_classes_training_data_path} '
                    f'and it was missing!')
                use_default_flat_classification_data = ask_true_false_question(
                    "Do you want to use classification data that comes with this package? \n"
                    "(It might not be compatible with your microscopes.)"
                )
                if use_default_flat_classification_data:
                    image_classes_training_data_path = Path(__file__).parent / "image_classes.csv"
                    print(f"default classification data path is:\n"
                          f"{image_classes_training_data_path.absolute()}")
                else:
                    print("You need classification data for flat image generation!")
                    raise RuntimeError
    need_raw_to_tiff_conversion = ask_true_false_question("Are images in raw format?")
    if need_raw_to_tiff_conversion:
        de_striped_posix += "_tif"
        what_for += "tif "
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
    down_sampling_factor = (int(voxel_size_z // voxel_size_y), int(voxel_size_z // voxel_size_x))
    need_down_sampling = False
    new_tile_size = None
    if down_sampling_factor < (1, 1):
        # Down-sampling makes sense if x-axis and y-axis voxel sizes were smaller than z axis voxel size
        # Down-sampling is disabled.
        down_sampling_factor = None
    else:
        need_down_sampling = ask_true_false_question(
            "Do you need to down-sample images for isotropic voxel generation before stitching?")
        if not need_down_sampling:
            down_sampling_factor = None
        else:
            de_striped_posix += "_downsampled"
            what_for += "down-sampling "
            new_tile_size = (
                int(round(tile_size[0] * voxel_size_y / voxel_size_z, 0)),
                int(round(tile_size[1] * voxel_size_x / voxel_size_z, 0))
            )
            voxel_size_x = voxel_size_y = voxel_size_z
    need_compression = ask_true_false_question("Do you need to compress temporary tif files?")
    preprocessed_path = source_path.parent / (source_path.name + de_striped_posix)
    continue_process_pystripe = False
    stitched_path = source_path.parent / (source_path.name + "_stitched")
    print_input_file_names = False
    if need_destriping or need_flat_image_application or need_raw_to_tiff_conversion or need_16bit_to_8bit_conversion \
            or need_down_sampling:
        print_input_file_names = False  # ask_true_false_question(
        # "Do you need to print raw or tif file names to find corrupt files during preprocessing stage?")
        preprocessed_path, continue_process_pystripe = get_destination_path(
            source_path.name,
            what_for=what_for + "files",
            posix=de_striped_posix,
            default_path=preprocessed_path)
    else:
        preprocessed_path = source_path
    stitched_path, continue_process_terastitcher = get_destination_path(
        source_path.name, what_for="stitched files", posix="_stitched", default_path=stitched_path)

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
    if len(all_channels) > 1:
        need_merged_channels = ask_true_false_question("Do you need to merge channels to RGB color tiff?")
    need_imaris_conversion = ask_true_false_question("Do you need to convert to Imaris format?")
    # Start ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    start_time = time()
    memory_ram = virtual_memory().total // 1024 ** 3  # in GB
    p_log(
        f"{datetime.now().isoformat(timespec='seconds', sep=' ')}: stitching started"
        f"\n\tRun on computer: {platform.node()}"
        f"\n\tTotal physical memory: {memory_ram} GB"
        f"\n\tPhysical CPU core count: {cpu_physical_core_count}"
        f"\n\tLogical CPU core count: {cpu_logical_core_count}"
        f"\n\tSource folder path:\n\t\t{source_path}"
        f"\n\tPreprocessed folder path:\n\t\t{preprocessed_path}"
        f"\n\tStitched folder path:\n\t\t{stitched_path}"
    )

    # stitch :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # channels need reconstruction will be stitched first to start slow TeraFly conversion as soon as possible

    all_channels = reorder_list(all_channels, channels_need_tera_fly_conversion)
    stitched_tif_paths, channel_volume_shapes = [], []
    queue = Queue()
    running_processes: int = 0
    for channel in all_channels:
        stitched_tif_path, shape, running_processes_addition = process_channel(
            source_path,
            channel,
            preprocessed_path,
            stitched_path,
            voxel_size_x,
            voxel_size_y,
            voxel_size_z,
            objective,
            queue,
            memory_ram,
            need_tera_fly_conversion=channel in channels_need_tera_fly_conversion,
            need_flat_image_application=need_flat_image_application,
            image_classes_training_data_path=image_classes_training_data_path,
            need_raw_to_tiff_conversion=need_raw_to_tiff_conversion,
            need_lightsheet_cleaning=channel in channels_need_lightsheet_cleaning,
            artifact_length=150,
            need_destriping=need_destriping,
            need_compression=need_compression,
            need_16bit_to_8bit_conversion=need_16bit_to_8bit_conversion,
            right_bit_shift=right_bit_shift[channel],
            continue_process_pystripe=continue_process_pystripe,
            continue_process_terastitcher=continue_process_terastitcher,
            down_sampling_factor=down_sampling_factor,
            tile_size=tile_size,
            new_tile_size=new_tile_size,
            print_input_file_names=print_input_file_names
        )
        stitched_tif_paths += [stitched_tif_path]
        channel_volume_shapes += [shape]
        running_processes += running_processes_addition

    if not channel_volume_shapes.count(channel_volume_shapes[0]) == len(channel_volume_shapes):
        p_log(
            f"{PrintColors.WARNING}warning: channels had different shapes:\n\t" + "\n\t".join(
                map(lambda p: f"channel {p[0]}: volume shape={p[1]}",
                    zip(all_channels, channel_volume_shapes))) + PrintColors.ENDC
        )

    # merge channels to RGB ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    merged_tif_paths = stitched_tif_paths
    if need_merged_channels and len(stitched_tif_paths) > 1:
        p_log(f"{datetime.now().isoformat(timespec='seconds', sep=' ')}: merging channels to RGB started ...\n\t"
              f"time elapsed so far {timedelta(seconds=time() - start_time)}")
        merged_tif_paths = [stitched_path / "merged_channels_tif"]
        order_of_colors: str = ""
        for channel in all_channels:
            order_of_colors += channel_color_dict[channel]
        for channel, color in AllChannels:
            if color not in order_of_colors:
                order_of_colors += color

        if 1 < len(stitched_tif_paths) < 4:
            merge_all_channels(stitched_tif_paths, merged_tif_paths[0],
                               channel_volume_shapes=channel_volume_shapes,
                               order_of_colors=order_of_colors,
                               workers=cpu_logical_core_count,
                               resume=continue_process_terastitcher)
        elif len(stitched_tif_paths) >= 4:
            p_log("Warning: since number of channels are more than 3 merging channels is impossible.\n\t"
                  "merging the first 3 channels instead.")
            merge_all_channels(stitched_tif_paths[0:3], merged_tif_paths[0],
                               channel_volume_shapes=channel_volume_shapes,
                               order_of_colors=order_of_colors,
                               workers=cpu_logical_core_count,
                               resume=continue_process_terastitcher)
            merged_tif_paths += stitched_tif_paths[3:]
        else:
            merged_tif_paths = []

    # Imaris File Conversion :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    progress_bar = []
    if need_imaris_conversion:
        p_log(f"{datetime.now().isoformat(timespec='seconds', sep=' ')}: started ims conversion  ...")
        for idx, path in enumerate(merged_tif_paths):
            command = get_imaris_command(path, voxel_size_x, voxel_size_y, voxel_size_z, workers=cpu_logical_core_count)
            MultiProcess(queue, command, pattern=r"(WriteProgress:)\s+(\d*.\d+)\s*$", position=idx).start()
            running_processes += 1
            progress_bar += [
                tqdm(total=100, ascii=True, position=idx, unit="%", smoothing=0.05,
                     desc=f"imaris {(idx + 1) if len(merged_tif_paths) > 1 else ''}")]

    # waite for TeraFly and Imaris conversion to finish ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    if need_tera_fly_conversion:
        p_log(f"{datetime.now().isoformat(timespec='seconds', sep=' ')}: waiting for TeraFly conversion to finish.\n\t"
              f"time elapsed so far {timedelta(seconds=time() - start_time)}")
    while running_processes > 0:
        try:
            [percent_addition, position, return_code, command] = queue.get()
            if return_code is not None:
                if return_code > 0:
                    print(f"\nFollowing command failed:\n\t{command}\n\treturn code: {return_code}\n")
                else:
                    print(f"\nFollowing command succeeded:\n\t{command}\n")
                running_processes -= 1
            if position is not None and 0 < len(progress_bar) <= position + 1:
                progress_bar[position].update(percent_addition)
        except Empty:
            sleep(1)  # waite one second before checking the queue again

    # Done :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    p_log(f"{datetime.now().isoformat(timespec='seconds', sep=' ')}: done.\n\t"
          f"Time elapsed: {timedelta(seconds=time() - start_time)}")


if __name__ == '__main__':
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    freeze_support()
    FlatNonFlatTrainingData = "image_classes.csv"
    cpu_physical_core_count = cpu_count(logical=False)
    cpu_logical_core_count = cpu_count(logical=True)
    cpu_instruction = select_among_multiple_options(
        "Select the best CPU instruction supported by your CPU:",
        ["SSE2", "AVX", "AVX2", "AVX512"]
    )
    PyScriptPath = Path(r"./TeraStitcher/pyscripts")
    if sys.platform == "win32":
        print("Windows is detected.")
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
        CacheDriveExample = "D:\\"  # "W:\\3D_stitched\\"
        TeraStitcherPath = Path(r"./TeraStitcher/Windows") / cpu_instruction
        os.environ["PATH"] = f"{os.environ['PATH']};{TeraStitcherPath.as_posix()}"
        os.environ["PATH"] = f"{os.environ['PATH']};{PyScriptPath.as_posix()}"
        terastitcher = "terastitcher.exe"
        teraconverter = "teraconverter.exe"
    elif sys.platform == 'linux':
        if 'microsoft' in uname().release.lower():
            print("Windows subsystem for Linux is detected.")
            CacheDriveExample = "/mnt/d/"
        else:
            print("Linux is detected.")
            CacheDriveExample = "/mnt/scratch"
        psutil.Process().nice(value=19)
        TeraStitcherPath = Path(r"./TeraStitcher/Linux")
        os.environ["PATH"] = f"{os.environ['PATH']}:{TeraStitcherPath.as_posix()}"
        os.environ["PATH"] = f"{os.environ['PATH']}:{PyScriptPath.as_posix()}"
        terastitcher = "terastitcher"
        teraconverter = "teraconverter"
        os.environ["TERM"] = "xterm"
        os.environ["USECUDA_X_NCC"] = "1"  # set to 0 to stop GPU acceleration
        if os.environ["USECUDA_X_NCC"] == "1":
            if Path("/usr/lib/jvm/java-11-openjdk-amd64/lib/server").exists():
                os.environ["LD_LIBRARY_PATH"] = "/usr/lib/jvm/java-11-openjdk-amd64/lib/server"
            else:
                log.error("Error: JAVA path not found")
                raise RuntimeError
            cuda_version = input("What is your cuda version (for example 11.6)?")
            if Path(f"/usr/local/cuda-{cuda_version}/").exists() and \
                    Path(f"/usr/local/cuda-{cuda_version}/bin").exists():
                os.environ["CUDA_ROOT_DIR"] = f"/usr/local/cuda-{cuda_version}/"
            else:
                log.error(f"Error: CUDA path not found in {os.environ['CUDA_ROOT_DIR']}")
                raise RuntimeError
            os.environ["PATH"] = f"{os.environ['PATH']}:{os.environ['CUDA_ROOT_DIR']}/bin"
            os.environ["LD_LIBRARY_PATH"] = f"{os.environ['LD_LIBRARY_PATH']}:{os.environ['CUDA_ROOT_DIR']}/lib64"
            # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # to train on a specific GPU on a multi-gpu machine
    else:
        log.error("yet unsupported OS")
        raise RuntimeError

    terastitcher = TeraStitcherPath / terastitcher
    if not terastitcher.exists():
        log.error("Error: TeraStitcher not found")
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

    imaris_converter = Path(r"./imaris/ImarisConvertiv.exe")
    if not imaris_converter.exists():
        log.error("Error: ImarisConvertiv.exe not found")
        raise RuntimeError

    if len(sys.argv) == 1:
        main(source_path=Path(__file__).parent.absolute())
    elif len(sys.argv) == 2:
        if Path(sys.argv[1]).exists():
            main(source_path=Path(sys.argv[1]).absolute())
        else:
            print("The entered path is not valid")
