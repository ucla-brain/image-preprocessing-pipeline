# ::::::::::::::::::::::: For Stitching Light Sheet data:::::::::::::
# Version 4 by Keivan Moradi on July, 2021
# Please read the readme file for more information:
# https://github.com/ucla-brain/image-preprocessing-pipeline/blob/main/README.md
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
import re
import os
import sys
import numpy as np
import psutil
import shutil
import platform
import subprocess
import logging as log
import pystripe_forked as pystripe
from pathlib import Path
from multiprocessing import freeze_support, Pool
from flat import create_flat_img
from datetime import datetime
from time import time
from platform import uname
import mpi4py
import tsv
from tsv.convert import convert_to_2D_tif

# experiment setup: user needs to set them right
AllChannels = ["Ex_488_Em_525", "Ex_561_Em_600", "Ex_642_Em_680"]  # the order determines color ["R", "G", "B"]?
VoxelSizeX_4x, VoxelSizeY_4x = 1.835, 1.835
VoxelSizeX_10x, VoxelSizeY_10x = 0.661, 0.661
VoxelSizeX_15x, VoxelSizeY_15x = 0.422, 0.422
FlatNonFlatTrainingData = "image_classes.csv"
cpu_physical_core_count = psutil.cpu_count(logical=False)
cpu_logical_core_count = psutil.cpu_count(logical=True)

if sys.platform == "win32":
    # print("Windows is detected.")
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    CacheDriveExample = "W:\\3D_stitched\\"
    TeraStitcherPath = Path(r"./TeraStitcher_windows_avx2")
    os.environ["PATH"] = f"{os.environ['PATH']};{TeraStitcherPath.as_posix()}"
    os.environ["PATH"] = f"{os.environ['PATH']};{TeraStitcherPath.joinpath('pyscripts').as_posix()}"
    terastitcher = "terastitcher.exe"
    teraconverter = "teraconverter.exe"
elif sys.platform == 'linux' and 'microsoft' not in uname().release.lower():
    print("Linux is detected.")
    psutil.Process().nice(value=19)
    CacheDriveExample = "/mnt/scratch"
    TeraStitcherPath = Path(r"./TeraStitcher_linux")
    os.environ["PATH"] = f"{os.environ['PATH']}:{TeraStitcherPath.as_posix()}"
    os.environ["PATH"] = f"{os.environ['PATH']}:{TeraStitcherPath.joinpath('pyscripts').as_posix()}"
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
        if Path("/usr/local/cuda-11.4/").exists() and Path("/usr/local/cuda-11.4/bin").exists():
            os.environ["CUDA_ROOT_DIR"] = "/usr/local/cuda-11.4/"
        elif Path("/usr/local/cuda-10.1/").exists() and Path("/usr/local/cuda-10.1/bin").exists():
            os.environ["CUDA_ROOT_DIR"] = "/usr/local/cuda-10.1/"
        else:
            log.error("Error: CUDA path not found")
            raise RuntimeError
        os.environ["PATH"] = f"{os.environ['PATH']}:{os.environ['CUDA_ROOT_DIR']}/bin"
        os.environ["LD_LIBRARY_PATH"] = f"{os.environ['LD_LIBRARY_PATH']}:{os.environ['CUDA_ROOT_DIR']}/lib64"
        # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # to train on a specific GPU on a multi-gpu machine
elif sys.platform == 'linux' and 'microsoft' in uname().release.lower():
    print("Windows subsystem for Linux is detected.")
    psutil.Process().nice(value=19)
    CacheDriveExample = "/mnt/d/"
    TeraStitcherPath = Path(r"./TeraStitcher_linux")
    os.environ["PATH"] = f"{os.environ['PATH']}:{TeraStitcherPath.as_posix()}"
    os.environ["PATH"] = f"{os.environ['PATH']}:{TeraStitcherPath.joinpath('pyscripts').as_posix()}"
    terastitcher = "terastitcher"
    teraconverter = "teraconverter"
    os.environ["TERM"] = "xterm"
    os.environ["USECUDA_X_NCC"] = "0"  # set to 0 to stop GPU acceleration
    if os.environ["USECUDA_X_NCC"] == "1":
        if Path("/usr/lib/jvm/java-11-openjdk-amd64/lib/server").exists():
            os.environ["LD_LIBRARY_PATH"] = "/usr/lib/jvm/java-11-openjdk-amd64/lib/server"
        else:
            log.error("Error: JAVA path not found")
            raise RuntimeError
        if Path("/usr/local/cuda-11.4/").exists() and Path("/usr/local/cuda-11.4/bin").exists():
            os.environ["CUDA_ROOT_DIR"] = "/usr/local/cuda-11.4/"
        elif Path("/usr/local/cuda-10.1/").exists() and Path("/usr/local/cuda-10.1/bin").exists():
            os.environ["CUDA_ROOT_DIR"] = "/usr/local/cuda-10.1/"
        else:
            log.error("Error: CUDA path not found")
            raise RuntimeError
        os.environ["PATH"] = f"{os.environ['PATH']}:{os.environ['CUDA_ROOT_DIR']}/bin"
        os.environ["LD_LIBRARY_PATH"] = f"{os.environ['LD_LIBRARY_PATH']}:{os.environ['CUDA_ROOT_DIR']}/lib64"
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # to train on a specific GPU on a multi-gpu machine
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

parastitcher = TeraStitcherPath / "pyscripts" / "Parastitcher.py"
if not parastitcher.exists():
    log.error("Error: ParaStitcher not found")
    raise RuntimeError

parasconverter = TeraStitcherPath / "pyscripts" / "paraconverter.py"
if not parastitcher.exists():
    log.error("Error: ParaStitcher not found")
    raise RuntimeError

imaris_converter = Path(r"./imaris") / "ImarisConvertiv.exe"
if not imaris_converter.exists():
    log.error("Error: ImarisConvertiv.exe not found")
    raise RuntimeError


def ask_true_false_question(message):
    answer = ''
    while answer not in {"1", "2"}:
        answer = input(
            '\n'
            f'{message}\n'
            '1 = Yes\n'
            '2 = No\n').strip()
    if answer == "1":
        return True
    else:
        return False


def select_one_channel(question: str):
    num_channel = str()
    while num_channel not in {"1", "2", "3"}:
        num_channel = input(
            f"\n"
            f"{question}\n"
            f"1 = {AllChannels[0]}\n"
            f"2 = {AllChannels[1]}\n"
            f"3 = {AllChannels[2]}\n").strip()
    most_informative_channel = AllChannels[int(num_channel)-1]
    return most_informative_channel


def get_voxel_sizes():
    objective = ""
    while objective not in {"1", "2", "3", "4", "5"}:
        objective = input(
            f'\nWhat is the Objective?\n'
            f'1 = 4x: Voxel Size X = {VoxelSizeX_4x}, Y = {VoxelSizeY_4x}, tile_size = 1600 x 2000\n'
            f'2 = 10x: Voxel Size X = {VoxelSizeX_10x}, Y = {VoxelSizeY_10x}, tile_size = 1850 x 1850\n'
            f'3 = 15x: Voxel Size X = {VoxelSizeX_15x}, Y = {VoxelSizeY_15x}, tile_size = 1850 x 1850\n'
            f'4 = 15x 1/2 sample: Voxel Size X = {VoxelSizeX_15x*2}, Y = {VoxelSizeY_15x*2}, tile_size = 925 x 925\n'
            f'5 = other: allows entering custom voxel sizes for custom tile_size\n'
        ).strip()

    if objective == "1":
        objective = "4x"
        voxel_size_x = VoxelSizeX_4x
        voxel_size_y = VoxelSizeY_4x
        tile_size = (1600, 2000)  # y, x = tile_size
    elif objective == "2":
        objective = "10x"
        voxel_size_x = VoxelSizeX_10x
        voxel_size_y = VoxelSizeY_10x
        tile_size = (1850, 1850)
    elif objective == "3":
        objective = "15x"
        voxel_size_x = VoxelSizeX_15x
        voxel_size_y = VoxelSizeY_15x
        tile_size = (1850, 1850)
    elif objective == "4":
        objective = "15x"
        voxel_size_x = VoxelSizeX_15x * 2
        voxel_size_y = VoxelSizeY_15x * 2
        tile_size = (925, 925)
    elif objective == "5":
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
        f"\n"
        f"Enter destination path for {what_for}.\n"
        f"for example: {CacheDriveExample}\n"
        f"If nothing entered, {default_path.absolute()} will be used.\n").strip()
    drive_path = Path(input_path)
    while not drive_path.exists():
        input_path = input(
            f"\n"
            f"Enter a valid destination path for {what_for}. "
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
                "No means start over from the beginning and overwrite files.\n")
        else:
            i = 0
            while destination_path.exists():
                i += 1
                destination_path = destination_path.parent / (destination_path.name + '_' + str(i))
    print(f"\nDestination path for {what_for} is:\n{destination_path.absolute()}\n")
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
    p = re.compile(r"/mnt/(.)/")
    new_path = p.sub(r'\1:\\\\', str(filepath))
    new_path = new_path.replace(" ", r"\ ").replace("(", r"\(").replace(")", r"\)").replace("/", "\\\\")
    return new_path


def p_log(txt):
    print(txt)
    for _ in range(10):
        try:
            log.info(txt)
        except OSError or PermissionError:
            sleep(0.1)
            continue
        break


def worker(command: str):
    result = subprocess.call(command, shell=True)
    print(f"\nfinished:\n{command}\nresult:\n{result}\n")
    return result


def merge_channels(r: np.ndarray, g: np.ndarray, b: np.ndarray):
    assert r.shape[0] == g.shape[0] == b.shape[0]
    assert r.shape[1] == g.shape[1] == b.shape[1]

    """Add the channels to the needed image one by one"""
    # method 1
    # needed_multi_channel_img = np.zeros((a.shape[0], a.shape[1], 3))
    # needed_multi_channel_img[:, :, 0] = r
    # needed_multi_channel_img[:, :, 1] = g
    # needed_multi_channel_img[:, :, 2] = b

    # method 2
    needed_multi_channel_img = np.dstack((r, g, b))

    return needed_multi_channel_img


def main(source_folder):
    log_file = source_folder / "log.txt"
    log.basicConfig(filename=str(log_file), level=log.INFO)
    log.FileHandler(str(log_file), mode="w")  # rewrite the file instead of appending
    # ::::::::::::::::::::: Ask questions :::::::::::::::::::::
    objective, voxel_size_x, voxel_size_y, voxel_size_z, tile_size = get_voxel_sizes()
    most_informative_channel = select_one_channel(
        "Choose the most informative channel for stitching alignment:")
    p_log(f"most informative channel is {most_informative_channel}\n")
    channels_need_reconstruction = [select_one_channel(
        "Choose channel contains dendrites to apply lightsheet cleaning:")]
    p_log(f"channels needs reconstruction is {channels_need_reconstruction}\n")
    de_striped_posix, what_for = "", ""
    img_flat = None
    image_classes_training_data_path = source_folder / FlatNonFlatTrainingData
    need_lightsheet_cleaning = True  # ask_true_false_question("Do you need to computationally clean images?")
    need_destriping = False  # ask_true_false_question("Do you need to remove stripes from images?")
    if need_destriping:
        de_striped_posix += "_destriped"
        what_for += "destriped "
    need_flat_image_application = False  # ask_true_false_question("Do you need to apply a flat image?")
    if need_flat_image_application:
        de_striped_posix += "_flat_applied"
        flat_img_not_exist = []
        for Channel in AllChannels:
            flat_img_created_already = source_folder.joinpath(Channel + '_flat.tif')
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
    need_raw_to_tiff_conversion = False  # ask_true_false_question("Are images in raw format?")
    if need_raw_to_tiff_conversion:
        de_striped_posix += "_tif"
        what_for += "tif "
    need_16bit_to_8bit_conversion = ask_true_false_question(
        "Do you need to convert 16-bit images to 8-bit before stitching?")
    right_bit_shift: int = 8
    if need_16bit_to_8bit_conversion:
        right_bit_shift_str = ""
        while right_bit_shift_str not in ["0", "1", "2", "3", "4", "5", "6", "7", "8"]:
            right_bit_shift_str = input(
                "Enter right bit shift [0 to 8] for 8-bit conversion. \n"
                "Values smaller than 8 will increase the pixel brightness. \n"
                "We suggest a value between 3 to 6 for 3D images and 8 for max projection. \n"
                "The smaller the value the brighter the pixels.\n"
            ).strip()
        right_bit_shift = int(right_bit_shift_str)
        de_striped_posix += f"_8b_{right_bit_shift_str}bsh"

    down_sampling_factor = (int(voxel_size_z // voxel_size_y), int(voxel_size_z // voxel_size_x))
    need_down_sampling, new_tile_size = False, None
    if down_sampling_factor < (1, 1):
        down_sampling_factor = None
        print("Down-sampling makes sense if x and y axis voxel sizes were smaller than z axis voxel size.\n"
              "Down-sampling is disabled.")
    else:
        need_down_sampling = ask_true_false_question(
            "Do you need to down-sample images for isotropic voxel generation before stitching?")
        if not need_down_sampling:
            down_sampling_factor = None
        else:
            de_striped_posix += "_ds"
            what_for += "down-sampling "
            new_tile_size = (
                int(round(tile_size[0] * voxel_size_y / voxel_size_z, 0)),
                int(round(tile_size[1] * voxel_size_x / voxel_size_z, 0))
            )
            voxel_size_x = voxel_size_y = voxel_size_z
    need_compression = False  # ask_true_false_question("Do you need to compress temporary tif files?")
    need_tera_fly_conversion = ask_true_false_question("Do you need to convert to teraFly format?")
    de_striped_dir = source_folder.parent / (source_folder.name + de_striped_posix)
    continue_process_pystripe = False
    dir_stitched = source_folder.parent / (source_folder.name + "_stitched_v4")
    if need_destriping or need_flat_image_application or need_raw_to_tiff_conversion or need_16bit_to_8bit_conversion \
            or need_down_sampling:
        de_striped_dir, continue_process_pystripe = get_destination_path(
            source_folder.name,
            what_for=what_for + "files",
            posix=de_striped_posix,
            default_path=de_striped_dir)
    else:
        de_striped_dir = source_folder
    dir_stitched, continue_process_terastitcher = get_destination_path(
        source_folder.name, what_for="stitched files", posix="_stitched_v4", default_path=dir_stitched)
    # ::::::::::::::::::::::::::::::::::::: Start :::::::::::::::::::::::::::::::::::

    log.info(f"{datetime.now()}: stitching started")
    log.info(f"Run on computer: {platform.node()}")
    log.info(f"Total physical memory: {psutil.virtual_memory().total // 1024 ** 3} GB")
    log.info(f"Physical CPU core count: {cpu_physical_core_count}")
    log.info(f"Logical CPU core count: {cpu_logical_core_count}")
    p_log(f"Source folder path:\n{source_folder}")
    p_log(f"Destriped or tif files path:\n{de_striped_dir}")
    p_log(f"Stitched folder path:\n{dir_stitched}")
    # ::::::::::::::::: RUN PyStripe  ::::::::::::::::
    start_time = time()
    if need_destriping or need_flat_image_application or need_raw_to_tiff_conversion or need_16bit_to_8bit_conversion \
            or need_down_sampling:
        for Channel in AllChannels:
            source_channel_folder = source_folder / Channel
            if source_channel_folder.exists():
                dark = (110 if Channel in channels_need_reconstruction else 511) if need_flat_image_application else 0
                if need_flat_image_application:
                    flat_img_created_already = source_folder.joinpath(Channel + '_flat.tif')
                    if flat_img_created_already.exists():
                        img_flat = pystripe.imread(str(flat_img_created_already))
                        with open(source_folder.joinpath(Channel + '_dark.txt'), "r") as f:
                            dark = int(f.read())
                        p_log(f"{datetime.now()}: {Channel}: using the existing flat image:\n"
                              f"{flat_img_created_already.absolute()}.")
                    else:
                        p_log(f"{datetime.now()}: {Channel}: creating a new flat image.")
                        img_flat, dark = create_flat_img(
                            source_channel_folder,
                            image_classes_training_data_path,
                            tile_size,
                            max_images=1024,  # the number of flat images averaged
                            batch_size=cpu_logical_core_count,
                            patience_before_skipping=cpu_logical_core_count-1,
                            # the number of non-flat images found successively before skipping
                            skips=256,  # the number of images should be skipped before testing again
                            sigma_spatial=1,  # the de-noising parameter
                            save_as_tiff=True
                        )
                p_log(f"\nBackground dark level is {dark} for {Channel} channel.")
                p_log(f"\n{datetime.now()}: DeStripe program started processing channel {Channel}.")
                pystripe.batch_filter(
                    source_channel_folder,
                    de_striped_dir / Channel,
                    workers=cpu_logical_core_count if cpu_logical_core_count < 61 else 61,
                    chunks=128,
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
                    lightsheet=True if Channel in channels_need_reconstruction and need_lightsheet_cleaning else False,
                    artifact_length=int(150 / (voxel_size_z // voxel_size_x + voxel_size_z // voxel_size_y) * 2),
                    # percentile=0.25,
                    # convert_to_16bit=False,  # defaults to False
                    convert_to_8bit=need_16bit_to_8bit_conversion,
                    bit_shift_to_right=right_bit_shift,
                    continue_process=continue_process_pystripe,
                    down_sample=down_sampling_factor,
                    new_size=new_tile_size
                )
                p_log(f"{datetime.now()}: {Channel}: DeStripe program is done.")

    # ::::::::::::::::: Stitching ::::::::::::::::
    # generates one multichannel tiff file: GPU accelerated & parallel

    log.info(f"{datetime.now()}: stitching to 3D TIFF started ...")
    print("running terastitcher ... ")
    print("Light Sheet Data Stitching (Version 4) is Running ...")
    existing_channels = []
    for Channel in AllChannels:
        channel_dir = de_striped_dir / Channel
        if channel_dir.exists():
            existing_channels += [Channel]
            log.info(f"{datetime.now()}: {channel_dir} folder exists importing for stitching...")
            proj_out = dir_stitched / f'{Channel}_xml_import_step_1.xml'
            command = [
                f"{terastitcher}",
                "-1",
                "--ref1=H",
                "--ref2=V",
                "--ref3=D",
                f"--vxl1={voxel_size_y}",
                f"--vxl2={voxel_size_x}",
                f"--vxl3={voxel_size_z}",
                "--sparse_data",
                f"--volin={channel_dir}",
                f"--projout={proj_out}",
                "--noprogressbar"
            ]
            p_log("import command:\n" + " ".join(command))
            subprocess.run(command, check=True)
            assert proj_out.exists()
            if Channel == most_informative_channel and \
                    not dir_stitched.joinpath(Channel + '_xml_import_step_' + str(5) + '.xml').exists():
                for step in [2, 3, 4, 5]:
                    p_log(f"{datetime.now()}: starting step {step} of stitching for most informative channel ...")
                    proj_in = dir_stitched / f"{Channel}_xml_import_step_{step - 1}.xml"
                    proj_out = dir_stitched / f"{Channel}_xml_import_step_{step}.xml"
                    assert proj_in.exists()
                    if step == 2:
                        command = [f"mpiexec -np {cpu_logical_core_count} python -m mpi4py {parastitcher}"]
                    else:
                        command = [f"{terastitcher}"]
                    command += [
                        f"-{step}",
                        "--threshold=0.7",
                        f"--projin={proj_in}",
                        f"--projout={proj_out}",
                    ]
                    p_log("stitching command:\n" + " ".join(command))
                    subprocess.call(" ".join(command), shell=True)  # subprocess.run(command)
                    assert proj_out.exists()
                    proj_in.unlink(missing_ok=False)
        else:
            p_log(f"{datetime.now()}: {channel_dir} did not exist and not imported ...")

    p_log(f"{datetime.now()}: importing all channels ...")

    # p = dir_stitched.rglob("*")
    # if p:
    #     for x in p:
    #         if x.is_dir():
    #             shutil.rmtree(x)
    #     del p, x
    vol_xml_import_path = dir_stitched / 'vol_xml_import.xml'
    vol_xml_import_path.unlink(missing_ok=True)
    if len(existing_channels) > 1:
        command = [
            f"{terastitcher}",
            "-1",
            "--ref1=H",
            "--ref2=V",
            "--ref3=D",
            f"--vxl1={voxel_size_y}",
            f"--vxl2={voxel_size_x}",
            f"--vxl3={voxel_size_z}",
            "--sparse_data",
            "--volin_plugin=MultiVolume",
            f"--volin={dir_stitched}",
            f"--projout={vol_xml_import_path}",
            "--imin_channel=all",
            "--noprogressbar",
        ]
        p_log("import command:\n" + " ".join(command))
        subprocess.run(command)

        p_log(f"{datetime.now()}: running parastitcher on {cpu_logical_core_count} physical cores ... ")
        command = [
            f"mpiexec -np {cpu_logical_core_count} python -m mpi4py {parastitcher}",
            "-6",
            f"--projin={vol_xml_import_path}",
            f"--volout={dir_stitched}",
            # "--volout_plugin=\"TiledXY|3Dseries\"",
            "--volout_plugin=\"TiledXY|2Dseries\"",
            "--slicewidth=100000",
            "--sliceheight=100000",
            "--slicedepth=100000",
            "--isotropic",
            "--halve=max"
        ]

        # command = [
        #     f"mpiexec -np {cpu_logical_core_count} python -m mpi4py {parasconverter}",
        #     "--sfmt=\"TIFF (unstitched, 3D)\"",
        #     "--dfmt=\"TIFF (tiled, 3D)\"",
        #     # "--dfmt=\"TIFF (tiled, 2D)\"",
        #     # "--dfmt=\"Vaa3D raw (tiled, 3D)\"",
        #     f"-s={dir_stitched.joinpath('vol_xml_import.xml')}",
        #     f"-d={dir_stitched}",
        #     f"--width=100000",
        #     f"--height=100000",
        #     f"--depth=100000",
        #     "--isotropic",
        #     "--halve=max",
        #     "--dsfactor=2,  # Down sampling factor to be used to read the source volume (only for series of 2D slices)
        #     "--noprogressbar",
        #     "--sparse_data",
        # ]
        p_log("merge command:\n" + " ".join(command))
        subprocess.call(" ".join(command), shell=True)
    else:
        dir_tif = dir_stitched / 'tif'
        dir_tif.mkdir(exist_ok=True)
        vol_xml_import_path = dir_stitched / f'{existing_channels[0]}_xml_import_step_5.xml'
        volume = tsv.volume.TSVVolume.load(vol_xml_import_path)
        convert_to_2D_tif(
            volume,
            str(dir_tif/"img_{z:06d}.tif"),
            compression=("ZLIB", 1),
            cores=cpu_logical_core_count if cpu_logical_core_count < 61 else 61
        )

    # ::::::::::::::::: File Conversion  ::::::::::::::::

    p = dir_stitched.rglob("*.tif")
    files = [x for x in p if x.is_file()]
    dir_tif = files[0].parent.rename(dir_stitched / 'tif')
    p = dir_tif.rglob("*.tif")
    files = sorted([x for x in p if x.is_file()])
    del p

    file = files[0]
    work = []
    if imaris_converter.exists() and len(files) > 0:
        p_log(f"{datetime.now()}: found Imaris View: converting {file.name} to ims ... ")
        ims_file_path = dir_stitched / f'{source_folder.name}.ims'
        command = [
            f"" if sys.platform == "win32" else "wine",
            f"{correct_path_for_cmd(imaris_converter)}",
            f"--input {correct_path_for_cmd(file)}",
            f"--output {ims_file_path}",
            # f"--log {log_file}" if sys.platform == "win32" else "",
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
            f"--nthreads {cpu_physical_core_count}",
            f"--compression 1",
            # f"--defaultcolorlist BBRRGG"
        ]
        p_log(f"tiff to ims conversion command:\n{' '.join(command)}\n")
        # subprocess.call(" ".join(command), shell=True)
        work += [" ".join(command)]
    else:
        if len(files) > 0:
            p_log("not found Imaris View: not converting tiff to ims ... ")
        else:
            p_log("no tif file found to convert to ims!")

    for channel in channels_need_reconstruction:
        if need_tera_fly_conversion and channel in existing_channels:
            dir_tera_fly = dir_stitched / f'TeraFly_{channel}'
            dir_tera_fly.mkdir(exist_ok=True)
            command = [
                f"mpiexec -np {4} python -m mpi4py {parasconverter}",
                "--sfmt=\"TIFF (series, 2D)\"",
                "--dfmt=\"TIFF (tiled, 3D)\"",
                "--resolutions=\"012345\"",
                f"--clist={existing_channels.index(channel)}",
                "--halve=max",
                "--noprogressbar",
                "--sparse_data",
                f"-s={dir_tif}",
                f"-d={dir_tera_fly}",
            ]
            p_log(f"TeraFly conversion command:\n{' '.join(command)}\n")
            # subprocess.call(" ".join(command), shell=True)
            work += [" ".join(command)]

    with Pool(processes=cpu_physical_core_count) as pool:
        list(pool.imap_unordered(worker, work))

    # ::::::::::::::::::::::::::::: Done ::::::::::::::::::::::::::::::

    p_log(f"done at {datetime.now()}")
    p_log(f"Time elapsed: {time() - start_time}")


if __name__ == '__main__':
    freeze_support()
    if len(sys.argv) == 1:
        main(source_folder=Path(__file__).parent.absolute())
    elif len(sys.argv) == 2:
        if Path(sys.argv[1]).exists():
            main(source_folder=Path(sys.argv[1]).absolute())
        else:
            print("The entered path is not valid")
