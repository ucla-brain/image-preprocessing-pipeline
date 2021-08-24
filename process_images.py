# ::::::::::::::::::::::: For Stitching Light Sheet data:::::::::::::
# Version 4 by Keivan Moradi on July, 2021
# Please read the readme file for more information:
# https://github.com/ucla-brain/image-preprocessing-pipeline/blob/main/README.md
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
import os
import sys
import psutil
import pathlib
import platform
import subprocess
import logging as log
import pystripe_forked as pystripe
from multiprocessing import freeze_support
from flat import create_flat_img
from datetime import datetime
from time import time

# experiment setup: user needs to set them right
AllChannels = ["Ex_488_Em_0", "Ex_561_Em_1", "Ex_642_Em_2"]  # the order determines color ["R", "G", "B"]?
VoxelSizeX_4x, VoxelSizeY_4x, VoxelSizeZ_4x = 1.835, 1.835, 4.0
VoxelSizeX_10x, VoxelSizeY_10x, VoxelSizeZ_10x = 0.661, 0.661, 1.0
VoxelSizeX_15x, VoxelSizeY_15x, VoxelSizeZ_15x = 0.422, 0.422, 1.0
FlatNonFlatTrainingData = "image_classes.csv"
cpu_physical_core_count = psutil.cpu_count(logical=False)
cpu_logical_core_count = psutil.cpu_count(logical=True)

if sys.platform == "win32":
    CacheDriveExample = "C:\\"
    if pathlib.Path(r"C:\TeraStitcher").exists():
        TeraStitcherPath = pathlib.Path(r"C:\TeraStitcher")
    elif pathlib.Path(r"C:\Programs\TeraStitcher").exists():
        TeraStitcherPath = pathlib.Path(r"C:\Programs\TeraStitcher")
    elif pathlib.Path(r"C:\Program Files\TeraStitcher-Qt5-standalone 1.10.18\bin").exists():
        TeraStitcherPath = pathlib.Path(r"C:\Program Files\TeraStitcher-Qt5-standalone 1.10.18\bin")
    else:
        log.error("Error: TeraStitcher path not found")
        raise RuntimeError
    os.environ["PATH"] = f"{os.environ['PATH']};{TeraStitcherPath.as_posix()}"
    os.environ["PATH"] = f"{os.environ['PATH']};{TeraStitcherPath.joinpath('pyscripts').as_posix()}"
    terastitcher = "terastitcher.exe"
    teraconverter = "teraconverter.exe"
    ImarisConverterPath = pathlib.Path(r"C:\Program Files\Bitplane\ImarisViewer x64 9.7.2")
elif sys.platform == 'linux':
    CacheDriveExample = "/mnt/scratch"
    os.environ["HOME"] = r"/home/kmoradi"
    # TeraStitcherPath = pathlib.Path(f"{os.environ['HOME']}/apps/ExM-Studio/stitching/bin")
    TeraStitcherPath = pathlib.Path(f"{os.environ['HOME']}/apps/ExM-Studio/stitching_native/bin")
    os.environ["PATH"] = f"{os.environ['PATH']}:{TeraStitcherPath.as_posix()}"
    os.environ["PATH"] = f"{os.environ['PATH']}:{TeraStitcherPath.joinpath('pyscripts').as_posix()}"
    terastitcher = "terastitcher"
    teraconverter = "teraconverter"
    if not TeraStitcherPath.exists():
        log.error("Error: TeraStitcher path not found")
        raise RuntimeError
    if pathlib.Path("/usr/lib/jvm/java-11-openjdk-amd64/lib/server").exists():
        os.environ["LD_LIBRARY_PATH"] = "/usr/lib/jvm/java-11-openjdk-amd64/lib/server"
    else:
        log.error("Error: JAVA path not found")
        raise RuntimeError
    os.environ["TERM"] = "xterm"
    os.environ["USECUDA_X_NCC"] = "1"  # set to 0 to stop GPU acceleration
    if os.environ["USECUDA_X_NCC"] != "0":
        if pathlib.Path("/usr/local/cuda-10.1/").exists() and pathlib.Path("/usr/local/cuda-10.1/bin").exists():
            os.environ["CUDA_ROOT_DIR"] = "/usr/local/cuda-10.1/"
        elif pathlib.Path("/usr/local/cuda-11.4/").exists() and pathlib.Path("/usr/local/cuda-11.4/bin").exists():
            os.environ["CUDA_ROOT_DIR"] = "/usr/local/cuda-11.4/"
        else:
            log.error("Error: CUDA path not found")
            raise RuntimeError
        os.environ["PATH"] = f"{os.environ['PATH']}:{os.environ['CUDA_ROOT_DIR']}/bin"
        os.environ["LD_LIBRARY_PATH"] = f"{os.environ['LD_LIBRARY_PATH']}:{os.environ['CUDA_ROOT_DIR']}/lib64"
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # to train on a specific GPU on a multi-gpu machine

    ImarisConverterPath = pathlib.Path(
        f"{os.environ['HOME']}/.wine/drive_c/Program Files/Bitplane/ImarisViewer x64 9.7.2/")
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

imaris_converter = ImarisConverterPath / "ImarisConvertiv.exe"
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
            '2 = No\n')
    if answer == "1":
        return True
    else:
        return False


def get_most_informative_channel():
    num_channel = ""
    while num_channel not in {"0", "1", "2"}:
        num_channel = input(
            f"\n"
            f"Choose the most informative channel from the list of channels:\n"
            f"0 = {AllChannels[0]}\n"
            f"1 = {AllChannels[1]}\n"
            f"2 = {AllChannels[2]}\n")
    most_informative_channel = AllChannels[int(num_channel)]
    print(f"most informative channel is {most_informative_channel}")
    log.info(f"most informative channel is {most_informative_channel}")
    return most_informative_channel


def get_voxel_sizes():
    objective = ""
    while objective not in {"1", "2", "3"}:
        objective = input(
            '\n'
            'What is the Objective?\n'
            '1 = 4x\n'
            '2 = 10x\n'
            '3 = 15x\n')

    if objective == "1":
        objective = "4x"
        voxel_size_x = VoxelSizeX_4x
        voxel_size_y = VoxelSizeY_4x
        voxel_size_z = VoxelSizeZ_4x
        tile_size = (2000, 1600)
    elif objective == "2":
        objective = "10x"
        voxel_size_x = VoxelSizeX_10x
        voxel_size_y = VoxelSizeY_10x
        voxel_size_z = VoxelSizeZ_10x
        tile_size = (1850, 1850)
    elif objective == "3":
        objective = "15x"
        voxel_size_x = VoxelSizeX_15x
        voxel_size_y = VoxelSizeY_15x
        voxel_size_z = VoxelSizeZ_15x
        tile_size = (1850, 1850)
    else:
        print("Error: unsupported objective")
        log.error("Error: unsupported objective")
        raise RuntimeError
    print(
        f"Objective is {objective} so voxel sizes are x = {voxel_size_x}, y = {voxel_size_y}, and z = {voxel_size_z}")
    log.info(
        f"Objective is {objective} so voxel sizes are x = {voxel_size_x}, y = {voxel_size_y}, and z = {voxel_size_z}")
    return objective, voxel_size_x, voxel_size_y, voxel_size_z, tile_size


def get_destination_path(folder_name_prefix, what_for='tif', posix='', default_path=pathlib.Path('')):
    input_path = input(
        f"\n"
        f"Enter destination path for {what_for}.\n"
        f"for example: {CacheDriveExample}\n"
        f"If nothing entered, {default_path.absolute()} will be used.\n")
    drive_path = pathlib.Path(input_path)
    while not drive_path.exists():
        input_path = input(
            f"\n"
            f"Enter a valid destination path for {what_for}. "
            f"for example: {CacheDriveExample}\n"
            f"If nothing entered, {default_path.absolute()} will be used.\n")
        drive_path = pathlib.Path(input_path)
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


def p_log(txt):
    print(txt)
    log.info(txt)


def main(source_folder):
    log_file = source_folder / "log.txt"
    log.basicConfig(filename=str(log_file), level=log.INFO)
    log.FileHandler(str(log_file), mode="w")  # rewrite the file instead of appending
    # ::::::::::::::::::::: Ask questions :::::::::::::::::::::
    de_striped_posix, what_for = "", ""
    need_flat_image_subtraction, img_flat = False, None
    image_classes_training_data_path = source_folder / FlatNonFlatTrainingData
    need_raw_to_tiff_conversion = False
    need_destriping = ask_true_false_question("Do you need to remove stripes from images?")
    if need_destriping:
        need_flat_image_subtraction = ask_true_false_question("Do you need flat image subtraction?")
        de_striped_posix += "_destriped"
        what_for += "destriped "
    else:
        need_raw_to_tiff_conversion = ask_true_false_question("Are images in raw format?")
        if need_raw_to_tiff_conversion:
            de_striped_posix += "_tif"
            what_for += "tif "
    if need_flat_image_subtraction:
        de_striped_posix += "_flat_subtracted"
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
                    image_classes_training_data_path = pathlib.Path(__file__).parent / "image_classes.csv"
                    print(f"default classification data path is:\n"
                          f"{image_classes_training_data_path.absolute()}")
                else:
                    print("You need classification data for flat image generation!")
                    raise RuntimeError

    de_striped_dir = source_folder.parent / (source_folder.name + de_striped_posix)
    continue_process_pystripe = False
    dir_stitched = source_folder.parent / (source_folder.name + "_stitched_v4")
    img_16bit_to_8bit, right_bit_shift = False, 8
    if need_destriping or need_raw_to_tiff_conversion:
        img_16bit_to_8bit = ask_true_false_question("Do you need to convert 16-bit images to 8 bit before stitching?")
        if img_16bit_to_8bit:
            right_bit_shift = int(input(
                "Enter right bit shift [0 to 8] for 8-bit conversion. \n"
                "Values smaller than 8 will increase the pixel brightness. \n"
                "We suggest a value between 3 to 6. \n"
                "The smaller the value the brighter the pixels.\n"))
        de_striped_dir, continue_process_pystripe = get_destination_path(
            source_folder.name,
            what_for=what_for+"files",
            posix=de_striped_posix,
            default_path=de_striped_dir)
    else:
        de_striped_dir = source_folder
    dir_stitched, continue_process_terastitcher = get_destination_path(
        source_folder.name, what_for="stitched files", posix="_stitched_v4", default_path=dir_stitched)
    # print(dir_stitched)

    most_informative_channel = get_most_informative_channel()
    objective, voxel_size_x, voxel_size_y, voxel_size_z, tile_size = get_voxel_sizes()

    # ::::::::::::::::::::::::::::::::::::: Start :::::::::::::::::::::::::::::::::::

    log.info(f"{datetime.now()} ... stitching started")
    log.info(f"Run on Computer: {platform.node()}")
    log.info(f"Total physical memory: {psutil.virtual_memory().total // 1024 ** 3} GB")
    log.info(f"Physical CPU core count: {cpu_physical_core_count}")
    log.info(f"Logical CPU core count: {cpu_logical_core_count}")
    p_log(f"Source folder path:\n{source_folder}")
    p_log(f"Destriped or tif files path:\n{de_striped_dir}")
    p_log(f"Stitched folder path:\n{dir_stitched}")
    # ::::::::::::::::: RUN PyStripe  ::::::::::::::::
    start_time = time()
    if need_destriping or need_raw_to_tiff_conversion:
        for Channel in AllChannels:
            source_channel_folder = source_folder / Channel
            if source_channel_folder.exists():
                if need_flat_image_subtraction:
                    flat_img_created_already = source_folder.joinpath(Channel+'_flat.tif')
                    if flat_img_created_already.exists():
                        img_flat = pystripe.imread(str(flat_img_created_already))
                        p_log(f"{datetime.now()}: {Channel}: using the existing flat image:\n"
                              f"{flat_img_created_already.absolute()}.")
                    else:
                        p_log(f"{datetime.now()}: {Channel}: creating a new flat image.")
                        img_flat = create_flat_img(
                            source_channel_folder,
                            image_classes_training_data_path,
                            cpu_physical_core_count=cpu_physical_core_count,
                            cpu_logical_core_count=cpu_logical_core_count,
                            max_images=1024,  # the number of flat images averaged
                            patience_before_skipping=10,  # the # of non-flat images found successively before skipping
                            skips=100,  # the number of images should be skipped before testing again
                            sigma_spatial=1,  # the de-noising parameter
                            save_as_tiff=True
                        )
                p_log(f"\n{datetime.now()}: {Channel}: ... DeStripe program started")
                pystripe.batch_filter(
                    source_channel_folder,
                    de_striped_dir / Channel,
                    workers=cpu_logical_core_count,  # if need_destriping else cpu_physical_core_count
                    chunks=1,
                    # sigma=[foreground, background] Default is [0, 0], indicating no de-striping.
                    sigma=((32, 32) if objective == "4x" else (256, 256)) if need_destriping else (0, 0),
                    # level=0,
                    wavelet="db10",
                    crossover=10,
                    # threshold=-1,
                    compression=('ZLIB', 1),  # ('ZLIB', 1) ('ZSTD', 1) conda install imagecodecs
                    flat=img_flat,
                    dark=100.0,  # 100.0
                    # z_step=voxel_size_z,  # z-step in micron. Only used for DCIMG files.
                    # rotate=False,
                    # lightsheet=True if need_destriping else False,  # default to False
                    # artifact_length=150,
                    # percentile=0.25,
                    # dont_convert_16bit=True,  # defaults to False
                    convert_to_8bit=img_16bit_to_8bit,
                    bit_shift_to_right=right_bit_shift,
                    continue_process=continue_process_pystripe
                )
                p_log(f"{datetime.now()}: {Channel}: DeStripe program is done.")

    # ::::::::::::::::: Stitching ::::::::::::::::
    # generates one multichannel tiff file: GPU accelerated & parallel

    log.info(f"{datetime.now()}: stitching to 3D TIFF started ...")
    print("running terastitcher ... ")
    print("Light Sheet Data Stitching (Version 4) is Running ...")
    for Channel in AllChannels:
        channel_dir = de_striped_dir / Channel
        if channel_dir.exists():
            log.info(f"{datetime.now()}: {channel_dir} folder exists importing for stitching...")
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
                f"--projout={dir_stitched.joinpath(Channel + '_xml_import_step_1.xml')}",
                "--noprogressbar"
            ]
            log.info("import command:\n" + " ".join(command))
            subprocess.run(command, check=True)
            if Channel == most_informative_channel:
                for step in [2, 3, 4, 5]:
                    log.info(f"{datetime.now()}: starting step {step} of stitching for most informative channel ...")
                    if step == 2:
                        command = [f"mpiexec -np {cpu_physical_core_count} python -m mpi4py {parastitcher}"]
                    else:
                        command = [f"{terastitcher}"]
                    command += [
                        f"-{step}",
                        "--threshold=0.5",
                        "--isotropic",
                        f"--projin={dir_stitched.joinpath(Channel + '_xml_import_step_' + str(step - 1) + '.xml')}",
                        f"--projout={dir_stitched.joinpath(Channel + '_xml_import_step_' + str(step) + '.xml')}",
                    ]
                    log.info("stitching command:\n" + " ".join(command))
                    subprocess.call(" ".join(command), shell=True)  # subprocess.run(command)
                    dir_stitched.joinpath(Channel + '_xml_import_step_' + str(step - 1) + '.xml').unlink()
        else:
            log.warning(f"{datetime.now()}: {channel_dir} did not exist and not imported ...")

    log.info(f"{datetime.now()}: importing all channels ...")
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
        f"--projout={dir_stitched.joinpath('vol_xml_import.xml')}",
        "--imin_channel=all",
        "--noprogressbar",
        "--isotropic"
    ]
    log.info("import command:\n" + " ".join(command))
    subprocess.run(command)

    log.info(f"{datetime.now()}: running parastitcher on {cpu_physical_core_count} physical cores ... ")
    command = [
        f"mpiexec -np {cpu_physical_core_count + 1}",
        "python -m mpi4py",
        f"{parastitcher}",
        "-6",
        f"--projin={dir_stitched.joinpath('vol_xml_import.xml')}",
        f"--volout={dir_stitched}",
        "--volout_plugin=\"TiledXY|3Dseries\"",
        # "--volout_plugin=\"HDF5 (Imaris IMS)\"",
        "--slicewidth=100000",
        "--sliceheight=150000",
        "--slicedepth=50000",
        "--isotropic"
    ]
    log.info("stitching command:\n" + " ".join(command))
    subprocess.call(" ".join(command), shell=True)

    p = dir_stitched.rglob("*.tif")
    files = [x for x in p if x.is_file()]
    del p
    for file in files:
        file = file.rename(
            dir_stitched / (source_folder.name + (
                '_' + file.name if len(files) > 1 else '') + '_V4.tif'))  # move tiff files
        if imaris_converter.exists():
            print(f"found Imaris View: converting {file.name} to ims ... ")
            command = [
                f"wine" if not sys.platform == "win32" else "",
                f"{correct_path_for_cmd(imaris_converter)}",
                f"--input {correct_path_for_cmd(file)}",
                f"--output {dir_stitched / (source_folder.name+('_'+file.name if len(files) > 1 else '')+'_v4.ims')}",
                f"--log {log_file}",
                f"--nthreads {cpu_logical_core_count}",
                f"--compression 1",
            ]
            log.info("tiff to ims conversion command:\n" + " ".join(command))
            subprocess.call(" ".join(command), shell=True)
        else:
            log.warning("not found Imaris View: not converting tiff to ims ... ")

    # ::::::::::::::::::::::::::::: Done ::::::::::::::::::::::::::::::

    p_log(f"done at {datetime.now()}")
    p_log(f"Time elapsed: {time() - start_time}")


if __name__ == '__main__':
    freeze_support()
    if len(sys.argv) == 1:
        main(source_folder=pathlib.Path(__file__).parent.absolute())
    elif len(sys.argv) == 2:
        if pathlib.Path(sys.argv[1]).exists():
            main(source_folder=pathlib.Path(sys.argv[1]).absolute())
        else:
            print("The entered path does not exists")
    else:
        print("More than one argument is entered. This program accepts only path as argument.")
