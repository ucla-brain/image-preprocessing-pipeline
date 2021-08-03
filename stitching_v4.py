# ::::::::::::::::::::::: For Stitching Light Sheet data:::::::::::::
# Version 4 by Keivan Moradi on July, 2021
# Please read the readme file for more information:
# https://github.com/ucla-brain/image-preprocessing-pipeline/blob/main/README.md
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

import logging as log
import pathlib
import platform
import psutil
import subprocess
import os
import sys
import pystripe
from skimage.transform import resize
from datetime import datetime
from time import time
from tifffile import imread


# experiment setup: user needs to set them right
# ChanelColor =
AllChannels = ["Ex_488_Em_0", "Ex_561_Em_1", "Ex_642_Em_2"]  # the order determines color ["R", "G", "B"]
VoxelSizeX_4x, VoxelSizeY_4x, VoxelSizeZ_4x = 1.835, 1.835, 4.0
VoxelSizeX_10x, VoxelSizeY_10x, VoxelSizeZ_10x = 0.661, 0.661, 2.0
VoxelSizeX_15x, VoxelSizeY_15x, VoxelSizeZ_15x = 0.422, 0.422, 2.0
LogFileName = "StitchingLog.txt"
cpu_physical_core_count = psutil.cpu_count(logical=False)
cpu_logical_core_count = psutil.cpu_count(logical=True)


if sys.platform == "win32":
    CacheDriveExample = "C:"
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
    TeraStitcherPath = pathlib.Path(f"{os.environ['HOME']}/apps/ExM-Studio/stitching/bin")
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


def get_data_format():
    data_format = ""
    while data_format not in {"1", "2"}:
        data_format = input(
            '\n'
            'What is the input data format?\n'
            '1 = RAW\n'
            '2 = TIF\n')
    if data_format == "1":
        data_format = "RAW"
    elif data_format == "2":
        data_format = "TIF"
    else:
        print("Error: unsupported data format")
        log.error("Error: unsupported data format")
        raise RuntimeError
    return data_format


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


def get_cache_drive(file_type='tif', posix='', disabled_path=pathlib.Path('')):
    drive = input(
        f"\n"
        f"Enter cache drive path for {file_type} files.\n"
        f"for example: {CacheDriveExample}\n"
        f"If nothing entered, the feature will be disable.\n")
    cache_drive = pathlib.Path(drive)
    while not cache_drive.exists():
        drive = input(
            f"\n"
            f"Enter a valid cache drives path for {file_type} files. "
            f"for example: {CacheDriveExample}\n"
            f"Disable the feature if nothing entered.\n")
        cache_drive = pathlib.Path(drive)
    if cache_drive.name == '':
        print(f"\nCache drive for {file_type} files is disabled.\n")
        cache_drive = disabled_path
    else:
        print(f"\nCache drive for {file_type} files = {cache_drive}\n")
        cache_drive = cache_drive.joinpath("Cache").joinpath(CaseName + posix)
    cache_drive.mkdir(exist_ok=True, parents=True)
    return cache_drive


def correct_path_for_cmd(filepath):
    if sys.platform == "win32":
        return f"\"{filepath}\""
    else:
        return str(filepath).replace(" ", r"\ ").replace("(", r"\(").replace(")", r"\)")


def main():

    # ::::::::::::::::: RUN PyStripe to convert RAW to TIFF ::::::::::::::::

    start_time = time()
    if DataFormat == "RAW":
        print(f"{datetime.now()} ... converting RAW to TIFF started")
        log.info(f"{datetime.now()} ... converting RAW to TIFF started")
        # DirTif.mkdir(exist_ok=True)
        for Channel in AllChannels:
            source_channel_folder = SourceFolder / Channel
            if source_channel_folder.exists():
                pystripe.batch_filter(
                    source_channel_folder,
                    DirTif / Channel,
                    workers=cpu_logical_core_count,
                    chunks=1,
                    # sigma=[foreground, background] Default is [0, 0], indicating no de-striping.
                    sigma=[32, 32] if Objective == "4x" else [256, 256],
                    # level=0,
                    wavelet="db10",
                    crossover=10,
                    # threshold=-1,
                    # compression=3,  # not working yet. It should be like ('zstd', 3)
                    flat=resize(imread(SourceFolder / "flat.tif"), TileSize),
                    dark=100.0,
                    # zstep=VoxelSizeZ,  # Z-step in micron. Only used for DCIMG files.
                    # rotate=False,
                    # lightsheet=True,  # default to False
                    # artifact_length=150,
                    # percentile=0.25,
                    # dont_convert_16bit=True  # defaults to False
                )
        # subprocess.run(["pystripe", "-i", SourceFolder, "-o", DirTif])
        print(f"{datetime.now()}: Done converting RAW to TIFF.")
        log.info(f"{datetime.now()}: Done converting RAW to TIFF.")

    # ::::::::::::::::: Stitching ::::::::::::::::

    log.info(f"{datetime.now()}: stitching to 3D TIFF started ...")
    # DirStitched.mkdir(exist_ok=True)
    print("running terastitcher ... ")

    """"
    # ::::::::::::::::::: Stitching V2: RUN TeraConverter to Stitching Tiffs & Convert To IMS ::::::::::::::::::::::::::

    # print("Light Sheet Data Stitching (Version 2) is Running ...")
    # for Channel in AllChannels:
    #     channel_dir = DirTif.joinpath(Channel)
    #     if channel_dir.exists():
    #         log.info(f"{datetime.now()}: {channel_dir} folder exists importing for stitching...")
    #         command = [
    #             f"{terastitcher}",
    #             "-1",
    #             "--ref1=H",
    #             "--ref2=V",
    #             "--ref3=D",
    #             f"--vxl1={VoxelSizeY}",
    #             f"--vxl2={VoxelSizeX}",
    #             f"--vxl3={VoxelSizeZ}",
    #             "--sparse_data",
    #             "--threshold=0.5",
    #             "--libtiff_bigtiff",
    #             f"--volin={channel_dir}",
    #             f"--volout={DirStitched.joinpath(Channel + '_stitched.tiff')}",
    #             f"--projout={DirStitched.joinpath(Channel + '_xml_import.xml')}"]
    #         log.info("import command: " + " ".join(command))
    #         subprocess.run(command)
    #     else:
    #         log.warning(f"{datetime.now()}: {channel_dir} folder did not exist and not imported for stitching...")
    #
    # log.info(f"{datetime.now()}: importing all channels ...")
    # command = [
    #     f"{terastitcher}",
    #     "-1",
    #     "--ref1=H",
    #     "--ref2=V",
    #     "--ref3=D",
    #     f"--vxl1={VoxelSizeY}",
    #     f"--vxl2={VoxelSizeX}",
    #     f"--vxl3={VoxelSizeZ}",
    #     "--sparse_data",
    #     "--volin_plugin=MultiVolume",
    #     f"--volin={DirStitched}",
    #     f"--projout={DirStitched.joinpath('Vol_xml_import.xml')}"]
    # log.info("import command: " + " ".join(command))
    # subprocess.run(command)
    #
    # print(f"{datetime.now()}: running teraconverter ... ")
    # command = [
    #     f"{teraconverter}",
    #     "--sfmt=TIFF (unstitched, 3D)",
    #     "--dfmt=HDF5 (Imaris IMS)",
    #     "--mdata_fname=null",
    #     "--sparse_data",
    #     f"--resolutions={Resolution}",
    #     f"--imout_plugin=IMS_HDF5",
    #     "--imout_plugin_params=Max=65535, HistogramMax=65535, img_bytes_x_chan=2",
    #     "-f=graylevel",
    #     f"-s={DirStitched.joinpath('Vol_xml_import.xml')}",
    #     f"-d={DirStitched.joinpath(CaseName + '_R' + Resolution + '_V2.ims')}"]
    # log.info("stitching command" + " ".join(command))
    # subprocess.run(command)

    # ::::::::::::::::: Stitching V1: RUN terastitcher first then convert To IMS with teraconverter ::::::::::::::::::::

    # print("Light Sheet Data Stitching (Version 1) is Running ...")
    # for Channel, Color in zip(AllChannels, ChanelColor):
    #     channel_dir = DirTif.joinpath(Channel)
    #     # DirStitchedChannel = DirStitched.joinpath(Channel)
    #     log.info(f"{datetime.now()} to check if {channel_dir} exist before stitching...")
    #     if channel_dir.exists():
    #         # print(f"{DirStitchedChannel}")
    #         # DirStitchedChannel.mkdir(exist_ok=True)
    #         for step in range(1, 6):  # steps 1 to 5
    #             print(f"step = {step}")
    #             subprocess.run([
    #                 f"{terastitcher}",
    #                 f"-{step}",
    #                 f"--volin={channel_dir}" if step == 1 else f"--projin={DirStitched.joinpath(Channel + '_step_' + str(step-1) + '.xml')}",  # "--volin_plugin=TiledXY|3Dseries",
    #                 "--ref1=H",  "--ref2=V", "--ref3=D", f"--vxl1={VoxelSizeY}", f"--vxl2={VoxelSizeX}", f"--vxl3={VoxelSizeZ}", "--sparse_data",  # -1
    #                 "--subvoldim=600", "--sV=50", "--sH=50", "--sD=0",  # -2
    #                 "--threshold=0.5",  # -4
    #                 "--volout_plugin=TiledXY|3Dseries", "--slicewidth=100000", "--sliceheight=150000",  # -6 TiledXY|2Dseries
    #                 f"--imin_channel={Color}",
    #                 "--libtiff_bigtiff",
    #                 f"--volout={DirStitched}",
    #                 f"--projout={DirStitched.joinpath(Channel + '_step_' + str(step) + '.xml')}"])
    #
    #         for step in range(1, 5):
    #             DirStitched.joinpath(Channel + '_step_' + str(step) + '.xml').unlink()
    #
    # subprocess.run([
    #     f"{terastitcher}",
    #     "-1",
    #     "--ref1=H",
    #     "--ref2=V",
    #     "--ref3=D",
    #     f"--vxl1={VoxelSizeY}",
    #     f"--vxl2={VoxelSizeX}",
    #     f"--vxl3={VoxelSizeZ}",
    #     f"--volin={DirStitched}", "--volin_plugin=MultiVolume",
    #     f"--projout={DirStitched.joinpath('Vol_xml_import.xml')}"])
    #
    # print("running teraconverter ... ")
    # subprocess.run([
    #     f"{teraconverter}",
    #     "--sfmt=TIFF (unstitched, 3D)",
    #     "--dfmt=HDF5 (Imaris IMS)",
    #     "--mdata_fname=null",
    #     "--sparse_data",
    #     f"--resolutions={Resolution}",
    #     "--imout_plugin=IMS_HDF5",
    #     # "--imout_plugin_params=Max=65535, HistogramMax=65535, img_bytes_x_chan=2",
    #     "-f=graylevel",
    #     f"-s={DirStitched.joinpath('Vol_xml_import.xml')}",
    #     f"-d={DirStitched.joinpath(CaseName + '_R' + Resolution + '_V2.ims')}"])

    # ::::::::::::::::::::: V3 Keivan :::::::::::::::::::::::::::::::::::
    # generates separate stitched tiff files for each channel
    
    # print("Light Sheet Data Stitching (Version 3) is Running ...")
    # for Channel, Color in zip(AllChannels, ChanelColor):
    #     channel_dir = DirTif.joinpath(Channel)
    #     DirStitchedChannel = DirStitched.joinpath(Channel)
    #     log.info(f"{datetime.now()} to check if {channel_dir} exist before stitching...")
    #     if channel_dir.exists():
    #         print(f"{DirStitchedChannel}")
    #         DirStitchedChannel.mkdir(exist_ok=True)
    #         subprocess.run([
    #             f"{terastitcher}",
    #             f"-S",
    #             f"--volin={channel_dir}",
    #             # "--volin_plugin=TiledXY|3Dseries",
    #             "--ref1=H", "--ref2=V", "--ref3=D", f"--vxl1={VoxelSizeY}", f"--vxl2={VoxelSizeX}",
    #             f"--vxl3={VoxelSizeZ}", "--sparse_data",  # -1
    #             "--subvoldim=600", "--sV=50", "--sH=50", "--sD=0",  # -2
    #             "--threshold=0.5",  # -4
    #             "--volout_plugin=TiledXY|3Dseries", "--slicewidth=100000", "--sliceheight=150000",  # -6 TiledXY|2Dseries
    #             # f"--imin_channel={Color}",
    #             "--libtiff_bigtiff",
    #             f"--volout={DirStitchedChannel}",
    #             f"--projout={DirStitched.joinpath(Channel + '_step_1_to_6.xml')}"])
    """
    # ::::::::::::::::::::: V4 Keivan :::::::::::::::::::::::::::::::::::
    # generates one multichannel tiff file: GPU accelerated & parallel

    print("Light Sheet Data Stitching (Version 4) is Running ...")
    for Channel in AllChannels:
        channel_dir = DirTif / Channel
        if channel_dir.exists():
            log.info(f"{datetime.now()}: {channel_dir} folder exists importing for stitching...")
            command = [
                f"{terastitcher}",
                "-1",
                "--ref1=H",
                "--ref2=V",
                "--ref3=D",
                f"--vxl1={VoxelSizeY}",
                f"--vxl2={VoxelSizeX}",
                f"--vxl3={VoxelSizeZ}",
                "--sparse_data",
                f"--volin={channel_dir}",
                f"--projout={DirStitched.joinpath(Channel + '_xml_import_step_1.xml')}",
                "--noprogressbar"
            ]
            log.info("import command:\n" + " ".join(command))
            subprocess.run(command)
            if Channel == MostInformativeChannel:
                for step in [2, 3, 4, 5]:
                    log.info(f"{datetime.now()}: starting step {step} of stitching for most informative channel ...")
                    command = [
                        f"mpiexec -np {cpu_physical_core_count} python -m mpi4py {parastitcher}" if step == 2 else f"{terastitcher}",
                        f"-{step}",
                        "--threshold=0.5",
                        "--isotropic",
                        f"--projin={DirStitched.joinpath(Channel + '_xml_import_step_' + str(step - 1) + '.xml')}",
                        f"--projout={DirStitched.joinpath(Channel + '_xml_import_step_' + str(step) + '.xml')}",
                    ]
                    log.info("stitching command:\n" + " ".join(command))
                    subprocess.call(" ".join(command), shell=True)  # subprocess.run(command)
                    DirStitched.joinpath(Channel + '_xml_import_step_' + str(step - 1) + '.xml').unlink()
        else:
            log.warning(f"{datetime.now()}: {channel_dir} did not exist and not imported ...")

    log.info(f"{datetime.now()}: importing all channels ...")
    command = [
        f"{terastitcher}",
        "-1",
        "--ref1=H",
        "--ref2=V",
        "--ref3=D",
        f"--vxl1={VoxelSizeY}",
        f"--vxl2={VoxelSizeX}",
        f"--vxl3={VoxelSizeZ}",
        "--sparse_data",
        "--volin_plugin=MultiVolume",
        f"--volin={DirStitched}",
        f"--projout={DirStitched.joinpath('vol_xml_import.xml')}",
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
        f"--projin={DirStitched.joinpath('vol_xml_import.xml')}",
        f"--volout={DirStitched}",
        "--volout_plugin=\"TiledXY|3Dseries\"",
        "--slicewidth=100000",
        "--sliceheight=150000",
        "--slicedepth=50000",
        "--isotropic"
    ]
    log.info("stiching command:\n" + " ".join(command))
    subprocess.call(" ".join(command), shell=True)

    p = DirStitched.rglob("*.tif")
    files = [x for x in p if x.is_file()]
    del p
    for file in files:
        file = file.rename(
            DirStitched / (CaseName + ('_' + file.name if len(files) > 1 else '') + '_V4.tif'))  # move tiff files
        if imaris_converter.exists():
            print(f"found Imaris View: converting {file.name} to ims ... ")
            command = [
                f"wine" if not sys.platform == "win32" else "",
                f"{correct_path_for_cmd(imaris_converter)}",
                f"--input {correct_path_for_cmd(file)}",
                f"--output {DirStitched.joinpath(CaseName + ('_' + file.name if len(files) > 1 else '') + '_V4.ims')}",
                f"--log {LogFile}",
                f"--nthreads {cpu_logical_core_count}",
                f"--compression 1",
            ]
            log.info("tiff to ims conversion command:\n" + " ".join(command))
            subprocess.call(" ".join(command), shell=True)
        else:
            log.warning("not found Imaris View: not converting tiff to ims ... ")

    # ::::::::::::::::::::::::::::: Done ::::::::::::::::::::::::::::::

    print(f"done at {datetime.now()}")
    log.info(f"done at {datetime.now()}")
    print(f"Time elapsed: {time() - start_time}")


if __name__ == '__main__':

    # :::::::::::::::::::::::path setting for processing:::::::::::::::::::::::::

    DataFormat = get_data_format()
    SourceFolder = pathlib.Path(__file__).parent.absolute()
    DirTif = pathlib.Path(str(SourceFolder) + "_Tiffs")
    DirStitched = pathlib.Path(str(SourceFolder) + "_Stitched_V4")
    if DataFormat == "TIF":
        DirTif = SourceFolder
    CaseName = SourceFolder.name
    LogFile = SourceFolder.joinpath(LogFileName)
    log.basicConfig(filename=str(LogFile), level=log.INFO)
    log.FileHandler(str(LogFile), mode="w")  # rewrite the file instead of appending
    if DataFormat != "TIF":
        print("raw files should be converted to tiff and stored in the cache drive.")
        DirTif = get_cache_drive(file_type="tif", posix="_Tiffs", disabled_path=DirTif)
        print(DirTif)
    DirStitched = get_cache_drive(file_type="stitched", posix="_Stitched_V4", disabled_path=DirStitched)
    print(DirStitched)

    MostInformativeChannel = get_most_informative_channel()
    Objective, VoxelSizeX, VoxelSizeY, VoxelSizeZ, TileSize = get_voxel_sizes()

    # Resolution = None
    # while Resolution not in {"1", "2", "3", "4", "5", "6"}:
    #     Resolution = input(
    #         '\nChoose IMS Format Resolution levels: \n'
    #         '1 = ~16 Layers/max Proj, \n'
    #         '2 = ~1000 Layers/ROI, \n'
    #         '3 = ~2000 Layers/4X imaging, \n'
    #         '4 = ~4000 Layers/15X imaging, \n'
    #         '5  = 5, \n'
    #         '6  = 6: \n')
    # if Resolution == "1":
    #     Resolution = "01"
    # elif Resolution == "2":
    #     Resolution = "012"
    # elif Resolution == "3":
    #     Resolution = "0123"
    # elif Resolution == "4":
    #     Resolution = "01234"
    # elif Resolution == "5":
    #     Resolution = "012345"
    # elif Resolution == "6":
    #     Resolution = "0123456"
    # else:
    #     print("unsupported resolution")

    # ::::::::::::::::::::::::::::::::::::: Start :::::::::::::::::::::::::::::::::::

    log.info(f"{datetime.now()} ... stitching started")
    log.info(f"Run on Computer: {platform.node()}")
    log.info(f"Total physical memory: {psutil.virtual_memory().total // 1024 ** 3} GB")
    log.info(f"Physical CPU core count: {cpu_physical_core_count}")
    print(f"CaseName = {CaseName}")
    log.info(f"CaseName: {CaseName}")
    print(f"Source Folder = {SourceFolder}")
    log.info(f"Source Folder: {SourceFolder}")
    print(f"TIff folder = {DirTif}")
    log.info(f"TIff folder: {DirTif}")
    print(f"Stitched Folder = {DirStitched}")
    log.info(f"Stitched Folder: {DirStitched}")

    main()
