# Image Preprocessing Pipeline
Python code for stitching and image enhancement of Light Sheet data

Version 4, peaced together by Keivan Moradi on July, 2021

# Usage:
* Install TeraStitcher portable >=1.11.
* On Linux also make sure Java server (e.g., openjdk), Nvidia drivers and CUDA >10.1 are installed.
* Install Imaris Viewer (on Linux use wine) and set the path (ImarisConverterPath) in the python file.
* Install anaconda python distribution: https://www.anaconda.com/products/individual
* make a dedicated python environment for stitching:

   `conda create -n stitching -c conda-forge python=3.8 psutil mpi4py`
   
   On windows, the conda version of mpi4py is not functional. Instead, install the latest Microsoft MPI from GitHub: https://github.com/microsoft/Microsoft-MPI. Then, install mpi4py with `pip install mpi4py` command.
* activate stitching environment in anaconda: `conda activate stitching`
* install PyStripe: `pip install https://github.com/chunglabmit/pystripe/archive/master.zip`
* Make sure the location of packages are set correctly in the python script.
* Copy the python file to the root folder of data.
* run it in the root folder of data: `python stitching_v4.py`
* select the resolution, most informative channel, and scratch folder and  wait for the results.


# Configurations

* Set the terastitcher path. The default path on Windows is `C:\TeraStitcher` folder, but you may edit the python file and change the `TeraStitcherPath` variable to any path you like. For example,

`TeraStitcherPath = r"C:\TeraStitcher"` (Windows)

`TeraStitcherPath = r"/path/to/terastitcher"` (Linux)

* Set up Imaris Viewer path:

`ImarisConverterPath = pathlib.Path(r"C:\Program Files\Bitplane\ImarisViewer x64 9.7.2")` (Windows)

`os.environ["HOME"] = r"/home/username"`, then `ImarisConverterPath = pathlib.Path(f"{os.environ['HOME']}/.wine/drive_c/Program Files/Bitplane/ImarisViewer x64 9.7.2/")` (Linux)

* to enable GPU acceleration set `os.environ["USECUDA_X_NCC"] = "1"` (Linux Only). For more information about GPU accelerated stitching go to https://github.com/abria/TeraStitcher/wiki/Multi-GPU-parallelization-using-CUDA.
* need to set libjvm.so path:

   `os.environ["LD_LIBRARY_PATH"] = "/usr/lib/jvm/java-11-openjdk-amd64/lib/server/"` (Linux)

* need to set path for CUDA Libraries: (Linux Only)

`os.environ["CUDA_ROOT_DIR"] = "/usr/local/cuda-11.4/"` (update the cuda version as needed)

* need to set visible GPU device on a multiGPU machine, which starts from zero.

`os.environ["CUDA_VISIBLE_DEVICES"] = "1"`

* default channel folder names and voxel dimentions are set up like this:

`AllChannels = ["Ex_488_Em_0", "Ex_561_Em_1", "Ex_642_Em_2"]`

`VoxelSizeX_4x, VoxelSizeY_4x, VoxelSizeZ_4x = 1.835, 1.835, 4.0`

`VoxelSizeX_10x, VoxelSizeY_10x, VoxelSizeZ_10x = 0.661, 0.661, 2.0`

`VoxelSizeX_15x, VoxelSizeY_15x, VoxelSizeZ_15x = 0.422, 0.422, 2.0`

 Please change them as needed.


# Functions and steps:
* Step 1: convert RAW 2D tiles => 2D TIFF tiles with PyStripe and remove stripes.
* Step 2: Import de-striped TIFF file and align the most informative channel with Parastitcher.
* Step 3: Stitch 3 channel data to multichannel 3D TIFF with Parastitcher
* Step 4: convert multichannel 3D TIFF to IMS using ImarisConvertiv.exe.

## terastitcher steps:
* Step 1: import a volume into TeraStitcher and prepare it for processing.
* Step 2: compute pairwise stacks displacements.
* Step 3: project existing displacements along Z axis for each stack by selecting the most reliable one.
* Step 4: threshold displacements using the given reliability threshold.
* Step 5: places tiles in the image space using a globally optimal tiles placement algorithm.
* Step 6: merges tiles at different resolutions.

# Features:
1. Works for different objective.
2. Works for both TIFF or RAW data.
3. VoxelSizeX can be different from VoxelSizeY.
4. Can stitch multi channels data to a single IMS.
5. Support cache/scratch drives to reduce network overhead.
6. Can manually choose resolution.
7. Log information.
8. multi-process stitching.
9. GPU acceleration in Linux.
10. isometric voxels.
11. 16-bit to 8-bit conversion (in development).
