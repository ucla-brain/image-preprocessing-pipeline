# Image Preprocessing Pipeline
Python code for stitching and image enhancement of Light Sheet data

# Installation:
* Install [Imaris Viewer](https://viewer.imaris.com/download/ImarisViewer9_9_1w64.exe) (on Linux use [wine](https://vitux.com/how-to-install-wine-on-ubuntu/)).
* Install 64 bit version of [Microsoft Visual C++ 2010 Service Pack 1 Redistributable Package](https://www.microsoft.com/en-us/download/confirmation.aspx?id=26999), specifically in Wine.
* On Linux make sure Java server (e.g., [openjdk](https://openjdk.java.net/install/)), and [Nvidia drivers and CUDA >10.1](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation) are installed.
* Install [anaconda python distribution](https://www.anaconda.com/products/individual):
  make a dedicated python environment for stitching:

   `conda create -n stitching -c conda-forge python=3.10 psutil tqdm tifffile numpy scipy scikit-image scikit-learn matplotlib pyqt pandas imagecodecs git mpi4py hdf5plugin h5py`
   
   `conda activate stitching`
    
   `pip install dcimg`

   `pip install cpufeature`
   
   **Note:** The last time I checked, Microsoft MPI installed fom conda-forge was functional. However, if mpi4py was not functional on Windows, try installing the latest [Microsoft MPI from GitHub](https://github.com/microsoft/Microsoft-MPI).
   
   **Note:** If installing anaconda as user did not work on Windows, you may test installing it as the system python.

* clone image processing pipeline:

  `git clone https://github.com/ucla-brain/image-preprocessing-pipeline.git`

  `cd image-preprocessing-pipeline`

* Make sure the location of packages are set correctly in the python script `process_images.py`:

## Computer and Microscope Specific Configurations:

You may edit `process_images.py` file to enable or disable some functionalities or set environment variables.

* To enable GPU acceleration set `os.environ["USECUDA_X_NCC"] = "1"` (Linux Only). For more information about GPU accelerated stitching go to https://github.com/abria/TeraStitcher/wiki/Multi-GPU-parallelization-using-CUDA.

* need to set libjvm.so path:

   `os.environ["LD_LIBRARY_PATH"] = "/usr/lib/jvm/java-11-openjdk-amd64/lib/server/"` (Linux)

* need to set path for CUDA Libraries: (Linux Only)

   `os.environ["CUDA_ROOT_DIR"] = "/usr/local/cuda-11.6/"` (update the cuda version as needed)

* need to set visible GPU device on a multiGPU machine, which starts from zero.

   `os.environ["CUDA_VISIBLE_DEVICES"] = "1"`

* default channel folder names and voxel dimensions are set up like (`channel folder name`, `rgb color`):

   `AllChannels = [("Ex_488_Em_525", "b"), ("Ex_561_Em_600", "g"), ("Ex_642_Em_680", "r")]`

   `VoxelSizeX_4x, VoxelSizeY_4x = 1.835, 1.835`

   `VoxelSizeX_10x, VoxelSizeY_10x = 0.6, 0.6`

   `VoxelSizeX_15x, VoxelSizeY_15x = 0.422, 0.422`

   Please change them as needed. Different microscopes have different naming for channels. The rgb color code will be used only if you choose to merge channel colors into RGB format. If you do that rgb colored 2D tiff series and the final imaris file will be generated. Voxel sizes have impact on down-sampling, stitching and imaris file.

# Usage:
* activate stitching environment in anaconda: `conda activate stitching`.
* From inside the `image-preprocessing-pipeline` folder run: `python process_images.py /path/to/image/folder`
* answer the questions wait for the results.

# Functions and steps:
* Step 1: Inspect all channels for missing files and get a list of all files.
* Step 2: generate flat image (optional).
* Step 3: convert raw files to tif using DeStripe (optional read DeStripe section bellow).
* Step 4: align tiles using ParaStitcher.
* Step 5: merge tiles to 2D tif series using TSV.
* Step 6: convert 2D tif series to TeraFly format using paraconverter (optional).
* Step 7: down-sample in xyz for the registration of brain to common brain coordinates (optional).

Repeat step 2-5 for each channel separately.
* Step 8: merge channels to multichannel RGB colored 2D tif series (optional).
* Step 9: convert 2D tif series to imaris format using ImarisConvertiv.exe (optional).

## DeStripe
We modified original DeStripe code to add the following functionalities:
* Inspect corrupted raw or tif files and replace them with a blank image (zeros).
* Timeout function which prevents program halting encountering corrupt raw files.
* Down-sampling in XY to produce isotropic images.
* Improved parallel processing model to be faster and more scalable. For example, using more than 61 CPU processes in Windows is possible now.
* Fixed a bug regarding dark leveling.
* conversion of 16bit images to 8bit and right bit shifting.
* Gaussian filter.

You need to use PyStripe step if you need: 
* Raw to tif conversion.
* dark level subtraction that sets all values smaller than dark level threshold to zero to remove camera noise, clears background and allows more compressible images,
* flat image application to correct camera artifacts,
* lightsheet cleaning that clears the background and helps reduce final file sizes up 5-folds factor, 
* down-sampling for isotropic voxel generation that improves automated neuronal reconstruction and allows you estimate neuronal diameters,
* 8-bit conversion and right bit shifting that reduces file sizes, memory requirement during alignment stage and provides brighter pixels for neuronal reconstruction, or
* compressed 2D tif tiles.
* Gaussian filter.

## TeraStitcher steps:
We patched terastitcher and teraconverter so that they can read data from mounted NAS drives and send a message if failed to do so.
* Step 1: import a volume into TeraStitcher and prepare it for processing.
* Step 2: compute pairwise stacks displacements (alignment).
* Step 3: project existing displacements along Z axis for each stack by selecting the most reliable one.
* Step 4: threshold displacements using the given reliability threshold.
* Step 5: places tiles in the image space using a globally optimal tiles placement algorithm.

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
10. Flat image subtraction and proper DeStripe settings as suggested in www.biorxiv.org/content/10.1101/576595
11. Isotropic voxels.
12. 16-bit to 8-bit conversion.
13. Generate training data for flat non-flat images.
14. Use machine learning to find flat images with 98.5% accuracy.
15. Generate flat images for each channel.
16. Deconvolution (still researching).
17. Display progress bar.
18. CLI interface.
19. Selecting colors for different channels.
20. Resume support.

# Data flow and parallel processing design
![image](https://user-images.githubusercontent.com/18602635/157983557-9511a194-2cd6-4d54-8c50-f2956609ebe8.png)
![image](https://user-images.githubusercontent.com/18602635/157983654-061ab655-8545-4857-9405-31e11e0b95f5.png)


# Practical stitching guide
In this section, I explain the workflow in our lab.

If your lightsheet microscope generates maximum intensity projection (MIP) images, we suggest you stitch those images, first. The stitched MIP image allows you to choose correct dark levels and right bitshift values. For example, in the main folder of our images, we have a set of folders that end with `*_MIP`: `Ex_488_Em_525_MIP`, `Ex_561_Em_600_MIP`, and `Ex_642_Em_680_MIP`.
* Stitch the `MIP` image: for Linux, `python process_images.py /path/to/image` or for Windows `python process_images.py X:\path\to\image`.
* First, select the CPU architecture of your stitching computer: `SSE2`, `AVX`, `AVX2`, `AVX512`. At the moment, a relatively new AMD CPU supports up to AVX2, and Intel CPUs support AVX512. For older computers, you may choose AVX, and for ancient computers `SSE2`.
* Choose objective. For example 15x.
* Input the z-step size. Our microscope generates MIP images for every 600 um z-steps.
* Set the tile overlap percentage, which is a setting in microscope software (for example 10).
* For question `Do you need to stitch the MIP image first?` --> answer `Yes`.
* For questions about lightsheet cleaning, raw format, 8-bit conversion, down-sampling, and tif compression choose `NO`, whenever you stitch MIP images.
* Enter the location where you want to save the stitched files (destination). You need to enter the parent folder path (for example, `Y:\path\to`), or you may press enter to choose the default suggested path. We suggest you use a different drive as destination to optimize read and write. For example, stitching to/from a NAS, we use NFS mount for reading, and SMB mount as destination. On a local disk, you may read from `C:\` and write on `D:\`.
* convert to TeraFly --> `No`
* merge to RGB --> `Yes`
* covert to Imaris --> `Yes`

Wait for the program to finish stitching. In the `*_stitched` folder, you can find a `merged_channels_MIP_tif.ims` file. Open it in Imaris Viewer. Press `CTRL + D` to see `Display Adjustment` dialog. Press `advanced` button. Click on a channel, then try to find the largest `MIN` value and the smallest `MAX` value with which your images look perfectly visible. For the `MAX` value try only these numbers: 256, 512, 1024, 2048, 4096, 8192, 16384, 32768 and 65536. The `MIN` value is a number that depends on microscope camera. For our camera, 100 to 120 could be a good value. However, sometimes the signal-to-noise ratio of the image is not good and smaller values might be used. Write the `MIN` and `MAX` values for each channel. 

After setting DarkThreshold values, start stitching the main image. For example,
* `python process_images.py /path/to/image`
* `CPU` --> AVX2
* `Objective` --> 15x
* `z-step` --> 1. Our images have 15x objective with 1 um z-step.
* `tile overlap` --> 10
* `lightsheet cleaning` --> Yes. You may choose all channels or one of the channels. Lightsheet cleaning algorithm is computationally expensive, but I usually choose all channels to reduce the final image size.
* `RAW` --> Yes
* `16-bit to 8-bit conversion` --> Yes. 8-bit images align better, use less memory during alignment stage of stitching (8-GB/thread compared with 16-GB/thread), and compression algorithms can compress them more efficiently.
* `bit shift` --> Considering the `MAX` values of each channel (that we found by inspecting the MIP image in the previous step), choose the correct right bit-shift value.
* Downsampling for isotropic voxel generation --> Yes. This downsampling happens only in xy-plane, and the goal is to increase x and y voxel sizes so that they match the z-step. Isotropic images are better for (manual or automated) neuronal reconstruction and estimation of dendritic diameter.
* `Dark` --> enter the `MIN` values of each channel (that we found by inspecting the MIP image in the previous step).
* `Gaussian filter` --> We recommend saying `Yes` since it reduces stitching artifacts and final file size.
* compress tif --> Yes or No. If you have a slow cache drive, if you want to permanently store processed tif files, or you have a limited disk space, choose YES. Otherwise, choose NO to reduce CPU overhead. This applies only to PyStripe stage. 2D stitched tif series will be compressed no mater what you choose here.

By answering YES to any of the above questions, you enable PyStripe module. You need to enter destination path for processed tif files. This can be a fast cache/scratch drive. You may delete processed tif files after stitching was done.
* `destination path for lightsheet cleaning tif down-sampling files` --> For example, let us assume `D:\` is your cache/scratch drive.
* `destination path for stitched files` --> You may just press enter this time since stitching happens from cache/scratch drive to the original location. You may also choose any other location.
* `convert a channel to TeraFly format` --> Yes.
* select a channel or multiple channels you wish to visualize/reconstruct in Vaa3D/TeraFly.
* merge channels to RGB color tiff --> If you say NO, a set of stitched 2D grayscale tif series will be generated for each channel. Each channel will be converted to Imaris format if you ask for Imaris conversion. If you say YES, stitched 2D grayscale tif series will be merged into stitched 2D RGB color tif series, and RGB tif will be converted to Imaris.
* `Convert to Imaris` --> Yes.

Stitching starts by inspecting your images for missing tiles. If you saw a warning in this regard, you may stop stitching to correct the issue. For example, there might be file transfer/conversion issues that you can fix. As a last resort, you may choose to duplicate neighboring tiles of the missing tiles and give them an appropriate name.


# Compiling terastitcher (optional)
For CUDA 11.6 and newer GPUs, in addition to [original compilation documentation instructions](https://github.com/abria/TeraStitcher/wiki/Get-and-build-source-code) do the following:

* `export HDF5_DIR=/path/to/mcp3d/src/3rd_party/hdf5`

* edit `/path/to/TeraStitcher/src/crossmips/CMakeLists.txt` and change `-arch sm30` to `sm60`

* configure in command line:

   ```
   cmake ../src/ -DWITH_UTILITY_MODULE_mergedisplacements:BOOL="1" -DWITH_CUDA:BOOL="0" -DWITH_UTILITY_MODULE_terastitcher2:BOOL="1" -DWITH_HDF5:BOOL="0" -DWITH_IO_PLUGIN_IMS_HDF5:BOOL="0" -DWITH_UTILITY_MODULE_example:BOOL="1" -DWITH_IO_PLUGIN_bioformats2D:BOOL="0" -DWITH_UTILITY_MODULE_pyscripts:BOOL="0" -DWITH_UTILITY_MODULE_subvolextractor:BOOL="1" -DWITH_NEW_MERGE:BOOL="1" -DWITH_UTILITY_MODULE_teraconverter:BOOL="1" -DWITH_IO_PLUGIN_exampleplugin2D:BOOL="1" -DWITH_HDF5:BOOL="0" -DWITH_UTILITY_MODULE_virtualvolume:BOOL="1" -DWITH_RESUME_STATUS:BOOL="1" -DWITH_UTILITY_MODULE_mdatagenerator:BOOL="1" -DCMAKE_INSTALL_PREFIX="/home/brain/TeraStitcher/install" -DCMAKE_C_FLAGS="-Ofast -march=native -fomit-frame-pointer -mfpmath=both -pipe -fPIC -frecord-gcc-switches -flto" -DCMAKE_CXX_FLAGS="-Ofast -march=native -fomit-frame-pointer -mfpmath=both -pipe -fPIC -frecord-gcc-switches -flto"
   ```
