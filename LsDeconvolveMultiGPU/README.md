# LsDeconvolveMultiGPU

**Deconvolution of Light Sheet Microscopy Stacks with Multi-GPU Support**

[![View on GitHub](https://img.shields.io/badge/GitHub-View%20Project-blue?logo=github)](https://github.com/ucla-brain/image-preprocessing-pipeline/tree/main/LsDeconvolveMultiGPU)

---

## Overview

This project provides a powerful, multi-GPU-capable implementation of light sheet deconvolution for microscopy image stacks. Originally developed at TU Wien in MATLAB 2018b, it has been significantly extended and maintained by Keivan Moradi at UCLA B.R.A.I.N (Dong Lab) using MATLAB 2025a.

It supports large-scale image data processing using GPU acceleration, automatic resuming of incomplete runs, destriping filters, and custom 3D Gaussian pre-filtering.

---

## Features

- ✅ Multi-GPU deconvolution  
- ✅ Resume incomplete deconvolution jobs  
- ✅ 3D Gaussian pre-filtering  
- ✅ Z-axis destriping  
- ✅ Fully scriptable with Python CLI wrapper  
- ✅ Parallel processing and speed optimizations

---
## Optimizations
| Module Name         | Module Function            | Language     | Optimization Method                             | Speedup vs MATLAB               |
|---------------------|----------------------------|--------------|-------------------------------------------------|---------------------------------|
| load_bl_tif         | 2D tiff to 3D block loader | C++17        | Queue–Based Multithreading                      | 2.5x to 5x                      |
| save_lz4            | Caching: 3D block save     | C            | Lz4 compression                                 | >50x                            |
| load_lz4            | Caching: 3D block load     | C            | Lz4 decompression                               | >4x                             |
| semaphore           | Semaphore                  | C            | Multi-GPU processing                            | #GPUx → 2x–8x in our lab        |
| gauss3d             | 3D Gaussian filter         | C/CUDA       | GPU acceleration / no extra padding             | 1.5x – 50x                      |
| edgetaper_3d        | Edge taper                 | C/CUDA       | GPU acceleration                                | 3x – 4x \| fixes edge artifacts |
| otf_gpu_mex         | OTF calculator             | C/CUDA       | GPU acceleration / bug fix                      | 2x MATLAB GPU                   |
| deconFFT            | FFT based deconvolution    | MATLAB       | GPU acceleration                                | 2x – 3x Spatial Method          |
| filter_subband_3d_z | Destriping                 | MATLAB       | GPU acceleration                                | 8x                              |
| decwrap             | Python Wrapper             | Python       | Optimal block size calculation                  | Processes larger blocks         |
| postproceses_save   | 8bit ↔16bit conversion     | MATLAB       | 8bit → Float32 → Decon → 16bit                  | High Quality 16bit              |

---

## Noteworthy Implementation Details

### 🔹 Dynamic GPU Memory Management
The `decwrap.py` script dynamically computes `block_size_max` based on available GPU VRAM. This prevents memory overflows and allows the code to adapt efficiently across various GPU hardware setups.

### 🔹 Seamless MATLAB-Python Integration
The system integrates MATLAB for GPU-based deconvolution and Python for parameter control and orchestration, combining MATLAB’s numerical power with Python’s flexibility.

### 🔹 Robust Error Handling
Key scripts check file paths, validate image data, and provide meaningful error messages, which is critical for large-scale batch processing.

---

## Requirements

- MATLAB 2025a (tested)
- CUDA-compatible NVIDIA GPUs (with at least 12 GB vRAM recommended)
- `nvidia-smi` available in system path (for Python wrapper)
- MATLAB Parallel Computing, Image Processing, and Wavelet Toolboxe

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ucla-brain/image-preprocessing-pipeline.git
   cd image-preprocessing-pipeline/LsDeconvolveMultiGPU
   ```

2. **Compile the MEX file**

   In MATLAB:
   ```matlab
   run('build_mex.m')
   ```
   This will compile the MEX files needed for interprocess communication.

3. Install numactl:

    From command line:
   ```bash
   sudo apt install numactl
   ```
---

## ⚠️ Windows CUDA/Visual Studio Compatibility

To compile CUDA MEX files with MATLAB on Windows, you must have a **supported Microsoft Visual C++ (MSVC) toolset** installed.

### Quick Steps

1. **Open Visual Studio Installer**  
   - Click **Modify** for your Visual Studio 2022 installation.  
   - Under **Individual Components**, install **MSVC v14.39** (or any version ≤ 14.39).

2. **No manual path setup required**  
   - This repository provides a pre-configured `nvcc_msvcpp2022.xml` options file, which automatically directs MATLAB/nvcc to the correct MSVC toolset.

3. **Build your CUDA MEX files as usual using this script.**  
   - The build script uses the custom XML to select the proper compiler.

### Why is this necessary?

- CUDA does **not support the very latest MSVC versions** until NVIDIA releases a CUDA update.  
- The provided XML ensures compatibility, but you must still have a *supported* MSVC toolset installed.

### Reference

- See [NVIDIA’s CUDA on Windows Compatibility](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#system-requirements) for details on supported MSVC versions.

---

## Python Wrapper

The `decwrap.py` script provides a convenient Python interface for running the MATLAB-based deconvolution pipeline directly from the command line. **Using this Python wrapper is strongly recommended over launching MATLAB manually**, as it includes critical features such as:

- Automatic calculation of optimal block sizes based on available system memory and image dimensions.
- Intelligent handling of multi-socket systems to ensure NUMA-aware execution and efficient CPU-GPU data movement.
- Streamlined integration with shell scripts or batch processing environments.

This ensures more reliable performance and reproducibility, especially on high-performance computing setups.


### Example:
```bash
python decwrap.py -i /mnt/data/2D_tif_series --dxy 0.4 --dz 1.2 --lambda-ex 561 --lambda_em 600 --gpu-indices 1 2 --gpu-workers-per-gpu 4 --resume --gaussian-sigma 0.5 0.5 1.5 --gaussian-filter-size 5 5 15
```

### Key Features of the Wrapper:
- Auto-detects available GPUs
- Dynamically computes `block_size_max` based on GPU VRAM
- Supports CPU fallback
- Compatible with both Windows and Linux
- Provides `--dry-run` option for debugging command setup

---

## Usage Notes

- The code has been tested on:
  > HGX server with 256 cores, 4 TB RAM, and 8x NVIDIA Tesla A100 GPUs

- Supported input format: **2D `.tif` series**

- After completion, results are saved in a subfolder named `deconvolved` inside the input folder.

- **Important GPU Memory Notes:**
  - Restart MATLAB between runs if GPU runs out of memory.
  - Some files may remain in GPU memory unless MATLAB is restarted.
  - Use `nvidia-smi` to monitor GPU memory usage.

---
## 🧠 System Configuration for Optimal Memory Management (Linux)

**LsDeconv** benefits significantly from a well-tuned Linux memory management setup, 
particularly on systems with large amounts of RAM and high I/O throughput.

### 🔧 Why This Matters

- **Large memory systems** (e.g., ≥ 256 GB RAM) can suffer from fragmentation that prevents large memory block allocations.
- **GPU allocations**, **transparent hugepages (THP)**, and **CUDA workloads** require large contiguous physical memory regions.
- **Proactive memory compaction** is essential and **requires swap to be enabled**, even if the system rarely uses it.

---

### ✅ Enable Continuous Memory Compaction

You can manually trigger memory compaction once using:

```bash
echo 1 | sudo tee /proc/sys/vm/compact_memory
```

To keep compaction active in the background, set:

```conf
# Enable proactive compaction (0–100)
vm.compaction_proactiveness = 80

# Start compacting when fragmentation index falls below this (0–1000)
vm.extfrag_threshold = 300
```

---

### 🛠 Recommended `sysctl.conf` Settings

Add these settings to a file such as `/etc/sysctl.d/99-lsdeconv.conf` and apply them using:

```bash
sudo sysctl --system
```

---

#### 🔄 Dual Socket System (4 TB RAM, Enterprise SSDs)

```conf
# Aggressive inode/dentry cache pruning
vm.vfs_cache_pressure = 200

# Lower latency writeback for dirty pages
vm.dirty_writeback_centisecs = 10
vm.dirty_expire_centisecs = 500
vm.dirty_ratio = 5
vm.dirty_background_ratio = 3

# Improve memory compaction for large page allocation
vm.compaction_proactiveness = 80
vm.extfrag_threshold = 300
```

---

#### 💻 Single Socket System (512 GB RAM, User-grade SSDs)

```conf
# Moderate cache retention for mid-range systems
vm.vfs_cache_pressure = 125

# Balanced I/O writeback for consumer SSDs
vm.dirty_writeback_centisecs = 100
vm.dirty_expire_centisecs = 1000
vm.dirty_ratio = 4
vm.dirty_background_ratio = 2

# Improve memory compaction for large page allocation
vm.compaction_proactiveness = 80
vm.extfrag_threshold = 300
```

---

### 💾 Swap is Required

Linux requires swap to perform memory compaction effectively, even if you don’t expect to use it.

**To check if swap is enabled:**

```bash
swapon --show
```

**To create a 4 GB swapfile:**

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**To make it persistent:**

```bash
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

These settings will help prevent memory allocation failures during GPU-intensive processing and ensure more reliable performance for large dataset deconvolution.

# Notes on FFT-Based Deconvolution

## Performance vs Memory Tradeoff

LsDeconvMultiGPU supports both **FFT-based** and **spatial domain** deconvolution. Each method has advantages and tradeoffs:

- **FFT-based deconvolution** is significantly faster, but it requires **approximately 4× more VRAM**. This limits the maximum block size that can be processed at once.
- **Spatial domain deconvolution**, while more memory-efficient, is slower and computationally more expensive when calculating convolutions directly.



## ⚙️ Performance Comparison: FFT vs Spatial Deconvolution

Deconvolution of a 3D volume with **8266 × 12778 × 7912 = 835,688,764,576 voxels**


| GPUs | vRAM  | Deconvolution Method   | Destriping | Elapsed Time | Speedup |
|------|-------|------------------------|------------|--------------|---------|
| 2    | 11 GB | Spatial Domain         | No         | 13h:32m      |         |
| 2    | 11 GB | Frequency Domain (FFT) | No         | 05h:05m      | 2.3x    |

## Licensing and Attribution

```
Program for Deconvolution of Light Sheet Microscopy Stacks.

Initial Copyright TU-Wien 2019, Klaus Becker (klaus.becker@tuwien.ac.at)

Modified by Keivan Moradi, Hongwei Dong Lab (B.R.A.I.N) at UCLA
Contact: kmoradi@mednet.ucla.edu

Main Modifications:
  * Parallel processing
  * Multi-GPU support
  * Resume support
  * 3D Gaussian filtering
  * Z-axis destriping
  * Speed enhancement

LsDeconv is free software.
You can redistribute it and/or modify it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This software is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

See the GNU General Public License for more details.
If not, see <http://www.gnu.org/licenses/>.
```

---

## Citation

If you use this software in your research, please cite the following papers.

---

## Reference

- Becker, K., Saghafi, S., Pende, M., Sabdyusheva-Litschauer, I., Hahn, C. M., Foroughipour, M., Jährling, N., & Dodt, H.-U. (2019). *Deconvolution of light sheet microscopy recordings*. Scientific Reports, **9**(1), 17625. [https://doi.org/10.1038/s41598-019-53875-y](https://doi.org/10.1038/s41598-019-53875-y)

- Marrett, K., Moradi, K., Park, C. S., Yan, M., Choi, C., Zhu, M., Akram, M., Nanda, S., Xue, Q., Mun, H.-S., Gutierrez, A. E., Rudd, M., Zingg, B., Magat, G., Wijaya, K., Dong, H., Yang, X. W., & Cong, J. (2024). *Gossamer: Scaling Image Processing and Reconstruction to Whole Brains*. bioRxiv. [View article](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=Ypb3C2gAAAAJ&sortby=pubdate&citation_for_view=Ypb3C2gAAAAJ:Y5dfb0dijaUC)

