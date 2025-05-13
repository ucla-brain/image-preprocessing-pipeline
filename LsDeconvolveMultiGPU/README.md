# LsDeconvolveMultiGPU

**Deconvolution of Light Sheet Microscopy Stacks with Multi-GPU Support**

[![View on GitHub](https://img.shields.io/badge/GitHub-View%20Project-blue?logo=github)](https://github.com/ucla-brain/image-preprocessing-pipeline/tree/main/LsDeconvolveMultiGPU)

---

## Overview

This project provides a powerful, multi-GPU-capable implementation of light sheet deconvolution for microscopy image stacks. Originally developed at TU Wien in MATLAB 2018b, it has been significantly extended and maintained by Keivan Moradi at UCLA B.R.A.I.N (Dong Lab) using MATLAB 2023a.

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

## Noteworthy Implementation Details

### 🔹 Dynamic GPU Memory Management
The `decwrap.py` script dynamically computes `block_size_max` based on available GPU VRAM. This prevents memory overflows and allows the code to adapt efficiently across various GPU hardware setups.

### 🔹 Seamless MATLAB-Python Integration
The system integrates MATLAB for GPU-based deconvolution and Python for parameter control and orchestration, combining MATLAB’s numerical power with Python’s flexibility.

### 🔹 Robust Error Handling
Key scripts check file paths, validate image data, and provide meaningful error messages, which is critical for large-scale batch processing.

---

## Requirements

- MATLAB 2023a (tested)
- CUDA-compatible NVIDIA GPUs (with at least 16 GB VRAM recommended)
- `nvidia-smi` available in system path (for Python wrapper)
- MATLAB Parallel Computing Toolbox

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

   This will compile the CUDA-accelerated MEX files needed for GPU execution.

---

## Python Wrapper

The `decwrap.py` script allows easy execution of MATLAB-based deconvolution from the command line using Python.

### Example:
```bash
python decwrap.py   -i /mnt/data/my_sample_stack   --dxy 0.4 --dz 1.2   --lambda_ex 561 --lambda_em 600   --gpu-indices 1 2 --gpu-workers-per-gpu 4   --resume --sigma 0.5 0.5 1.5 --filter_size 5 5 15
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

- For very large samples requiring block splitting, border artifacts may appear (⚠️ to-do).

---

## Suggested Enhancements

1. **Detailed Installation Instructions**  
   Expand on MATLAB and Python environment setup, including dependencies, CUDA toolkit versions, and `nvidia-smi` usage.

2. **Usage Examples with Sample Data**  
   Provide links to or include sample datasets to help users test the full pipeline end-to-end.

3. **Visualization of Workflow**  
   Add a flowchart or schematic to illustrate how different components (MATLAB scripts, MEX files, Python wrapper) interact.

---

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

If you use this software in your research, please cite the original TU Wien implementation and acknowledge modifications by UCLA B.R.A.I.N.

---

## Reference

- Becker, K., Saghafi, S., Pende, M., Sabdyusheva-Litschauer, I., Hahn, C. M., Foroughipour, M., Jährling, N., & Dodt, H.-U. (2019). *Deconvolution of light sheet microscopy recordings*. Scientific Reports, **9**(1), 17625. [https://doi.org/10.1038/s41598-019-53875-y](https://doi.org/10.1038/s41598-019-53875-y)

- Marrett, K., Moradi, K., Park, C. S., Yan, M., Choi, C., Zhu, M., Akram, M., Nanda, S., Xue, Q., Mun, H.-S., Gutierrez, A. E., Rudd, M., Zingg, B., Magat, G., Wijaya, K., Dong, H., Yang, X. W., & Cong, J. (2024). *Gossamer: Scaling Image Processing and Reconstruction to Whole Brains*. bioRxiv. [View article](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=Ypb3C2gAAAAJ&sortby=pubdate&citation_for_view=Ypb3C2gAAAAJ:Y5dfb0dijaUC)

- Goodwin, B., Jones, S. A., Price, R. R., Watson, M. A., McKee, D. D., Moore, L. B., Galardi, C., Wilson, J. G., Lewis, M. C., Roth, M. E., Maloney, P. R., Willson, T. M., & Kliewer, S. A. (2000). *A regulatory cascade of the nuclear receptors FXR, SHP-1, and LRH-1 represses bile acid biosynthesis*. Molecular Cell, **6**(3), 517–526. [https://doi.org/10.1016/s1097-2765(00)00051-4](https://europepmc.org/abstract/MED/11030332)
