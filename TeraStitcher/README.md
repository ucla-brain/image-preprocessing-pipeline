TeraStitcher
===========================================================

A tool for fast automatic 3D-stitching of teravoxel-sized 
microscopy images (BMC Bioinformatics 2012, 13:316)

Exploiting multi-level parallelism for stitching very large 
microscopy images (Frontiers in Neuroinformatics, 13, 2019)

===========================================================

Before using this software, you MUST accept the LICENSE.txt

Documentation,  help and  other info  are available on  our 
GitHub wiki at http://abria.github.io/TeraStitcher/.

===========================================================

Contributors

- Alessandro Bria (email: a.bria@unicas.it).
  Post-doctoral Fellow at University of Cassino (Italy).
  Main developer.

- Giulio Iannello (email: g.iannello@unicampus.it).
  Full Professor at University Campus Bio-Medico of Rome (italy).
  Supervisor and co-developer.
  
===========================================================

Main features

- designed for images exceeding the TeraByte size
- fast and reliable 3D stitching based on a multi-MIP approach
- typical memory requirement below 4 GB (8 at most)
- 2D stitching (single slice images) supported
- regular expression based matching for image file names
- data subset selection
- sparse data support
- i/o plugin-based architecture
- stitching of multichannel images
- support for big tiff files (> 4 GB)
- HDF5-based formats
- parallelization on multi-core platform
- fast alignment computation on NVIDIA GPUs

===========================================================

GPU acceleration

```bash
mkdir build && cd build
```

`
cmake ../src/ -DWITH_UTILITY_MODULE_mergedisplacements:BOOL="1" -DWITH_CUDA:BOOL="1" -DWITH_UTILITY_MODULE_terastitcher2:BOOL="1" -DWITH_HDF5:BOOL="0" -DWITH_IO_PLUGIN_IMS_HDF5:BOOL="0" -DWITH_UTILITY_MODULE_example:BOOL="0" -DWITH_IO_PLUGIN_bioformats2D:BOOL="0" -DWITH_UTILITY_MODULE_pyscripts:BOOL="0" -DWITH_UTILITY_MODULE_subvolextractor:BOOL="1" -DWITH_NEW_MERGE:BOOL="1" -DWITH_UTILITY_MODULE_teraconverter:BOOL="1" -DWITH_UTILITY_MODULE_virtualvolume:BOOL="1" -DWITH_RESUME_STATUS:BOOL="1" -DWITH_UTILITY_MODULE_mdatagenerator:BOOL="1" -DCMAKE_INSTALL_PREFIX="../install" -DCMAKE_C_FLAGS="-Ofast -march=native -fomit-frame-pointer -mfpmath=both -pipe -fPIC -frecord-gcc-switches -flto -w" -DCMAKE_CXX_FLAGS="-Ofast -march=native -fomit-frame-pointer -mfpmath=both -pipe -fPIC -frecord-gcc-switches -flto -w" && make -j20
`