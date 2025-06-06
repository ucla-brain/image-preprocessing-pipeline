% ===============================
% build_mex.m
% ===============================
% Compile semaphore, LZ4, and GPU MEX files using Anaconda's libtiff.
% Requires MATLAB R2018a+ (-R2018a MEX API) and Anaconda (CONDA_PREFIX).

debug = false;

if verLessThan('matlab', '9.4')
    error('This script requires MATLAB R2018a or newer (for -R2018a MEX API)');
end

if exist('mexcuda', 'file') ~= 2
    error('mexcuda not found. Ensure CUDA is set up correctly.');
end

% Source files
src_semaphore = 'semaphore.c';
src_lz4_save  = 'save_lz4_mex.c';
src_lz4_load  = 'load_lz4_mex.c';
src_lz4_c     = 'lz4.c';
src_gauss3d   = 'gauss3d_mex.cu';
src_conv3d    = 'conv3d_mex.cu';
src_otf_gpu   = 'otf_gpu_mex.cu';
src_deconFFT  = 'deconFFT_mex.cu';
src_load_bl   = 'load_bl_tif.cpp';

% LZ4 download if missing
lz4_c_url = 'https://raw.githubusercontent.com/lz4/lz4/dev/lib/lz4.c';
lz4_h_url = 'https://raw.githubusercontent.com/lz4/lz4/dev/lib/lz4.h';

if ~isfile('lz4.c')
    fprintf('Downloading lz4.c ...\n');
    try, websave('lz4.c', lz4_c_url); catch, error('Failed to download lz4.c'); end
end
if ~isfile('lz4.h')
    fprintf('Downloading lz4.h ...\n');
    try, websave('lz4.h', lz4_h_url); catch, error('Failed to download lz4.h'); end
end

% MEX optimization flags (CPU)
mex_flags = {'-R2018a'};
if debug
    if ispc
        mex_flags_cpu = [mex_flags, {'COMPFLAGS="$COMPFLAGS /Od /Zi /openmp"'}];
    else
        mex_flags_cpu = [mex_flags, {'CFLAGS="$CFLAGS -O0 -g -fopenmp"', ...
                                     'CXXFLAGS="$CXXFLAGS -O0 -g -fopenmp"'}];
    end
else
    if ispc
        mex_flags_cpu = [mex_flags, {'COMPFLAGS="$COMPFLAGS /O2 /arch:AVX2 /openmp"'}];
    else
        mex_flags_cpu = [mex_flags, {'CFLAGS="$CFLAGS -O3 -march=native -fomit-frame-pointer -fopenmp"', ...
                                     'CXXFLAGS="$CXXFLAGS -O3 -march=native -fomit-frame-pointer -fopenmp"'}];
    end
end

% Use Anaconda-provided libtiff for all platforms
conda_prefix = getenv('CONDA_PREFIX');
assert(~isempty(conda_prefix) && isfolder(conda_prefix), ...
    'CONDA_PREFIX is not set or does not point to a valid directory.');

tiff_include = {['-I', fullfile(conda_prefix, 'include')]};
tiff_lib     = {['-L', fullfile(conda_prefix, 'lib')]};
tiff_link    = {'-ltiff'};

fprintf('Using Anaconda libtiff from: %s\n', conda_prefix);

% Build CPU MEX files
%mex(mex_flags_cpu{:}, src_semaphore);
%mex(mex_flags_cpu{:}, src_lz4_save, src_lz4_c);
%mex(mex_flags_cpu{:}, src_lz4_load, src_lz4_c);
mex(mex_flags_cpu{:}, src_load_bl, tiff_include{:}, tiff_lib{:}, tiff_link{:});

% CUDA optimization flags
if debug
    if ispc
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -G -std=c++17 -Xcompiler ""/Od,/Zi"" "';
    else
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -G -std=c++17 -Xcompiler ''-O0,-g'' "';
    end
else
    if ispc
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -O2 -std=c++17 -Xcompiler ""/O2,/arch:AVX2,/openmp"" "';
    else
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -O2 -std=c++17 -Xcompiler ''-O2,-march=native,-fomit-frame-pointer,-fopenmp'' "';
    end
end

% CUDA include dirs
root_dir = '.'; include_dir = './mex_incubator';

% Windows: use custom nvcc config
if ispc
    xmlfile = fullfile(fileparts(mfilename('fullpath')), 'nvcc_msvcpp2022.xml');
    assert(isfile(xmlfile), 'nvcc_msvcpp2022.xml not found!');
    cuda_mex_flags = {'-f', xmlfile};
else
    cuda_mex_flags = {};
end

% Build CUDA MEX files
%mexcuda(cuda_mex_flags{:}, mex_flags{:}, src_gauss3d , ['-I', root_dir], ['-I', include_dir], nvccflags);
%mexcuda(cuda_mex_flags{:}, mex_flags{:}, src_conv3d  , ['-I', root_dir], ['-I', include_dir], nvccflags);
%mexcuda(cuda_mex_flags{:}, mex_flags{:}, src_otf_gpu , ['-I', root_dir], ['-I', include_dir], nvccflags, '-L/usr/local/cuda/lib64', '-lcufft');
%mexcuda(cuda_mex_flags{:}, mex_flags{:}, src_deconFFT, ['-I', root_dir], ['-I', include_dir], nvccflags, '-L/usr/local/cuda/lib64', '-lcufft');

fprintf('All MEX files built successfully.\n');
