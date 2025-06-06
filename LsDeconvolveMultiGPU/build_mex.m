% ===============================
% build_mex.m
% ===============================
% Compile semaphore, LZ4, and GPU MEX files using Anaconda's libtiff or fallback to libtiff built from source.
% Requires MATLAB R2018a+ (-R2018a MEX API) and Anaconda (CONDA_PREFIX).
% On all platforms, will automatically download and compile libtiff from source
% if the required library is missing.

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

% Use Anaconda-provided libtiff or fallback to source build
conda_prefix = getenv('CONDA_PREFIX');
assert(~isempty(conda_prefix) && isfolder(conda_prefix), ...
    'CONDA_PREFIX is not set or does not point to a valid directory.');

libtiff_root = fullfile(pwd, 'tiff_src', 'tiff-4.6.0');

% --- Platform-specific linking and flags ---
if ispc
    built_lib = fullfile(conda_prefix, 'lib', 'libtiff.lib');
    tiff_header = fullfile(conda_prefix, 'include', 'tiffio.h');
    needs_build = ~(isfile(built_lib) && isfile(tiff_header));

    if needs_build
        fprintf('libtiff.lib or tiffio.h not found. Downloading and building libtiff from source...\n');
        if ~isfolder(libtiff_root)
            system('curl -L -o tiff.zip https://download.osgeo.org/libtiff/tiff-4.6.0.zip');
            unzip('tiff.zip', 'tiff_src');
            delete('tiff.zip');
        end
        cd(libtiff_root);
        status = system(['cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="', conda_prefix, '" . && ' ...
                         'cmake --build build --config Release --target install']);
        cd('../../');
        if status ~= 0
            error('Failed to build libtiff from source on Windows.');
        end
    else
        fprintf('Using existing libtiff.lib and headers from conda_prefix.\n');
    end

    % Use installed include dir to find tiffconf.h
    tiff_include = {['-I', fullfile(conda_prefix, 'include')]} ;
    tiff_lib     = {['-L', fullfile(conda_prefix, 'lib')]} ;
    tiff_link    = {'-ltiff'};

    if debug
        mex_flags_cpu = {'-R2018a', 'COMPFLAGS="$COMPFLAGS /Od /Zi /openmp"'};
    else
        mex_flags_cpu = {'-R2018a', 'COMPFLAGS="$COMPFLAGS /O2 /arch:AVX2 /openmp"'};
    end
else
    built_lib = fullfile(conda_prefix, 'lib', 'libtiff.so');
    tiff_header = fullfile(conda_prefix, 'include', 'tiffio.h');
    needs_build = ~(isfile(built_lib) && isfile(tiff_header));

    if needs_build
        fprintf('libtiff.so or tiffio.h not found. Downloading and building libtiff from source...\n');
        if ~isfolder('tiff-4.6.0')
            system('curl -L -o tiff.tar.gz https://download.osgeo.org/libtiff/tiff-4.6.0.tar.gz');
            system('tar -xzf tiff.tar.gz');
            delete('tiff.tar.gz');
        end
        cd('tiff-4.6.0');
        status = system(['./configure --prefix=', conda_prefix, ' && make && make install']);
        cd('..');
        if status ~= 0
            error('Failed to build libtiff from source on Linux.');
        end
    else
        fprintf('Using existing libtiff.so and headers from conda_prefix.\n');
    end

    % Use installed include dir to find tiffconf.h
    tiff_include = {['-I', fullfile(conda_prefix, 'include')]} ;
    tiff_lib     = {['-L', fullfile(conda_prefix, 'lib')]} ;
    tiff_link    = {'-ltiff'};

    if debug
        mex_flags_cpu = {'-R2018a', 'CFLAGS="$CFLAGS -O0 -g -fopenmp"', ...
                         'CXXFLAGS="$CXXFLAGS -O0 -g -fopenmp"'};
    else
        mex_flags_cpu = {'-R2018a', ...
            'CFLAGS="$CFLAGS -O3 -march=native -fomit-frame-pointer -fopenmp"', ...
            'CXXFLAGS="$CXXFLAGS -O3 -march=native -fomit-frame-pointer -fopenmp"'};
    end
end

fprintf('Using libtiff from: %s\n', fullfile(conda_prefix, 'lib'));

% Build CPU MEX files
% mex(mex_flags_cpu{:}, src_semaphore);
% mex(mex_flags_cpu{:}, src_lz4_save, src_lz4_c);
% mex(mex_flags_cpu{:}, src_lz4_load, src_lz4_c);
mex(mex_flags_cpu{:}, src_load_bl, tiff_include{:}, tiff_lib{:}, tiff_link{:});

% CUDA optimization flags
if ispc
    if debug
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -G -std=c++17 -Xcompiler ""/Od,/Zi"" "'; %#ok<NASGU>
    else
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -O2 -std=c++17 -Xcompiler ""/O2,/arch:AVX2,/openmp"" "'; %#ok<NASGU>
    end
else
    if debug
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -G -std=c++17 -Xcompiler ''-O0,-g'' "'; %#ok<NASGU>
    else
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -O2 -std=c++17 -Xcompiler ''-O2,-march=native,-fomit-frame-pointer,-fopenmp'' "'; %#ok<NASGU>
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
% mexcuda(cuda_mex_flags{:}, '-R2018a', src_gauss3d , ['-I', root_dir], ['-I', include_dir], nvccflags);
% mexcuda(cuda_mex_flags{:}, '-R2018a', src_conv3d  , ['-I', root_dir], ['-I', include_dir], nvccflags);
% mexcuda(cuda_mex_flags{:}, '-R2018a', src_otf_gpu , ['-I', root_dir], ['-I', include_dir], nvccflags, '-L/usr/local/cuda/lib64', '-lcufft');
% mexcuda(cuda_mex_flags{:}, '-R2018a', src_deconFFT, ['-I', root_dir], ['-I', include_dir], nvccflags, '-L/usr/local/cuda/lib64', '-lcufft');

fprintf('All MEX files built successfully.\n');
