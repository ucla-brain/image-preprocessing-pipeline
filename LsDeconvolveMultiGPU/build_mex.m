% ===============================
% build_mex.m
% ===============================
% Compile semaphore, LZ4, and GPU MEX files using Anaconda's libtiff.
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

% Use Anaconda-provided libtiff for all platforms, or build if missing
conda_prefix = getenv('CONDA_PREFIX');
assert(~isempty(conda_prefix) && isfolder(conda_prefix), ...
    'CONDA_PREFIX is not set or does not point to a valid directory.');

tiff_include = {['-I', fullfile(conda_prefix, 'include')]} ;
tiff_lib     = {['-L', fullfile(conda_prefix, 'lib')]} ;

% --- Platform-specific linking and flags ---
if ispc
    tiff_libfile = fullfile(conda_prefix, 'lib', 'libtiff.lib');
    if ~isfile(tiff_libfile)
        % Auto-build libtiff from source if missing
        fprintf('libtiff.lib not found. Downloading and building libtiff from source...\n');
        system('curl -L -o tiff.zip https://download.osgeo.org/libtiff/tiff-4.6.0.zip');
        unzip('tiff.zip', 'tiff_src');
        delete('tiff.zip');
        cd(fullfile('tiff_src', 'tiff-4.6.0'));

        % Configure and build with Visual Studio
        build_cmd = [
            'cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="', conda_prefix, '" . && ', ...
            'cmake --build build --config Release --target install'
        ];
        status = system(build_cmd);
        cd ../..;

        if status ~= 0
            error('Failed to build libtiff from source on Windows.');
        end
    end
    tiff_link = {'-ltiff'};

    if debug
        mex_flags_cpu = {'-R2018a', 'COMPFLAGS="$COMPFLAGS /Od /Zi /openmp"'};
    else
        mex_flags_cpu = {'-R2018a', 'COMPFLAGS="$COMPFLAGS /O2 /arch:AVX2 /openmp"'};
    end
else
    % On Linux/macOS
    tiff_libfile = fullfile(conda_prefix, 'lib', 'libtiff.so');
    if ~isfile(tiff_libfile)
        fprintf('libtiff.so not found. Downloading and building libtiff from source...\n');
        system('curl -L -o tiff.tar.gz https://download.osgeo.org/libtiff/tiff-4.6.0.tar.gz');
        system('tar -xzf tiff.tar.gz');
        delete('tiff.tar.gz');
        cd('tiff-4.6.0');
        configure_cmd = ['./configure --prefix=', conda_prefix, ' && make && make install'];
        status = system(configure_cmd);
        cd('..');

        if status ~= 0
            error('Failed to build libtiff from source on Linux.');
        end
    end
    tiff_link = {'-ltiff'};

    if debug
        mex_flags_cpu = {'-R2018a', 'CFLAGS="$CFLAGS -O0 -g -fopenmp"', ...
                         'CXXFLAGS="$CXXFLAGS -O0 -g -fopenmp"'};
    else
        mex_flags_cpu = {'-R2018a', ...
            'CFLAGS="$CFLAGS -O3 -march=native -fomit-frame-pointer -fopenmp"', ...
            'CXXFLAGS="$CXXFLAGS -O3 -march=native -fomit-frame-pointer -fopenmp"'};
    end
end

fprintf('Using Anaconda libtiff from: %s\n', conda_prefix);

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
