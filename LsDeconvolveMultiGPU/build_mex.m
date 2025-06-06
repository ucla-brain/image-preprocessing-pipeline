% ===============================
% build_mex.m
% ===============================
% Compile semaphore, LZ4, and GPU MEX files using Anaconda's libtiff.
% Requires MATLAB R2018a+ (-R2018a MEX API) and Anaconda (CONDA_PREFIX).
% On Windows or Linux, if libtiff is missing, downloads and compiles from source.

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

% Use Anaconda-provided libtiff for all platforms if available
conda_prefix = getenv('CONDA_PREFIX');
assert(~isempty(conda_prefix) && isfolder(conda_prefix), ...
    'CONDA_PREFIX is not set or does not point to a valid directory.');

tiff_include = {['-I', fullfile(conda_prefix, 'include')]} ;
tiff_lib     = {['-L', fullfile(conda_prefix, 'lib')]} ;

tiff_found = false;
tiff_link = {};

if ispc
    libfile = fullfile(conda_prefix, 'lib', 'libtiff.lib');
    if isfile(libfile)
        tiff_link = {libfile};
        tiff_found = true;
    end
else
    libfile = fullfile(conda_prefix, 'lib', 'libtiff.so');
    if isfile(libfile)
        tiff_link = {'-ltiff'};
        tiff_found = true;
    end
end

% If libtiff was not found, try building it from source
if ~tiff_found
    fprintf('libtiff not found, attempting to build from source ...\n');
    tiff_url = 'http://download.osgeo.org/libtiff/tiff-4.6.0.tar.gz';
    tiff_tar = 'tiff-4.6.0.tar.gz';
    tiff_dir = 'tiff-4.6.0';

    try
        if ~isfolder(tiff_dir)
            if ~isfile(tiff_tar)
                fprintf('Downloading libtiff source ...\n');
                websave(tiff_tar, tiff_url);
            end
            untar(tiff_tar);
        end

        % Build libtiff using system compiler
        old_dir = pwd;
        cd(tiff_dir);
        if ispc
            configure_cmd = 'cmake -Bbuild -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON';
            build_cmd = 'cmake --build build --config Release';
        else
            configure_cmd = './configure --prefix=install';
            build_cmd = 'make && make install';
        end
        fprintf('Running: %s\n', configure_cmd);
        assert(system(configure_cmd) == 0, 'Configuration failed.');
        fprintf('Running: %s\n', build_cmd);
        assert(system(build_cmd) == 0, 'Build failed.');
        cd(old_dir);

        % Update include/lib paths after build
        if ispc
            tiff_include = {['-I', fullfile(tiff_dir, 'build')]};
            tiff_lib = {['-L', fullfile(tiff_dir, 'build', 'Release')]};
            tiff_link = {fullfile(tiff_dir, 'build', 'Release', 'libtiff.lib')};
        else
            tiff_include = {['-I', fullfile(tiff_dir, 'install', 'include')]};
            tiff_lib = {['-L', fullfile(tiff_dir, 'install', 'lib')]};
            tiff_link = {'-ltiff'};
        end

        fprintf('libtiff built successfully.\n');
    catch ME
        error('Failed to build libtiff from source: %s', ME.message);
    end
end

% --- Platform-specific compiler flags ---
if ispc
    if debug
        mex_flags_cpu = {'-R2018a', 'COMPFLAGS="$COMPFLAGS /Od /Zi /openmp"'};
    else
        mex_flags_cpu = {'-R2018a', 'COMPFLAGS="$COMPFLAGS /O2 /arch:AVX2 /openmp"'};
    end
else
    if debug
        mex_flags_cpu = {'-R2018a', 'CFLAGS="$CFLAGS -O0 -g -fopenmp"', ...
                         'CXXFLAGS="$CXXFLAGS -O0 -g -fopenmp"'};
    else
        mex_flags_cpu = {'-R2018a', ...
            'CFLAGS="$CFLAGS -O3 -march=native -fomit-frame-pointer -fopenmp"', ...
            'CXXFLAGS="$CXXFLAGS -O3 -march=native -fomit-frame-pointer -fopenmp"'};
    end
end

fprintf('Using libtiff from: %s\n', libfile);

% Build CPU MEX files
% mex(mex_flags_cpu{:}, src_semaphore);
% mex(mex_flags_cpu{:}, src_lz4_save, src_lz4_c);
% mex(mex_flags_cpu{:}, src_lz4_load, src_lz4_c);
mex(mex_flags_cpu{:}, src_load_bl, tiff_include{:}, tiff_lib{:}, tiff_link{:});

% CUDA flags
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
