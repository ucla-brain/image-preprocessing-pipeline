% ===============================
% build_mex.m (Patched to always use local libtiff)
% ===============================
% Compile semaphore, LZ4, and GPU MEX files using locally-compiled libtiff.
% Always use the version in tiff_build/libtiff and never the system or Anaconda version.

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

% Use locally-compiled libtiff, always.
libtiff_src_version = '4.6.0'; % or whatever you are building
libtiff_root = fullfile(pwd, 'tiff_src', ['tiff-', libtiff_src_version]);
libtiff_install_dir = fullfile(pwd, 'tiff_build', 'libtiff');
stamp_file = fullfile(libtiff_install_dir, '.libtiff_installed');

use_fallback = false;
if ~isfile(stamp_file)
    fprintf('Building libtiff from source with optimized flags...\n');
    if ~try_build_libtiff(libtiff_root, libtiff_install_dir)
        error('Failed to build local libtiff. Please check the build log.');
    else
        if ~isfolder(libtiff_install_dir), error('libtiff install failed!'); end
        fclose(fopen(stamp_file, 'w'));
    end
end

tiff_include = {['-I', fullfile(libtiff_install_dir, 'include')]};
tiff_lib     = {['-L', fullfile(libtiff_install_dir, 'lib')]};
tiff_link    = {'-ltiff'};

fprintf('Using libtiff from: %s\n', fullfile(libtiff_install_dir, 'lib'));

% CPU compile flags
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

% ===============================
% Function: try_build_libtiff
% ===============================
function ok = try_build_libtiff(libtiff_root, libtiff_install_dir)
    if ispc
        archive = 'tiff.zip';
        if ~isfolder(libtiff_root)
            system(['curl -L -o ', archive, ' https://download.osgeo.org/libtiff/tiff-4.7.0.zip']);
            unzip(archive, 'tiff_src'); delete(archive);
        end
        cd(libtiff_root);
        % Install to local libtiff_install_dir, NOT conda_prefix
        status = system(['cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="', libtiff_install_dir, '" . && ' ...
                         'cmake --build build --config Release --target install']);
        cd('../../');
    else
        archive = 'tiff.tar.gz';
        if ~isfolder(['tiff-', '4.6.0'])
            system(['curl -L -o ', archive, ' https://download.osgeo.org/libtiff/tiff-4.7.0.tar.gz']);
            system(['tar -xzf ', archive]); delete(archive);
        end
        cd(['tiff-', '4.6.0']);
        % Install to local libtiff_install_dir, NOT conda_prefix
        status = system(['./configure --prefix=', libtiff_install_dir, ' && make -j4 && make install']);
        cd('..');
    end
    ok = (status == 0);
end
