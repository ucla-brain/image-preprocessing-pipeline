% ===============================
% build_mex.m
% ===============================
% Compile semaphore, queue, chunked LZ4, and GPU Gaussian MEX files.
% Downloads lz4.c/.h from GitHub if missing.
% Requires MATLAB R2018a+ (-R2018a mxArray API).
debug = false;
if verLessThan('matlab', '9.4')
    error('This script requires MATLAB R2018a or newer (for -R2018a MEX API)');
end

src_semaphore = 'semaphore.c';
src_queue = 'queue.c';
src_lz4_save = 'save_lz4_mex.c';
src_lz4_load = 'load_lz4_mex.c';
src_lz4_c = 'lz4.c';
src_gauss3d = 'gauss3d_mex.cu';

lz4_c_url  = 'https://raw.githubusercontent.com/lz4/lz4/dev/lib/lz4.c';
lz4_h_url  = 'https://raw.githubusercontent.com/lz4/lz4/dev/lib/lz4.h';

% Download lz4.c/.h if missing
if ~isfile('lz4.c')
    fprintf('Downloading lz4.c ...\n');
    try, websave('lz4.c', lz4_c_url);
    catch, error('Failed to download lz4.c'); end
end
if ~isfile('lz4.h')
    fprintf('Downloading lz4.h ...\n');
    try, websave('lz4.h', lz4_h_url);
    catch, error('Failed to download lz4.h'); end
end

% MEX optimization flags (for CPU builds)
mex_flags = {'-R2018a'};
if debug
    if ispc && ~ismac
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -G -std=c++14 -Xcompiler ""/Od,/Zi"" "';
    else
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -G -std=c++14 -Xcompiler ''-O0,-g'' "';
    end
else
    if ispc && ~ismac
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -O2 -std=c++14 -Xcompiler ""/O2,/arch:AVX2,/openmp"" "';
    else
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -O2 -std=c++14 -Xcompiler ''-O2,-march=native,-fomit-frame-pointer,-fopenmp'' "';
    end
end


% Build semaphore/queue/lz4 MEX files (CPU)
mex(mex_flags{:}, src_semaphore);
mex(mex_flags{:}, src_queue);
mex(mex_flags{:}, src_lz4_save, src_lz4_c);
mex(mex_flags{:}, src_lz4_load, src_lz4_c);

% CUDA optimization flags (for mexcuda)
if ispc && ~ismac
    nvccflags = 'NVCCFLAGS="$NVCCFLAGS -G -std=c++14 -Xcompiler ""/O2,/arch:AVX2,/openmp"" "';
else
    nvccflags = 'NVCCFLAGS="$NVCCFLAGS -G -std=c++14 -Xcompiler ''-O2,-march=native,-fopenmp'' "';
end

% CUDA include dirs (if any)
root_dir = '.'; include_dir = './cuda_kernels';

% Build CUDA Gaussian 3D MEX file (GPU)
mexcuda('-R2018a', src_gauss3d, ['-I', root_dir], ['-I', include_dir], nvccflags);

fprintf('All MEX files built successfully.\n');
