% ===============================
% build_mex.m
% ===============================
% Compile semaphore, queue, chunked LZ4, and GPU Gaussian MEX files.
% Downloads lz4.c/.h from GitHub if missing.
% Requires MATLAB R2018a+ (-R2018a mxArray API).

% --- PATCH: Force mexcuda -setup to use local nvcc_msvcpp2022.xml on Windows ---
if ispc
    this_xml = fullfile(fileparts(mfilename('fullpath')), 'nvcc_msvcpp2022.xml');
    assert(isfile(this_xml), 'nvcc_msvcpp2022.xml not found!');
    fprintf('Please run the following command ONCE in MATLAB to register your custom CUDA config:\n');
    fprintf('  mexcuda -setup "%s"\n', this_xml);
    % Or try interactive selection:
    mexcuda('-setup');
    % ...user picks the config from menu
end

debug = false;
if verLessThan('matlab', '9.4')
    error('This script requires MATLAB R2018a or newer (for -R2018a MEX API)');
end

src_semaphore = 'semaphore.c';
% src_queue = 'queue.c';
src_lz4_save = 'save_lz4_mex.c';
src_lz4_load = 'load_lz4_mex.c';
src_lz4_c = 'lz4.c';
src_gauss3d = 'gauss3d_mex.cu';
src_conv3d = 'conv3d_mex.cu';
src_otf_gpu = 'otf_gpu_mex.cu';

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
        % Windows debug
        mex_flags_cpu = [mex_flags, {'COMPFLAGS="$COMPFLAGS /Od /Zi /openmp"'}];
    else
        % POSIX debug
        mex_flags_cpu = [mex_flags, {'CFLAGS="$CFLAGS -O0 -g -fopenmp"', 'CXXFLAGS="$CXXFLAGS -O0 -g -fopenmp"'}];
    end
else
    if ispc && ~ismac
        % Windows release
        mex_flags_cpu = [mex_flags, {'COMPFLAGS="$COMPFLAGS /O2 /arch:AVX2 /openmp"'}];
    else
        % POSIX release (Linux/macOS)
        mex_flags_cpu = [mex_flags, {'CFLAGS="$CFLAGS -O3 -march=native -fomit-frame-pointer -fopenmp"', ...
                                     'CXXFLAGS="$CXXFLAGS -O3 -march=native -fomit-frame-pointer -fopenmp"'}];
    end
end

% Build semaphore/queue/lz4 MEX files (CPU)
% mex(mex_flags_cpu{:}, src_queue);
mex(mex_flags_cpu{:}, src_semaphore);
mex(mex_flags_cpu{:}, src_lz4_save, src_lz4_c);
mex(mex_flags_cpu{:}, src_lz4_load, src_lz4_c);

% CUDA optimization flags (for mexcuda)
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

% CUDA include dirs (if any)
root_dir = '.'; include_dir = './mex_incubator';

% Build CUDA Gaussian 3D MEX file (GPU)
mexcuda(mex_flags{:}, src_gauss3d, ['-I', root_dir], ['-I', include_dir], nvccflags);
mexcuda(mex_flags{:}, src_conv3d , ['-I', root_dir], ['-I', include_dir], nvccflags);
mexcuda(mex_flags{:}, src_otf_gpu, ['-I', root_dir], ['-I', include_dir], nvccflags, '-L/usr/local/cuda/lib64', '-lcufft');

fprintf('All MEX files built successfully.\n');
