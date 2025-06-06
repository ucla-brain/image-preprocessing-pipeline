% ===============================
% build_mex.m
% ===============================
% Compile semaphore, LZ4, and GPU MEX files using Anaconda's libtiff.
% Requires MATLAB R2018a+ (-R2018a MEX API) and Anaconda (CONDA_PREFIX).
% On Windows, will automatically generate libtiff.lib from tiff.dll if needed,
% using lib.exe (from Visual Studio).

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

% Use Anaconda-provided libtiff for all platforms
conda_prefix = getenv('CONDA_PREFIX');
assert(~isempty(conda_prefix) && isfolder(conda_prefix), ...
    'CONDA_PREFIX is not set or does not point to a valid directory.');

tiff_include = {['-I', fullfile(conda_prefix, 'include')]} ;
tiff_lib     = {['-L', fullfile(conda_prefix, 'lib')]} ;

% --- Platform-specific linking and flags ---
if ispc
    % On Windows: check for libtiff.lib or tiff.lib, generate if needed.
    tiff_libfile  = fullfile(conda_prefix, 'lib', 'libtiff.lib');
    tiff_libfile2 = fullfile(conda_prefix, 'lib', 'tiff.lib');
    need_generate = false;

    if isfile(tiff_libfile)
        tiff_link = {tiff_libfile};
    elseif isfile(tiff_libfile2)
        tiff_link = {tiff_libfile2};
    else
        % Try to auto-generate libtiff.lib from tiff.dll using lib.exe
        tiff_dll = fullfile(conda_prefix, 'Library', 'bin', 'tiff.dll');
        tiff_def = fullfile(conda_prefix, 'lib', 'tiff.def');
        generated_lib = fullfile(conda_prefix, 'lib', 'libtiff.lib');

        if ~isfile(tiff_dll)
            error('tiff.dll not found (looked for %s)', tiff_dll);
        end

        fprintf('Attempting to generate libtiff.lib from tiff.dll ...\n');

        % Use dumpbin and editbin approach if needed in future
        [s1, out1] = system(sprintf('dumpbin /exports "%s" > "%s"', tiff_dll, tiff_def));
        if s1 ~= 0
            disp(out1);
            error('Failed to generate tiff.def. Make sure dumpbin is available.');
        end

        % Generate libtiff.lib using lib.exe (Visual Studio must be in PATH)
        [s2, out2] = system(sprintf('lib /def:"%s" /out:"%s" /machine:x64', tiff_def, generated_lib));
        if s2 ~= 0 || ~isfile(generated_lib)
            disp(out2);
            error(['Failed to generate libtiff.lib using lib.exe. ' ...
                'Make sure you are running MATLAB from a "Developer Command Prompt for VS" or that lib.exe is on your PATH.']);
        end

        fprintf('libtiff.lib successfully generated in %s\n', fullfile(conda_prefix, 'lib'));
        if exist(tiff_def, 'file'), delete(tiff_def); end

        tiff_link = {generated_lib};
    end

    if debug
        mex_flags_cpu = {'-R2018a', 'COMPFLAGS="$COMPFLAGS /Od /Zi /openmp"'};
    else
        mex_flags_cpu = {'-R2018a', 'COMPFLAGS="$COMPFLAGS /O2 /arch:AVX2 /openmp"'};
    end
else
    % On Linux/Mac, typically need libtiff.so
    tiff_libfile = fullfile(conda_prefix, 'lib', 'libtiff.so');
    if ~isfile(tiff_libfile)
        error(['libtiff.so not found in: ' fullfile(conda_prefix, 'lib')]);
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
