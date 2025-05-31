% ===============================
% build_mex.m
% ===============================
% Compile CUDA MEX files for wavedec2 and waverec2

% Set source files
src_semaphore = 'semaphore.c';
src_queue = 'queue.c';
src_lz4_save = 'save_lz4_mex.c';
src_lz4_load = 'load_lz4_mex.c';
src_lz4_c = 'lz4.c';

% URLs for LZ4
lz4_c_url  = 'https://raw.githubusercontent.com/lz4/lz4/dev/lib/lz4.c';
lz4_h_url  = 'https://raw.githubusercontent.com/lz4/lz4/dev/lib/lz4.h';

% Download lz4.c and lz4.h if missing
if ~isfile('lz4.c')
    fprintf('Downloading lz4.c ...\n');
    websave('lz4.c', lz4_c_url);
end
if ~isfile('lz4.h')
    fprintf('Downloading lz4.h ...\n');
    websave('lz4.h', lz4_h_url);
end

% Optimization flags for gcc/clang (Linux/macOS) and fallback for MSVC/Windows
opt_flags = '-O2 -march=native -fomit-frame-pointer';

% Use different flags for Windows/MSVC (ignores -march/-fomit-frame-pointer)
if ispc && ~ismac
    mex_cmd_base = 'mex -v CFLAGS="$CFLAGS /O2 /arch:AVX2" CXXFLAGS="$CXXFLAGS /O2 /arch:AVX2" ';
else
    mex_cmd_base = ['mex -v CFLAGS="$CFLAGS ', opt_flags, '" CXXFLAGS="$CXXFLAGS ', opt_flags, '" '];
end

% Compile semaphore.c
eval([mex_cmd_base, ' ', src_semaphore]);

% Compile queue.c
eval([mex_cmd_base, ' ', src_queue]);

% Compile save_lz4_mex.c + lz4.c
eval([mex_cmd_base, ' ', src_lz4_save, ' ', src_lz4_c]);

% Compile load_lz4_mex.c + lz4.c
eval([mex_cmd_base, ' ', src_lz4_load, ' ', src_lz4_c]);

% CUDA/mexcuda examples (uncomment & edit paths as needed)
% nvcc_flags = '-O2 -Xcompiler -march=native -Xcompiler -fomit-frame-pointer';
% build_xml = "C:/Program Files/MATLAB/R2023a/toolbox/parallel/gpu/extern/src/mex/win64/nvcc_msvcpp2022.xml";
% root_dir = '.'; include_dir = './cuda_kernels';
% mexcuda('-v', '-R2018a', '-f', build_xml, src_wavedec2, ['-I', root_dir], ['-I', include_dir], ...
%     'CFLAGS="$CFLAGS -O2 -march=native -fomit-frame-pointer"', ...
%     'CXXFLAGS="$CXXFLAGS -O2 -march=native -fomit-frame-pointer"', ...
%     'NVCCFLAGS="$NVCCFLAGS -O2 -Xcompiler -march=native -Xcompiler -fomit-frame-pointer"');
% ... (repeat for other .cu files as needed)