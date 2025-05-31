% ===============================
% build_mex.m
% ===============================
% Compile CUDA MEX files for wavedec2 and waverec2

src_semaphore = 'semaphore.c';
src_queue = 'queue.c';
src_lz4_save = 'save_lz4_mex.c';
src_lz4_load = 'load_lz4_mex.c';
src_lz4_c = 'lz4.c';

lz4_c_url  = 'https://raw.githubusercontent.com/lz4/lz4/dev/lib/lz4.c';
lz4_h_url  = 'https://raw.githubusercontent.com/lz4/lz4/dev/lib/lz4.h';

% Download lz4.c and lz4.h if missing, with error handling
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

opt_flags = '-O2 -march=native -fomit-frame-pointer';

if ispc && ~ismac
    mex_cmd_base = 'mex CFLAGS="$CFLAGS /O2 /arch:AVX2" CXXFLAGS="$CXXFLAGS /O2 /arch:AVX2" ';
else
    mex_cmd_base = ['mex CFLAGS="$CFLAGS ', opt_flags, '" CXXFLAGS="$CXXFLAGS ', opt_flags, '" '];
end

eval([mex_cmd_base, ' ', src_semaphore]);
eval([mex_cmd_base, ' ', src_queue]);
eval([mex_cmd_base, ' ', src_lz4_save, ' ', src_lz4_c]);
eval([mex_cmd_base, ' ', src_lz4_load, ' ', src_lz4_c]);

% CUDA/mexcuda examples (uncomment as needed)
% nvcc_flags = '-O2 -Xcompiler -march=native -Xcompiler -fomit-frame-pointer';
% build_xml = "...";
% root_dir = '.'; include_dir = './cuda_kernels';
% mexcuda('-R2018a', '-f', build_xml, src_wavedec2, ['-I', root_dir], ['-I', include_dir], ...
%     'CFLAGS="$CFLAGS -O2 -march=native -fomit-frame-pointer"', ...
%     'CXXFLAGS="$CXXFLAGS -O2 -march=native -fomit-frame-pointer"', ...
%     'NVCCFLAGS="$NVCCFLAGS -O2 -Xcompiler -march=native -Xcompiler -fomit-frame-pointer"');
