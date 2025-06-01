% ===============================
% build_mex.m
% ===============================
% Compile semaphore, queue, and chunked LZ4 MEX files.
% Downloads lz4.c/.h from GitHub if missing.
% Requires MATLAB R2018a+ (-R2018a mxArray API).

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

% Compile
mex_flags = {'-R2018a'};
if ispc && ~ismac
    mex_flags = [mex_flags, 'CFLAGS="$CFLAGS /O2 /arch:AVX2"', 'CXXFLAGS="$CXXFLAGS /O2 /arch:AVX2"'];
else
    mex_flags = [mex_flags, 'CFLAGS="$CFLAGS -O2 -march=native -fomit-frame-pointer"', ...
                           'CXXFLAGS="$CXXFLAGS -O2 -march=native -fomit-frame-pointer"'];
end

%mex(mex_flags{:}, src_semaphore);
%mex(mex_flags{:}, src_queue);
%mex(mex_flags{:}, src_lz4_save, src_lz4_c);
%mex(mex_flags{:}, src_lz4_load, src_lz4_c);

% nvcc_flags = '-O2 -Xcompiler -march=native -Xcompiler -fomit-frame-pointer';
% 'CFLAGS="$CFLAGS -O2 -march=native -fomit-frame-pointer"', ...
% 'CXXFLAGS="$CXXFLAGS -O2 -march=native -fomit-frame-pointer"', ...
% 'NVCCFLAGS="$NVCCFLAGS -O2 -Xcompiler -march=native -Xcompiler -fomit-frame-pointer"'
% build_xml = "..."; '-f', build_xml,
root_dir = '.'; include_dir = './cuda_kernels';
mexcuda('-R2018a', src_gauss3d, ['-I', root_dir], ['-I', include_dir]);
