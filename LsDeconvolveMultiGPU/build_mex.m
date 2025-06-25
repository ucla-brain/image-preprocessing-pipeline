% build_mex.m — static, LTO-enabled build of zlib-ng, libdeflate, zstd,
%               libtiff and all project MEX files for Linux & Windows 10/11
%
% Version : 2025-06-24  (libdeflate support, correct codec disable flags)
% Author  : Keivan Moradi  (with ChatGPT-o3 assistance)
% License : MIT
%
% -- Features --
%  • Builds every dependency in-tree, statically, with full LTO
%  • Disables *all* unneeded TIFF codecs (JPEG, JBIG, LZMA, WebP, PixarLog,
%    LERC) while *retaining* Deflate+zlib-ng and ZSTD for compression
%  • Links libtiff together with libdeflate, zstd and zlib-ng in
%    dependency order, so no undefined-symbol errors remain
%  • One-command build:  >> build_mex            % release, native -O3
%                         >> build_mex(true)     % debug   (-g, no LTO)
%
% Notes
%  • Requires: CMake ≥ 3.19, a C++17 compiler, git/curl/wget, make/ninja
%  • CUDA MEX rules stay untouched; they assume CUDA 12.8 is installed
%  • Windows builds use MSVC 2022 (cl /GL for LTO), Linux uses GCC/Clang

function build_mex(debug)
%———————————————————————————————————————————————————————————————————————

if nargin < 1,  debug = false;  end
if verLessThan('matlab','9.4')
    error('Requires MATLAB R2018a or newer.'); end
if exist('mexcuda','file') ~= 2
    error('mexcuda not found – check CUDA/toolkit installation.'); end

% ——— Dependency versions ————————————————————————————————
zlibng_v     = '2.2.4';
libdeflate_v = '1.20';
zstd_v       = '1.5.7';
libtiff_v    = '4.7.0';

% ——— Directory layout ————————————————————————————————————————
root        = pwd;
thirdparty  = fullfile(root,'thirdparty');
build_root  = fullfile(root,'tiff_build');

% Sources
zlibng_src     = fullfile(thirdparty, ['zlib-ng-'     zlibng_v]);
libdeflate_src = fullfile(thirdparty, ['libdeflate-'  libdeflate_v]);
zstd_src       = fullfile(thirdparty, ['zstd-'        zstd_v]);
lz4_src        = fullfile(thirdparty, 'lz4');
libtiff_src    = fullfile(thirdparty, ['tiff-'        libtiff_v]);

% Install prefixes
zlibng_inst     = fullfile(build_root,'zlib-ng');
libdeflate_inst = fullfile(build_root,'libdeflate');
zstd_inst       = fullfile(build_root,'zstd');
lz4_inst        = fullfile(build_root,'lz4');        % headers only
libtiff_inst    = fullfile(build_root,'libtiff');

mkdir(thirdparty);  mkdir(build_root);

% ——— Helper: run cmake && build/install ————————————————
function status = cmake_build(src,bld,inst,args)
    if ~exist(bld,'dir'), mkdir(bld); end
    old = cd(bld);
    cleanup = onCleanup(@() cd(old));

    cfg = sprintf('cmake "%s" -DCMAKE_INSTALL_PREFIX="%s" %s',src,inst,args);
    status = system(cfg);
    if status ~= 0,  return;  end

    if ispc
        status = system('cmake --build . --config Release --target INSTALL');
    else
        status = system(sprintf('cmake --build . -- -j%d install', feature('numCores')));
    end
end

%======================================================================
% 1) LZ4 – single-file header+source, no build needed
%======================================================================
lz4_c = fullfile(lz4_src,'lz4.c');   lz4_h = fullfile(lz4_src,'lz4.h');
if ~isfile(lz4_c)
    fprintf('[LZ4] downloading …\n');
    mkdir(lz4_src);
    websave(lz4_c,'https://raw.githubusercontent.com/lz4/lz4/dev/lib/lz4.c');
    websave(lz4_h,'https://raw.githubusercontent.com/lz4/lz4/dev/lib/lz4.h');
end

%======================================================================
% 2) zlib-ng (zlib replacement, static, PIC)
%======================================================================
stamp = fullfile(zlibng_inst,'.built');
if ~isfile(stamp)
    fprintf('[zlib-ng] building …\n');
    if ~exist(zlibng_src,'dir')
        tgz = fullfile(thirdparty, ['zlib-ng-' zlibng_v '.tar.gz']);
        websave(tgz, ['https://github.com/zlib-ng/zlib-ng/archive/refs/tags/'
                      zlibng_v '.tar.gz']);
        untar(tgz,thirdparty); delete(tgz);
    end
    args = [ ...
        '-DBUILD_SHARED_LIBS=OFF ', ...
        '-DZLIB_COMPAT=ON ', ...
        '-DCMAKE_POSITION_INDEPENDENT_CODE=ON ', ...
        '-DCMAKE_BUILD_TYPE=Release ', ...
        '-DCMAKE_C_FLAGS_RELEASE="-O3 -march=native -flto -fPIC" ' ...
    ];
    if cmake_build(zlibng_src, fullfile(zlibng_src,'build'), zlibng_inst, args)
        error('zlib-ng build failed.');
    end
    fclose(fopen(stamp,'w'));
end

%======================================================================
% 3) zstd (static)
%======================================================================
stamp = fullfile(zstd_inst,'.built');
if ~isfile(stamp)
    fprintf('[zstd] building …\n');
    if ~exist(zstd_src,'dir')
        tgz = fullfile(thirdparty, ['zstd-' zstd_v '.tar.gz']);
        websave(tgz, ['https://github.com/facebook/zstd/archive/refs/tags/v'
                      zstd_v '.tar.gz']);
        untar(tgz,thirdparty); delete(tgz);
    end
    if ispc
        args = [ ...
            '-DBUILD_SHARED_LIBS=OFF ', ...
            '-DCMAKE_POSITION_INDEPENDENT_CODE=ON ', ...
            '-DCMAKE_BUILD_TYPE=Release ', ...
            '-DZSTD_PROGRAMS=OFF ', ...
            '-DCMAKE_C_FLAGS_RELEASE="/O2 /GL" ' ...
        ];
        if cmake_build(zstd_src, fullfile(zstd_src,'build'), zstd_inst, args)
            error('zstd build failed (Windows).');
        end
    else
        old = cd(zstd_src);  cleanup = onCleanup(@()cd(old));
        system(sprintf('make -j%d lib', feature('numCores')));
        system(sprintf('make PREFIX="%s" install', zstd_inst));
    end
    fclose(fopen(stamp,'w'));
end

%======================================================================
% 4) libdeflate (needed by libtiff Deflate codec)
%======================================================================
stamp = fullfile(libdeflate_inst,'.built');
if ~isfile(stamp)
    fprintf('[libdeflate] building …\n');
    if ~exist(libdeflate_src,'dir')
        tgz = fullfile(thirdparty, ['libdeflate-' libdeflate_v '.tar.gz']);
        websave(tgz, ['https://github.com/ebiggers/libdeflate/archive/refs/tags/v'
                      libdeflate_v '.tar.gz']);
        untar(tgz, thirdparty); delete(tgz);
    end
    if ispc
        if cmake_build(libdeflate_src, fullfile(libdeflate_src,'build'), ...
                       libdeflate_inst, ...
                       '-DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release')
            error('libdeflate build failed (Windows).');
        end
    else % Makefile build → copy .a and headers
        old = cd(libdeflate_src);  cleanup = onCleanup(@()cd(old));
        system(sprintf('make -j%d libdeflate.a', feature('numCores')));
        mkdir(fullfile(libdeflate_inst,'lib')); mkdir(fullfile(libdeflate_inst,'include'));
        copyfile('libdeflate.a', fullfile(libdeflate_inst,'lib'));
        copyfile('libdeflate.h', fullfile(libdeflate_inst,'include'));
    end
    fclose(fopen(stamp,'w'));
end

%======================================================================
% 5) libtiff (static, unwanted codecs OFF)
%======================================================================
stamp = fullfile(libtiff_inst,'.built');
if ~isfile(stamp)
    fprintf('[libtiff] building … (JPEG/JBIG/LZMA/WebP/LERC off)\n');
    if ~exist(libtiff_src,'dir')
        tgz = fullfile(thirdparty, ['tiff-' libtiff_v '.tar.gz']);
        websave(tgz, ['https://download.osgeo.org/libtiff/tiff-' libtiff_v '.tar.gz']);
        untar(tgz, thirdparty); delete(tgz);
    end
    args = sprintf([ ...
        '-DBUILD_SHARED_LIBS=OFF ', ...
        '-DCMAKE_POSITION_INDEPENDENT_CODE=ON ', ...
        '-Djbig=OFF -Djpeg=OFF -Dold-jpeg=OFF -Dlzma=OFF ', ...
        '-Dwebp=OFF -Dlerc=OFF -Dpixarlog=OFF ', ...
        '-Dlibdeflate=ON ', ...
        '-DZLIB_LIBRARY=%s -DZLIB_INCLUDE_DIR=%s ', ...
        '-DZSTD_LIBRARY=%s -DZSTD_INCLUDE_DIR=%s ', ...
        '-DLIBDEFLATE_LIBRARY=%s -DLIBDEFLATE_INCLUDE_DIR=%s ', ...
        '-DCMAKE_C_FLAGS_RELEASE="-O3 -march=native -flto -fPIC" ', ...
        '-DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -flto -fPIC" ' ...
        ], ...
        fullfile(zlibng_inst,'lib','libz.a'),      fullfile(zlibng_inst,'include'), ...
        fullfile(zstd_inst,'lib','libzstd.a'),     fullfile(zstd_inst,'include'), ...
        fullfile(libdeflate_inst,'lib','libdeflate.a'), fullfile(libdeflate_inst,'include'));
    if cmake_build(libtiff_src, fullfile(libtiff_src,'build'), libtiff_inst, args)
        error('libtiff build failed.');
    end
    fclose(fopen(stamp,'w'));
end

%======================================================================
% 6) MEX compilation flags
%======================================================================
if ispc
    if debug
        mex_cpu = {'-R2018a', ...
                   'COMPFLAGS="$COMPFLAGS /std:c++17 /Od /Zi"', ...
                   'LINKFLAGS="$LINKFLAGS /DEBUG"'};
    else
        mex_cpu = {'-R2018a', ...
                   'COMPFLAGS="$COMPFLAGS /std:c++17 /O2 /arch:AVX2 /GL"', ...
                   'LINKFLAGS="$LINKFLAGS /LTCG"'};
    end
else
    if debug
        mex_cpu = {'-R2018a', ...
                   'CFLAGS="$CFLAGS -O0 -g"', ...
                   'CXXFLAGS="$CXXFLAGS -O0 -g"', ...
                   'LDFLAGS="$LDFLAGS -g"'};
    else
        mex_cpu = {'-R2018a', ...
                   'CFLAGS="$CFLAGS -O3 -march=native -flto"', ...
                   'CXXFLAGS="$CXXFLAGS -O3 -march=native -flto"', ...
                   'LDFLAGS="$LDFLAGS -flto"'};
    end
end

%———————————————————————————————————————————————————————————————————————
% 7) Include & link lines for TIFF-dependent MEX files
%———————————————————————————————————————————————————————————————————————
inc_tiff  = ['-I' fullfile(libtiff_inst,'include')];
link_tiff = { ...
    fullfile(libtiff_inst,'lib','libtiffxx.a'), ...
    fullfile(libtiff_inst,'lib','libtiff.a'), ...
    fullfile(libdeflate_inst,'lib','libdeflate.a'), ...
    fullfile(zstd_inst,'lib','libzstd.a'), ...
    fullfile(zlibng_inst,'lib','libz.a') };

%———————————————————————————————————————————————————————————————————————
% 8) Build CPU MEX files
%———————————————————————————————————————————————————————————————————————
fprintf('\n[MEX] Compiling CPU modules …\n');
mex(mex_cpu{:}, 'semaphore.c');

mex(mex_cpu{:}, 'save_lz4_mex.c',   lz4_c, ['-I' lz4_src]);
mex(mex_cpu{:}, 'load_lz4_mex.c',   lz4_c, ['-I' lz4_src]);
mex(mex_cpu{:}, 'load_slab_lz4.cpp',lz4_c, ['-I' lz4_src]);

mex(mex_cpu{:}, inc_tiff, 'load_bl_tif.cpp', link_tiff{:});
mex(mex_cpu{:}, inc_tiff, 'save_bl_tif.cpp', link_tiff{:});

%———————————————————————————————————————————————————————————————————————
% 9) CUDA MEX files (unchanged from user’s original script)
%———————————————————————————————————————————————————————————————————————
if ispc
    if debug
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -G -std=c++17 -Xcompiler "/Od,/Zi"" ';
    else
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -O2 -std=c++17 -Xcompiler "/O2,/arch:AVX2"" ';
    end
    xmlfile = fullfile(fileparts(mfilename('fullpath')), 'nvcc_msvcpp2022.xml');
    assert(isfile(xmlfile),'nvcc_msvcpp2022.xml not found!');
    cuda_mex_flags = {'-f', xmlfile};
else
    if debug
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -G -std=c++17 -Xcompiler ''-O0,-g''" ';
    else
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -O3 -std=c++17 -Xcompiler ''-O3,-march=native -flto''" ';
    end
    cuda_mex_flags = {};
end

root_dir = '.';   include_dir = './mex_incubator';
mexcuda(cuda_mex_flags{:}, '-R2018a', 'gauss3d_mex.cu', ...
        ['-I',root_dir], ['-I',include_dir], nvccflags);
mexcuda(cuda_mex_flags{:}, '-R2018a', 'conv3d_mex.cu', ...
        ['-I',root_dir], ['-I',include_dir], nvccflags);
mexcuda(cuda_mex_flags{:}, '-R2018a', 'otf_gpu_mex.cu', ...
        ['-I',root_dir], ['-I',include_dir], nvccflags, ...
        '-L/usr/local/cuda/lib64', '-lcufft');

fprintf('\n✅  All MEX files built successfully.\n');
end
