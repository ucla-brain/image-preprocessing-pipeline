% build_mex.m — static, LTO-enabled build of zlib-ng, zstd (Makefile),
%               libtiff and all project MEX files for Linux & Windows
%
% Version : 2025-06-24  (zstd via Make on Linux; zlib-ng for Deflate)
% Author  : Keivan Moradi  (with ChatGPT-o3 assistance)
% License : MIT
function build_mex(debug)
    ncores = feature('numCores');
    if nargin<1, debug=false; end
    if verLessThan('matlab','9.4')
        error('Requires MATLAB R2018a or newer.'); end
    if exist('mexcuda','file')~=2
        error('mexcuda not found – ensure CUDA is configured.'); end

    % Versions
    zlibng_v  = '2.2.4';
    zstd_v    = '1.5.7';
    libtiff_v = '4.7.0';

    % Paths
    root        = pwd;
    thirdparty  = fullfile(root,'thirdparty');
    build_root  = fullfile(root,'tiff_build');
    if ~exist(thirdparty,'dir'), mkdir(thirdparty); end
    if ~exist(build_root,'dir'), mkdir(build_root); end

    zlibng_src   = fullfile(thirdparty, ['zlib-ng-'   zlibng_v]);
    zstd_src     = fullfile(thirdparty, ['zstd-'      zstd_v]);
    lz4_src      = fullfile(thirdparty, 'lz4');
    libtiff_src  = fullfile(thirdparty, ['tiff-'      libtiff_v]);

    zlibng_inst  = fullfile(build_root,'zlib-ng');
    zstd_inst    = fullfile(build_root,'zstd');
    libtiff_inst = fullfile(build_root,'libtiff');

    % Helper to run CMake + install (for zlib-ng & libtiff)
    function status = cmake_build(src,bld,inst,args)
        if ~exist(bld,'dir'), mkdir(bld); end
        old = cd(bld); onCleanup(@() cd(old));
        if system(sprintf('cmake "%s" -DCMAKE_INSTALL_PREFIX="%s" %s',src,inst,args))~=0
            status=1; return;
        end
        if ispc
            status = system('cmake --build . --config Release --target INSTALL');
        else
            status = system(sprintf('cmake --build . -- -j%d install', ncores));
        end
    end

    %% 1) LZ4 (single-file)
    lz4_c = fullfile(lz4_src,'lz4.c');
    lz4_h = fullfile(lz4_src,'lz4.h');
    if ~isfile(lz4_c)
        fprintf('[LZ4] downloading…\n');
        mkdir(lz4_src);
        websave(lz4_c,'https://raw.githubusercontent.com/lz4/lz4/dev/lib/lz4.c');
        websave(lz4_h,'https://raw.githubusercontent.com/lz4/lz4/dev/lib/lz4.h');
    end

    %% 2) zlib-ng (static, zlib-compatible)
    stamp = fullfile(zlibng_inst,'.built');
    if ~isfile(stamp)
        fprintf('[zlib-ng] building…\n');
        if ~exist(zlibng_src,'dir')
            tgz = fullfile(thirdparty,sprintf('zlib-ng-%s.tar.gz',zlibng_v));
            websave(tgz,sprintf(['https://github.com/zlib-ng/zlib-ng/archive/refs/tags/' zlibng_v '.tar.gz']));
            untar(tgz,thirdparty); delete(tgz);
        end
        args = [ ...
          '-DBUILD_SHARED_LIBS=OFF ', ...
          '-DZLIB_COMPAT=ON ', ...
          '-DCMAKE_POSITION_INDEPENDENT_CODE=ON ', ...
          '-DCMAKE_BUILD_TYPE=Release ', ...
          sprintf('-DCMAKE_C_FLAGS_RELEASE="-O3 -march=native -flto=%d -fPIC"', ncores) ...
        ];
        if cmake_build(zlibng_src,fullfile(zlibng_src,'build'),zlibng_inst,args)
            error('zlib-ng build failed.'); end
        fclose(fopen(stamp,'w'));
    end

    %% 3) zstd (Makefile on Linux, CMake on Windows)
    stamp = fullfile(zstd_inst,'.built');
    if ~isfile(stamp)
        fprintf('[zstd] building…\n');
        if ~exist(zstd_src,'dir')
            tgz = fullfile(thirdparty,sprintf('zstd-%s.tar.gz',zstd_v));
            websave(tgz,sprintf(['https://github.com/facebook/zstd/archive/refs/tags/v' zstd_v '.tar.gz']));
            untar(tgz,thirdparty); delete(tgz);
        end
        if ispc
            % CMake-based on Windows
            args = [ ...
              '-DBUILD_SHARED_LIBS=OFF ', ...
              '-DZSTD_PROGRAMS=OFF ', ...
              '-DCMAKE_POSITION_INDEPENDENT_CODE=ON ', ...
              '-DCMAKE_BUILD_TYPE=Release ', ...
              '-DCMAKE_C_FLAGS_RELEASE="/O2 /GL" ', ...
              '-DCMAKE_CXX_FLAGS_RELEASE="/O2 /GL"' ...
            ];
            if cmake_build(zstd_src,fullfile(zstd_src,'build'),zstd_inst,args)
                error('zstd build failed (Windows).'); end
        else
            % Makefile-based on Linux/macOS
            old = cd(zstd_src); onCleanup(@() cd(old));
            if system(sprintf('make -j%d', ncores))~=0
                error('zstd make failed');
            end
            % install static lib + headers
            mkdir(fullfile(zstd_inst,'lib'));
            mkdir(fullfile(zstd_inst,'include'));
            copyfile(fullfile(zstd_src,'lib','libzstd.a'), ...
                     fullfile(zstd_inst,'lib','libzstd.a'));
            copyfile(fullfile(zstd_src,'lib','*.h'), ...
                     fullfile(zstd_inst,'include'));
        end
        fclose(fopen(stamp,'w'));
    end

    %% 4) libtiff (disable JPEG/JBIG/LZMA/WebP/LERC/PixarLog; use zlib-ng + zstd)
    stamp = fullfile(libtiff_inst,'.built');
    if ~isfile(stamp)
        fprintf('[libtiff] building… (codecs off)\n');
        if ~exist(libtiff_src,'dir')
            tgz = fullfile(thirdparty,sprintf('tiff-%s.tar.gz',libtiff_v));
            websave(tgz,sprintf(['https://download.osgeo.org/libtiff/tiff-' libtiff_v '.tar.gz']));
            untar(tgz,thirdparty); delete(tgz);
        end
        args = sprintf([ ...
          '-DBUILD_SHARED_LIBS=OFF ', ...
          '-DCMAKE_POSITION_INDEPENDENT_CODE=ON ', ...
          '-Djbig=OFF -Djpeg=OFF -Dold-jpeg=OFF -Dlzma=OFF -Dwebp=OFF -Dlerc=OFF -Dpixarlog=OFF -Dlibdeflate=OFF -Dtiff-opengl=OFF ', ...
          '-DZLIB_LIBRARY=%s -DZLIB_INCLUDE_DIR=%s ', ...
          '-DZSTD_LIBRARY=%s -DZSTD_INCLUDE_DIR=%s ', ...
          '-DCMAKE_C_FLAGS_RELEASE="-O3 -march=native -flto=%d -fPIC" ', ...
          '-DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -flto=%d -fPIC"' ...
        ], ...
          fullfile(zlibng_inst,'lib','libz.a'),    fullfile(zlibng_inst,'include'), ...
          fullfile(zstd_inst,  'lib','libzstd.a'), fullfile(zstd_inst,  'include'), ...
          ncores, ncores);
        if cmake_build(libtiff_src,fullfile(libtiff_src,'build'),libtiff_inst,args)
            error('libtiff build failed.'); end
        fclose(fopen(stamp,'w'));
    end

    %% 5) MEX compilation flags
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
              sprintf('CFLAGS="$CFLAGS -O3 -march=native -flto=%d"'    , ncores), ...
              sprintf('CXXFLAGS="$CXXFLAGS -O3 -march=native -flto=%d"', ncores), ...
              sprintf('LDFLAGS="$LDFLAGS -flto=%d"'                    , ncores)};
        end
    end

    %% 6) Include & link for TIFF MEXs
    inc_tiff  = ['-I' fullfile(libtiff_inst,'include')];
    link_tiff = { ...
        fullfile(libtiff_inst,'lib','libtiffxx.a'), ...
        fullfile(libtiff_inst,'lib','libtiff.a'), ...
        fullfile(zstd_inst,  'lib','libzstd.a'), ...
        fullfile(zlibng_inst,'lib','libz.a') };

    %% 7) Build CPU MEX files
    fprintf('\n[MEX] Compiling CPU modules …\n');
    mex(mex_cpu{:}, 'semaphore.c');
    mex(mex_cpu{:}, 'save_lz4_mex.c',    lz4_c, ['-I' lz4_src]);
    mex(mex_cpu{:}, 'load_lz4_mex.c',    lz4_c, ['-I' lz4_src]);
    mex(mex_cpu{:}, 'load_slab_lz4.cpp', lz4_c, ['-I' lz4_src]);
    mex(mex_cpu{:}, inc_tiff, 'load_bl_tif.cpp', link_tiff{:});
    mex(mex_cpu{:}, inc_tiff, 'save_bl_tif.cpp', link_tiff{:});

    %% 8) CUDA MEX files (unchanged)
    if ispc
      xmlfile = fullfile(fileparts(mfilename('fullpath')), 'nvcc_msvcpp2022.xml');
      assert(isfile(xmlfile),'nvcc_msvcpp2022.xml not found!');
      cuda_mex_flags = {'-f',xmlfile};
    else
      cuda_mex_flags = {};
    end

    if debug
      % keep only the debug flag (-G), no -std or -O here
      nvccflags = 'NVCCFLAGS="$NVCCFLAGS -G"';
    else
      % only pass host flags; leave -O and -std to mexcuda
      nvccflags = sprintf('NVCCFLAGS="$NVCCFLAGS -Xcompiler -march=native -Xcompiler -flto=%d"', ncores);
    end

    root_dir    = '.';
    include_dir = './mex_incubator';

    mexcuda(cuda_mex_flags{:}, '-R2018a', nvccflags, 'gauss3d_mex.cu',  ['-I',root_dir], ['-I',include_dir]);

    mexcuda(cuda_mex_flags{:}, '-R2018a', nvccflags, 'conv3d_mex.cu',  ['-I',root_dir], ['-I',include_dir]);

    mexcuda(cuda_mex_flags{:}, '-R2018a', nvccflags, 'otf_gpu_mex.cu', ['-I',root_dir], ['-I',include_dir], ...
           '-L/usr/local/cuda/lib64','-lcufft');

    fprintf('\n✅  All MEX files built successfully.\n');
end
