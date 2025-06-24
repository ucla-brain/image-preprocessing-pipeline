% build_mex.m - Static build of libtiff, zstd, zlib-ng, lz4 and MEX files with LTO
% Supports Linux and Windows (Visual Studio 2022)

function build_mex(debug)
    if nargin<1, debug = false; end

    % MATLAB version check
    if verLessThan('matlab','9.4')
        error('Requires MATLAB R2018a or newer');
    end
    if exist('mexcuda','file')~=2
        error('mexcuda not found. Ensure CUDA is configured');
    end

    % Versions
    zlibng_v  = '2.2.4';
    zstd_v    = '1.5.7';
    libtiff_v = '4.7.0';

    % Paths
    root          = pwd;
    thirdparty    = fullfile(root,'thirdparty');
    build_root    = fullfile(root,'tiff_build');
    zlibng_src    = fullfile(thirdparty,['zlib-ng-',zlibng_v]);
    zstd_src      = fullfile(thirdparty,['zstd-',zstd_v]);
    lz4_src       = fullfile(thirdparty,'lz4');
    libtiff_src   = fullfile(thirdparty,['tiff-',libtiff_v]);
    zlibng_inst   = fullfile(build_root,'zlib-ng');
    zstd_inst     = fullfile(build_root,'zstd');
    lz4_inst      = fullfile(build_root,'lz4');
    libtiff_inst  = fullfile(build_root,'libtiff');

    if ~exist(thirdparty,'dir'), mkdir(thirdparty); end
    if ~exist(build_root,'dir'), mkdir(build_root); end

    % Helper: CMake build & install
    function status = try_cmake(src, bld, inst, args)
        if ~exist(bld,'dir'), mkdir(bld); end
        old = pwd; cd(bld);
        onCleanup(@() cd(old));
        cmd = sprintf('cmake \"%s\" -DCMAKE_INSTALL_PREFIX=\"%s\" %s', src, inst, args);
        status = system(cmd);
        if status, cd(old); return; end
        if ispc
            cmd = 'cmake --build . --config Release --target INSTALL';
        else
            cmd = sprintf('cmake --build . -- -j%d install', feature('numCores'));
        end
        status = system(cmd);
        cd(old);
    end

    %% 1) LZ4 download & verify
    if ~exist(lz4_src,'dir'), mkdir(lz4_src); end
    lz4_c = fullfile(lz4_src,'lz4.c');
    lz4_h = fullfile(lz4_src,'lz4.h');
    if ~isfile(lz4_c)
        try websave(lz4_c,'https://raw.githubusercontent.com/lz4/lz4/dev/lib/lz4.c');
        catch, error('Failed to download lz4.c'); end
    end
    if ~isfile(lz4_h)
        try websave(lz4_h,'https://raw.githubusercontent.com/lz4/lz4/dev/lib/lz4.h');
        catch, error('Failed to download lz4.h'); end
    end
    if ~isfile(lz4_c)||~isfile(lz4_h)
        error('Missing lz4 sources in %s',lz4_src);
    end

    % Single token include flag for LZ4
    lz4_include = {['-I', lz4_src]};

    %% 2) zlib-ng
    stamp = fullfile(zlibng_inst,'.built');
    if ~isfile(stamp)
        fprintf('Building zlib-ng...\n');
        if ~exist(zlibng_src,'dir')
            archive = fullfile(thirdparty,sprintf('zlib-ng-%s.tar.gz',zlibng_v));
            websave(archive,sprintf('https://github.com/zlib-ng/zlib-ng/archive/refs/tags/%s.tar.gz',zlibng_v));
            untar(archive,thirdparty); delete(archive);
        end
        args = [ ...
            '-DBUILD_SHARED_LIBS=OFF ', ...
            '-DCMAKE_POSITION_INDEPENDENT_CODE=ON ', ...
            '-DCMAKE_BUILD_TYPE=Release ', ...
            '-DZLIB_COMPAT=ON ', ...
            '-DCMAKE_C_FLAGS_RELEASE="-O3 -march=native -flto" ', ...
            '-DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -flto"' ];
        if try_cmake(zlibng_src,fullfile(zlibng_src,'build'),zlibng_inst,args)
            error('zlib-ng build failed');
        end
        fclose(fopen(stamp,'w'));
    end

    %% 3) zstd
    stamp = fullfile(zstd_inst,'.built');
    if ~isfile(stamp)
        fprintf('Building zstd...\n');
        if ~exist(zstd_src,'dir')
            archive = fullfile(thirdparty,sprintf('zstd-%s.tar.gz',zstd_v));
            websave(archive,sprintf('https://github.com/facebook/zstd/archive/refs/tags/v%s.tar.gz',zstd_v));
            untar(archive,thirdparty); delete(archive);
        end
        if ispc
            args = [ ...
                '-DBUILD_SHARED_LIBS=OFF ', ...
                '-DCMAKE_POSITION_INDEPENDENT_CODE=ON ', ...
                '-DCMAKE_BUILD_TYPE=Release ', ...
                '-DCMAKE_C_FLAGS_RELEASE="-O3 -arch:AVX2 -GL" ', ...
                '-DCMAKE_CXX_FLAGS_RELEASE="-O3 -arch:AVX2 -GL"' ];
            if try_cmake(zstd_src,fullfile(zstd_src,'build'),zstd_inst,args)
                error('zstd build failed');
            end
        else
            old=pwd; cd(zstd_src);
            system(sprintf('make -j%d',feature('numCores')));
            system(sprintf('make PREFIX="%s" install',zstd_inst));
            cd(old);
        end
        fclose(fopen(stamp,'w'));
    end

    %% 4) libtiff (LZW,LZ4,ZLIB,ZSTD)
    stamp = fullfile(libtiff_inst,'.built');
    if ~isfile(stamp)
        fprintf('Building libtiff...\n');
        if ~exist(libtiff_src,'dir')
            archive = fullfile(thirdparty,sprintf('tiff-%s.tar.gz',libtiff_v));
            websave(archive,sprintf('https://download.osgeo.org/libtiff/tiff-%s.tar.gz',libtiff_v));
            untar(archive,thirdparty); delete(archive);
        end
        cm_args = sprintf([ ...
            '-DBUILD_SHARED_LIBS=OFF ', ...
            '-DTIFF_DISABLE_JPEG=ON ', ...
            '-DTIFF_DISABLE_OLD_JPEG=ON ', ...
            '-DTIFF_DISABLE_JBIG=ON ', ...
            '-DTIFF_DISABLE_LZMA=ON ', ...
            '-DTIFF_DISABLE_WEBP=ON ', ...
            '-DTIFF_DISABLE_LERC=ON ', ...
            '-DTIFF_DISABLE_PIXARLOG=ON ', ...
            '-DTIFF_ENABLE_LZW=ON ', ...
            '-DTIFF_ENABLE_LZ4=ON ', ...
            '-DTIFF_ENABLE_ZLIB=ON ', ...
            '-DTIFF_ENABLE_ZSTD=ON ', ...
            '-Dtiff-opengl=OFF ', ...
            '-DLZ4_LIBRARY="%s" ', ...
            '-DLZ4_INCLUDE_DIR="%s" ', ...
            '-DZLIB_LIBRARY="%s" ', ...
            '-DZLIB_INCLUDE_DIR="%s" ', ...
            '-DZSTD_LIBRARY="%s" ', ...
            '-DZSTD_INCLUDE_DIR="%s" ', ...
            '-DCMAKE_C_FLAGS_RELEASE="-O3 -march=native -flto" ', ...
            '-DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -flto"' ], ...
            fullfile(lz4_inst,'lib','liblz4.a'), ...
            fullfile(lz4_inst,'include'), ...
            fullfile(zlibng_inst,'lib','libz.a'), ...
            fullfile(zlibng_inst,'include'), ...
            fullfile(zstd_inst,'lib','libzstd.a'), ...
            fullfile(zstd_inst,'include'));
        if try_cmake(libtiff_src,fullfile(libtiff_src,'build'),libtiff_inst,cm_args)
            error('libtiff build failed');
        end
        fclose(fopen(stamp,'w'));
    end

    %% 5) MEX flags
    if ispc
        if debug
            mex_cpu = {'-R2018a','COMPFLAGS="$COMPFLAGS /std:c++17 /Od /Zi"','LINKFLAGS="$LINKFLAGS /DEBUG"'};
        else
            mex_cpu = {'-R2018a','COMPFLAGS="$COMPFLAGS /std:c++17 /O2 /arch:AVX2 /GL"','LINKFLAGS="$LINKFLAGS /LTCG"'};
        end
    else
        if debug
            mex_cpu = {'-R2018a','CFLAGS="$CFLAGS -O0 -g"','CXXFLAGS="$CXXFLAGS"','LDFLAGS="$LDFLAGS -g"'};
        else
            mex_cpu = {'-R2018a','CFLAGS="$CFLAGS -O3 -march=native -flto"','CXXFLAGS="$CXXFLAGS -O3 -march=native -flto"','LDFLAGS="$LDFLAGS -flto"'};
        end
    end

    %% 6) Include & link for TIFF
    include_flags = {'-I',fullfile(libtiff_inst,'include')};
    lib_flags     = {'-L',fullfile(libtiff_inst,'lib'),'-Wl,-Bstatic','-ltiff','-llz4','-lzstd','-lz','-Wl,-Bdynamic'};

    %% 7) Build CPU MEX with LZ4 include
    cpu_sources = { 'semaphore.c', lz4_c, 'save_lz4_mex.c', 'load_lz4_mex.c', 'load_slab_lz4.cpp' };
    fprintf('Building CPU MEX files...\n');
    for k=1:numel(cpu_sources)
        mex(mex_cpu{:}, cpu_sources{k}, lz4_include{:});
    end

    %% 8) TIFF MEX
    mex(mex_cpu{:},'load_bl_tif.cpp',include_flags{:},lib_flags{:});
    mex(mex_cpu{:},'save_bl_tif.cpp',include_flags{:},lib_flags{:});

    %% 9) CUDA MEX compilation (unchanged)
    if ispc
        if debug
            nvccflags = 'NVCCFLAGS="$NVCCFLAGS -G -std=c++17 -Xcompiler \"/Od,/Zi\"\" ';
        else
            nvccflags = 'NVCCFLAGS="$NVCCFLAGS -O2 -std=c++17 -Xcompiler \"/O2,/arch:AVX2\"\" ';
        end
    else
        if debug
            nvccflags = 'NVCCFLAGS="$NVCCFLAGS -G -std=c++17 -Xcompiler ''-O0,-g''\" ';
        else
            nvccflags = 'NVCCFLAGS="$NVCCFLAGS -O3 -std=c++17 -Xcompiler ''-O3,-march=native,-fomit-frame-pointer''\" ';
        end
    end
    root_dir = '.'; include_dir = './mex_incubator';
    if ispc
        xmlfile = fullfile(fileparts(mfilename('fullpath')),'nvcc_msvcpp2022.xml');
        assert(isfile(xmlfile),'nvcc_msvcpp2022.xml not found!');
        cuda_mex_flags = {'-f',xmlfile};
    else
        cuda_mex_flags = {};
    end
    mexcuda(cuda_mex_flags{:},'-R2018a','gauss3d_mex.cu',['-I',root_dir],['-I',include_dir],nvccflags);
    mexcuda(cuda_mex_flags{:},'-R2018a','conv3d_mex.cu',['-I',root_dir],['-I',include_dir],nvccflags);
    mexcuda(cuda_mex_flags{:},'-R2018a','otf_gpu_mex.cu',['-I',root_dir],['-I',include_dir],nvccflags,'-L/usr/local/cuda/lib64','-lcufft');

    fprintf('All MEX files built successfully.\\n');
end
