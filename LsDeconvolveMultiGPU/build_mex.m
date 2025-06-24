% build_mex.m - Static build of libtiff, zstd, zlib-ng and MEX files with LTO
% Supports Linux and Windows (Visual Studio 2022)

function build_mex_static(debug)
% Usage: build_mex_static(true|false)
if nargin<1, debug = false; end

% MATLAB version check
if verLessThan('matlab','9.4')
    error('Requires MATLAB R2018a or newer');
end
if exist('mexcuda','file')~=2
    error('mexcuda not found. Ensure CUDA is configured');
end

% Third-party versions
zlibng_version  = '2.2.4';
zstd_version    = '1.5.7';
libtiff_version = '4.7.0';

% Paths
root            = pwd;
thirdparty      = fullfile(root,'thirdparty');
build_root      = fullfile(root,'tiff_build');
zlibng_src      = fullfile(thirdparty,['zlib-ng-',zlibng_version]);
zstd_src        = fullfile(thirdparty,['zstd-',zstd_version]);
libtiff_src     = fullfile(thirdparty,['tiff-',libtiff_version]);
zlibng_install  = fullfile(build_root,'zlib-ng');
zstd_install    = fullfile(build_root,'zstd');
libtiff_install = fullfile(build_root,'libtiff');

% Ensure directories
if ~exist(thirdparty,'dir'), mkdir(thirdparty); end
if ~exist(build_root,'dir'), mkdir(build_root); end

%% Helper: CMake build & install
function status = try_cmake_build(src_dir, build_dir, install_dir, cmake_args)
    if ~exist(build_dir,'dir'), mkdir(build_dir); end
    old = pwd; cd(build_dir);
    cleanupObj = onCleanup(@() cd(old));
    cmd = sprintf('cmake "%s" -DCMAKE_INSTALL_PREFIX="%s" %s', src_dir, install_dir, cmake_args);
    status = system(cmd);
    if status~=0, cd(old); return; end
    if ispc
        cmd = 'cmake --build . --config Release --target INSTALL';
    else
        cmd = sprintf('cmake --build . -- -j%d install', feature('numCores'));
    end
    status = system(cmd);
    cd(old);
end

%% Build zlib-ng via CMake
zlibng_stamp = fullfile(zlibng_install,'.built');
if ~isfile(zlibng_stamp)
    fprintf('Building zlib-ng...\n');
    if ~exist(zlibng_src,'dir')
        archive = fullfile(thirdparty,sprintf('zlib-ng-%s.tar.gz',zlibng_version));
        url = sprintf('https://github.com/zlib-ng/zlib-ng/archive/refs/tags/v%s.tar.gz',zlibng_version);
        websave(archive,url); untar(archive,thirdparty); delete(archive);
    end
    cmake_args = sprintf([ ...
        '-DBUILD_SHARED_LIBS=OFF ',...
        '-DCMAKE_POSITION_INDEPENDENT_CODE=ON ',...
        '-DCMAKE_BUILD_TYPE=Release ',...
        '-DZLIB_COMPAT=ON ',...
        '-DCMAKE_C_FLAGS_RELEASE="-O3 -march=native -flto" ',...
        '-DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -flto"' ]);
    if try_cmake_build(zlibng_src, fullfile(zlibng_src,'build'), zlibng_install, cmake_args)~=0
        error('zlib-ng build failed');
    end
    fclose(fopen(zlibng_stamp,'w'));
end

%% Build zstd via Make on Linux, CMake on Windows
zstd_stamp = fullfile(zstd_install,'.built');
if ~isfile(zstd_stamp)
    fprintf('Building zstd...\n');
    if ~exist(zstd_src,'dir')
        archive = fullfile(thirdparty,sprintf('zstd-%s.tar.gz',zstd_version));
        url = sprintf('https://github.com/facebook/zstd/archive/refs/tags/v%s.tar.gz',zstd_version);
        websave(archive,url); untar(archive,thirdparty); delete(archive);
    end
    if ispc
        % Use CMake on Windows
        cmake_args = sprintf([ ...
            '-DBUILD_SHARED_LIBS=OFF ',...
            '-DCMAKE_POSITION_INDEPENDENT_CODE=ON ',...
            '-DCMAKE_BUILD_TYPE=Release ',...
            '-DCMAKE_C_FLAGS_RELEASE="-O3 -arch:AVX2 -GL" ',...
            '-DCMAKE_CXX_FLAGS_RELEASE="-O3 -arch:AVX2 -GL"' ]);
        if try_cmake_build(zstd_src, fullfile(zstd_src,'build'), zstd_install, cmake_args)~=0
            error('zstd build failed on Windows');
        end
    else
        % Linux: use Makefile
        old = pwd; cd(zstd_src);
        system(sprintf('make -j%d', feature('numCores')));
        system(sprintf('make PREFIX="%s" install', zstd_install));
        cd(old);
    end
    fclose(fopen(zstd_stamp,'w'));
end

%% Build libtiff via CMake
libtiff_stamp = fullfile(libtiff_install,'.built');
if ~isfile(libtiff_stamp)
    fprintf('Building libtiff...\n');
    if ~exist(libtiff_src,'dir')
        archive = fullfile(thirdparty,sprintf('tiff-%s.tar.gz',libtiff_version));
        url = sprintf('https://download.osgeo.org/libtiff/tiff-%s.tar.gz',libtiff_version);
        websave(archive,url); untar(archive,thirdparty); delete(archive);
    end
    cmake_args = sprintf([ ...
        '-DBUILD_SHARED_LIBS=OFF ',...
        '-DTIFF_DISABLE_JPEG=ON ',...
        '-DTIFF_DISABLE_JBIG=ON ',...
        '-DTIFF_DISABLE_LZMA=ON ',...
        '-DTIFF_DISABLE_WEBP=ON ',...
        '-DTIFF_DISABLE_LERC=ON ',...
        '-DTIFF_DISABLE_PIXARLOG=ON ',...
        '-DZLIB_LIBRARY="%s" ',...
        '-DZLIB_INCLUDE_DIR="%s" ',...
        '-DZSTD_LIBRARY="%s" ',...
        '-DZSTD_INCLUDE_DIR="%s" ',...
        '-DCMAKE_C_FLAGS_RELEASE="-O3 -march=native -flto" ',...
        '-DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -flto"' ],...
        fullfile(zlibng_install,'lib','libz.a'),...
        fullfile(zlibng_install,'include'),...
        fullfile(zstd_install,'lib','libzstd.a'),...
        fullfile(zstd_install,'include'));
    if try_cmake_build(libtiff_src, fullfile(libtiff_src,'build'), libtiff_install, cmake_args)~=0
        error('libtiff build failed');
    end
    fclose(fopen(libtiff_stamp,'w'));
end

%% MEX compile flags
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

%% Include and link flags for static libs
include_flags = {'-I',fullfile(libtiff_install,'include')};
lib_flags     = {'-L',fullfile(libtiff_install,'lib'), '-Wl,-Bstatic','-ltiff','-lzstd','-lz','-Wl,-Bdynamic'};

%% Build CPU MEX files
cpu_sources = {'semaphore.c','lz4.c','save_lz4_mex.c','load_lz4_mex.c','load_slab_lz4.cpp'};
fprintf('Building CPU MEX files...\n');
for k=1:numel(cpu_sources)
    mex(mex_cpu{:}, cpu_sources{k});
end
mex(mex_cpu{:}, 'load_bl_tif.cpp', include_flags{:}, lib_flags{:});
mex(mex_cpu{:}, 'save_bl_tif.cpp', include_flags{:}, lib_flags{:});

%% CUDA MEX compilation (unchanged)
if ispc
    if debug
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -G -std=c++17 -Xcompiler "/Od,/Zi"" ';
    else
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -O2 -std=c++17 -Xcompiler "/O2,/arch:AVX2"" ';
    end
else
    if debug
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -G -std=c++17 -Xcompiler ''-O0,-g''" ';
    else
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -O3 -std=c++17 -Xcompiler ''-O3,-march=native,-fomit-frame-pointer''" ';
    end
end
root_dir='.'; include_dir='./mex_incubator';
if ispc
    xmlfile=fullfile(fileparts(mfilename('fullpath')),'nvcc_msvcpp2022.xml'); assert(isfile(xmlfile)); cuda_mex_flags={'-f',xmlfile};
else
    cuda_mex_flags={};
end
mexcuda(cuda_mex_flags{:},'-R2018a','gauss3d_mex.cu',['-I',root_dir],['-I',include_dir],nvccflags);
mexcuda(cuda_mex_flags{:},'-R2018a','conv3d_mex.cu',['-I',root_dir],['-I',include_dir],nvccflags);
mexcuda(cuda_mex_flags{:},'-R2018a','otf_gpu_mex.cu',['-I',root_dir],['-I',include_dir],nvccflags,'-L/usr/local/cuda/lib64','-lcufft');

fprintf('All MEX files built successfully.\n');
end
