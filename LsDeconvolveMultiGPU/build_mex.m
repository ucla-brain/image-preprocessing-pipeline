% ===============================
% build_mex.m (Patched to always use local libtiff + LTO + dependencies)
% ===============================
% Compile semaphore, LZ4, and GPU MEX files using locally-compiled libtiff and dependencies.
% Always use the version in tiff_build/libtiff and never the system or Anaconda version.

debug = false;
force_rebuild = ~isempty(getenv('FORCE_REBUILD'));

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

% CPU compile flags
if ispc
    if debug
        mex_flags_cpu = {
            '-R2018a', ...
            'COMPFLAGS="$COMPFLAGS /std:c++17 /Od /Zi /openmp"', ...
            'LINKFLAGS="$LINKFLAGS /DEBUG"'
        };
    else
        mex_flags_cpu = {
            '-R2018a', ...
            'COMPFLAGS="$COMPFLAGS /std:c++17 /O2 /arch:AVX2 /Ot /GL /openmp"', ...
            'LINKFLAGS="$LINKFLAGS /LTCG"'
        };
    end
else
    if debug
        mex_flags_cpu = {
            '-R2018a', ...
            'CFLAGS="$CFLAGS -O0 -g -fopenmp"', ...
            'CXXFLAGS="$CXXFLAGS -O0 -g -fopenmp"', ...
            'LDFLAGS="$LDFLAGS -g -fopenmp"'
        };
    else
        mex_flags_cpu = {
            '-R2018a', ...
            'CFLAGS="$CFLAGS -O2 -march=native -fomit-frame-pointer -fopenmp -flto"', ...
            'CXXFLAGS="$CXXFLAGS -O2 -march=native -fomit-frame-pointer -fopenmp -flto"', ...
            'LDFLAGS="$LDFLAGS -flto -fopenmp"'
        };
    end
end

% Build all third-party libraries using unified try_build_library
libs = {'libtiff', 'libjpeg', 'zlib', 'liblzma', 'libjbig', 'libdeflate'};
for i = 1:numel(libs)
    lib = libs{i};
    install_dir = fullfile(pwd, 'tiff_build', lib);
    stamp_file = fullfile(install_dir, ['.', lib, '_installed']);
    src_dir = fullfile(pwd, 'tiff_src', lib);
    if force_rebuild || ~isfile(stamp_file)
        fprintf('Building %s from source with optimized flags...\n', lib);
        if ~try_build_library(lib, src_dir, install_dir, mex_flags_cpu)
            error('Failed to build %s', lib);
        else
            fid = fopen(stamp_file, 'w'); if fid > 0, fclose(fid); end
        end
    end
end

% Set include and lib paths
libtiff_include = {['-I', fullfile(libtiff_install_dir, 'include')]};
libtiff_lib     = {['-L', fullfile(libtiff_install_dir, 'lib')]};
dep_includes = {}; dep_libs = {};
for i = 1:numel(libs)
    inc = ['-I', fullfile(pwd, 'tiff_build', libs{i}, 'include')];
    lib = ['-L', fullfile(pwd, 'tiff_build', libs{i}, 'lib')];
    dep_includes{end+1} = inc;
    dep_libs{end+1} = lib;
end

% Combined link flags
tiff_link = {
    '-ltiff', '-ljpeg', '-lz', '-llzma', '-ljbig', '-ldeflate'
};

fprintf('Using libtiff from: %s\n', fullfile(libtiff_install_dir, 'lib'));

% Build CPU MEX files
mex(mex_flags_cpu{:}, src_semaphore);
mex(mex_flags_cpu{:}, src_lz4_save, src_lz4_c);
mex(mex_flags_cpu{:}, src_lz4_load, src_lz4_c);
mex(mex_flags_cpu{:}, src_load_bl, ...
    libtiff_include{:}, dep_includes{:}, ...
    libtiff_lib{:}, dep_libs{:}, ...
    tiff_link{:});
% CUDA optimization flags
if ispc
    if debug
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -G -std:c++17 -Xcompiler ""/Od,/Zi"" "'; %#ok<NASGU>
    else
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -O2 -std:c++17 -Xcompiler ""/O2,/arch:AVX2,/openmp"" "'; %#ok<NASGU>
    end
else
    if debug
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -G -std:c++17 -Xcompiler ''-O0,-g'' "'; %#ok<NASGU>
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
mexcuda(cuda_mex_flags{:}, '-R2018a', src_gauss3d , ['-I', root_dir], ['-I', include_dir], nvccflags);
mexcuda(cuda_mex_flags{:}, '-R2018a', src_conv3d  , ['-I', root_dir], ['-I', include_dir], nvccflags);
mexcuda(cuda_mex_flags{:}, '-R2018a', src_otf_gpu , ['-I', root_dir], ['-I', include_dir], nvccflags, '-L/usr/local/cuda/lib64', '-lcufft');

fprintf('All MEX files built successfully.\n');

% ===============================
% Function: try_build_library (generic for libtiff + deps)
% ===============================
function ok = try_build_library(lib, src_dir, install_dir, mex_flags_cpu)
    orig_dir = pwd;
    ok = false;

    % === Parse CFLAGS, CXXFLAGS, LDFLAGS from mex_flags_cpu ===
    CFLAGS = ''; CXXFLAGS = ''; LDFLAGS = '';
    for i = 1:numel(mex_flags_cpu)
        token = mex_flags_cpu{i};
        if contains(token, 'CFLAGS=')
            m = regexp(token, 'CFLAGS="\$CFLAGS ([^"]+)"', 'tokens'); if ~isempty(m), CFLAGS = strtrim(m{1}{1}); end
        elseif contains(token, 'CXXFLAGS=')
            m = regexp(token, 'CXXFLAGS="\$CXXFLAGS ([^"]+)"', 'tokens'); if ~isempty(m), CXXFLAGS = strtrim(m{1}{1}); end
        elseif contains(token, 'LDFLAGS=')
            m = regexp(token, 'LDFLAGS="\$LDFLAGS ([^"]+)"', 'tokens'); if ~isempty(m), LDFLAGS = strtrim(m{1}{1}); end
        end
    end

    % === Source download settings ===
    archive = ''; folder_name = ''; url = '';
    switch lib
        case 'libtiff'
            archive = 'tiff-4.7.0.tar.gz';
            folder_name = 'tiff-4.7.0';
            url = 'https://download.osgeo.org/libtiff/tiff-4.7.0.tar.gz';
        case 'libjpeg'
            archive = 'jpegsrc.v9e.tar.gz';
            folder_name = 'jpeg-9e';
            url = 'https://ijg.org/files/jpegsrc.v9e.tar.gz';
        case 'zlib'
            archive = 'zlib-1.3.1.tar.gz';
            folder_name = 'zlib-1.3.1';
            url = 'https://zlib.net/zlib-1.3.1.tar.gz';
        case 'liblzma'
            archive = 'xz-5.4.2.tar.gz';
            folder_name = 'xz-5.4.2';
            url = 'https://tukaani.org/xz/xz-5.4.2.tar.gz';
        case 'libjbig'
            archive = 'jbigkit-2.1.tar.gz';
            folder_name = 'jbigkit-2.1';
            url = 'https://www.cl.cam.ac.uk/~mgk25/jbigkit/jbigkit-2.1.tar.gz';
        case 'libdeflate'
            archive = 'libdeflate-1.19.zip';
            folder_name = 'libdeflate-1.19';
            url = 'https://github.com/ebiggers/libdeflate/archive/refs/tags/v1.19.zip';
        otherwise
            error('Unknown library: %s', lib);
    end

    % === Download and extract ===
    if ~isfolder(folder_name)
        fprintf('Downloading %s...\n', lib);
        if ~isfile(archive)
            system(sprintf('curl -L -o "%s" "%s"', archive, url));
        end
        if endsWith(archive, '.tar.gz')
            untar(archive);
        elseif endsWith(archive, '.zip')
            unzip(archive);
        else
            error('Unsupported archive format: %s', archive);
        end
        delete(archive);
    end

    cd(folder_name);

    % === Build on Windows using CMake ===
    if ispc
        setenv('CFLAGS', CFLAGS); setenv('CXXFLAGS', CXXFLAGS); setenv('LDFLAGS', LDFLAGS);
        cmake_flags = [
            '-DCMAKE_BUILD_TYPE=Release ', ...
            '-DCMAKE_INSTALL_PREFIX="', install_dir, '" ', ...
            '-DBUILD_SHARED_LIBS=OFF ', ...
            '-DCMAKE_C_FLAGS_RELEASE="/O2 /GL" ', ...
            '-DCMAKE_CXX_FLAGS_RELEASE="/O2 /GL" ', ...
            '-DCMAKE_SHARED_LINKER_FLAGS_RELEASE="/LTCG" ', ...
            '-DCMAKE_EXE_LINKER_FLAGS_RELEASE="/LTCG" '
        ];
        status = system(['cmake -B build ', cmake_flags, ' . && cmake --build build --config Release --target install']);
        setenv('CFLAGS', ''); setenv('CXXFLAGS', ''); setenv('LDFLAGS', '');
    else
        % === Build on Linux using ./configure or Make ===
        prefix = '';
        if ~isempty(CFLAGS), prefix = [prefix, 'CFLAGS="', CFLAGS, '" ']; end
        if ~isempty(CXXFLAGS), prefix = [prefix, 'CXXFLAGS="', CXXFLAGS, '" ']; end
        if ~isempty(LDFLAGS), prefix = [prefix, 'LDFLAGS="', LDFLAGS, '" ']; end

        if strcmp(lib, 'libdeflate')
            cmd = ['make -j4 CFLAGS="', CFLAGS, '" && make install PREFIX=', install_dir];
        elseif strcmp(lib, 'libjbig')
            cmd = ['make -j4 CFLAGS="', CFLAGS, '" lib && make install PREFIX=', install_dir];
        else
            cmd = [prefix, './configure --disable-shared --enable-static --prefix=', install_dir, ...
                   ' && make -j4 && make install'];
        end

        status = system(cmd);
    end

    cd(orig_dir);
    ok = (status == 0);

    % === Optional: libtiff MEX test ===
    if ok && strcmp(lib, 'libtiff')
        test_c = 'libtiff_test_mex.c';
        fid = fopen(test_c, 'w');
        fprintf(fid, '#include "mex.h"\n#include "tiffio.h"\nvoid mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {\nprintf("LIBTIFF version: %%s\\n", TIFFGetVersion());\n}');
        fclose(fid);
        include_flag = ['-I', fullfile(install_dir, 'include')];
        lib_flag     = ['-L', fullfile(install_dir, 'lib')];
        link_flag    = {'-ltiff'};
        try
            mex(mex_flags_cpu{:}, test_c, include_flag, lib_flag, link_flag{:});
            delete(test_c);
        catch ME
            warning('Post-build MEX libtiff test failed: %s', ME.message);
            ok = false;
        end
    end
end
