% ===============================
% build_mex.m (Patched to always use local libtiff + LTO + dependencies)
% ===============================
% Compile semaphore, LZ4, and GPU MEX files using locally-compiled libtiff and dependencies.
% Always use the version in tiff_build/libtiff and never the system or Anaconda version.

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

% Dependency versions and install dirs
libtiff_src_version = '4.7.0';
libtiff_root = fullfile(pwd, 'tiff_src', ['tiff-', libtiff_src_version]);
libtiff_install_dir = fullfile(pwd, 'tiff_build', 'libtiff');

deps = {'libjpeg', 'zlib', 'liblzma', 'libjbig', 'libdeflate'};
for i = 1:numel(deps)
    eval([deps{i}, '_install_dir = fullfile(pwd, ''tiff_build'', ''deps'', ''', deps{i}, ''');']);
end

% Build all dependencies
stamp_file = fullfile(libtiff_install_dir, '.libtiff_installed');
if ~isfile(stamp_file)
    fprintf('Building libtiff from source with optimized flags...\n');
    if ~try_build_libtiff(libtiff_root, libtiff_install_dir, mex_flags_cpu, libtiff_src_version)
        error('Failed to build local libtiff. Please check the build log.');
    else
        if ~isfolder(libtiff_install_dir), error('libtiff install failed!'); end
        fclose(fopen(stamp_file, 'w'));
    end
end

for i = 1:numel(deps)
    dep = deps{i};
    dep_stamp = fullfile(eval([dep, '_install_dir']), ['.', dep, '_installed']);
    if ~isfile(dep_stamp)
        fprintf('Building %s from source with optimized flags...\n', dep);
        if ~try_build_dep(dep, eval([dep, '_install_dir']), mex_flags_cpu)
            error(['Failed to build ', dep]);
        else
            fclose(fopen(dep_stamp, 'w'));
        end
    end
end

% Set include and lib paths
libtiff_include = {['-I', fullfile(libtiff_install_dir, 'include')]};
libtiff_lib     = {['-L', fullfile(libtiff_install_dir, 'lib')]};
dep_includes = {}; dep_libs = {};
for i = 1:numel(deps)
    inc = ['-I', fullfile(eval([deps{i}, '_install_dir']), 'include')];
    lib = ['-L', fullfile(eval([deps{i}, '_install_dir']), 'lib')];
    dep_includes{end+1} = inc; %#ok<AGROW>
    dep_libs{end+1} = lib; %#ok<AGROW>
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
% Function: try_build_dep (for libjpeg, zlib, etc.)
% ===============================
function ok = try_build_dep(dep_name, install_dir, mex_flags_cpu)
    orig_dir = pwd;
    src_url_base = 'https://github.com/';
    archive = [dep_name, '.zip'];
    src_folder = fullfile('tiff_src', dep_name);

    % === Parse CFLAGS and CXXFLAGS from mex_flags_cpu ===
    CFLAGS = ''; CXXFLAGS = '';
    for i = 1:numel(mex_flags_cpu)
        token = mex_flags_cpu{i};
        if contains(token, 'CFLAGS=')
            m = regexp(token, 'CFLAGS="\$CFLAGS ([^"]+)"', 'tokens');
            if ~isempty(m), CFLAGS = strtrim(m{1}{1}); end
        elseif contains(token, 'CXXFLAGS=')
            m = regexp(token, 'CXXFLAGS="\$CXXFLAGS ([^"]+)"', 'tokens');
            if ~isempty(m), CXXFLAGS = strtrim(m{1}{1}); end
        end
    end

    % === Download and extract ===
    if ~isfolder(src_folder)
        url = '';  % fallback default
        switch dep_name
            case 'libjpeg'
                url = 'https://ijg.org/files/jpegsrc.v9e.tar.gz';
                archive = 'jpegsrc.v9e.tar.gz';
                src_folder = 'jpeg-9e';
            case 'zlib'
                url = 'https://zlib.net/zlib-1.3.tar.gz';
                archive = 'zlib-1.3.tar.gz';
                src_folder = 'zlib-1.3';
            case 'liblzma'
                url = 'https://tukaani.org/xz/xz-5.4.2.tar.gz';
                archive = 'xz-5.4.2.tar.gz';
                src_folder = 'xz-5.4.2';
            case 'libjbig'
                url = 'https://www.cl.cam.ac.uk/~mgk25/jbigkit/jbigkit-2.1.tar.gz';
                archive = 'jbigkit-2.1.tar.gz';
                src_folder = 'jbigkit-2.1';
            case 'libdeflate'
                url = 'https://github.com/ebiggers/libdeflate/archive/refs/tags/v1.19.zip';
                archive = 'libdeflate-1.19.zip';
                src_folder = fullfile('libdeflate-1.19');
        end

        fprintf('Downloading %s...\n', dep_name);
        if endsWith(archive, '.tar.gz')
            system(['curl -L -o ', archive, ' ', url]);
            system(['tar -xzf ', archive]);
        else
            system(['curl -L -o ', archive, ' ', url]);
            unzip(archive, 'tiff_src');
        end
        delete(archive);
    end
    cd(src_folder);

    % === Build ===
    if ispc
        setenv('CFLAGS', CFLAGS);
        setenv('CXXFLAGS', CXXFLAGS);
        cmd = [
            'cmake -B build -DCMAKE_BUILD_TYPE=Release ' ...
            '-DBUILD_SHARED_LIBS=OFF ' ...
            '-DCMAKE_C_FLAGS_RELEASE="/O2 /GL" ' ...
            '-DCMAKE_CXX_FLAGS_RELEASE="/O2 /GL" ' ...
            '-DCMAKE_INSTALL_PREFIX="', install_dir, '" . && ' ...
            'cmake --build build --config Release --target install'
        ];
        status = system(cmd);
        setenv('CFLAGS', '');
        setenv('CXXFLAGS', '');
    else
        prefix = '';
        if ~isempty(CFLAGS), prefix = [prefix, 'CFLAGS="', CFLAGS, '" ']; end
        if ~isempty(CXXFLAGS), prefix = [prefix, 'CXXFLAGS="', CXXFLAGS, '" ']; end
        conf = './configure';
        if strcmp(dep_name, 'liblzma'), conf = './configure --disable-shared'; end
        if strcmp(dep_name, 'libdeflate'), conf = ''; end  % uses Makefile directly
        if ~isempty(conf)
            cmd = [prefix, conf, ' --prefix=', install_dir, ...
                   ' && make -j4 && make install'];
        else
            cmd = ['make -j4 CFLAGS="', CFLAGS, '" && make install PREFIX=', install_dir];
        end
        status = system(cmd);
    end

    cd(orig_dir);
    ok = (status == 0);
end
