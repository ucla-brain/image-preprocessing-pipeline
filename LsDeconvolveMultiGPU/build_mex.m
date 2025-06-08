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

    % === Parse compiler flags ===
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

    % === Get library info ===
    [archive, folder_name, url, version] = get_library_info(lib);
    cd(fileparts(mfilename('fullpath')));

    % === Download and extract ===
    if ~isfolder(folder_name)
        if ~isfile(archive)
            fprintf('Downloading %s...\n', archive);
            system(sprintf('curl -L -o "%s" "%s"', archive, url));
        end

        if endsWith(archive, '.tar.gz')
            system(sprintf('tar -xzf "%s"', archive));
            actual_folder = get_top_level_folder(archive);
            if ~strcmp(actual_folder, folder_name)
                movefile(actual_folder, folder_name);
            end
        elseif endsWith(archive, '.zip')
            unzip(archive);
            actual_folder = get_top_level_folder(archive);
            if ~strcmp(actual_folder, folder_name)
                movefile(actual_folder, folder_name);
            end
        else
            error('Unsupported archive format: %s', archive);
        end

        actual_folder = get_top_level_folder(archive);
        if ~strcmp(actual_folder, folder_name)
            movefile(actual_folder, folder_name);
        end

        delete(archive);
    end

    cd(folder_name);

    % === Build phase ===
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
        prefix = '';
        if ~isempty(CFLAGS), prefix = [prefix, 'CFLAGS="', CFLAGS, '" ']; end
        if ~isempty(CXXFLAGS), prefix = [prefix, 'CXXFLAGS="', CXXFLAGS, '" ']; end
        if ~isempty(LDFLAGS), prefix = [prefix, 'LDFLAGS="', LDFLAGS, '" ']; end

        if strcmp(lib, 'libdeflate')
            fprintf('Building libdeflate using CMake...\n');

            cmake_flags = [
                '-DCMAKE_BUILD_TYPE=Release ', ...
                '-DCMAKE_INSTALL_PREFIX="', install_dir, '" ', ...
                '-DBUILD_SHARED_LIBS=OFF '
            ];

            if ispc
                setenv('CFLAGS', CFLAGS); setenv('CXXFLAGS', CXXFLAGS); setenv('LDFLAGS', LDFLAGS);
                cmd = ['cmake -B build ', cmake_flags, ' . && cmake --build build --config Release --target install'];
                status = system(cmd);
                setenv('CFLAGS', ''); setenv('CXXFLAGS', ''); setenv('LDFLAGS', '');
            else
                cmd = sprintf('cmake -B build %s . && cmake --build build --target install -- -j4', cmake_flags);
                status = system(cmd);
            end
        elseif strcmp(lib, 'libjbig')
            fprintf('Building libjbig using make -C libjbig...\n');
            cmd = ['make -j4 -C libjbig CFLAGS="', CFLAGS, '"'];
        else
            cmd = [prefix, './configure --disable-shared --enable-static --prefix=', install_dir, ...
                   ' && make -j4 && make install'];
        end

        [status, output] = system(cmd);
        if status ~= 0
            warning('%s build failed:\n%s', lib, output);
        end
    end

    ok = (status == 0);

    % === Manual install fallback ===
    if ok && ~isfolder(fullfile(install_dir, 'lib'))
        fprintf('Manual install fallback for %s...\n', lib);
        try
            mkdir(fullfile(install_dir, 'lib'));
            mkdir(fullfile(install_dir, 'include'));
            headers = dir(fullfile(pwd, '**', '*.h'));
            static_libs = dir(fullfile(pwd, '**', '*.a'));
            for k = 1:numel(headers)
                dst = fullfile(install_dir, 'include', headers(k).name);
                copyfile(fullfile(headers(k).folder, headers(k).name), dst);
            end
            for k = 1:numel(static_libs)
                dst = fullfile(install_dir, 'lib', static_libs(k).name);
                copyfile(fullfile(static_libs(k).folder, static_libs(k).name), dst);
            end
        catch ME
            warning('Manual install of %s failed: %s', lib, ME.message);
            ok = false;
        end
    end

    % === Clean shared libs ===
    if exist(fullfile(install_dir, 'lib'), 'dir')
        delete(fullfile(install_dir, 'lib', '*.so*'));
        delete(fullfile(install_dir, 'lib', '*.dylib'));
    end

    cd(orig_dir);

    % === Optional libtiff validation test ===
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

function [archive, folder_name, url, version] = get_library_info(lib)
    switch lib
        case 'libtiff'
            version = '4.7.0';
            archive = ['tiff-', version, '.tar.gz'];
            folder_name = ['tiff-', version];
            url = ['https://download.osgeo.org/libtiff/', archive];

        case 'libjpeg'
            version = '9e';
            archive = ['jpegsrc.v', version, '.tar.gz'];
            folder_name = ['jpeg-', version];
            url = ['https://ijg.org/files/', archive];

        case 'zlib'
            version = '1.3.1';
            archive = ['zlib-', version, '.tar.gz'];
            folder_name = ['zlib-', version];
            url = ['https://zlib.net/', archive];

        case 'liblzma'
            version = '5.4.2';
            archive = ['xz-', version, '.tar.gz'];
            folder_name = ['xz-', version];
            url = ['https://tukaani.org/xz/', archive];

        case 'libjbig'
            version = '2.1';
            archive = ['jbigkit-', version, '.tar.gz'];
            folder_name = ['jbigkit-', version];
            url = ['https://www.cl.cam.ac.uk/~mgk25/jbigkit/download/', archive];

        case 'libdeflate'
            version = '1.24';
            archive = ['libdeflate-', version, '.tar.gz'];
            folder_name = ['libdeflate-', version];
            url = ['https://github.com/ebiggers/libdeflate/releases/download/v', version, '/', archive];

        otherwise
            error('Unknown library: %s', lib);
    end
end

function top = get_top_level_folder(archive_file)
    % Extracts the top-level folder name from a tar.gz or zip archive
    [~, tmp_name] = fileparts(tempname);
    tmp_dir = fullfile(tempdir, tmp_name);
    mkdir(tmp_dir);

    list_path = fullfile(tmp_dir, 'list.txt');

    if endsWith(archive_file, '.tar.gz')
        cmd = sprintf('tar -tzf "%s" > "%s"', archive_file, list_path);
    elseif endsWith(archive_file, '.zip')
        cmd = sprintf('unzip -l "%s" > "%s"', archive_file, list_path);
    else
        error('Unsupported archive format: %s', archive_file);
    end

    status = system(cmd);
    if status ~= 0 || ~isfile(list_path)
        error('Failed to list contents of archive: %s', archive_file);
    end

    lines = strsplit(fileread(list_path), '\n');

    % Extract top-level folders from paths
    folders = {};
    for i = 1:numel(lines)
        line = strtrim(lines{i});
        if isempty(line) || startsWith(line, 'Length') || startsWith(line, 'Archive')
            continue;
        end
        % Match something like "foldername/file" or "folder/"
        match = regexp(line, '^([^/\\]+)[/\\]', 'tokens', 'once');
        if ~isempty(match)
            folders{end+1} = match{1}; %#ok<AGROW>
        end
    end

    folders = unique(folders);
    delete(list_path); rmdir(tmp_dir);

    if isempty(folders)
        % Fallback: use archive name (no top-level folder found)
        [~, top, ~] = fileparts(archive_file);
        fprintf('No top-level folder found. Using fallback: %s\n', top);
    elseif numel(folders) == 1
        top = folders{1};
    else
        error('Archive contains multiple top-level folders: %s', strjoin(folders, ', '));
    end
end
