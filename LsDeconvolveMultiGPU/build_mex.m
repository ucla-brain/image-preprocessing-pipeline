function build_mex()
    % ===============================
    % build_mex.m — Modular MEX builder for local libtiff and dependencies
    % ===============================
    % Builds:
    %   - libtiff and all its dependencies (libjpeg, zlib, etc.)
    %   - Various MEX files (LZ4, semaphore, load_bl_tif, CUDA modules)
    % ===============================

    % Debug mode toggle
    debug = false;

    % === Compiler flags (CPU) ===
    mex_flags_cpu = get_mex_flags_cpu(debug);

    % === List of required libraries ===
    libs = {'libtiff', 'libjpeg', 'zlib', 'liblzma', 'libjbig', 'libdeflate'};
    lib_dirs = struct();

    % === Build third-party libraries ===
    for i = 1:numel(libs)
        lib = libs{i};
        install_dir = fullfile(pwd, 'tiff_build', lib);
        src_dir = fullfile(pwd, 'tiff_src', lib);
        stamp_file = fullfile(install_dir, ['.', lib, '_installed']);

        if ~isfile(stamp_file)
            fprintf('\nBuilding %s from source...\n', lib);
            ok = try_build_library(lib, src_dir, install_dir, mex_flags_cpu);
            if ~ok
                error('❌ Failed to build %s', lib);
            else
                fclose(fopen(stamp_file, 'w'));
            end
        end
        lib_dirs.(lib) = install_dir;
    end

    % === Set link/include flags ===
    [include_flags, lib_flags, link_flags] = get_link_flags(lib_dirs);

    % === Build MEX files ===
    build_all_mex_files(mex_flags_cpu, include_flags, lib_flags, link_flags);

    fprintf('\n✅ All MEX files built successfully.\n');
end

function build_all_mex_files(mex_flags_cpu, ...
                              libtiff_install_dir, dep_install_dirs, ...
                              src_files, nvccflags, cuda_mex_flags)

    % === MEX Sources ===
    src_semaphore = src_files.semaphore;
    src_lz4_save  = src_files.lz4_save;
    src_lz4_load  = src_files.lz4_load;
    src_lz4_c     = src_files.lz4_c;
    src_load_bl   = src_files.load_bl;
    src_gauss3d   = src_files.gauss3d;
    src_conv3d    = src_files.conv3d;
    src_otf_gpu   = src_files.otf_gpu;
    src_deconFFT  = src_files.deconFFT;

    % === Include & Link Paths ===
    libtiff_include = {['-I', fullfile(libtiff_install_dir, 'include')]};
    libtiff_lib     = {['-L', fullfile(libtiff_install_dir, 'lib')]};

    dep_includes = {};
    dep_libs     = {};
    for i = 1:numel(dep_install_dirs)
        dep_includes{end+1} = ['-I', fullfile(dep_install_dirs{i}, 'include')]; %#ok<AGROW>
        dep_libs{end+1}     = ['-L', fullfile(dep_install_dirs{i}, 'lib')];     %#ok<AGROW>
    end

    link_flags = {
        '-ltiff', '-ljpeg', '-lz', '-llzma', '-ljbig', '-ldeflate'
    };

    % === Build CPU MEX files ===
    mex(mex_flags_cpu{:}, src_semaphore);
    mex(mex_flags_cpu{:}, src_lz4_save, src_lz4_c);
    mex(mex_flags_cpu{:}, src_lz4_load, src_lz4_c);
    mex(mex_flags_cpu{:}, src_load_bl, ...
        libtiff_include{:}, dep_includes{:}, ...
        libtiff_lib{:}, dep_libs{:}, ...
        link_flags{:});

    % === CUDA MEX Files ===
    root_dir    = '.';
    include_dir = './mex_incubator';

    mexcuda(cuda_mex_flags{:}, '-R2018a', src_gauss3d , ...
        ['-I', root_dir], ['-I', include_dir], nvccflags);

    mexcuda(cuda_mex_flags{:}, '-R2018a', src_conv3d  , ...
        ['-I', root_dir], ['-I', include_dir], nvccflags);

    mexcuda(cuda_mex_flags{:}, '-R2018a', src_otf_gpu , ...
        ['-I', root_dir], ['-I', include_dir], nvccflags, ...
        '-L/usr/local/cuda/lib64', '-lcufft');

    fprintf('✅ All MEX files built successfully.\n');
end

% ===============================
% Function: try_build_library (generic for libtiff + deps)
% ===============================
function ok = try_build_library(lib, src_dir, install_dir, mex_flags_cpu)
    orig_dir = pwd;
    ok = false;

    % === Parse compiler flags ===
    [CFLAGS, CXXFLAGS, LDFLAGS] = parse_mex_flags(mex_flags_cpu);

    % === Get library info ===
    [archive, folder_name, url, version] = get_library_info(lib);
    cd(fileparts(mfilename('fullpath')));

    % === Download and extract ===
    if ~isfolder(folder_name)
        if ~isfile(archive)
            fprintf('Downloading %s...\n', archive);
            status = system(sprintf('curl -L -o "%s" "%s"', archive, url));
            if status ~= 0, error('Download failed for %s', archive); end
        end

        if endsWith(archive, '.tar.gz')
            system(sprintf('tar -xzf "%s"', archive));
        elseif endsWith(archive, '.zip')
            unzip(archive);
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

    % === Build ===
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
            cmd = ['cmake -DCMAKE_INSTALL_PREFIX=', install_dir, ' -DBUILD_SHARED_LIBS=OFF . && make -j4 && make install'];
        elseif strcmp(lib, 'libjbig')
            fprintf('Building libjbig using make -C libjbig...\n');
            cmd = ['make -j4 -C libjbig CFLAGS="', CFLAGS, '"'];
        else
            cmd = [prefix, './configure --disable-shared --enable-static --prefix=', install_dir, ...
                   ' && make -j4 && make install'];
        end
        status = system(cmd);
    end

    ok = (status == 0);

    % === Fallback: Manual install for headers and static libs ===
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

    % === Remove shared libraries ===
    if exist(fullfile(install_dir, 'lib'), 'dir')
        delete(fullfile(install_dir, 'lib', '*.so*'));
        delete(fullfile(install_dir, 'lib', '*.dylib'));
    end

    cd(orig_dir);

    % === Optional test for libtiff ===
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

function top_folder = get_top_level_folder(archive)
    % Attempt to determine the top-level folder of an archive
    [~, ~, ext] = fileparts(archive);

    if endsWith(archive, '.tar.gz')
        [~, output] = system(sprintf('tar -tzf "%s"', archive));
    elseif endsWith(archive, '.zip')
        zipInfo = unzip(archive, tempname); % dry-run unzip to inspect
        % Detect top-level folder from file paths
        top_dirs = cellfun(@(x) regexp(x, '^[^/\\]+', 'match', 'once'), zipInfo, 'UniformOutput', false);
        top_dirs = unique(top_dirs(~cellfun('isempty', top_dirs)));
        if numel(top_dirs) ~= 1
            error('Unable to determine top-level folder from zip: %s', archive);
        end
        top_folder = top_dirs{1};
        return;
    else
        error('Unsupported archive format: %s', archive);
    end

    lines = strsplit(strtrim(output), newline);
    tokens = regexp(lines, '^([^/\\]+)/', 'tokens');
    top_dirs = unique(cellfun(@(x) x{1}, tokens(~cellfun('isempty', tokens)), 'UniformOutput', false));

    if numel(top_dirs) ~= 1
        error('Could not determine top-level folder from archive: %s', archive);
    end

    top_folder = top_dirs{1};
end

function flags = get_mex_flags_cpu(debug)
    if ispc
        if debug
            flags = {
                '-R2018a', ...
                'COMPFLAGS="$COMPFLAGS /std:c++17 /Od /Zi /openmp"', ...
                'LINKFLAGS="$LINKFLAGS /DEBUG"'
            };
        else
            flags = {
                '-R2018a', ...
                'COMPFLAGS="$COMPFLAGS /std:c++17 /O2 /arch:AVX2 /Ot /GL /openmp"', ...
                'LINKFLAGS="$LINKFLAGS /LTCG"'
            };
        end
    else
        if debug
            flags = {
                '-R2018a', ...
                'CFLAGS="$CFLAGS -O0 -g -fopenmp"', ...
                'CXXFLAGS="$CXXFLAGS -O0 -g -fopenmp"', ...
                'LDFLAGS="$LDFLAGS -g -fopenmp"'
            };
        else
            flags = {
                '-R2018a', ...
                'CFLAGS="$CFLAGS -O2 -march=native -fomit-frame-pointer -fopenmp -flto"', ...
                'CXXFLAGS="$CXXFLAGS -O2 -march=native -fomit-frame-pointer -fopenmp -flto"', ...
                'LDFLAGS="$LDFLAGS -flto -fopenmp"'
            };
        end
    end
end

function [include_flags, lib_flags, link_flags] = get_link_flags(libs, base_dir)
    include_flags = {};
    lib_flags     = {};
    link_flags    = {};

    for i = 1:numel(libs)
        lib = libs{i};
        install_dir = fullfile(base_dir, lib);
        include_flags{end+1} = ['-I', fullfile(install_dir, 'include')]; %#ok<AGROW>
        lib_flags{end+1}     = ['-L', fullfile(install_dir, 'lib')];     %#ok<AGROW>
    end

    link_flags = {
        '-ltiff', '-ljpeg', '-lz', '-llzma', '-ljbig', '-ldeflate'
    };
end

