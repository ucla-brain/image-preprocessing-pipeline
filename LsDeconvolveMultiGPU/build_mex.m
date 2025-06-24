% ===============================
% build_mex.m (Patched to always use local libtiff)
% ===============================
% Compile semaphore, LZ4, and GPU MEX files using locally-compiled libtiff.
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
src_load_bl   = 'load_bl_tif.cpp';
src_save_bl   = 'save_bl_tif.cpp';
src_load_slab_lz4 = 'load_slab_lz4.cpp';

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
            'COMPFLAGS="$COMPFLAGS /std:c++17 /Od /Zi"', ...
            'LINKFLAGS="$LINKFLAGS /DEBUG"'
        };
    else
        mex_flags_cpu = {
            '-R2018a', ...
            'COMPFLAGS="$COMPFLAGS /std:c++17 /O3 /arch:AVX2"', ...
            'LINKFLAGS="$LINKFLAGS"'
        };
    end
else
    if debug
        mex_flags_cpu = {
            '-R2018a', ...
            'CFLAGS="$CFLAGS -O0 -g"', ...
            'CXXFLAGS="$CFLAGS"', ...
            'LDFLAGS="$LDFLAGS -g"'
        };
    else
        mex_flags_cpu = {
            '-R2018a', ...
            'CFLAGS="$CFLAGS -O3 -march=native -fomit-frame-pointer"', ...
            'CXXFLAGS="$CFLAGS"', ...
            'LDFLAGS="$LDFLAGS"'
        };
    end
end

% Use locally-compiled libtiff, always.
libtiff_src_version = '4.7.0'; % or whatever you are building
libtiff_root = fullfile(pwd, 'tiff_src', ['tiff-', libtiff_src_version]);
libtiff_install_dir = fullfile(pwd, 'tiff_build', 'libtiff');
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

tiff_include = {['-I', fullfile(libtiff_install_dir, 'include')]};
tiff_lib     = {['-L', fullfile(libtiff_install_dir, 'lib')]};
tiff_link    = {'-ltiff'};
fprintf('Using libtiff from: %s\n', fullfile(libtiff_install_dir, 'lib'));

% Build CPU MEX files
mex(mex_flags_cpu{:}, src_semaphore);
mex(mex_flags_cpu{:}, src_lz4_save, src_lz4_c);
mex(mex_flags_cpu{:}, src_lz4_load, src_lz4_c);
mex(mex_flags_cpu{:}, src_load_slab_lz4, src_lz4_c);
mex(mex_flags_cpu{:}, src_load_bl, tiff_include{:}, tiff_lib{:}, tiff_link{:});
mex(mex_flags_cpu{:}, src_save_bl, tiff_include{:}, tiff_lib{:}, tiff_link{:});

% CUDA optimization flags
if ispc
    if debug
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -G -std=c++17 -Xcompiler ""/Od,/Zi"" "'; %#ok<NASGU>
    else
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -O2 -std=c++17 -Xcompiler ""/O2,/arch:AVX2"" "'; %#ok<NASGU>
    end
else
    if debug
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -G -std=c++17 -Xcompiler ''-O0,-g'' "'; %#ok<NASGU>
    else
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -O3 -std=c++17 -Xcompiler ''-O3,-march=native,-fomit-frame-pointer'' "'; %#ok<NASGU>
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
% Function: try_build_libtiff
% ===============================
function ok = try_build_libtiff(libtiff_root, libtiff_install_dir, mex_flags_cpu, version)
    if nargin < 3, mex_flags_cpu = {}; end
    if nargin < 4 || isempty(version), version = '4.7.0'; end
    orig_dir = pwd;

    % === Parse CFLAGS from mex_flags_cpu ===
    CFLAGS = '';
    for i = 1:numel(mex_flags_cpu)
        token = mex_flags_cpu{i};
        if contains(token, 'CFLAGS=')
            m = regexp(token, 'CFLAGS="\$CFLAGS ([^"]+)"', 'tokens');
            if ~isempty(m), CFLAGS = strtrim(m{1}{1}); break; end
        end
    end
    CXXFLAGS = CFLAGS;

    % === Try to locate LZ4 and ZSTD ===
    extra_flags = '';
    incs = {};
    libs = {};
    % Try conda-forge default install paths
    guess_paths = {
        fullfile(getenv('CONDA_PREFIX'), 'include'), ...
        fullfile(getenv('CONDA_PREFIX'), 'Library', 'include')  % Windows
    };
    found_lz4 = false; found_zstd = false;

    for i = 1:numel(guess_paths)
        incdir = guess_paths{i};
        if isfile(fullfile(incdir, 'lz4.h'))
            incs{end+1} = ['-I', incdir]; %#ok<AGROW>
            extra_flags = [extra_flags, ' --with-lz4']; %#ok<AGROW>
            found_lz4 = true;
        end
        if isfile(fullfile(incdir, 'zstd.h'))
            incs{end+1} = ['-I', incdir]; %#ok<AGROW>
            extra_flags = [extra_flags, ' --with-zstd']; %#ok<AGROW>
            found_zstd = true;
        end
    end

    if ~found_lz4,  error('lz4.h not found in conda or expected paths'); end
    if ~found_zstd, error('zstd.h not found in conda or expected paths'); end

    % === Start build ===
    if ispc
        archive = ['tiff-', version, '.zip'];
        if ~isfolder(libtiff_root)
            url = ['https://download.osgeo.org/libtiff/tiff-', version, '.zip'];
            system(['curl -L -o ', archive, ' ', url]);
            unzip(archive, 'tiff_src'); delete(archive);
        end
        cd(libtiff_root);
        setenv('CFLAGS',   CFLAGS);
        setenv('CXXFLAGS', CXXFLAGS);

        status = system([
            'cmake -B build -DCMAKE_BUILD_TYPE=Release ' ...
            '-DCMAKE_INSTALL_PREFIX="', libtiff_install_dir, '" ' ...
            '-Dtiff-tools=OFF -Dtiff-tests=OFF . && ' ...
            'cmake --build build --config Release --target install'
        ]);
        setenv('CFLAGS', ''); setenv('CXXFLAGS', '');
        cd(orig_dir);
    else
        archive = ['tiff-', version, '.tar.gz'];
        if ~isfolder(['tiff-', version])
            url = ['https://download.osgeo.org/libtiff/', archive];
            system(['curl -L -o ', archive, ' ', url]);
            system(['tar -xzf ', archive]); delete(archive);
        end
        cd(['tiff-', version]);

        cmd = sprintf('%s CFLAGS="%s" CXXFLAGS="%s" ./configure --disable-jpeg --disable-old-jpeg --disable-cxx --disable-tools --enable-shared --disable-static --with-pic%s --prefix=%s && make -j%d && make install', ...
            strjoin(incs, ' '), CFLAGS, CXXFLAGS, extra_flags, libtiff_install_dir, feature('numCores'));
        status = system(cmd);
        cd(orig_dir);
    end

    ok = (status == 0);

    % === Post build test ===
    if ok
        fprintf('✔️  libtiff built successfully with:\n');
        fprintf('   • CFLAGS/CXXFLAGS: %s\n', CFLAGS);
        fprintf('   • LZ4 enabled:  %s\n', string(found_lz4));
        fprintf('   • ZSTD enabled: %s\n', string(found_zstd));
        if ~isempty(dir(fullfile(libtiff_install_dir, 'lib', '*zstd*.a')))
            fprintf('   • Static libzstd available ✅\n');
        else
            fprintf('   • Static libzstd not found ❌\n');
        end
        if ~isempty(dir(fullfile(libtiff_install_dir, 'lib', '*z.a')))
            fprintf('   • Static zlib available ✅\n');
        else
            fprintf('   • Static zlib not found ❌\n');
        end

        % MEX sanity test
        test_file = 'libtiff_test_mex.c';
        fid = fopen(test_file, 'w');
        fprintf(fid, '#include "mex.h"\n#include "tiffio.h"\nvoid mexFunction(int nlhs,mxArray*plhs[],int nrhs,const mxArray*prhs[]){printf("TIFF: %%s\\n",TIFFGetVersion());}\n');
        fclose(fid);
        try
            mex(mex_flags_cpu{:}, test_file, ['-I', fullfile(libtiff_install_dir, 'include')], ...
                ['-L', fullfile(libtiff_install_dir, 'lib')], '-ltiff');
            delete(test_file);
        catch ME
            warning('Post-build libtiff test failed: %s', ME.message);
            ok = false;
        end
    else
        warning('❌ libtiff build failed');
    end
end
