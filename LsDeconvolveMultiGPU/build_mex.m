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
%TRY_BUILD_LIBTIFF Fetch, configure and build libtiff with only LZW, LZ4, ZSTD, DEFLATE.
% 1) Detect CONDA_PREFIX and fall back to local headers (lz4.h, zstd.h, zlib.h)
% 2) Fail early if required headers missing.
% 3) Use identical CFLAGS and CXXFLAGS.
% 4) Pass -j<cores> from feature('numCores').
% 5) Disable all codecs except the needed four.
% 6) Report if static libs for zstd or zlib were found.

    orig_dir = pwd;
    ok = false;

    % === Parse CFLAGS/CXXFLAGS from mex_flags_cpu ===
    CFLAGS = '';
    for tok = mex_flags_cpu
        s = tok{1};
        m = regexp(s, 'CFLAGS="\$CFLAGS\s+([^"]+)"', 'tokens');
        if ~isempty(m), CFLAGS = m{1}{1}; end
    end
    CXXFLAGS = CFLAGS;

    % === Locate Conda include/lib or local headers ===
    conda = getenv('CONDA_PREFIX');
    inc_dirs = {};
    lib_dirs = {};
    if ~isempty(conda)
        inc_dirs{end+1} = fullfile(conda,'include');
        lib_dirs{end+1} = fullfile(conda,'lib');
    end
    % always include the local root for downloaded headers
    inc_dirs{end+1} = fullfile(orig_dir);

    % ensure headers exist
    needed = {'lz4.h','zstd.h','zlib.h','tiff.h'};
    missing = {};
    for h = needed
        found = false;
        for d = inc_dirs
            if isfile(fullfile(d{1},h{1}))
                found = true; break;
            end
        end
        if ~found, missing{end+1} = h{1}; end
    end
    if ~isempty(missing)
        error("Missing required headers: %s", strjoin(missing,', '));
    end

    % === Fetch and extract libtiff source if needed ===
    if ~isfolder(libtiff_root)
        archive = sprintf('tiff-%s.tar.gz', version);
        url = sprintf('https://download.osgeo.org/libtiff/tiff-%s.tar.gz', version);
        system(sprintf('curl -L -o %s %s', archive, url));
        system(sprintf('tar xf %s'), archive);
        delete(archive);
    end
    cd(libtiff_root);

    % === Configure only needed codecs ===
    % Disable all optional codecs except LZW, DEFLATE, LZ4, ZSTD
    configure_cmd = sprintf([...
        'CFLAGS="%s" CXXFLAGS="%s" ./configure '...
        '--enable-shared --disable-static '...
        '--without-jpeg --without-webp --without-jbig '...
        '--without-pixarlog --without-lzma --without-old-jpeg '...
        '--with-included-zlib '...         % ensure zlib support
        '--with-zstd '...                  % rely on zstd detection
        '--with-lz4 '...                   % rely on lz4 detection
        '--prefix=%s'], ...
        CFLAGS, CXXFLAGS, libtiff_install_dir);

    % === Build and install ===
    cores = feature('numCores');
    make_cmd = sprintf('make -j%d && make install', cores);

    [status, out] = system([configure_cmd ' && ' make_cmd]);
    cd(orig_dir);
    if status ~= 0
        fprintf('Configure/Build output:\n%s\n', out);
        return;
    end

    % === Report static libs availability ===
    report = {};
    for libn = {'libzstd.a','libz.a'}
        p = fullfile(libtiff_install_dir,'lib', libn{1});
        if isfile(p)
            report{end+1} = sprintf('Found static %s', libn{1}); end
    end
    if ~isempty(report)
        fprintf('=== Static library report ===\n');
        fprintf('%s\n', report{:});
    end

    % === Simple smoketest of TIFFGetVersion via MEX ===
    if ~isempty(mex_flags_cpu)
        test_c = 'tiff_version_test_mex.c';
        fid = fopen(test_c,'w');
        fprintf(fid,['#include "mex.h"\n#include "tiffio.h"\n' ...
            'void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { '...
            ' mexPrintf("libtiff: %s\\n", TIFFGetVersion()); }\n']);
        fclose(fid);
        inc_flag = ['-I', fullfile(libtiff_install_dir,'include')];
        lib_flag = ['-L', fullfile(libtiff_install_dir,'lib')];
        try
            mex(mex_flags_cpu{:}, test_c, inc_flag, lib_flag, '-ltiff');
            delete(test_c);
            ok = true;
        catch ME
            warning('libtiff smoketest failed: %s', ME.message);
            ok = false;
        end
    else
        ok = true;
    end
end
