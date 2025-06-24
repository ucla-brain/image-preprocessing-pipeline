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
    if nargin<3, mex_flags_cpu = {}; end
    if nargin<4 || isempty(version), version = '4.7.0'; end

    orig_dir = pwd;

    %--- 1) Extract CFLAGS & CXXFLAGS from mex_flags_cpu ---
    CFLAGS = '';
    CXXFLAGS = '';
    for i = 1:numel(mex_flags_cpu)
        tok = mex_flags_cpu{i};
        % look for CFLAGS
        m = regexp(tok, 'CFLAGS="\$CFLAGS\s*([^"]+)"', 'tokens');
        if ~isempty(m)
            CFLAGS = strtrim(m{1}{1});
        end
        % look for CXXFLAGS
        m2 = regexp(tok, 'CXXFLAGS="\$CFLAGS\s*([^"]+)"', 'tokens');
        if ~isempty(m2)
            % if user actually wrote CXXFLAGS="$CFLAGS …"
            CXXFLAGS = strtrim(m2{1}{1});
        end
        m3 = regexp(tok, 'CXXFLAGS="\$CXXFLAGS\s*([^"]+)"', 'tokens');
        if ~isempty(m3)
            % or CXXFLAGS explicitly
            CXXFLAGS = strtrim(m3{1}{1});
        end
    end
    % ensure both are set and identical if only one was given
    if isempty(CXXFLAGS)
        CXXFLAGS = CFLAGS;
    end

    %--- 2) Download & extract source if needed ---
    if ispc
        archive = fullfile(pwd, sprintf('tiff-%s.zip', version));
        if ~isfolder(libtiff_root)
            fprintf('Downloading libtiff %s …\n', version);
            system(sprintf('curl -L -o "%s" "https://download.osgeo.org/libtiff/tiff-%s.zip"', archive, version));
            unzip(archive, fullfile(pwd,'tiff_src'));
            delete(archive);
        end
        cd(libtiff_root);

        % set env vars for cmake
        setenv('CFLAGS',  CFLAGS);
        setenv('CXXFLAGS',CXXFLAGS);

        % configure & build
        status = system([
          'cmake -B build -DCMAKE_BUILD_TYPE=Release '          ...
          '-DCMAKE_INSTALL_PREFIX="', libtiff_install_dir, '" ' ...
          '-DENABLE_LZ4=ON -DENABLE_ZSTD=ON '                   ...
          '. && cmake --build build --config Release --target install'
        ]);

        % reset
        setenv('CFLAGS','');
        setenv('CXXFLAGS','');
        cd(orig_dir);
    else
        archive = fullfile(pwd, sprintf('tiff-%s.tar.gz', version));
        src_dir = fullfile(pwd, sprintf('tiff-%s', version));
        if ~isfolder(src_dir)
            fprintf('Downloading libtiff %s …\n', version);
            system(sprintf('curl -L -o "%s" "https://download.osgeo.org/libtiff/tiff-%s.tar.gz"', archive, version));
            system(sprintf('tar xf "%s"', archive));
            delete(archive);
        end
        cd(src_dir);

        % pass CFLAGS & CXXFLAGS as prefix
        prefix = '';
        if ~isempty(CFLAGS)
            prefix = sprintf('CFLAGS="%s" CXXFLAGS="%s" ', CFLAGS, CXXFLAGS);
        end

        cmd = [ prefix,                          ...
            './configure --enable-shared --disable-static ', ...
            '--with-lz4 --with-zstd --with-pic ',           ...
            '--prefix=', libtiff_install_dir, ' && '        ...
            'make -j', round(feature('numCores')), ' && make install' ];

        status = system(cmd);
        cd(orig_dir);
    end

    ok = (status == 0);

    %--- 3) Optional smoke-test of the new libtiff via MEX ---
    if ok && ~isempty(mex_flags_cpu)
        test_c = 'libtiff_test_mex.c';
        fid = fopen(test_c,'w');
        fprintf(fid, [
            '#include "mex.h"\n' ...
            '#include "tiffio.h"\n' ...
            'void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {\n' ...
            '    mexPrintf("TIFF version: %s\\n", TIFFGetVersion());\n' ...
            '}\n']);
        fclose(fid);

        include_flag = ['-I', fullfile(libtiff_install_dir,'include')];
        lib_flag     = ['-L', fullfile(libtiff_install_dir,'lib')];
        try
            mex(mex_flags_cpu{:}, test_c, include_flag, lib_flag, '-ltiff');
            delete(test_c);
        catch ME
            warning('libtiff MEX smoke-test failed: %s', ME.message);
            ok = false;
        end
    end
end
