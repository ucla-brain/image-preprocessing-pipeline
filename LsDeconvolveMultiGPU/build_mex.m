function build_mex(debug)
    ncores = feature('numCores');
    if nargin<1, debug=false; end
    if verLessThan('matlab','9.4')
        error('Requires MATLAB R2018a or newer.'); end
    if exist('mexcuda','file')~=2
        error('mexcuda not found – ensure CUDA is configured.'); end

    % Get compiler settings and MSVC bin path for Windows
    if ispc
        [cmake_gen, cmake_arch, vs_env_bat] = get_vs_cmake_info();
        cc = mex.getCompilerConfigurations('C++', 'Selected');
        msvc_bin = get_msvc_bin_from_setenv(cc.Details.SetEnv, cc);
    else
        cmake_gen = ''; cmake_arch = ''; vs_env_bat = ''; msvc_bin = '';
    end

    % Versions
    zlibng_v  = '2.2.4';
    libtiff_v = '4.7.0';

    % Paths
    root        = pwd;
    thirdparty  = fullfile(root,'thirdparty');
    build_root  = fullfile(root,'tiff_build');
    if ~exist(thirdparty,'dir'), mkdir(thirdparty); end
    if ~exist(build_root,'dir'), mkdir(build_root); end

    zlibng_src   = fullfile(thirdparty, ['zlib-ng-'   zlibng_v]);
    lz4_src      = fullfile(thirdparty, 'lz4');
    libtiff_src  = fullfile(thirdparty, ['tiff-'      libtiff_v]);

    zlibng_inst  = fullfile(build_root,'zlib-ng');
    libtiff_inst = fullfile(build_root,'libtiff');

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
        if ispc
            args = [
                '-DBUILD_SHARED_LIBS=OFF ', ...
                '-DZLIB_COMPAT=ON ', ...
                '-DZLIB_ENABLE_TESTS=OFF -DZLIBNG_ENABLE_TESTS=OFF -DWITH_GTEST=OFF ', ...
                '-DWITH_BENCHMARKS=OFF -DWITH_BENCHMARK_APPS=OFF ', ...
                '-DCMAKE_POSITION_INDEPENDENT_CODE=ON ', ...
                '-DWITH_NATIVE_INSTRUCTIONS=ON -DWITH_NEW_STRATEGIES=ON -DWITH_AVX2=ON ', ...
                '-DCMAKE_C_FLAGS_RELEASE="/O2 /GL /arch:AVX2" ', ...
                '-DCMAKE_STATIC_LINKER_FLAGS_RELEASE="/LTCG" '
            ];
        else
            args = [ ...
                '-DBUILD_SHARED_LIBS=OFF ', ...
                '-DZLIB_COMPAT=ON ', ...
                '-DZLIB_ENABLE_TESTS=OFF -DZLIBNG_ENABLE_TESTS=OFF -DWITH_GTEST=OFF ', ...
                '-DWITH_BENCHMARKS=OFF -DWITH_BENCHMARK_APPS=OFF ', ...
                '-DZLIBNG_ENABLE_TESTS=OFF ', ...
                '-DZLIBNG_ENABLE_BENCHMARKS=OFF ', ...
                '-DCMAKE_POSITION_INDEPENDENT_CODE=ON ', ...
                '-DCMAKE_BUILD_TYPE=Release ', ...
                '-DWITH_NATIVE_INSTRUCTIONS=ON -DWITH_NEW_STRATEGIES=ON -DWITH_AVX2=ON ', ...
                sprintf('-DCMAKE_C_FLAGS_RELEASE="-O3 -march=native -flto=%d -fPIC"', ncores) ...
            ];
        end
        if cmake_build(zlibng_src, fullfile(zlibng_src,'build'), zlibng_inst, cmake_gen, cmake_arch, args, vs_env_bat, msvc_bin)
            error('zlib-ng build failed.');
        end
        fclose(fopen(stamp,'w'));
    end

    %% 3) libtiff (disable JPEG/JBIG/LZMA/WebP/LERC/PixarLog; use zlib-ng)
    stamp = fullfile(libtiff_inst,'.built');
    if ~isfile(stamp)
        fprintf('[libtiff] building… (codecs off)\n');
        if ~exist(libtiff_src,'dir')
            tgz = fullfile(thirdparty,sprintf('tiff-%s.tar.gz',libtiff_v));
            websave(tgz,sprintf(['https://download.osgeo.org/libtiff/tiff-' libtiff_v '.tar.gz']));
            untar(tgz,thirdparty); delete(tgz);
        end
        if ispc
            args = sprintf([
                '-DBUILD_SHARED_LIBS=OFF ', ...
                '-DCMAKE_POSITION_INDEPENDENT_CODE=ON ', ...
                '-DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>" ' ...
                '-Djbig=OFF -Djpeg=OFF -Dold-jpeg=OFF -Dlzma=OFF -Dwebp=OFF -Dlerc=OFF -Dpixarlog=OFF -Dlibdeflate=OFF ', ...
                '-Dtiff-tests=OFF -Dtiff-opengl=OFF -Dtiff-contrib=OFF -Dtiff-tools=OFF ', ...
                '-DZLIB_LIBRARY=%s -DZLIB_INCLUDE_DIR=%s ', ...
                '-DCMAKE_C_FLAGS_RELEASE="/O2 /GL /arch:AVX2" ', ...
                '-DCMAKE_STATIC_LINKER_FLAGS_RELEASE="/LTCG" ', ...
                '-DCMAKE_EXE_LINKER_FLAGS_RELEASE="/LTCG" ', ...
                '-DCMAKE_SHARED_LINKER_FLAGS_RELEASE="/LTCG" '
            ], ...
                fullfile(zlibng_inst,'lib','zlibstatic.lib'),    fullfile(zlibng_inst,'include') ...
            );
        else
            args = sprintf([ ...
                '-DBUILD_SHARED_LIBS=OFF ', ...
                '-DCMAKE_POSITION_INDEPENDENT_CODE=ON ', ...
                '-Djbig=OFF -Djpeg=OFF -Dold-jpeg=OFF -Dlzma=OFF -Dwebp=OFF -Dlerc=OFF -Dpixarlog=OFF -Dlibdeflate=OFF ', ...
                '-Dtiff-tests=OFF -Dtiff-opengl=OFF ', ...
                '-DZLIB_LIBRARY=%s -DZLIB_INCLUDE_DIR=%s ', ...
                '-DCMAKE_C_FLAGS_RELEASE="-O3 -march=native -flto=%d -fPIC" ', ...
                '-DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -flto=%d -fPIC"' ...
            ], ...
                fullfile(zlibng_inst,'lib','libz.a'),    fullfile(zlibng_inst,'include'), ...
                ncores, ncores);
        end
        if cmake_build(libtiff_src,fullfile(libtiff_src,'build'),libtiff_inst, cmake_gen, cmake_arch, args, vs_env_bat, msvc_bin)
            error('libtiff build failed.'); end
        fclose(fopen(stamp,'w'));
    end

    %% 4) MEX compilation flags
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

    %% 5) Include & link for TIFF MEXs
    inc_tiff = ['-I' fullfile(libtiff_inst,'include')];
    if ispc
        link_tiff = {
            fullfile(libtiff_inst,'lib','tiffxx.lib'), ...
            fullfile(libtiff_inst,'lib','tiff.lib'), ...
            fullfile(zlibng_inst,'lib','zlibstatic.lib')     % zlib-ng CMake default is 'zlibstatic.lib'
        };
    else
        link_tiff = {
            fullfile(libtiff_inst,'lib','libtiffxx.a'), ...
            fullfile(libtiff_inst,'lib','libtiff.a'), ...
            fullfile(zlibng_inst,'lib','libz.a')
        };
    end

    %% 6) Build CPU MEX files
    fprintf('\n[MEX] Compiling CPU modules …\n');
    mex(mex_cpu{:}, 'semaphore.c');
    mex(mex_cpu{:}, 'save_lz4_mex.c',    lz4_c, ['-I' lz4_src]);
    mex(mex_cpu{:}, 'load_lz4_mex.c',    lz4_c, ['-I' lz4_src]);
    mex(mex_cpu{:}, 'load_slab_lz4.cpp', lz4_c, ['-I' lz4_src]);
    mex(mex_cpu{:}, inc_tiff, 'load_bl_tif.cpp', link_tiff{:});
    mex(mex_cpu{:}, inc_tiff, 'save_bl_tif.cpp', link_tiff{:});

    %% 7) CUDA MEX files (unchanged)
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

    mexcuda(cuda_mex_flags{:}, '-R2018a', nvccflags, 'gauss3d_gpu.cu', ['-I',root_dir], ['-I',include_dir]);
    mexcuda(cuda_mex_flags{:}, '-R2018a', nvccflags, 'conv3d_gpu.cu' , ['-I',root_dir], ['-I',include_dir]);
    mexcuda(cuda_mex_flags{:}, '-R2018a', nvccflags, 'conj_gpu.cu'   , ['-I',root_dir], ['-I',include_dir]);
    mexcuda(cuda_mex_flags{:}, '-R2018a', nvccflags, 'otf_gpu.cu'    , ['-I',root_dir], ['-I',include_dir], ...
           '-L/usr/local/cuda/lib64','-lcufft');

    fprintf('\n✅  All MEX files built successfully.\n');
end

function [cmake_gen, cmake_arch, vs_env_bat] = get_vs_cmake_info()
    cc = mex.getCompilerConfigurations('C++', 'Selected');
    cmake_gen = '';
    cmake_arch = '';
    vs_env_bat = '';
    % Figure out VS version and set generator
    if contains(cc.Name, 'Visual C++ 2022')
        cmake_gen = '-G "Visual Studio 17 2022" -T v143';
        cmake_arch = '-A x64';
    elseif contains(cc.Name, 'Visual C++ 2019')
        cmake_gen = '-G "Visual Studio 16 2019" -T v142';
        cmake_arch = '-A x64';
    else
        error('Unsupported Visual Studio version for MEX: %s', cc.Name);
    end

    % Attempt to find vs_env_bat by walking up from cc.Location
    if isfield(cc,'Location')
        loc = cc.Location;
        % walk up to find "VC" folder (should be .../VC/Tools/MSVC/xx.x.xxxxx/bin/Hostx64/x64)
        parts = strsplit(loc, filesep);
        idx = find(strcmpi(parts, 'VC'), 1, 'last');
        if ~isempty(idx)
            vsroot = fullfile(parts{1:idx});
            vs_env_bat_try = fullfile(vsroot, 'Auxiliary', 'Build', 'vcvars64.bat');
            if exist(vs_env_bat_try, 'file')
                vs_env_bat = vs_env_bat_try;
            end
        end
    end
end

function msvc_bin = get_msvc_bin_from_setenv(setenvstr, cc)
    % Try to extract MSVC bin dir from SetEnv string
    msvc_bin = '';
    pat = 'set PATH=([^\;]+)\\bin\\HostX64\\x64\;';
    m = regexp(setenvstr, pat, 'tokens', 'once');
    if ~isempty(m)
        msvc_bin = m{1};
        return;
    end
    % Fallback: try cc.Location (go up to ...\MSVC\XX.XX.XXXXX)
    if isfield(cc, 'Location')
        loc = cc.Location;
        % should be ...\VC\Tools\MSVC\XX.XX.XXXXX\bin\Hostx64\x64
        % so take the folder 4 levels up
        [d1, p1] = fileparts(loc); % x64
        [d2, p2] = fileparts(d1);  % Hostx64
        [d3, p3] = fileparts(d2);  % bin
        [msvc_bin, p4] = fileparts(d3); % XX.XX.XXXXX
        % msvc_bin now ends with ...\VC\Tools\MSVC\XX.XX.XXXXX
        if ~isempty(msvc_bin) && contains(msvc_bin, 'MSVC')
            return;
        end
    end
    % If still not found, print a warning and leave empty
    warning(['Could not find MSVC bin path from mex settings; ' ...
             'continuing without explicit path. You may get a compiler mismatch if multiple MSVC versions are installed.']);
    if isempty(msvc_bin)
    % Try hard-coded override as a last resort:
    override_bin = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207';
    if exist(fullfile(override_bin,'bin','Hostx64','x64','cl.exe'),'file')
        warning('Using manual override for MSVC bin path: %s', override_bin);
        msvc_bin = override_bin;
    end
end

end

% Helper to run CMake + install (for zlib-ng & libtiff)
function status = cmake_build(src, bld, inst, cmake_gen, cmake_arch, args, vs_env_bat, msvc_bin)
    if ~exist(bld, 'dir'), mkdir(bld); end
    old = cd(bld); onCleanup(@() cd(old));
    ncores = feature('numCores');
    if ispc
        % Prepend MSVC bin dir to PATH for this call
        orig_path = getenv('PATH');
        cleanup = onCleanup(@() setenv('PATH', orig_path));
        if ~isempty(msvc_bin)
            setenv('PATH', [msvc_bin filesep 'bin' filesep 'HostX64' filesep 'x64' pathsep orig_path]);
        end
        cmake_cmd = sprintf('cmake %s %s "%s" -DCMAKE_INSTALL_PREFIX="%s" %s', cmake_gen, cmake_arch, src, inst, args);
        build_cmd = sprintf('cmake --build . --config Release --target INSTALL -- /m:%d', ncores);
        if exist(vs_env_bat, 'file')
            cmake_cmd = sprintf('call "%s" && %s', vs_env_bat, cmake_cmd);
            build_cmd = sprintf('call "%s" && %s', vs_env_bat, build_cmd);
        end
        if system(cmake_cmd) ~= 0
            status = 1; return;
        end
        status = system(build_cmd);
    else
        cmake_cmd = sprintf('cmake "%s" -DCMAKE_INSTALL_PREFIX="%s" %s', src, inst, args);
        if system(cmake_cmd) ~= 0
            status = 1; return;
        end
        status = system(sprintf('cmake --build . -- -j%d install', ncores));
    end
end
