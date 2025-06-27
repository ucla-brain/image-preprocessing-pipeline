function build_mex(debug)
    % build_mex  Compile all C/C++/CUDA MEX files and static libraries.
    % Keivan Moradi, 2024-2025

    % --------- USER: SET THIS PATH ON WINDOWS ---------
    MSVC_BASE = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.33519';
    % --------------------------------------------------

    ncores = feature('numCores');
    if nargin < 1, debug = false; end
    if verLessThan('matlab','9.4')
        error('Requires MATLAB R2018a or newer.'); end
    if exist('mexcuda','file') ~= 2
        error('mexcuda not found – ensure CUDA is configured.'); end

    isWin = ispc;
    % ---- BEGIN WINDOWS TOOLCHAIN ENV PATCH ----
    if isWin
        BIN     = fullfile(MSVC_BASE, 'bin', 'HostX64', 'x64');
        INCLUDE = fullfile(MSVC_BASE, 'include');
        LIB     = fullfile(MSVC_BASE, 'lib', 'x64');
        VCINSTALLDIR    = fileparts(fileparts(fileparts(MSVC_BASE))); % ...VC
        VCToolsInstallDir = MSVC_BASE;

        orig_path = getenv('PATH');
        orig_include = getenv('INCLUDE');
        orig_lib = getenv('LIB');
        orig_vcinstalldir = getenv('VCINSTALLDIR');
        orig_vctools = getenv('VCToolsInstallDir');

        setenv('PATH', [BIN pathsep orig_path]);
        setenv('INCLUDE', INCLUDE);
        setenv('LIB', LIB);
        setenv('VCINSTALLDIR', VCINSTALLDIR);
        setenv('VCToolsInstallDir', VCToolsInstallDir);

        cleanupMexEnv = onCleanup(@() restore_mex_env(orig_path, orig_include, orig_lib, orig_vcinstalldir, orig_vctools));
    end
    % ---- END WINDOWS TOOLCHAIN ENV PATCH ----

    % ---- Toolchain detection (unchanged) ----
    if isWin
        cc = mex.getCompilerConfigurations('C++', 'Selected');
        [cmake_gen, cmake_arch, msvc] = get_vs_and_msvc(cc, MSVC_BASE);
    else
        cmake_gen = ''; cmake_arch = ''; msvc = [];
    end

    % ---- Version numbers ----
    zlibng_v  = '2.2.4';
    libtiff_v = '4.7.0';

    % ---- Paths ----
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

    %% --- 1) LZ4 (single-file) ---
    lz4_c = fullfile(lz4_src,'lz4.c');
    lz4_h = fullfile(lz4_src,'lz4.h');
    if ~isfile(lz4_c)
        fprintf('[LZ4] downloading…\n');
        mkdir(lz4_src);
        websave(lz4_c,'https://raw.githubusercontent.com/lz4/lz4/dev/lib/lz4.c');
        websave(lz4_h,'https://raw.githubusercontent.com/lz4/lz4/dev/lib/lz4.h');
    end

    %% --- 2) zlib-ng (static) ---
    if isWin
        stamp = fullfile(zlibng_inst, ['.built_', msvc.ver_full]);
    else
        stamp = fullfile(zlibng_inst, '.built');
    end
    if ~exist(zlibng_src,'dir')
        tgz = fullfile(thirdparty, sprintf('zlib-ng-%s.tar.gz',zlibng_v));
        fprintf('[zlib-ng] Downloading source: %s\n', tgz);
        websave(tgz, sprintf('https://github.com/zlib-ng/zlib-ng/archive/refs/tags/%s.tar.gz',zlibng_v));
        untar(tgz, thirdparty); delete(tgz);
    end
    if ~isfile(stamp)
        fprintf('[zlib-ng] building…\n');
        builddir = fullfile(zlibng_src,'build');
        if ~exist(builddir,'dir'), mkdir(builddir); end
        if ~exist(zlibng_inst,'dir'), mkdir(zlibng_inst); end
        if isWin
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
                '-DCMAKE_POSITION_INDEPENDENT_CODE=ON ', ...
                '-DCMAKE_BUILD_TYPE=Release ', ...
                '-DWITH_NATIVE_INSTRUCTIONS=ON -DWITH_NEW_STRATEGIES=ON -DWITH_AVX2=ON ', ...
                sprintf('-DCMAKE_C_FLAGS_RELEASE="-O3 -march=native -flto=%d -fPIC"', ncores) ...
            ];
        end
        if cmake_build(zlibng_src, builddir, zlibng_inst, cmake_gen, cmake_arch, args, msvc)
            error('zlib-ng build failed.');
        end
        fclose(fopen(stamp,'w'));
    end

    %% --- 3) libtiff ---
    if isWin
        stamp = fullfile(libtiff_inst, ['.built_', msvc.ver_full]);
    else
        stamp = fullfile(libtiff_inst, '.built');
    end
    if ~isfile(stamp)
        fprintf('[libtiff] building… (codecs off)\n');
        if ~exist(libtiff_src,'dir')
            tgz = fullfile(thirdparty, sprintf('tiff-%s.tar.gz',libtiff_v));
            websave(tgz, sprintf('https://download.osgeo.org/libtiff/tiff-%s.tar.gz',libtiff_v));
            untar(tgz, thirdparty); delete(tgz);
        end
        builddir = fullfile(libtiff_src,'build');
        if ~exist(builddir,'dir'), mkdir(builddir); end
        if ~exist(libtiff_inst,'dir'), mkdir(libtiff_inst); end
        if isWin
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
                fullfile(zlibng_inst,'lib','zlibstatic.lib'), ...
                fullfile(zlibng_inst,'include') ...
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
                fullfile(zlibng_inst,'lib','libz.a'), ...
                fullfile(zlibng_inst,'include'), ...
                ncores, ncores);
        end
        if cmake_build(libtiff_src, builddir, libtiff_inst, cmake_gen, cmake_arch, args, msvc)
            error('libtiff build failed.'); end
        fclose(fopen(stamp,'w'));
    end

    %% --- 4) MEX compilation flags ---
    if isWin
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
                sprintf('CFLAGS="$CFLAGS -O3 -march=native -flto=%d"', ncores), ...
                sprintf('CXXFLAGS="$CXXFLAGS -O3 -march=native -flto=%d"', ncores), ...
                sprintf('LDFLAGS="$LDFLAGS -flto=%d"', ncores)};
        end
    end

    %% --- 5) Include & link for TIFF MEXs ---
    inc_tiff = ['-I' fullfile(libtiff_inst,'include')];
    if isWin
        link_tiff = {
            fullfile(libtiff_inst,'lib','tiffxx.lib'), ...
            fullfile(libtiff_inst,'lib','tiff.lib'), ...
            fullfile(zlibng_inst,'lib','zlibstatic.lib')
        };
    else
        link_tiff = {
            fullfile(libtiff_inst,'lib','libtiffxx.a'), ...
            fullfile(libtiff_inst,'lib','libtiff.a'), ...
            fullfile(zlibng_inst,'lib','libz.a')
        };
    end

    %% --- 6) Build CPU MEX files ---
    fprintf('\n[MEX] Compiling CPU modules …\n');
    mex(mex_cpu{:}, 'semaphore.c');
    mex(mex_cpu{:}, 'save_lz4_mex.c',    lz4_c, ['-I' lz4_src]);
    mex(mex_cpu{:}, 'load_lz4_mex.c',    lz4_c, ['-I' lz4_src]);
    mex(mex_cpu{:}, 'load_slab_lz4.cpp', lz4_c, ['-I' lz4_src]);
    mex(mex_cpu{:}, inc_tiff, 'load_bl_tif.cpp', link_tiff{:});
    mex(mex_cpu{:}, inc_tiff, 'save_bl_tif.cpp', link_tiff{:});

    %% --- 7) CUDA MEX files (unchanged) ---
    if isWin
        xmlfile = fullfile(fileparts(mfilename('fullpath')), 'nvcc_msvcpp2022.xml');
        assert(isfile(xmlfile), 'nvcc_msvcpp2022.xml not found!');
        cuda_mex_flags = {'-f', xmlfile};
    else
        cuda_mex_flags = {};
    end

    if debug
        nvccflags = 'NVCCFLAGS="$NVCCFLAGS -G"';
    else
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

function restore_mex_env(path, include, lib, vcdir, vctools)
    setenv('PATH', path);
    setenv('INCLUDE', include);
    setenv('LIB', lib);
    setenv('VCINSTALLDIR', vcdir);
    setenv('VCToolsInstallDir', vctools);
end

function [cmake_gen, cmake_arch, m] = get_vs_and_msvc(cc, MSVC_BASE)
    fprintf('\n[get_vs_and_msvc] Selected compiler: %s\n', cc.Name);
    details = cc.Details;
    disp('[get_vs_and_msvc] cc.Details:');
    disp(details);

    % Compose paths from MSVC_BASE
    cl_path = fullfile(MSVC_BASE, 'bin', 'HostX64', 'x64', 'cl.exe');
    m.cl = cl_path;
    m.bin = fileparts(m.cl);
    m.msvc_root = MSVC_BASE;
    m.ver_full  = regexp(MSVC_BASE,'\d+\.\d+\.\d+','match','once');
    m.ver_pair  = regexp(m.ver_full,'^\d+\.\d+','match','once');

    vsroot = fileparts(fileparts(fileparts(MSVC_BASE))); % ...VC
    m.vcvars = fullfile(vsroot, 'Auxiliary', 'Build', 'vcvars64.bat');

    % CMake generator string
    if contains(cc.Name, '2022')
        cmake_gen  = '-G "Visual Studio 17 2022" -T v143';
        cmake_arch = '-A x64';
    elseif contains(cc.Name, '2019')
        cmake_gen  = '-G "Visual Studio 16 2019" -T v142';
        cmake_arch = '-A x64';
    else
        error('[get_vs_and_msvc] Unsupported Visual Studio version for MEX: %s', cc.Name);
    end

    fprintf('[get_vs_and_msvc] Using MSVC cl.exe: %s\n', m.cl);
    fprintf('[get_vs_and_msvc] Toolset root: %s\n', m.msvc_root);
    fprintf('[get_vs_and_msvc] Toolset version: %s\n', m.ver_full);
    fprintf('[get_vs_and_msvc] cmake_gen: %s\n', cmake_gen);
    fprintf('[get_vs_and_msvc] vcvars64.bat: %s\n', m.vcvars);
end

function status = cmake_build(src, bld, inst, cmake_gen, cmake_arch, args, msvc)
    % Ensure build and install dirs exist
    if ~exist(bld,'dir'), mkdir(bld); end
    if ~exist(inst,'dir'), mkdir(inst); end

    old = cd(bld); onCleanup(@() cd(old));
    ncores = feature('numCores');
    isWin = ispc;

    if isWin
        msvc_bin = msvc.bin;
        orig_path = getenv('PATH');
        setenv('PATH', [msvc_bin pathsep orig_path]);
        cleanup = onCleanup(@() setenv('PATH', orig_path));

        cmake_cfg = sprintf(['cmake %s %s "%s" -DCMAKE_INSTALL_PREFIX="%s" ' ...
            '-DCMAKE_C_COMPILER="%s" -DCMAKE_CXX_COMPILER="%s" %s'], ...
            cmake_gen, cmake_arch, src, inst, ...
            fullfile(msvc_bin, 'cl.exe'), ...
            fullfile(msvc_bin, 'cl.exe'), args);

        build_cfg = sprintf('cmake --build . --config Release --target INSTALL -- /m:%d', ncores);

        cmake_cmd = sprintf('call "%s" && %s', msvc.vcvars, cmake_cfg);
        build_cmd = sprintf('call "%s" && %s', msvc.vcvars, build_cfg);

        fprintf('[cmake_build] CWD: %s\n', bld);
        [~,out] = system('dir'); fprintf('[cmake_build] Directory listing before configure:\n%s\n', out);
        fprintf('[cmake_build] Running CMake configure command:\n%s\n', cmake_cmd);

        rc1 = system(cmake_cmd);
        if rc1 ~= 0
            fprintf('[cmake_build] CMake configure FAILED (code %d)\n', rc1);
            status = 1; return;
        end
        fprintf('[cmake_build] Running CMake build command:\n%s\n', build_cmd);

        rc2 = system(build_cmd);
        if rc2 ~= 0
            fprintf('[cmake_build] CMake build FAILED (code %d)\n', rc2);
            status = rc2; return;
        end
        status = 0;
    else
        % ---- LINUX/MAC PATH UNCHANGED ----
        cmake_cmd = sprintf('cmake "%s" -DCMAKE_INSTALL_PREFIX="%s" %s', src, inst, args);
        fprintf('[cmake_build] CWD: %s\n', bld);
        [~,out] = system('ls -lh'); fprintf('[cmake_build] Directory listing before configure:\n%s\n', out);
        fprintf('[cmake_build] Running CMake configure command:\n%s\n', cmake_cmd);
        if system(cmake_cmd) ~= 0
            status = 1; return;
        end
        build_cmd = sprintf('cmake --build . -- -j%d install', ncores);
        fprintf('[cmake_build] Running CMake build command:\n%s\n', build_cmd);
        status = system(build_cmd);
    end
end
