function build_mex(debug)
% build_mex  Compile all C/C++/CUDA MEX files and static libraries.
% Robust, cross-platform, production ready.
% Keivan Moradi, 2024-2025 (with ChatGPT-4o assistance)

    if nargin < 1, debug = false; end
    if verLessThan('matlab','9.4')
        error('Requires MATLAB R2018a or newer.'); end
    if exist('mexcuda','file') ~= 2
        error('mexcuda not found – ensure CUDA is configured.'); end

    ncores = feature('numCores');
    isWin = ispc;

    %% --------- Compiler Toolchain Discovery ---------
    % Always autodetect and force usage of the *exact* compiler/toolset as MEX

    if isWin
        cc = mex.getCompilerConfigurations('C++','Selected');
        if isempty(cc)
            error('No C++ compiler configured for MEX. Run "mex -setup".');
        end

        % Find cl.exe path from SetEnv
        setenv_lines = splitlines(cc.Details.SetEnv);
        cl_path = '';
        for i = 1:numel(setenv_lines)
            line = strtrim(setenv_lines{i});
            if startsWith(line, 'set PATH=', 'IgnoreCase', true)
                paths_str = extractAfter(line, '=');
                path_list = split(paths_str, ';');
                for j = 1:numel(path_list)
                    candidate = strtrim(path_list{j});
                    cl_candidate = fullfile(candidate, 'cl.exe');
                    if isfile(cl_candidate)
                        cl_path = cl_candidate;
                        break;
                    end
                end
            end
            if ~isempty(cl_path), break; end
        end
        if isempty(cl_path)
            error('Could not determine cl.exe location from mex config!');
        end

        % Infer MSVC_BASE and others from cl_path
        MSVC_BASE = fileparts(fileparts(fileparts(cl_path)));
        VSROOT = fileparts(fileparts(fileparts(fileparts(MSVC_BASE))));
        VCVARSALL = fullfile(VSROOT, 'Auxiliary', 'Build', 'vcvars64.bat');

        fprintf('[build_mex] MSVC_BASE:    %s\n', MSVC_BASE);
        fprintf('[build_mex] cl.exe path:  %s\n', cl_path);
        fprintf('[build_mex] VCVARSALL:    %s\n', VCVARSALL);
        fprintf('[build_mex] cc.Version:   %s\n', cc.Version);

        if ~isfile(VCVARSALL), error('vcvars64.bat not found at %s', VCVARSALL); end

        msvc_ver_full = regexp(MSVC_BASE,'\d+\.\d+\.\d+','match','once');
        VCVARS_CMD = sprintf('"%s" -vcvars_ver=%s && set', VCVARSALL, msvc_ver_full);
        [~, envout] = system(VCVARS_CMD);

        vars = splitlines(strtrim(envout));
        env_vars = ["PATH","INCLUDE","LIB","LIBPATH","VCINSTALLDIR","VCToolsInstallDir"];
        original_env = containers.Map();
        for k = 1:numel(vars)
            parts = split(vars{k}, '=');
            if numel(parts) ~= 2, continue; end
            key = upper(parts{1});
            value = parts{2};
            if ismember(key, env_vars)
                original_env(key) = getenv(key);
                setenv(key, value);
                fprintf('[build_mex] setenv %s\n', key);
            end
        end
        cleanupMexEnv = onCleanup(@() restore_mex_env(original_env));

        [~,cl_out] = system('where cl.exe');
        fprintf('[build_mex] where cl.exe:\n%s\n',cl_out);

    else
        MSVC_BASE = '';
        VCVARSALL = '';
        msvc_ver_full = '';
    end

    %% --------- CMake/MSVC Generator Strings ---------
    if isWin
        % Figure out the best generator/toolset for CMake
        if contains(cc.Name, '2022')
            cmake_gen  = '-G "Visual Studio 17 2022" -T v143';
            cmake_arch = '-A x64';
        elseif contains(cc.Name, '2019')
            cmake_gen  = '-G "Visual Studio 16 2019" -T v142';
            cmake_arch = '-A x64';
        else
            error('Unsupported Visual Studio version for MEX: %s', cc.Name);
        end
        msvc.vcvars = VCVARSALL;
        msvc.ver_full = msvc_ver_full;
        msvc.cl = cl_path;
        msvc.msvc_root = MSVC_BASE;
    else
        cmake_gen = '';
        cmake_arch = '';
        msvc = [];
    end

    %% --------- Library Version Numbers ---------
    zlibng_v  = '2.2.4';
    libtiff_v = '4.7.0';

    %% --------- Paths (never removed) ---------
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

    %% --------- 1) LZ4 (single-file) ---------
    lz4_c = fullfile(lz4_src,'lz4.c');
    lz4_h = fullfile(lz4_src,'lz4.h');
    if ~isfile(lz4_c)
        fprintf('[LZ4] downloading…\n');
        mkdir(lz4_src);
        websave(lz4_c,'https://raw.githubusercontent.com/lz4/lz4/dev/lib/lz4.c');
        websave(lz4_h,'https://raw.githubusercontent.com/lz4/lz4/dev/lib/lz4.h');
    end

    %% --------- 2) zlib-ng (static) ---------
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

    %% --------- 3) libtiff (static) ---------
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
                '-Dtiff-tests=OFF -Dtiff-opengl=OFF -Dtiff-contrib=OFF -Dtiff-tools=OFF -Dzstd=OFF ', ...
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
                '-Dtiff-tests=OFF -Dtiff-opengl=OFF -Dzstd=OFF ', ...
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

    %% --------- 4) MEX compilation flags (unchanged) ---------
    if isWin
        if debug
            mex_cpu = {'-R2018a', ...
                'COMPFLAGS="$COMPFLAGS /std:c++17 /Od /Zi"', ...
                'LINKFLAGS="$LINKFLAGS /DEBUG"' ...
            };
        else
            mex_cpu = {'-R2018a', ...
                'COMPFLAGS="$COMPFLAGS /std:c++17 /O2 /arch:AVX2 /GL"', ...
                'LINKFLAGS="$LINKFLAGS /LTCG"', ...
            };
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

    %% --------- 5) Include & link for TIFF MEXs (unchanged) ---------
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

    %% --------- 6) Build CPU MEX files (unchanged) ---------
    fprintf('\n[MEX] Compiling CPU modules …\n');
    mex(mex_cpu{:}, 'semaphore.c');
    mex(mex_cpu{:}, 'save_lz4_mex.c',    lz4_c, ['-I' lz4_src]);
    mex(mex_cpu{:}, 'load_lz4_mex.c',    lz4_c, ['-I' lz4_src]);
    mex(mex_cpu{:}, 'load_slab_lz4.cpp', lz4_c, ['-I' lz4_src]);
    mex(mex_cpu{:}, inc_tiff, 'load_bl_tif.cpp', link_tiff{:});
    mex(mex_cpu{:}, inc_tiff, 'save_bl_tif.cpp', link_tiff{:});

    %% --------- 7) CUDA MEX files (platform-specific) ---------
    if isWin
        xmlfile = fullfile(fileparts(mfilename('fullpath')), 'nvcc_msvcpp2022.xml');
        assert(isfile(xmlfile), 'nvcc_msvcpp2022.xml not found!');
        cuda_mex_flags = {'-f', xmlfile};

        % Set CUDA path if needed
        cuda_path = getenv('CUDA_PATH');
        if isempty(cuda_path)
            cuda_path = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9';
        end
        cufft_lib_path = fullfile(cuda_path, 'lib', 'x64');
        mexcuda_libflags = {['-L"', cufft_lib_path, '"'], '-lcufft'};

        if debug
            nvccflags = 'NVCCFLAGS="$NVCCFLAGS -allow-unsupported-compiler -G"';
        else
            nvccflags = 'NVCCFLAGS="$NVCCFLAGS -allow-unsupported-compiler -Xcompiler /O2 /arch:AVX2 /GL"';
        end
    else
        cuda_mex_flags = {};
        mexcuda_libflags = {'-L/usr/local/cuda/lib64','-lcufft'};
        if debug
            nvccflags = 'NVCCFLAGS="$NVCCFLAGS -G"';
        else
            nvccflags = [
                'NVCCFLAGS="$NVCCFLAGS -O3 -use_fast_math ' ...
                '-arch=sm_75 ' ...
                '-gencode=arch=compute_75,code=sm_75 ' ...
                '-gencode=arch=compute_80,code=sm_80 ' ...
                '-gencode=arch=compute_86,code=sm_86 ' ...
                '-gencode=arch=compute_89,code=sm_89"'
            ];
        end
    end

    mexcuda(cuda_mex_flags{:}, '-R2018a', nvccflags, 'gauss3d_gpu.cu');
    mexcuda(cuda_mex_flags{:}, '-R2018a', nvccflags, 'conv3d_gpu.cu' );
    mexcuda(cuda_mex_flags{:}, '-R2018a', nvccflags, 'otf_gpu.cu'    , mexcuda_libflags{:});

    fprintf('\n✅  All MEX files built successfully.\n');
end

function restore_mex_env(original_env)
    fprintf('[restore_mex_env] Restoring original environment variables...\n');
    keys = original_env.keys;
    for k = 1:numel(keys)
        setenv(keys{k}, original_env(keys{k}));
    end
end

function status = cmake_build(src, bld, inst, cmake_gen, cmake_arch, args, msvc)
    if ~exist(bld,'dir'), mkdir(bld); end
    if ~exist(inst,'dir'), mkdir(inst); end

    old = cd(bld); onCleanup(@() cd(old));
    ncores = feature('numCores');
    isWin = ispc;

    if isWin
        % Force correct cl.exe for CMake!
        [cl_bin_dir, ~, ~] = fileparts(msvc.cl);
        orig_path = getenv('PATH');
        new_path = [cl_bin_dir, ';', orig_path];
        setenv('PATH', new_path);
        setenv('CC', msvc.cl);
        setenv('CXX', msvc.cl);

        % CMake configure/build
        cmake_cfg = sprintf([
            'call "%s" && set CC="%s" && set CXX="%s" && cmake %s %s "%s" -DCMAKE_INSTALL_PREFIX="%s" %s && ', ...
            'cmake --build . --config Release --target INSTALL -- /m:%d'], ...
            msvc.vcvars, msvc.cl, msvc.cl, cmake_gen, cmake_arch, src, inst, args, ncores);

        fprintf('[cmake_build] Running combined configure & build:\n%s\n', cmake_cfg);
        rc = system(cmake_cfg);
        if rc ~= 0
            fprintf('[cmake_build] Combined CMake configure & build FAILED (code %d)\n', rc);
            status = rc;
            return;
        end
    else
        cmake_cmd = sprintf('cmake "%s" -DCMAKE_INSTALL_PREFIX="%s" %s', src, inst, args);
        fprintf('[cmake_build] Running CMake configure:\n%s\n', cmake_cmd);
        if system(cmake_cmd) ~= 0, status = 1; return; end
        build_cmd = sprintf('cmake --build . -- -j%d install', ncores);
        fprintf('[cmake_build] Running CMake build:\n%s\n', build_cmd);
        status = system(build_cmd);
    end

    status = 0;
end

