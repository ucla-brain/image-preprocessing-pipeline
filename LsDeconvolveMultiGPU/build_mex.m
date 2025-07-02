function build_mex(debug)
% build_mex  Compile all C/C++/CUDA MEX files and static libraries.
% Robust, cross-platform, production ready.
% Keivan Moradi, 2024-2025 (with ChatGPT-4o assistance)

    if nargin < 1, debug = false; end
    if verLessThan('matlab','9.4')
        error('Requires MATLAB R2018a or newer.'); end
    if exist('mexcuda','file') ~= 2
        error('mexcuda not found – ensure CUDA is configured.'); end

    %---- global constants ----
    policy_min   = 3.5;
    policy_flag  = sprintf('-DCMAKE_POLICY_VERSION_MINIMUM=%.1f',policy_min);
    ncores       = feature('numCores');
    isWin        = ispc;

    %---- helper to normalize paths for CMake ----
    function p = unixify(path)
        p = strrep(path,'\','/');
    end

    %---- helper for build-stamp files ----
    function stamp = getStamp(dir, tag)
        if nargin < 2 || isempty(tag)
            stamp = fullfile(dir, '.built');
        else
            stamp = fullfile(dir, ['.built_' tag]);
        end
    end

    %% --------- 0) CPU feature detection ---------
    if isWin
        mex('cpuid_mex.cpp');  % must exist in path
        f = cpuid_mex();
        if f.AVX512, instr = 'AVX512';
        elseif f.AVX2, instr = 'AVX2';
        else instr = 'SSE'; end
    else
        instr = 'native';  % Linux flags will use -march=native
    end

    %% --------- Compiler Toolchain Discovery ---------
    if isWin
        cc = mex.getCompilerConfigurations('C++','Selected');
        if isempty(cc)
            error('No C++ compiler configured for MEX. Run "mex -setup".');
        end
        % Extract cl.exe from cc.Details.SetEnv
        setenv_lines = splitlines(cc.Details.SetEnv);
        cl_path = '';
        for i=1:numel(setenv_lines)
            tline = strtrim(setenv_lines{i});
            if startsWith(tline,'set PATH=','IgnoreCase',true)
                paths = split(extractAfter(tline,'='),';');
                for p=paths'
                    c = fullfile(strtrim(p{1}),'cl.exe');
                    if isfile(c), cl_path = c; break; end
                end
            end
            if ~isempty(cl_path), break; end
        end
        if isempty(cl_path)
            error('Could not locate cl.exe from mex configuration.');
        end
        MSVC_BASE    = fileparts(fileparts(fileparts(cl_path)));
        VSROOT       = fileparts(fileparts(fileparts(fileparts(MSVC_BASE))));
        VCVARS64     = fullfile(VSROOT,'Auxiliary','Build','vcvars64.bat');
        if ~isfile(VCVARS64)
            error('vcvars64.bat not found at %s',VCVARS64);
        end
        % Prepare MSVC environment
        cc_ver       = regexp(MSVC_BASE,'\\d+\\.\\d+\\.\\d+','match','once');
        vcmd         = sprintf('"%s" -vcvars_ver=%s && set', VCVARS64, cc_ver);
        [~,envout]   = system(vcmd);
        env_lines    = splitlines(strtrim(envout));
        keys_keep    = ["PATH","INCLUDE","LIB","LIBPATH","VCINSTALLDIR","VCToolsInstallDir"];
        orig_env     = containers.Map();
        for L=env_lines'
            parts = split(L{1},'=');
            if numel(parts)~=2, continue; end
            k = upper(parts{1}); v = parts{2};
            if ismember(k, keys_keep)
                orig_env(k) = getenv(k);
                setenv(k,v);
            end
        end
        cleanupEnv = onCleanup(@() restoreEnv(orig_env));
        % Choose CMake generator
        if contains(cc.Name,'2022')
            cmake_gen  = '-G "Visual Studio 17 2022" -T v143';
        elseif contains(cc.Name,'2019')
            cmake_gen  = '-G "Visual Studio 16 2019" -T v142';
        else
            error('Unsupported VS version: %s',cc.Name);
        end
        cmake_arch = '-A x64';
        msvc = struct('vcvars',VCVARS64,'cl',cl_path,'tag',cc_ver);
    else
        cmake_gen  = '';
        cmake_arch = '';
        msvc        = [];
    end

    %% --------- 1) Paths & Versions ---------
    root       = pwd;
    thirdparty = fullfile(root,'thirdparty');    if ~exist(thirdparty,'dir'), mkdir(thirdparty); end
    build_root = fullfile(root,'tiff_build');    if ~exist(build_root,'dir'), mkdir(build_root); end

    zlibng_v   = '2.2.4';
    libtiff_v  = '4.7.0';

    zlibng_src  = fullfile(thirdparty,['zlib-ng-' zlibng_v]);
    libtiff_src = fullfile(thirdparty,['tiff-'    libtiff_v]);

    zlibng_inst  = fullfile(build_root,'zlib-ng');
    libtiff_inst = fullfile(build_root,'libtiff');

    %% --------- 2) Download & build LZ4 ---------
    lz4_src = fullfile(thirdparty,'lz4');
    if ~exist(fullfile(lz4_src,'lz4.c'),'file')
        try
            mkdir(lz4_src);
            websave(fullfile(lz4_src,'lz4.c'), 'https://raw.githubusercontent.com/lz4/lz4/dev/lib/lz4.c');
            websave(fullfile(lz4_src,'lz4.h'), 'https://raw.githubusercontent.com/lz4/lz4/dev/lib/lz4.h');
        catch err
            error('Failed to download LZ4: %s', err.message);
        end
    end

    %% --------- 3) Build zlib-ng (all original flags restored) ---------
    if isWin
        z_stamp = getStamp(zlibng_inst, msvc.tag);
    else
        z_stamp = getStamp(zlibng_inst, '');
    end
    if ~isfile(z_stamp)
        if ~exist(zlibng_src,'dir')
            tgz = fullfile(thirdparty,sprintf('zlib-ng-%s.tar.gz',zlibng_v));
            websave(tgz,sprintf('https://github.com/zlib-ng/zlib-ng/archive/refs/tags/%s.tar.gz',zlibng_v));
            untar(tgz,thirdparty); delete(tgz);
        end
        builddir = fullfile(zlibng_src,'build'); if ~exist(builddir,'dir'), mkdir(builddir); end
        if ~exist(zlibng_inst,'dir'), mkdir(zlibng_inst); end
        if isWin
            args = {policy_flag, ...
                '-DBUILD_SHARED_LIBS=OFF', ...
                '-DZLIB_COMPAT=ON', ...
                '-DZLIB_ENABLE_TESTS=OFF', '-DZLIBNG_ENABLE_TESTS=OFF', '-DWITH_GTEST=OFF', ...
                '-DWITH_BENCHMARKS=OFF', '-DWITH_BENCHMARK_APPS=OFF', ...
                '-DCMAKE_POSITION_INDEPENDENT_CODE=ON', ...
                '-DWITH_NATIVE_INSTRUCTIONS=ON', '-DWITH_NEW_STRATEGIES=ON', '-DWITH_AVX2=ON', ...
                sprintf('-DCMAKE_C_FLAGS_RELEASE="/O2 /GL /arch:%s" ', instr), ...
                '-DCMAKE_STATIC_LINKER_FLAGS_RELEASE="/LTCG" ', ...
                '-DCMAKE_EXE_LINKER_FLAGS_RELEASE="/LTCG" ', ...
                '-DCMAKE_SHARED_LINKER_FLAGS_RELEASE="/LTCG" ' ...
            };
        else
            args = {policy_flag, ...
                '-DBUILD_SHARED_LIBS=OFF', ...
                '-DZLIB_COMPAT=ON', ...
                '-DZLIB_ENABLE_TESTS=OFF', '-DZLIBNG_ENABLE_TESTS=OFF', '-DWITH_GTEST=OFF', ...
                '-DWITH_BENCHMARKS=OFF', '-DWITH_BENCHMARK_APPS=OFF', ...
                '-DCMAKE_POSITION_INDEPENDENT_CODE=ON', ...
                '-DCMAKE_BUILD_TYPE=Release', ...
                '-DWITH_NATIVE_INSTRUCTIONS=ON', '-DWITH_NEW_STRATEGIES=ON', '-DWITH_AVX2=ON', ...
                sprintf('-DCMAKE_C_FLAGS_RELEASE="-O3 -march=native -flto=%d -fPIC"', ncores) ...
            };
        end
        status = cmake_build(zlibng_src, builddir, zlibng_inst, cmake_gen, cmake_arch, args, msvc);
        if status~=0, error('zlib-ng build failed (code %d)',status); end
        fid = fopen(z_stamp, 'w'); if fid < 0, error('Cannot write stamp file: %s', z_stamp); end; fclose(fid);
    end

    %% --------- 4) Build libtiff (all original flags restored) ---------
    if isWin
        t_stamp = getStamp(libtiff_inst, msvc.tag);
    else
        t_stamp = getStamp(libtiff_inst, '');
    end
    if ~isfile(t_stamp)
        if ~exist(libtiff_src,'dir')
            tgz = fullfile(thirdparty,sprintf('tiff-%s.tar.gz',libtiff_v));
            websave(tgz,sprintf('https://download.osgeo.org/libtiff/tiff-%s.tar.gz',libtiff_v));
            untar(tgz,thirdparty); delete(tgz);
        end
        builddir = fullfile(libtiff_src,'build'); if ~exist(builddir,'dir'), mkdir(builddir); end
        if ~exist(libtiff_inst,'dir'), mkdir(libtiff_inst); end
        if isWin
            args = {policy_flag, ...
                '-DBUILD_SHARED_LIBS=OFF', ...
                '-DCMAKE_POSITION_INDEPENDENT_CODE=ON', ...
                '-DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>" ', ...
                '-Djbig=OFF', '-Djpeg=OFF', '-Dold-jpeg=OFF', '-Dlzma=OFF', '-Dwebp=OFF', '-Dlerc=OFF', '-Dpixarlog=OFF', '-Dlibdeflate=OFF', ...
                '-Dtiff-tests=OFF', '-Dtiff-opengl=OFF', '-Dtiff-contrib=OFF', '-Dtiff-tools=OFF', '-Dzstd=OFF', ...
                sprintf('-DZLIB_LIBRARY=%s', unixify(fullfile(zlibng_inst,'lib','zlibstatic.lib'))), ...
                sprintf('-DZLIB_INCLUDE_DIR=%s', unixify(fullfile(zlibng_inst,'include'))), ...
                sprintf('-DCMAKE_C_FLAGS_RELEASE="/O2 /GL /arch:%s" ', instr), ...
                '-DCMAKE_STATIC_LINKER_FLAGS_RELEASE="/LTCG" ', ...
                '-DCMAKE_EXE_LINKER_FLAGS_RELEASE="/LTCG" ', ...
                '-DCMAKE_SHARED_LINKER_FLAGS_RELEASE="/LTCG" ' ...
            };
        else
            args = {policy_flag, ...
                '-DBUILD_SHARED_LIBS=OFF', ...
                '-DCMAKE_POSITION_INDEPENDENT_CODE=ON', ...
                '-Djbig=OFF', '-Djpeg=OFF', '-Dold-jpeg=OFF', '-Dlzma=OFF', '-Dwebp=OFF', '-Dlerc=OFF', '-Dpixarlog=OFF', '-Dlibdeflate=OFF', ...
                '-Dtiff-tests=OFF', '-Dtiff-opengl=OFF', '-Dzstd=OFF', ...
                sprintf('-DZLIB_LIBRARY=%s', unixify(fullfile(zlibng_inst,'lib','libz.a'))), ...
                sprintf('-DZLIB_INCLUDE_DIR=%s', unixify(fullfile(zlibng_inst,'include'))), ...
                sprintf('-DCMAKE_C_FLAGS_RELEASE="-O3 -march=native -flto=%d -fPIC" ', ncores), ...
                sprintf('-DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -flto=%d -fPIC" ', ncores) ...
            };
        end
        status = cmake_build(libtiff_src, builddir, libtiff_inst, cmake_gen, cmake_arch, args, msvc);
        if status~=0, error('libtiff build failed (code %d)',status); end
        fid = fopen(t_stamp, 'w'); if fid < 0, error('Cannot write stamp file: %s', t_stamp); end; fclose(fid);
    end

    %% --------- 5) Prepare MEX flags ---------
    inc_tiff   = ['-I"' unixify(fullfile(libtiff_inst,'include')) '"'];
    if isWin
        link_tiff = {fullfile(libtiff_inst,'lib','tiffxx.lib'), fullfile(libtiff_inst,'lib','tiff.lib'), fullfile(zlibng_inst,'lib','zlibstatic.lib')};
        if debug
            mex_cpu = {'-R2018a', 'COMPFLAGS="$COMPFLAGS /std:c++17 /Od /Zi"', 'LINKFLAGS="$LINKFLAGS /DEBUG"'};
        else
            mex_cpu = {'-R2018a', sprintf('COMPFLAGS="$COMPFLAGS /std:c++17 /O2 /arch:%s /GL"',instr), 'LINKFLAGS="$LINKFLAGS /LTCG"'};
        end
    else
        link_tiff = {fullfile(libtiff_inst,'lib','libtiffxx.a'), fullfile(libtiff_inst,'lib','libtiff.a'), fullfile(zlibng_inst,'lib','libz.a')};
        if debug
            mex_cpu = {'-R2018a','CFLAGS="$CFLAGS -O0 -g"','CXXFLAGS="$CXXFLAGS -O0 -g"','LDFLAGS="$LDFLAGS -g"'};
        else
            mex_cpu = {'-R2018a', sprintf('CFLAGS="$CFLAGS -O3 -march=native -flto=%d"',ncores), sprintf('CXXFLAGS="$CXXFLAGS -O3 -march=native -flto=%d"',ncores), sprintf('LDFLAGS="$LDFLAGS -flto=%d"',ncores)};
        end
    end

    %% --------- 6) Build CPU MEX files ---------
    fprintf('\n[MEX] Compiling CPU modules …\n');
    mex(mex_cpu{:}, 'semaphore.c');
    mex(mex_cpu{:}, 'save_lz4_mex.c',    fullfile(lz4_src,'lz4.c'), ['-I"' unixify(lz4_src) '"']);
    mex(mex_cpu{:}, 'load_lz4_mex.c',    fullfile(lz4_src,'lz4.c'), ['-I"' unixify(lz4_src) '"']);
    mex(mex_cpu{:}, 'load_slab_lz4.cpp', fullfile(lz4_src,'lz4.c'), ['-I"' unixify(lz4_src) '"']);
    mex(mex_cpu{:}, inc_tiff, 'load_bl_tif.cpp', link_tiff{:});
    mex(mex_cpu{:}, inc_tiff, 'save_bl_tif.cpp', link_tiff{:});

    %% --------- 7) Build CUDA MEX files ---------
    archs_env = getenv('BUILD_SM_ARCHS');
    if isempty(archs_env)
        sm_list = {'75','80','86','89'};  % default SM architectures
    else
        sm_list = strsplit(archs_env,';');
    end
    gencode = cellfun(@(sm) sprintf('-gencode=arch=compute_%s,code=sm_%s',sm,sm), sm_list, 'UniformOutput',false);
    if isWin
        if debug
            nvccflags = 'NVCCFLAGS="$NVCCFLAGS -allow-unsupported-compiler -G -Xcompiler=/O2,/arch:%s,/GL,-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"';
        else
            % allow-unsupported-compiler silences CUDA vs MSVC mismatch warnings
            nvccflags = sprintf('NVCCFLAGS="$NVCCFLAGS -allow-unsupported-compiler -Xcompiler=/O2,/arch:%s,/GL,-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"',instr);
        end
    else
        if debug
            nvccflags = 'NVCCFLAGS="$NVCCFLAGS -G"';
        else
            nvcc_base = sprintf('NVCCFLAGS="$NVCCFLAGS -use_fast_math -Xcompiler -Ofast -Xcompiler -flto=%d"',ncores);
            nvccflags = strjoin([{nvcc_base}, gencode], ' ');
        end
    end
    mexcuda('-R2018a', nvccflags, 'gauss3d_gpu.cu');
    mexcuda('-R2018a', nvccflags, 'conv3d_gpu.cu');

    fprintf('\n✅  All MEX files built successfully.\n');
end

%%-------- helper to restore original env --------
function restoreEnv(orig)
    keys = orig.keys;
    for k = 1:numel(keys)
        setenv(keys{k}, orig(keys{k}));
    end
end

%%-------- CMake build wrapper --------
function status = cmake_build(src,bld,inst,gen,arch,args,msvc)
    if ~exist(bld,'dir'), mkdir(bld); end
    if ~exist(inst,'dir'), mkdir(inst); end
    oldp = cd(bld); onCleanup(@() cd(oldp));
    ncores = feature('numCores'); isWin = ispc;
    args_str = strjoin(args,' ');
    if isWin
        % ensure correct cl.exe in PATH
        [cl_dir,~,~] = fileparts(msvc.cl);
        setenv('PATH',[cl_dir ';' getenv('PATH')]);
        setenv('CC',msvc.cl); setenv('CXX',msvc.cl);
        cfg = sprintf('call "%s" && cmake %s %s "%s" -DCMAKE_INSTALL_PREFIX="%s" %s && cmake --build . --config Release --target INSTALL -- /m:%d', ...
                      msvc.vcvars, gen, arch, src, inst, args_str, ncores);
        status = system(cfg);
    else
        cfg = sprintf('cmake "%s" -DCMAKE_INSTALL_PREFIX="%s" %s', src, inst, args_str);
        if system(cfg)~=0, status=1; return; end
        buildcmd = sprintf('cmake --build . -- -j%d install', ncores);
        status = system(buildcmd);
    end
end
