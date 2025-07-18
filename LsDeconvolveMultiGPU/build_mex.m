function build_mex(debug)
% build_mex  Compile all C/C++/CUDA MEX files and static libraries.
% Robust, cross-platform, production ready.
% Centralized compile flags (edit ONCE for all libs).
% Keivan Moradi, 2025 (with ChatGPT-4o assistance)

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

    % === [0] Centralized CMake/Configure flags (EDIT HERE) ===
    cmake_flags_win = { ...
        '-DCMAKE_POSITION_INDEPENDENT_CODE=ON', ...
        '-DCMAKE_BUILD_TYPE=Release', ...
        '-DCMAKE_STATIC_LINKER_FLAGS_RELEASE="/LTCG"', ...
        '-DCMAKE_EXE_LINKER_FLAGS_RELEASE="/LTCG"', ...
        '-DCMAKE_SHARED_LINKER_FLAGS_RELEASE="/LTCG"'};
    cmake_flags_lin = { ...
        '-DCMAKE_POSITION_INDEPENDENT_CODE=ON', ...
        '-DCMAKE_BUILD_TYPE=Release'};
    win_c_flags  = '/O2 /GL'; % add arch below
    lin_c_flags  = sprintf('-O3 -march=native -flto=%d -fPIC', ncores);

    % === [1] Utility fns ===
    function p = unixify(path), p = strrep(path,'\','/'); end
    function stamp = getStamp(dir, tag)
        if nargin < 2 || isempty(tag)
            stamp = fullfile(dir, '.built');
        else
            stamp = fullfile(dir, ['.built_' tag]);
        end
    end

    % === [2] CPU feature detection ===
    if isWin
        if ~exist('cpuid_mex.mexw64', 'file')
            mex('cpuid_mex.cpp');
        end
        f = cpuid_mex();
        if f.AVX512,   instr = 'AVX512';
        elseif f.AVX2, instr = 'AVX2';
        else           instr = 'SSE'; end
    else
        instr = 'native';
    end

    % === [3] Compiler Toolchain Discovery ===
    if isWin
        % 1. Find and set up the Visual Studio environment for compilation
        cc = mex.getCompilerConfigurations('C++','Selected');
        if isempty(cc), error('No C++ compiler configured for MEX. Run "mex -setup".'); end
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
        if isempty(cl_path), error('Could not locate cl.exe from mex configuration.'); end
        MSVC_BASE    = fileparts(fileparts(fileparts(cl_path)));
        VSROOT       = fileparts(fileparts(fileparts(fileparts(MSVC_BASE)))); 
        VCVARS64     = fullfile(VSROOT,'Auxiliary','Build','vcvars64.bat');
        if ~isfile(VCVARS64), error('vcvars64.bat not found at %s', VCVARS64); end
        cc_ver   = regexp(MSVC_BASE,'\\d+\\.\\d+\\.\\d+','match','once');
        vcmd     = sprintf('"%s" -vcvars_ver=%s && set', VCVARS64, cc_ver);
        [~,envout] = system(vcmd);
        env_lines  = splitlines(strtrim(envout));
        keys_keep  = ["PATH","INCLUDE","LIB","LIBPATH","VCINSTALLDIR","VCToolsInstallDir"];
        orig_env   = containers.Map();
        for L=env_lines'
            parts = split(L{1},'=');
            if numel(parts)~=2, continue; end
            k = upper(parts{1}); v = parts{2};
            if ismember(k, keys_keep)
                orig_env(k) = getenv(k); setenv(k,v);
            end
        end
        cleanupEnv = onCleanup(@() restoreEnv(orig_env));
    
        % 2. Set msvc struct and cmake flags
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
        msvc = [];
    end

    % === [4] Paths & Versions ===
    root       = pwd;
    thirdparty = fullfile(root,'thirdparty');    if ~exist(thirdparty,'dir'), mkdir(thirdparty); end
    build_root = fullfile(root,'tiff_build');    if ~exist(build_root,'dir'), mkdir(build_root); end

    zlibng_v   = '2.2.4';
    libtiff_v  = '4.7.0';
    lz4_v      = '1.9.4';

    zlibng_src  = fullfile(thirdparty,['zlib-ng-' zlibng_v]);
    libtiff_src = fullfile(thirdparty,['tiff-'    libtiff_v]);
    lz4_src     = fullfile(thirdparty, ['lz4-' lz4_v]);

    zlibng_inst  = fullfile(build_root,'zlib-ng');
    libtiff_inst = fullfile(build_root,'libtiff');
    lz4_inst     = fullfile(build_root, 'lz4');

    % =========== 5) Build LZ4 (static) ==========
    if isWin, lz4_stamp = getStamp(lz4_inst, msvc.tag);
    else,     lz4_stamp = getStamp(lz4_inst, ''); end

    if ~isfile(lz4_stamp)
        if ~exist(lz4_src,'dir')
            tgz = fullfile(thirdparty,sprintf('lz4-%s.tar.gz',lz4_v));
            websave(tgz,sprintf('https://github.com/lz4/lz4/archive/refs/tags/v%s.tar.gz',lz4_v));
            untar(tgz,thirdparty); delete(tgz);
        end
        lz4_cmake_src = fullfile(lz4_src, 'build', 'cmake');
        builddir = fullfile(lz4_src, 'build_dir'); if ~exist(builddir,'dir'), mkdir(builddir); end
        if ~exist(lz4_inst, 'dir'), mkdir(lz4_inst); end
        if isWin
            args = [policy_flag, '-DBUILD_SHARED_LIBS=OFF', ...
                    cmake_flags_win, ...
                    sprintf('-DCMAKE_C_FLAGS_RELEASE="%s /arch:%s"', win_c_flags, instr), ...
                    sprintf('-DCMAKE_CXX_FLAGS_RELEASE="%s /arch:%s"', win_c_flags, instr)];
        else
            args = [policy_flag, '-DBUILD_SHARED_LIBS=OFF', ...
                    cmake_flags_lin, ...
                    sprintf('-DCMAKE_C_FLAGS_RELEASE="%s"', lin_c_flags), ...
                    sprintf('-DCMAKE_CXX_FLAGS_RELEASE="%s"', lin_c_flags)];
        end
        status = cmake_build(lz4_cmake_src, builddir, lz4_inst, cmake_gen, cmake_arch, args, msvc);
        if status~=0, error('lz4 build failed (code %d)',status); end
        fid = fopen(lz4_stamp, 'w'); if fid < 0, error('Cannot write stamp file: %s', lz4_stamp); end; fclose(fid);
    end

    % =========== 6) Build zlib-ng (static) ==========
    if isWin, z_stamp = getStamp(zlibng_inst, msvc.tag);
    else,     z_stamp = getStamp(zlibng_inst, ''); end
    cmake_common_flags_zlib = {policy_flag, '-DBUILD_SHARED_LIBS=OFF', '-DZLIB_COMPAT=ON', ...
        '-DZLIB_ENABLE_TESTS=OFF', '-DZLIBNG_ENABLE_TESTS=OFF', '-DWITH_GTEST=OFF', ...
        '-DWITH_BENCHMARKS=OFF', '-DWITH_BENCHMARK_APPS=OFF', ...
        '-DWITH_NATIVE_INSTRUCTIONS=ON', '-DWITH_NEW_STRATEGIES=ON', '-DWITH_OPTIM=ON', '-DWITH_AVX2=ON'};
    if ~isfile(z_stamp)
        if ~exist(zlibng_src,'dir')
            tgz = fullfile(thirdparty,sprintf('zlib-ng-%s.tar.gz',zlibng_v));
            websave(tgz,sprintf('https://github.com/zlib-ng/zlib-ng/archive/refs/tags/%s.tar.gz',zlibng_v));
            untar(tgz,thirdparty); delete(tgz);
        end
        builddir = fullfile(zlibng_src,'build'); if ~exist(builddir,'dir'), mkdir(builddir); end
        if ~exist(zlibng_inst,'dir'), mkdir(zlibng_inst); end
        if isWin
            args = [cmake_common_flags_zlib, ...
                    cmake_flags_win, ...
                    '-DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>" ', ...
                    sprintf('-DCMAKE_C_FLAGS_RELEASE="%s /arch:%s"', win_c_flags, instr), ...
                    sprintf('-DCMAKE_CXX_FLAGS_RELEASE="%s /arch:%s"', win_c_flags, instr)];
        else
            args = [cmake_common_flags_zlib, ...
                    cmake_flags_lin, ...
                    sprintf('-DCMAKE_C_FLAGS_RELEASE="%s"', lin_c_flags), ...
                    sprintf('-DCMAKE_CXX_FLAGS_RELEASE="%s"', lin_c_flags)];
        end
        status = cmake_build(zlibng_src, builddir, zlibng_inst, cmake_gen, cmake_arch, args, msvc);
        if status~=0, error('zlib-ng build failed (code %d)',status); end
        fid = fopen(z_stamp, 'w'); if fid < 0, error('Cannot write stamp file: %s', z_stamp); end; fclose(fid);
    end

    % =========== 7) Build libtiff (STATIC) ==========
    if isWin, t_stamp = getStamp(libtiff_inst, msvc.tag);
    else,     t_stamp = getStamp(libtiff_inst, ''); end
    cmake_common_flags_tiff = {policy_flag, ...
        '-Dtiff-tools=ON', '-DBUILD_SHARED_LIBS=OFF', ...
        '-Djbig=OFF', '-Djpeg=OFF', '-Dold-jpeg=OFF', '-Dlzma=OFF', '-Dwebp=OFF', ...
        '-Dlerc=OFF', '-Dpixarlog=OFF', ...
        '-Dtiff-tests=OFF', '-Dtiff-opengl=OFF', '-Dtiff-contrib=OFF', ...
        '-Dzstd=OFF', '-Dlibdeflate=OFF'};
    if ~isfile(t_stamp)
        if ~exist(libtiff_src,'dir')
            tgz = fullfile(thirdparty,sprintf('tiff-%s.tar.gz',libtiff_v));
            websave(tgz,sprintf('https://download.osgeo.org/libtiff/tiff-%s.tar.gz',libtiff_v));
            untar(tgz,thirdparty); delete(tgz);
        end
        builddir = fullfile(libtiff_src,'build'); if ~exist(builddir,'dir'), mkdir(builddir); end
        if ~exist(libtiff_inst,'dir'), mkdir(libtiff_inst); end
        if isWin
            args = [cmake_common_flags_tiff, ...
                cmake_flags_win, ...
                sprintf('-DZLIB_LIBRARY=%s',   unixify(fullfile(zlibng_inst,'lib','zlibstatic.lib'))), ...
                sprintf('-DZLIB_INCLUDE_DIR=%s',unixify(fullfile(zlibng_inst,'include'))), ...
                '-DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>" ', ...
                sprintf('-DCMAKE_C_FLAGS_RELEASE="%s /arch:%s"', win_c_flags, instr), ...
                sprintf('-DCMAKE_CXX_FLAGS_RELEASE="%s /arch:%s"', win_c_flags, instr)];
        else
            args = [cmake_common_flags_tiff, ...
                cmake_flags_lin, ...
                sprintf('-DZLIB_LIBRARY=%s',    unixify(fullfile(zlibng_inst,'lib','libz.a'))), ...
                sprintf('-DZLIB_INCLUDE_DIR=%s',unixify(fullfile(zlibng_inst,'include'))), ...
                sprintf('-DCMAKE_C_FLAGS_RELEASE="%s"', lin_c_flags), ...
                sprintf('-DCMAKE_CXX_FLAGS_RELEASE="%s"', lin_c_flags)];
        end
        status = cmake_build(libtiff_src, builddir, libtiff_inst, cmake_gen, cmake_arch, args, msvc);
        if status~=0, error('libtiff build failed (code %d)',status); end
        fid = fopen(t_stamp, 'w'); if fid < 0, error('Cannot write stamp file: %s', t_stamp); end; fclose(fid);
    end

    % =========== 8) Build hwloc (STATIC) ==========
    hwloc_v_major = '2.12'; hwloc_v_patch = '1'; hwloc_v = sprintf('%s.%s', hwloc_v_major, hwloc_v_patch);
    hwloc_src     = fullfile(thirdparty, ['hwloc-' hwloc_v]);
    hwloc_inst    = fullfile(build_root, 'hwloc');
    if isWin, hwloc_stamp = getStamp(hwloc_inst, msvc.tag);
    else, hwloc_stamp = getStamp(hwloc_inst, ''); end

    if ~isfile(hwloc_stamp)
        if ~exist(hwloc_src, 'dir')
            tgz_url = sprintf('https://download.open-mpi.org/release/hwloc/v%s/hwloc-%s.tar.gz', hwloc_v_major, hwloc_v);
            tgz = fullfile(thirdparty, sprintf('hwloc-%s.tar.gz', hwloc_v));
            fprintf('Downloading hwloc source...\n');
            websave(tgz, tgz_url); untar(tgz, thirdparty); delete(tgz);
        end
        if isWin
            cmake_dir = fullfile(hwloc_src, 'contrib', 'windows-cmake');
            cmake_txt = fullfile(cmake_dir, 'CMakeLists.txt');
            if ~exist(cmake_dir, 'dir'), mkdir(cmake_dir); end
            if ~exist(cmake_txt, 'file')
                fprintf('Downloading hwloc Windows CMakeLists.txt...\n');
                urlwrite('https://raw.githubusercontent.com/open-mpi/hwloc/refs/heads/master/contrib/windows-cmake/CMakeLists.txt', cmake_txt);
            end
            builddir = fullfile(hwloc_src, 'build'); if ~exist(builddir,'dir'), mkdir(builddir); end
            if ~exist(hwloc_inst, 'dir'), mkdir(hwloc_inst); end
            args = {policy_flag, '-DBUILD_SHARED_LIBS=OFF', '-DHWLOC_BUILD_STANDALONE=ON', ...
                '-DHWLOC_INSTALL_HEADERS=ON', ...
                '-DHWLOC_ENABLE_OPENCL=OFF', '-DHWLOC_ENABLE_CUDA=OFF', '-DHWLOC_ENABLE_NVML=OFF', ...
                '-DHWLOC_ENABLE_LIBXML2=OFF', '-DHWLOC_ENABLE_PCI=OFF', '-DHWLOC_ENABLE_TOOLS=OFF', ...
                '-DHWLOC_ENABLE_TESTING=OFF', '-DHWLOC_ENABLE_DOCS=OFF', ...
                cmake_flags_win, ...
                sprintf('-DCMAKE_C_FLAGS_RELEASE="%s /arch:%s"', win_c_flags, instr), ...
                sprintf('-DCMAKE_CXX_FLAGS_RELEASE="%s /arch:%s"', win_c_flags, instr)};
            status = cmake_build(cmake_dir, builddir, hwloc_inst, cmake_gen, cmake_arch, args, msvc);
            if status ~= 0, error('hwloc build failed (code %d)', status); end
        else
            if ~exist(hwloc_inst, 'dir'), mkdir(hwloc_inst); end
            cflags = lin_c_flags;
            cxxflags = lin_c_flags;
            build_cmd = sprintf(['cd "%s" && ' ...
                'CFLAGS="%s" CXXFLAGS="%s" ' ...
                './configure --prefix="%s" --enable-static --disable-shared ' ...
                '--disable-libxml2 --disable-pci --disable-cuda --disable-opencl --disable-nvml ' ...
                '--disable-libudev --disable-tools --disable-testing --disable-docs && ' ...
                'make -j%d && make install'], ...
                hwloc_src, cflags, cxxflags, hwloc_inst, ncores);
            status = system(build_cmd);
            if status ~= 0, error('hwloc build failed (code %d)', status); end
        end
        fid = fopen(hwloc_stamp, 'w'); if fid < 0, error('Cannot write stamp file: %s', hwloc_stamp); end; fclose(fid);
    end

    % Set include and link flags
    lz4_incflag = ['-I"' unixify(fullfile(lz4_inst, 'include')) '"'];
    inc_hwloc  = ['-I"' unixify(fullfile(hwloc_inst, 'include')) '"'];
    if ispc
        lz4_libfile = fullfile(lz4_inst, 'lib', 'lz4.lib');
        link_hwloc = {fullfile(hwloc_inst, 'lib', 'hwloc.lib')};
    else
        lz4_libfile = fullfile(lz4_inst, 'lib', 'liblz4.a');
        link_hwloc = {fullfile(hwloc_inst, 'lib', 'libhwloc.a')};
    end

    %% --------- Build CPU MEX files ---------
    inc_tiff   = ['-I"' unixify(fullfile(libtiff_inst,'include')) '"'];
    mex_cpu_win = {'-R2018a'};
    mex_cpu_linux = {'-R2018a'};
    if isWin
        link_tiff = {fullfile(libtiff_inst,'lib','tiffxx.lib'), fullfile(libtiff_inst,'lib','tiff.lib'), fullfile(zlibng_inst,'lib','zlibstatic.lib')};
        if debug
            mex_cpu_win = [mex_cpu_win, ...
                'COMPFLAGS="$COMPFLAGS /std:c++17 /Od /Zi /openmp"', ...
                'CXXFLAGS="$CXXFLAGS /std:c++17 /Od /Zi /openmp"', ...
                'LINKFLAGS="$LINKFLAGS /DEBUG /openmp"'];
        else
            mex_cpu_win = [mex_cpu_win, ...
                sprintf('COMPFLAGS="$COMPFLAGS /std:c++17 /O2 /arch:%s /GL /openmp"',instr), ...
                sprintf('CXXFLAGS="$CXXFLAGS /std:c++17 /O2 /arch:%s /GL /openmp"',instr), ...
                'LINKFLAGS="$LINKFLAGS /LTCG /openmp"'];
        end
        mex_cpu = mex_cpu_win;
    else
        link_tiff = {fullfile(libtiff_inst,'lib','libtiffxx.a'), fullfile(libtiff_inst,'lib','libtiff.a'), fullfile(zlibng_inst,'lib','libz.a')};
        if debug
            mex_cpu_linux = [mex_cpu_linux, ... 
                'CFLAGS="$CFLAGS -O0 -g -fopenmp"', ... 
                'CXXFLAGS="$CXXFLAGS -O0 -g -fopenmp"', ... 
                'LDFLAGS="$LDFLAGS -g -fopenmp"'];
        else
            mex_cpu_linux = [mex_cpu_linux, ...
                sprintf('CFLAGS="$CFLAGS -O3 -march=native -flto=%d -fopenmp"',ncores), ...
                sprintf('CXXFLAGS="$CXXFLAGS -O3 -march=native -flto=%d -fopenmp"',ncores), ...
                sprintf('LDFLAGS="$LDFLAGS -flto=%d -fopenmp"',ncores)];
        end
        mex_cpu = mex_cpu_linux;
    end
    
    fprintf('\n[MEX] Compiling CPU modules …\n');
    mex(mex_cpu{:}, 'fast_twin_tail_orderstat.cpp');
    mex(mex_cpu{:}, 'semaphore.c');
    mex(mex_cpu{:}, 'save_lz4_mex.c', lz4_incflag, lz4_libfile);
    mex(mex_cpu{:}, 'load_lz4_mex.c', lz4_incflag, lz4_libfile);
    mex(mex_cpu{:}, inc_tiff, inc_hwloc, 'load_bl_tif.cpp', 'mex_thread_utils.cpp', link_tiff{:}, link_hwloc{:});
    mex(mex_cpu{:}, inc_tiff, inc_hwloc, 'load_slab_lz4_save_as_tif.cpp', 'mex_thread_utils.cpp', lz4_incflag, lz4_libfile, link_tiff{:}, link_hwloc{:});
    
    % mex(mex_cpu{:}, inc_tiff, inc_hwloc, 'save_bl_tif.cpp', 'mex_thread_utils.cpp', link_tiff{:}, link_hwloc{:});
    % mex(mex_cpu{:}, 'load_slab_lz4.cpp', lz4_incflag, lz4_libfile);

    %% --------- Build CUDA MEX files ---------
    archs_env = getenv('BUILD_SM_ARCHS');
    if isempty(archs_env)
        sm_list = detect_sm_archs();
    else
        sm_list = strsplit(archs_env,';');
    end
    gencode_flags = strjoin(cellfun(@(sm) ...
        sprintf('-gencode=arch=compute_%s,code=sm_%s', sm, sm), sm_list, 'UniformOutput', false), ' ');

    if isWin
        if debug
            nvccflags = sprintf('NVCCFLAGS="$NVCCFLAGS -allow-unsupported-compiler -G %s"', gencode_flags);
        else
            %nvccflags = sprintf('NVCCFLAGS="$NVCCFLAGS -allow-unsupported-compiler -Xcompiler=/O2,/arch:%s,-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH %s"', instr, gencode_flags);
            nvccflags = sprintf(['NVCCFLAGS="$NVCCFLAGS -allow-unsupported-compiler -Xcompiler=/O2,/arch:%s,/openmp,-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH %s"'], ...
            instr, gencode_flags);
        end
    else
        if debug
            nvccflags = sprintf('NVCCFLAGS="$NVCCFLAGS -G %s"', gencode_flags);
        else
            %nvccflags = sprintf('NVCCFLAGS="$NVCCFLAGS -use_fast_math -Xcompiler=-Ofast,-flto=%d %s"', ncores, gencode_flags);
            nvccflags = sprintf(['NVCCFLAGS="$NVCCFLAGS -use_fast_math -Xcompiler=-Ofast,-flto=%d,-mavx2,-fopenmp %s"'], ncores, gencode_flags);
        end
    end
    % cufft_link = strsplit(strtrim(fft_lib()));
    mexcuda('-R2018a', nvccflags, 'conv3d_gpu.cu');

    % mexcuda('-R2018a', nvccflags, 'gauss3d_gpu.cu', cufft_link{:});
    % mexcuda('-R2018a', nvccflags, 'if_else.cu');

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
function status = cmake_build(src, bld, inst, gen, arch, args, msvc)
    % Ensure build/install directories exist
    if ~exist(bld, 'dir'), mkdir(bld); end
    if ~exist(inst, 'dir'), mkdir(inst); end
    oldp = cd(bld); onCleanup(@() cd(oldp));
    ncores = feature('numCores');
    isWin = ispc;

    % --- Robustly flatten args to a single cell array of char vectors
    flat_args = {};
    for k = 1:numel(args)
        a = args{k};
        if iscell(a)
            for j = 1:numel(a)
                flat_args{end+1} = char(a{j});
            end
        else
            flat_args{end+1} = char(a);
        end
    end
    args_str = strjoin(flat_args, ' ');

    if isWin
        [cl_dir,~,~] = fileparts(msvc.cl);
        setenv('PATH', [cl_dir ';' getenv('PATH')]);
        setenv('CC', msvc.cl); setenv('CXX', msvc.cl);
        cfg = sprintf(['call "%s" && cmake %s %s "%s" -DCMAKE_INSTALL_PREFIX="%s" %s && cmake --build . --config Release --target INSTALL -- /m:%d'], ...
            msvc.vcvars, gen, arch, src, inst, args_str, ncores);
        status = system(cfg);
    else
        cfg = sprintf('cmake "%s" -DCMAKE_INSTALL_PREFIX="%s" %s', src, inst, args_str);
        if system(cfg) ~= 0
            status = 1; return;
        end
        buildcmd = sprintf('cmake --build . -- -j%d install', ncores);
        status = system(buildcmd);
    end
end


function sm_list = detect_sm_archs()
    sm_env = getenv('BUILD_SM_ARCHS');
    if ~isempty(sm_env)
        sm_list = strsplit(strtrim(sm_env), {';',','});
        sm_list = cellfun(@strtrim, sm_list, 'UniformOutput', false);
        sm_list = unique(sm_list, 'stable');
        return;
    end
    [nvcc_status, nvcc_out] = system('nvcc --version');
    cuda_ver = [];
    if nvcc_status==0
        m = regexp(nvcc_out, 'release (\d+)\.(\d+)', 'tokens', 'once');
        if ~isempty(m)
            cuda_ver = str2double([m{1}, '.', m{2}]);
        end
    end
    arch_table = {
        11.0, {'52','60','61','70','75','80'};
        11.1, {'52','60','61','70','75','80'};
        11.2, {'52','60','61','70','75','80','86'};
        12.0, {'52','60','61','70','75','80','86','89'};
        12.1, {'52','60','61','70','75','80','86','89'};
        12.2, {'52','60','61','70','75','80','86','89'};
        12.3, {'52','60','61','70','75','80','86','89','90'};
        12.4, {'52','60','61','70','75','80','86','89','90'};
        12.5, {'52','60','61','70','75','80','86','89','90'};
        12.9, {'52','60','61','70','75','80','86','89','90'};
    };
    sm_list = {'75','80','86'};
    if ~isempty(cuda_ver)
        for i = size(arch_table,1):-1:1
            if cuda_ver >= arch_table{i,1}
                sm_list = arch_table{i,2};
                break;
            end
        end
    end
    try
        ngpus = gpuDeviceCount;
        for idx = 1:ngpus
            g = gpuDevice(idx);
            cc = g.ComputeCapabilityMajor*10 + g.ComputeCapabilityMinor;
            cc_str = num2str(cc);
            if ~ismember(cc_str, sm_list)
                sm_list{end+1} = cc_str;
            end
        end
    catch
    end
    sm_list = unique(sm_list, 'stable');
    sm_list_num = cellfun(@str2double, sm_list);
    [~, idx] = sort(sm_list_num);
    sm_list = sm_list(idx);
end

function cufft_link = fft_lib()
%FFT_LIB  Return platform-correct linker flags for cuFFT library.
%   Returns a string with -L and -lcufft for mexcuda.
%   Issues a warning if cuFFT is not found.

if ispc
    % Windows
    cuda_root = getenv('CUDA_PATH');
    if isempty(cuda_root)
        % Try common fallback location
        possible_roots = { ...
            'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA', ...
            'C:\CUDA' ...
        };
        cuda_root = '';
        for k = 1:numel(possible_roots)
            d = dir(fullfile(possible_roots{k}, 'v*'));
            if ~isempty(d)
                % Use the highest version
                [~, idx] = max([d.datenum]);
                cuda_root = fullfile(possible_roots{k}, d(idx).name);
                break;
            end
        end
    end
    if isempty(cuda_root) || ~isfolder(cuda_root)
        warning('fft_lib:cudaNotFound', 'Could not find CUDA Toolkit. Please set CUDA_PATH.');
        cufft_link = '';
        return;
    end
    cufft_lib = fullfile(cuda_root, 'lib', 'x64');
    if ~isfolder(cufft_lib)
        warning('fft_lib:libNotFound', 'Could not find cuFFT library folder: %s', cufft_lib);
        cufft_link = '';
        return;
    end
    cufft_link = sprintf('-L"%s" -lcufft', cufft_lib);
else
    % Linux/Mac
    cuda_root = getenv('CUDA_HOME');
    if isempty(cuda_root)
        cuda_root = getenv('CUDA_PATH');
    end
    if isempty(cuda_root)
        cuda_root = '/usr/local/cuda';
    end
    cufft_lib = fullfile(cuda_root, 'lib64');
    if ~isfolder(cufft_lib)
        warning('fft_lib:libNotFound', 'Could not find cuFFT library folder: %s', cufft_lib);
        cufft_link = '';
        return;
    end
    cufft_link = sprintf('-L%s -lcufft', cufft_lib);
end

end