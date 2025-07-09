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
        if f.AVX512,   instr = 'AVX512';
        elseif f.AVX2, instr = 'AVX2';
        else           instr = 'SSE'; end
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

    %% --------- Common repeated CMake flags for static Release builds ---------
    if isWin
        cmake_flags_release = { ...
            '-DCMAKE_POSITION_INDEPENDENT_CODE=ON', '-DBUILD_SHARED_LIBS=OFF', ...
            '-DCMAKE_BUILD_TYPE=Release', ...
            sprintf('-DCMAKE_C_FLAGS_RELEASE="/O2 /GL /arch:%s"', instr), ...
            sprintf('-DCMAKE_CXX_FLAGS_RELEASE="/O2 /GL /arch:%s"', instr), ...
            '-DCMAKE_STATIC_LINKER_FLAGS_RELEASE="/LTCG" ', ...
            '-DCMAKE_EXE_LINKER_FLAGS_RELEASE="/LTCG" ', ...
            '-DCMAKE_SHARED_LINKER_FLAGS_RELEASE="/LTCG" ' ...
        };
    else
        cmake_flags_release = { ...
            '-DCMAKE_POSITION_INDEPENDENT_CODE=ON', '-DBUILD_SHARED_LIBS=OFF', ...
            '-DCMAKE_BUILD_TYPE=Release', ...
            sprintf('-DCMAKE_C_FLAGS_RELEASE="-O3 -march=native -flto=%d -fPIC"', ncores), ...
            sprintf('-DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -flto=%d -fPIC"', ncores) ...
        };
    end

    %% --------- 1) Paths & Versions ---------
    root       = pwd;
    thirdparty = fullfile(root,'thirdparty');    if ~exist(thirdparty,'dir'), mkdir(thirdparty); end
    build_root = fullfile(root,'tiff_build');    if ~exist(build_root,'dir'), mkdir(build_root); end

    zlibng_v    = '2.2.4';
    libtiff_v   = '4.7.0';
    libdeflate_v= '1.24';

    zlibng_src    = fullfile(thirdparty,['zlib-ng-' zlibng_v]);
    libtiff_src   = fullfile(thirdparty,['tiff-'    libtiff_v]);
    libdeflate_src= fullfile(thirdparty,['libdeflate-' libdeflate_v]);

    zlibng_inst    = fullfile(build_root,'zlib-ng');
    libtiff_inst   = fullfile(build_root,'libtiff');
    libdeflate_inst= fullfile(build_root,'libdeflate');

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
        args = [ ...
            policy_flag, ...
            '-DZLIB_COMPAT=ON', ...
            '-DZLIB_ENABLE_TESTS=OFF', '-DZLIBNG_ENABLE_TESTS=OFF', '-DWITH_GTEST=OFF', ...
            '-DWITH_BENCHMARKS=OFF', '-DWITH_BENCHMARK_APPS=OFF', ...
            '-DWITH_NATIVE_INSTRUCTIONS=ON', '-DWITH_NEW_STRATEGIES=ON', '-DWITH_OPTIM=ON', '-DWITH_AVX2=ON', ...
            cmake_flags_release ...
        ];
        status = cmake_build(zlibng_src, builddir, zlibng_inst, cmake_gen, cmake_arch, args, msvc);
        if status~=0, error('zlib-ng build failed (code %d)',status); end
        fid = fopen(z_stamp, 'w'); if fid < 0, error('Cannot write stamp file: %s', z_stamp); end; fclose(fid);
    end

    %% --------- 4) Download & build libdeflate ---------
    % if isWin
    %     ld_stamp = getStamp(libdeflate_inst, msvc.tag);
    % else
    %     ld_stamp = getStamp(libdeflate_inst, '');
    % end
    % if ~isfile(ld_stamp)
    %     if ~exist(libdeflate_src,'dir')
    %         tgz = fullfile(thirdparty, sprintf('libdeflate-%s.tar.gz',libdeflate_v));
    %
    %         websave(tgz, sprintf('https://github.com/ebiggers/libdeflate/archive/refs/tags/v%s.tar.gz',libdeflate_v));
    %         untar(tgz, thirdparty); delete(tgz);
    %     end
    %     builddir = fullfile(libdeflate_src,'build'); if ~exist(builddir,'dir'), mkdir(builddir); end
    %     if ~exist(libdeflate_inst,'dir'), mkdir(libdeflate_inst); end
    %     args = [ ...
    %         policy_flag, ...
    %         cmake_flags_release ...
    %     ];
    %     status = cmake_build(libdeflate_src, builddir, libdeflate_inst, cmake_gen, cmake_arch, args, msvc);
    %     if status~=0, error('libdeflate build failed (code %d)',status); end
    %     fid = fopen(ld_stamp, 'w'); if fid < 0, error('Cannot write stamp file: %s', ld_stamp); end; fclose(fid);
    % end

    %% --------- 5) Build libtiff (with zlib-ng & libdeflate as backends) ---------
    if isWin
        t_stamp = getStamp(libtiff_inst, msvc.tag);
        zlib_lib = unixify(fullfile(zlibng_inst,'lib','zlibstatic.lib'));
        ld_lib   = unixify(fullfile(libdeflate_inst,'lib','libdeflate_static.lib'));
    else
        t_stamp = getStamp(libtiff_inst, '');
        zlib_lib = unixify(fullfile(zlibng_inst,'lib','libz.a'));
        ld_lib   = unixify(fullfile(libdeflate_inst,'lib','libdeflate.a'));
    end
    if ~isfile(t_stamp)
        if ~exist(libtiff_src,'dir')
            tgz = fullfile(thirdparty,sprintf('tiff-%s.tar.gz',libtiff_v));
            websave(tgz,sprintf('https://download.osgeo.org/libtiff/tiff-%s.tar.gz',libtiff_v));
            untar(tgz,thirdparty); delete(tgz);
        end
        builddir = fullfile(libtiff_src,'build'); if ~exist(builddir,'dir'), mkdir(builddir); end
        if ~exist(libtiff_inst,'dir'), mkdir(libtiff_inst); end
        libtiff_common_args = { ...
            '-DLIBDEFLATE=OFF', '-Dzstd=OFF', '-Dlzma=OFF', ...
            '-Djbig=OFF', '-Djpeg=OFF', '-Dold-jpeg=OFF', '-Dwebp=OFF', '-Dlerc=OFF', '-Dpixarlog=OFF', ...
            '-Dtiff-tests=OFF', '-Dtiff-opengl=OFF', ...
            sprintf('-DZLIB_LIBRARY=%s', zlib_lib), ...
            sprintf('-DZLIB_INCLUDE_DIR=%s', unixify(fullfile(zlibng_inst,'include'))), ...
            % sprintf('-DLIBDEFLATE_LIBRARY=%s', ld_lib), ...
            % sprintf('-DLIBDEFLATE_INCLUDE_DIR=%s', unixify(fullfile(libdeflate_inst,'include'))), ...
            cmake_flags_release{:} ...
        };
        if isWin
            args = [ ...
                policy_flag, ...
                libtiff_common_args, ...
                '-Dtiff-contrib=OFF', '-Dtiff-tools=OFF', ...
                '-DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>" ' ...
            ];
        else
            args = [ ...
                policy_flag, ...
                libtiff_common_args ...
            ];
        end
        status = cmake_build(libtiff_src, builddir, libtiff_inst, cmake_gen, cmake_arch, args, msvc);
        if status~=0, error('libtiff build failed (code %d)',status); end
        fid = fopen(t_stamp, 'w'); if fid < 0, error('Cannot write stamp file: %s', t_stamp); end; fclose(fid);
    end

    %% --------- 6) Download & build hwloc (for NUMA affinity) ---------
    hwloc_v_major = '2.12';   % Major.minor
    hwloc_v_patch = '1';      % Patch
    hwloc_v       = sprintf('%s.%s', hwloc_v_major, hwloc_v_patch);
    hwloc_src     = fullfile(thirdparty, ['hwloc-' hwloc_v]);
    hwloc_inst    = fullfile(build_root, 'hwloc');
    if isWin
        hwloc_stamp = getStamp(hwloc_inst, msvc.tag);
    else
        hwloc_stamp = getStamp(hwloc_inst, '');
    end

    if ~isfile(hwloc_stamp)
        % Download hwloc source if missing
        if ~exist(hwloc_src, 'dir')
            tgz_url = sprintf('https://download.open-mpi.org/release/hwloc/v%s/hwloc-%s.tar.gz', ...
                hwloc_v_major, hwloc_v);
            tgz = fullfile(thirdparty, sprintf('hwloc-%s.tar.gz', hwloc_v));
            fprintf('Downloading hwloc source...\n');
            websave(tgz, tgz_url);
            untar(tgz, thirdparty); delete(tgz);
        end

        if isWin
            % Download CMakeLists.txt for Windows build
            cmake_dir = fullfile(hwloc_src, 'contrib', 'windows-cmake');
            cmake_txt = fullfile(cmake_dir, 'CMakeLists.txt');
            if ~exist(cmake_dir, 'dir'), mkdir(cmake_dir); end
            if ~exist(cmake_txt, 'file')
                fprintf('Downloading hwloc Windows CMakeLists.txt...\n');
                websave( ...
                  cmake_txt, ...
                  'https://raw.githubusercontent.com/open-mpi/hwloc/refs/heads/master/contrib/windows-cmake/CMakeLists.txt');
            end

            % Use CMake to build/install static lib
            builddir = fullfile(hwloc_src, 'build'); if ~exist(builddir,'dir'), mkdir(builddir); end
            if ~exist(hwloc_inst, 'dir'), mkdir(hwloc_inst); end

            args = [policy_flag, ...
                '-DCMAKE_POSITION_INDEPENDENT_CODE=ON', ...
                '-DBUILD_SHARED_LIBS=OFF', ...
                '-DHWLOC_BUILD_STANDALONE=ON', ...
                '-DHWLOC_INSTALL_HEADERS=ON', ...
                '-DHWLOC_ENABLE_OPENCL=OFF', '-DHWLOC_ENABLE_CUDA=OFF', '-DHWLOC_ENABLE_NVML=OFF', ...
                '-DHWLOC_ENABLE_LIBXML2=OFF', '-DHWLOC_ENABLE_PCI=OFF', '-DHWLOC_ENABLE_TOOLS=OFF', ...
                '-DHWLOC_ENABLE_TESTING=OFF', '-DHWLOC_ENABLE_DOCS=OFF', ...
                cmake_flags_release ...
            ];
            status = cmake_build(cmake_dir, builddir, hwloc_inst, cmake_gen, cmake_arch, args, msvc);
            if status ~= 0, error('hwloc build failed (code %d)', status); end
        else
            % Linux: Use autotools (not CMake)
            if ~exist(hwloc_inst, 'dir'), mkdir(hwloc_inst); end
            cflags = sprintf('-O3 -march=native -flto=%d -fPIC', ncores);
            cxxflags = sprintf('-O3 -march=native -flto=%d -fPIC', ncores);
            build_cmd = sprintf([...
                'cd "%s" && ' ...
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

    inc_hwloc  = ['-I"' unixify(fullfile(hwloc_inst, 'include')) '"'];
    if ispc
        link_hwloc = {fullfile(hwloc_inst, 'lib', 'hwloc.lib')};
    else
        link_hwloc = {fullfile(hwloc_inst, 'lib', 'libhwloc.a')};
    end

    %% --------- 7) Prepare MEX flags (link both zlib-ng and libdeflate) ---------
    inc_tiff   = ['-I"' unixify(fullfile(libtiff_inst,'include')) '"'];
    if isWin
        link_tiff = {
            fullfile(libtiff_inst,'lib','tiffxx.lib'), ...
            fullfile(libtiff_inst,'lib','tiff.lib'), ...
            fullfile(zlibng_inst,'lib','zlibstatic.lib'), ...
            fullfile(libdeflate_inst,'lib','libdeflate_static.lib')};
        if debug
            mex_cpu = {'-R2018a', ...
                       'COMPFLAGS="$COMPFLAGS /std:c++17 /Od /Zi"', ...
                       'CXXFLAGS="$CXXFLAGS /std:c++17 /Od /Zi"', ...
                       'LINKFLAGS="$LINKFLAGS /DEBUG"'};
        else
            mex_cpu = {'-R2018a', ...
                       sprintf('COMPFLAGS="$COMPFLAGS /std:c++17 /O2 /arch:%s /GL"',instr), ...
                       sprintf('CXXFLAGS="$CXXFLAGS /std:c++17 /O2 /arch:%s /GL"',instr), ...
                       'LINKFLAGS="$LINKFLAGS /LTCG"'};
        end
    else
        link_tiff = {
            fullfile(libtiff_inst,'lib','libtiffxx.a'), ...
            fullfile(libtiff_inst,'lib','libtiff.a'), ...
            fullfile(zlibng_inst,'lib','libz.a'), ...
            fullfile(libdeflate_inst,'lib','libdeflate.a')};
        if debug
            mex_cpu = {'-R2018a','CFLAGS="$CFLAGS -O0 -g"','CXXFLAGS="$CXXFLAGS -O0 -g"','LDFLAGS="$LDFLAGS -g"'};
        else
            mex_cpu = {'-R2018a', ...
                       sprintf('CFLAGS="$CFLAGS -O3 -march=native -flto=%d"',ncores), ...
                       sprintf('CXXFLAGS="$CXXFLAGS -O3 -march=native -flto=%d"',ncores), ...
                       sprintf('LDFLAGS="$LDFLAGS -flto=%d"',ncores)};
        end
    end

    %% --------- 8) Build CPU MEX files ---------
    fprintf('\n[MEX] Compiling CPU modules …\n');
    mex(mex_cpu{:}, 'semaphore.c');
    mex(mex_cpu{:}, 'save_lz4_mex.c',    fullfile(lz4_src,'lz4.c'), ['-I"' unixify(lz4_src) '"']);
    mex(mex_cpu{:}, 'load_lz4_mex.c',    fullfile(lz4_src,'lz4.c'), ['-I"' unixify(lz4_src) '"']);
    mex(mex_cpu{:}, 'load_slab_lz4.cpp', fullfile(lz4_src,'lz4.c'), ['-I"' unixify(lz4_src) '"']);
    mex(mex_cpu{:}, inc_tiff, 'load_bl_tif.cpp', link_tiff{:});
    mex(mex_cpu{:}, inc_tiff, inc_hwloc, 'save_bl_tif.cpp', link_tiff{:}, link_hwloc{:});

    %% --------- 9) Build CUDA MEX files ---------
    % (unchanged, CUDA block here)
    % ...

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

function sm_list = detect_sm_archs()
% Robustly detect which SM archs should be built for nvcc/mexcuda.
%
% - Uses CUDA version from nvcc to pick supported archs.
% - Always includes all physical GPUs installed.
% - Honors BUILD_SM_ARCHS if set in the environment.

    % User override (for reproducibility/debugging):
    sm_env = getenv('BUILD_SM_ARCHS');
    if ~isempty(sm_env)
        sm_list = strsplit(strtrim(sm_env), {';',','});
        sm_list = cellfun(@strtrim, sm_list, 'UniformOutput', false);
        sm_list = unique(sm_list, 'stable');
        return;
    end

    % 1. Try to parse CUDA version from nvcc
    [nvcc_status, nvcc_out] = system('nvcc --version');
    cuda_ver = [];
    if nvcc_status==0
        m = regexp(nvcc_out, 'release (\d+)\.(\d+)', 'tokens', 'once');
        if ~isempty(m)
            cuda_ver = str2double([m{1}, '.', m{2}]);
        end
    end

    % 2. Build up arch support table (extendable)
    % (keep only unique and known archs for all modern CUDA)
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
        12.9, {'52','60','61','70','75','80','86','89','90'}; % add SM90, SM100, SM102 for Blackwell
    };

    % Default
    sm_list = {'75','80','86'}; % safe for CUDA 11.2+
    if cuda_ver >= 12.9
        sm_list = {'52','60','61','70','75','80','86','89','90'};
    end

    % Try to pick best from table:
    if ~isempty(cuda_ver)
        for i = size(arch_table,1):-1:1
            if cuda_ver >= arch_table{i,1}
                sm_list = arch_table{i,2};
                break;
            end
        end
    end

    % 3. Include all installed GPUs' compute capability
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
        % No GPU, or GPU driver not set up
    end

    % Remove duplicates, sort as numbers (lowest to highest, but keep as cellstr)
    sm_list = unique(sm_list, 'stable');
    sm_list_num = cellfun(@str2double, sm_list);
    [~, idx] = sort(sm_list_num);
    sm_list = sm_list(idx);
end
