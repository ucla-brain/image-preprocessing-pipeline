% ===============================
% build_mex.m (Patched for static libtiff, zlib, and zstd + LTO)
% ===============================
% Compile semaphore, LZ4, and GPU MEX files using locally-compiled static libtiff,
% zlib, and zstd, with link-time optimization. Cross-platform: Linux & Windows.

function build_mex(debug)
if nargin<1, debug = false; end

% MATLAB version check
if verLessThan('matlab', '9.4')
    error('Requires MATLAB R2018a or newer.');
end

% Workspace dirs
rootDir = pwd;
thirdDir = fullfile(rootDir, 'third_party');
if ~isfolder(thirdDir), mkdir(thirdDir); end

% Versions
zlibVer  = '1.2.13';
zstdVer  = '1.5.5';
tiffVer  = '4.7.0';

% Paths
zlibSrc    = fullfile(thirdDir, ['zlib-' zlibVer]);
zlibBuild  = fullfile(thirdDir, 'zlib-build');
zstdSrc    = fullfile(thirdDir, ['zstd-' zstdVer]);
zstdBuild  = fullfile(thirdDir, 'zstd-build');
tiffSrc    = fullfile(thirdDir, ['tiff-' tiffVer]);
tiffBuild  = fullfile(thirdDir, 'tiff-build');

% Stamp files
stampZ = fullfile(zlibBuild, '.built');
stampS = fullfile(zstdBuild, '.built');
stampT = fullfile(tiffBuild, '.built');

nCores = feature('numCores');

% 1) Build zlib static + LTO
if ~isfile(stampZ)
    fprintf('Building zlib %s\n', zlibVer);
    ok = try_build_zlib(zlibSrc, zlibBuild, nCores);
    if ~ok, error('zlib build failed'); end
    fclose(fopen(stampZ,'w'));
end

% 2) Build zstd static + LTO
if ~isfile(stampS)
    fprintf('Building zstd %s\n', zstdVer);
    ok = try_build_zstd(zstdSrc, zstdBuild, nCores);
    if ~ok, error('zstd build failed'); end
    fclose(fopen(stampS,'w'));
end

% 3) Build libtiff static, linking against zlib/zstd
if ~isfile(stampT)
    fprintf('Building libtiff %s\n', tiffVer);
    ok = try_build_libtiff(tiffSrc, tiffBuild, zlibBuild, zstdBuild, nCores);
    if ~ok, error('libtiff build failed'); end
    fclose(fopen(stampT,'w'));
end

% Prepare include & lib flags
includeFlags = {['-I', fullfile(tiffBuild,'include')]};

if ispc
    libPath = fullfile(tiffBuild,'lib');
    libFlags = {['-L', libPath], 'tiff', 'zlibstatic', 'zstd'};  % .lib names
else
    libPath = fullfile(tiffBuild,'lib');
    libFlags = {['-L', libPath], '-ltiff', '-lz', '-lzstd', '-llz4', '-static'};
end

% --- Configure mex flags with LTO and static runtime ---
if ispc
    if debug
        mexFlags = {
            '-R2018a', ...
            'COMPFLAGS="$COMPFLAGS /std:c++17 /Od /Zi /MTd"', ...
            ['LINKFLAGS="$LINKFLAGS /DEBUG /MTd /LTCG /LIBPATH:' libPath '"']
        };
    else
        mexFlags = {
            '-R2018a', ...
            'COMPFLAGS="$COMPFLAGS /std:c++17 /O2 /MT /GL"', ...
            ['LINKFLAGS="$LINKFLAGS /INCREMENTAL:NO /LTCG /OPT:REF /OPT:ICF /LIBPATH:' libPath '"']
        };
    end
else
    if debug
        mexFlags = {
            '-R2018a', ...
            'CFLAGS="$CFLAGS -O0 -g -flto"', ...
            'CXXFLAGS="$CXXFLAGS -O0 -g -flto"', ...
            'LDFLAGS="$LDFLAGS -g -flto -static"'
        };
    else
        mexFlags = {
            '-R2018a', ...
            'CFLAGS="$CFLAGS -O3 -march=native -fomit-frame-pointer -flto"', ...
            'CXXFLAGS="$CXXFLAGS -O3 -march=native -fomit-frame-pointer -flto"', ...
            'LDFLAGS="$LDFLAGS -flto -static"'
        };
    end
end

% Now libtiff test to confirm static build
testMex = fullfile(tempdir, 'tiff_test_mex.c');
fid = fopen(testMex,'w');
fprintf(fid, ["#include <stdio.h>\n#include 'tiffio.h'\nint main(){ printf('TIFF OK: %s\\n', TIFFGetVersion()); return 0;} "]);
fclose(fid);
try
    if ispc
        fprintf('Testing libtiff link on Windows...\n');
        mex(mexFlags{:}, testMex, includeFlags{:}, ['-L', libPath], 'tiff');
    else
        fprintf('Testing libtiff link on Linux...\n');
        mex(mexFlags{:}, testMex, includeFlags{:}, libFlags{:});
    end
    delete(testMex);
catch ME
    warning('Post-build libtiff test failed: %s', ME.message);
    error('Static libtiff link test failed');
end

fprintf('Static libtiff, zlib, and zstd built successfully!\n');

end  % build_mex


%% Helper: build zlib via CMake
function ok = try_build_zlib(srcDir, outDir, cores)
    orig = pwd;
    if ~isfolder(srcDir)
        url = sprintf('https://zlib.net/zlib-%s.tar.gz', srcDir(regexp(srcDir,'\d')));
        archive = fullfile(fileparts(srcDir), ['zlib-', srcDir(end-4:end), '.tar.gz']);
        system(sprintf('curl -L -o %s %s', archive, url));
        system(sprintf('tar xf %s -C %s', archive, fileparts(srcDir)));
    end
    if isfolder(outDir), rmdir(outDir,'s'); end
    mkdir(outDir);
    cd(outDir);
    cmd = sprintf(['cmake -S %s -B %s -DCMAKE_INSTALL_PREFIX=%s '...
                   '-DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON '...
                   '-DCMAKE_BUILD_TYPE=Release '...
                   '-DCMAKE_C_FLAGS_RELEASE="-O3 -flto"'], srcDir, outDir, outDir);
    status = system(cmd);
    if status==0, status = system(sprintf('cmake --build %s --target install -- -j%d', outDir, cores)); end
    cd(orig);
    ok = (status==0);
end

%% Helper: build zstd via CMake
function ok = try_build_zstd(srcDir, outDir, cores)
    orig = pwd;
    if ~isfolder(srcDir)
        url = sprintf('https://github.com/facebook/zstd/releases/download/v%s/zstd-%s.tar.gz', srcDir(end-3:end), srcDir(end-3:end));
        archive = fullfile(fileparts(srcDir), ['zstd-', srcDir(end-3:end), '.tar.gz']);
        system(sprintf('curl -L -o %s %s', archive, url));
        system(sprintf('tar xf %s -C %s', archive, fileparts(srcDir)));
    end
    if isfolder(outDir), rmdir(outDir,'s'); end
    mkdir(outDir);
    cd(outDir);
    cmd = sprintf(['cmake -S %s -B %s -DCMAKE_INSTALL_PREFIX=%s '...
                   '-DBUILD_SHARED_LIBS=OFF -DZSTD_BUILD_SHARED=OFF '...
                   '-DCMAKE_POSITION_INDEPENDENT_CODE=ON '...
                   '-DCMAKE_BUILD_TYPE=Release '...
                   '-DCMAKE_C_FLAGS_RELEASE="-O3 -flto"'], srcDir, outDir, outDir);
    status = system(cmd);
    if status==0, status = system(sprintf('cmake --build %s --target install -- -j%d', outDir, cores)); end
    cd(orig);
    ok = (status==0);
end

%% Helper: build libtiff via CMake, static
function ok = try_build_libtiff(srcDir, outDir, zlibDir, zstdDir, cores)
    orig = pwd;
    if ~isfolder(srcDir)
        url = sprintf('https://download.osgeo.org/libtiff/tiff-%s.tar.gz', srcDir(end-5:end));
        archive = fullfile(fileparts(srcDir), ['tiff-', srcDir(end-5:end), '.tar.gz']);
        system(sprintf('curl -L -o %s %s', archive, url));
        system(sprintf('tar xf %s -C %s', archive, fileparts(srcDir)));
    end
    if isfolder(outDir), rmdir(outDir,'s'); end
    mkdir(outDir);
    cd(outDir);
    prefixFlags = sprintf('-DCMAKE_PREFIX_PATH="%s;%s"', zlibDir, zstdDir);
    cmd = sprintf([...
        'cmake -S %s -B %s -DCMAKE_INSTALL_PREFIX=%s %s '...
        '-DBUILD_SHARED_LIBS=OFF -DENABLE_CXX=ON '...
        '-DENABLE_JPEG=OFF -DENABLE_JBIG=OFF -DENABLE_LZMA=OFF '...
        '-DENABLE_WEBP=OFF -DENABLE_LERC=OFF -DENABLE_PIXARLOG=OFF '...
        '-DENABLE_ZLIB=ON -DZLIB_LIBRARY=%s/lib/libz.a -DZLIB_INCLUDE_DIR=%s/include '...
        '-DENABLE_ZSTD=ON -DZSTD_LIBRARY=%s/lib/libzstd.a -DZSTD_INCLUDE_DIR=%s/include '...
        '-DENABLE_LZ4=ON -DLZ4_LIBRARY=%s/lib/liblz4.a -DLZ4_INCLUDE_DIR=%s/include '...
        '-DCMAKE_C_FLAGS_RELEASE="-O3 -flto" -DCMAKE_CXX_FLAGS_RELEASE="-O3 -flto" '...
        '-DCMAKE_BUILD_TYPE=Release'], ...
        srcDir, outDir, outDir, prefixFlags, zlibDir, zlibDir, zstdDir, zstdDir, zlibDir, zlibDir);
    status = system(cmd);
    if status==0
        status = system(sprintf('cmake --build %s --target install --config Release -- -j%d', outDir, cores));
    end
    cd(orig);
    ok = (status==0);
end
