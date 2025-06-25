function save_bl_tif_test
% Extended regression + benchmark for save_bl_tif MEX
%
% • Uses gamma-distributed, sparse test volumes.
% • Records both logical and physical (btrfs/zstd) file sizes.

fprintf("🧪  save_bl_tif extended test-suite\n");

%% ---------- sandbox (deleted on exit) ----------
tmpRoot = tempname;
mkdir(tmpRoot);
cSandbox = onCleanup(@() sandbox_cleanup(tmpRoot));

%% ---------- A. basic 2-D / 3-D-singleton ----------
rng(42);
vol2d = uint8(randi(255,[256 256]));
fn2d  = fullfile(tmpRoot,'basic_2d.tif');
save_bl_tif(vol2d,{fn2d},false,'none');
assert(isequal(readTiff(fn2d), vol2d));

vol3d = reshape(vol2d,256,256,1);
fn3d  = fullfile(tmpRoot,'basic_3d.tif');
save_bl_tif(vol3d,{fn3d},false,'none');
assert(isequal(readTiff(fn3d), vol3d(:,:,1)));

fprintf("   ✅ basic 2-D / 3-D paths ok\n");

%% ---------- B. full matrix: {layout × dtype × compression} + timing & sizes ----------
cfg.order = {'YXZ',false; 'XYZ',true};
cfg.dtype = {'uint8',@uint8; 'uint16',@uint16};
cfg.comp  = {'none','lzw','deflate','zstd'};
sz        = [2048 1024 4];

nLayouts   = size(cfg.order,1);
nTypes     = size(cfg.dtype,1);
nComps     = numel(cfg.comp);
times      = zeros(nLayouts,nTypes,nComps);
logicalMiB = zeros(nLayouts,nTypes,nComps);
physicalMiB= zeros(nLayouts,nTypes,nComps);

for o = 1:nLayouts
    fprintf("\n   🏁 Testing layout: %s\n", cfg.order{o,1});
    for d = 1:nTypes
        for c = 1:nComps
            comp = cfg.comp{c};
            V = generateTestData(sz, cfg.dtype{d,1});
            if cfg.order{o,2}
                V = permute(V,[2 1 3]);
            end
            tagSafe = regexprep(sprintf('%s_%s_%s',cfg.order{o,1},...
                          cfg.dtype{d,1},comp), '[^A-Za-z0-9]','_');
            files = arrayfun(@(k) fullfile(tmpRoot, ...
                       sprintf('t_%s_%02d.tif',tagSafe,k)), ...
                       1:sz(3), 'Uni',false);

            t0 = tic;
            save_bl_tif(V, files, cfg.order{o,2}, comp);
            tElapsed = toc(t0);

            bytesLogical  = 0;
            bytesPhysical = 0;
            for k = 1:sz(3)
                ref = V(:,:,k);
                if cfg.order{o,2}, ref = ref.'; end
                data = readTiff(files{k});
                assert(isequal(data, ref), ...
                       'Mismatch %s slice %d via Tiff', tagSafe, k);
                info = dir(files{k});
                bytesLogical = bytesLogical + info.bytes;
                [~,bcount] = system(sprintf('stat -c "%%b" "%s"', files{k}));
                [~,bsize ] = system(sprintf('stat -c "%%B" "%s"', files{k}));
                bytesPhysical = bytesPhysical + ...
                    str2double(strtrim(bcount)) * str2double(strtrim(bsize));
            end

            times(o,d,c)        = tElapsed;
            logicalMiB(o,d,c)   = bytesLogical  / 2^20;
            physicalMiB(o,d,c)  = bytesPhysical / 2^20;

            fprintf("      ✅ %-30s in %.2f s, logical %.1f MiB, physical %.1f MiB\n", ...
                strrep(sprintf('%s_%s_%s',cfg.order{o,1},...
                cfg.dtype{d,1},comp),'_',' | '), ...
                tElapsed, logicalMiB(o,d,c), physicalMiB(o,d,c));
        end
    end
end

%% ---------- C. guard-clause checks ----------
fprintf("\n   🛡  guard-clause checks\n");
try
    save_bl_tif(uint8(0), {'/no/way/out.tif'}, false,'lzw');
    error('invalid-path accepted');
catch, fprintf('      ✅ invalid path rejected\n'); end

roFile = fullfile(tmpRoot,'readonly.tif');
imwrite(uint8(1),roFile);  fileattrib(roFile,'-w');
cRO = onCleanup(@() restore_rw(roFile));
try
    save_bl_tif(uint8(0), {roFile}, false,'none');
    error('read-only overwrite accepted');
catch, fprintf('      ✅ read-only overwrite rejected\n'); end

%% ---------- D. comparison table ----------
fprintf("\n   📊 XYZ vs YXZ comparison (Time and Sizes):\n");
rows = {};
for d = 1:nTypes
    for c = 1:nComps
        t_YXZ   = times(1,d,c);
        t_XYZ   = times(2,d,c);
        sp      = t_YXZ / t_XYZ;
        lY      = logicalMiB(1,d,c);
        lX      = logicalMiB(2,d,c);
        pY      = physicalMiB(1,d,c);
        pX      = physicalMiB(2,d,c);
        rows(end+1,:) = {cfg.dtype{d,1}, cfg.comp{c}, t_YXZ, t_XYZ, sp, lY, lX, pY, pX};
    end
end

T = cell2table(rows, 'VariableNames', {'DataType','Compression', 'Time_YXZ_s', 'Time_XYZ_s', 'Speedup', 'Logical_YXZ_MiB','Logical_XYZ_MiB','Physical_YXZ_MiB','Physical_XYZ_MiB'});
disp(T);

fprintf("\n🎉  all save_bl_tif tests passed\n");
end

function data = readTiff(fname)
    % Primary: MATLAB Tiff class
    try
        t = Tiff(fname,'r');
        data = read(t);
        return;
    catch
        % Fail-safe: use locally built tiffcp with ZSTD support
    end
    tempName = [fname, '.dc.tif'];
    tiffcp_bin = fullfile(pwd, 'tiff_build', 'libtiff', 'bin', 'tiffcp');
    cmd = sprintf('%s -c none "%s" "%s"', tiffcp_bin, fname, tempName);
    status = system(cmd);
    if status ~= 0
        error('Decompression failed for %s', fname);
    end
    data = imread(tempName);
    delete(tempName);
end

function out = generateTestData(sz, dtype)
    alpha = 2; beta = 50;
    X = gamrnd(alpha, beta, sz);
    mask = rand(sz) > 0.10;
    X(~mask) = 0;
    X = X / max(X(:));
    switch dtype
        case 'uint8',  out = uint8(X * 255);
        case 'uint16', out = uint16(X * 65535);
        otherwise,     error("Unsupported dtype '%s'",dtype);
    end
end

function sandbox_cleanup(dirPath)
    fclose('all'); safe_rmdir(dirPath);
end

function restore_rw(p)
    if exist(p,'file'), fileattrib(p,'+w'); end
end

function safe_rmdir(p)
    if exist(p,'dir')
        try rmdir(p,'s'); catch, pause(0.1); if exist(p,'dir'), rmdir(p,'s'); end; end
    end
end
