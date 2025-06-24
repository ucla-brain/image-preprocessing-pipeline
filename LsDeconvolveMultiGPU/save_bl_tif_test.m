function save_bl_tif_test
% Extended regression + benchmark for save_bl_tif MEX
%
% • Uses gamma-distributed, sparse test volumes.
% • Records both logical and physical (btrfs/zstd) file sizes.

fprintf("🧪  save_bl_tif extended test-suite\n");

%% ---------- sandbox (deleted on exit) ----------
tmpRoot = tempname;   mkdir(tmpRoot);
cSandbox = onCleanup(@() sandbox_cleanup(tmpRoot));

%% ---------- A. basic 2-D / 3-D-singleton ----------
rng(42);
vol2d = uint8(randi(255,[256 256]));
fn2d  = fullfile(tmpRoot,'basic_2d.tif');
save_bl_tif(vol2d,{fn2d},false,'none');
assert(isequal(imread(fn2d),vol2d));

vol3d = reshape(vol2d,256,256,1);
fn3d  = fullfile(tmpRoot,'basic_3d.tif');
save_bl_tif(vol3d,{fn3d},false,'none');
assert(isequal(imread(fn3d),vol3d(:,:,1)));

fprintf("   ✅ basic 2-D / 3-D paths ok\n");

%% ---------- B. full matrix: {layout × dtype × compression} + timing & sizes ----------
cfg.order = {'YXZ',false; 'XYZ',true};
cfg.dtype = {'uint8',@uint8; 'uint16',@uint16};
cfg.comp  = {'none','lzw','deflate'};
sz        = [2048 2048 2048];  % ≥ 2 MiB slice

nLayouts = size(cfg.order,1);
nTypes   = size(cfg.dtype,1);
nComps   = numel(cfg.comp);

times            = zeros(nLayouts,nTypes,nComps);
logicalMiB       = zeros(nLayouts,nTypes,nComps);
physicalMiB      = zeros(nLayouts,nTypes,nComps);

for o = 1:nLayouts
    fprintf("\n   🏁 Testing layout: %s\n", cfg.order{o,1});
    for d = 1:nTypes
        for c = 1:nComps
            % 1) generate gamma data
            V = generateTestData(sz, cfg.dtype{d,1});
            if cfg.order{o,2}
                V = permute(V,[2 1 3]);
            end

            % 2) build file list
            tagSafe = regexprep(sprintf('%s_%s_%s',cfg.order{o,1},...
                          cfg.dtype{d,1},cfg.comp{c}), '[^A-Za-z0-9]','_');
            files = arrayfun(@(k) fullfile(tmpRoot, ...
                       sprintf('t_%s_%02d.tif',tagSafe,k)), ...
                       1:sz(3), 'Uni',false);

            % 3) write & time
            t0 = tic;
            save_bl_tif(V, files, cfg.order{o,2}, cfg.comp{c});
            tElapsed = toc(t0);

            % 4) verify & gather sizes
            bytesLogical  = 0;
            bytesPhysical = 0;
            for k = 1:sz(3)
                % verify
                ref = V(:,:,k);
                if cfg.order{o,2}, ref = ref.'; end
                assert(isequal(imread(files{k}), ref), ...
                       "Mismatch %s slice %d", tagSafe, k);

                % logical size (pre-FS compression)
                info = dir(files{k});
                bytesLogical = bytesLogical + info.bytes;

                % physical on-disk size (post-btrfs/zstd)
                cmdB = sprintf('stat -c "%%b" "%s"', files{k});
                [~,blkStr] = system(cmdB);
                blkCount = str2double(strtrim(blkStr));
                cmdS = sprintf('stat -c "%%B" "%s"', files{k});
                [~,bsStr] = system(cmdS);
                blkSize = str2double(strtrim(bsStr));
                bytesPhysical = bytesPhysical + blkCount * blkSize;
            end

            % 5) record
            times(o,d,c)       = tElapsed;
            logicalMiB(o,d,c)  = bytesLogical  / 2^20;
            physicalMiB(o,d,c) = bytesPhysical / 2^20;

            fprintf("      ✅ %-30s in %.2f s, logical %.1f MiB, physical %.1f MiB\n", ...
                strrep(sprintf('%s_%s_%s',cfg.order{o,1},...
                cfg.dtype{d,1},cfg.comp{c}),'_',' | '), ...
                tElapsed, logicalMiB(o,d,c), physicalMiB(o,d,c));
        end
    end
end

%% ---------- C. guard-clause checks ----------
fprintf("\n   🛡  guard-clause checks\n");
try
    save_bl_tif(uint8(0), {'/no/way/out.tif'}, false,'lzw');
    error("invalid-path accepted");
catch, fprintf("      ✅ invalid path rejected\n"); end

roFile = fullfile(tmpRoot,'readonly.tif');
imwrite(uint8(1),roFile);  fileattrib(roFile,'-w');
cRO = onCleanup(@() restore_rw(roFile));
try
    save_bl_tif(uint8(0), {roFile}, false,'none');
    error("read-only overwrite accepted");
catch, fprintf("      ✅ read-only overwrite rejected\n"); end

%% ---------- D. comparison table ----------
fprintf("\n   📊 XYZ vs YXZ comparison (Time and Sizes):\n");
rows = {};
for d = 1:nTypes
    for c = 1:nComps
        % gather metrics
        t_YXZ   = times(1,d,c);
        t_XYZ   = times(2,d,c);
        sp      = t_YXZ / t_XYZ;
        log_Y   = logicalMiB(1,d,c);
        log_X   = logicalMiB(2,d,c);
        phy_Y   = physicalMiB(1,d,c);
        phy_X   = physicalMiB(2,d,c);

        rows(end+1,:) = {
          cfg.dtype{d,1}, cfg.comp{c}, ...
          t_YXZ, t_XYZ, sp, ...
          log_Y, log_X, ...
          phy_Y, phy_X ...
        };
    end
end

T = cell2table(rows, 'VariableNames', {
    'DataType','Compression',   ...
    'Time_YXZ_s','Time_XYZ_s','Speedup',  ...
    'Logical_YXZ_MiB','Logical_XYZ_MiB',  ...
    'Physical_YXZ_MiB','Physical_XYZ_MiB' ...
});
disp(T);

fprintf("\n🎉  all save_bl_tif tests passed\n");
end

%% ---------- helpers --------------------------------------------------------
function out = generateTestData(sz, dtype)
    % Gamma distribution + ~10% zeros
    alpha = 2; beta = 50;
    X = gamrnd(alpha, beta, sz);
    mask = rand(sz) > 0.10;
    X(~mask) = 0;
    X = X / max(X(:));
    switch dtype
        case 'uint8';  out = uint8(X * 255);
        case 'uint16'; out = uint16(X * 65535);
        otherwise; error("Unsupported dtype '%s'",dtype);
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
