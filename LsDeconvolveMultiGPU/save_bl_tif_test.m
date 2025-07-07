function save_bl_tif_test
% Extended regression + benchmark for save_bl_tif MEX with TILED mode test

fprintf("🧪  save_bl_tif extended test-suite (with tiles)\n");

tmpRoot = tempname;
mkdir(tmpRoot);
cSandbox = onCleanup(@() sandbox_cleanup(tmpRoot));

%% ---------- A. basic 2-D / 3-D-singleton ----------
rng(42);
vol2d = uint8(randi(255,[256 256]));
fn2d  = fullfile(tmpRoot,'basic_2d.tif');
save_bl_tif(vol2d,{fn2d},false,'none',[],false); % strips
assert(isequal(readTiff(fn2d), vol2d));

vol3d = reshape(vol2d,256,256,1);
fn3d  = fullfile(tmpRoot,'basic_3d.tif');
save_bl_tif(vol3d,{fn3d},false,'none',[],false); % strips
assert(isequal(readTiff(fn3d), vol3d(:,:,1)));
fprintf("   ✅ basic 2-D / 3-D paths ok\n");

%% ---------- B. full matrix: {layout × dtype × compression} + timing & sizes ----------
cfg.order = {'YXZ',false; 'XYZ',true};
cfg.dtype = {'uint8',@uint8; 'uint16',@uint16};
cfg.comp  = {'none','lzw','deflate'};
sz        = [2048 1024 4];

tileModes = [false true];
nLayouts   = size(cfg.order,1);
nTypes     = size(cfg.dtype,1);
nComps     = numel(cfg.comp);
nTiles     = numel(tileModes);

times        = zeros(nLayouts,nTypes,nComps,nTiles);
logicalMiB   = zeros(nLayouts,nTypes,nComps,nTiles);
physicalMiB  = zeros(nLayouts,nTypes,nComps,nTiles);

for t = 1:nTiles
    tilemode = tileModes(t);
    tileMsg = {'STRIP','TILE'}{t};
    for o = 1:nLayouts
        fprintf("\n   🏁 Testing layout: %s (%s)\n", cfg.order{o,1}, tileMsg);
        for d = 1:nTypes
            for c = 1:nComps
                comp = cfg.comp{c};
                V = generateTestData(sz, cfg.dtype{d,1});
                if cfg.order{o,2}
                    V = permute(V,[2 1 3]);
                end
                tagSafe = regexprep(sprintf('%s_%s_%s_%s',cfg.order{o,1},...
                              cfg.dtype{d,1},comp,tileMsg), '[^A-Za-z0-9]','_');
                files = arrayfun(@(k) fullfile(tmpRoot, ...
                           sprintf('t_%s_%02d.tif',tagSafe,k)), ...
                           1:sz(3), 'Uni',false);

                t0 = tic;
                save_bl_tif(V, files, cfg.order{o,2}, comp, [], tilemode);
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

                times(o,d,c,t)        = tElapsed;
                logicalMiB(o,d,c,t)   = bytesLogical  / 2^20;
                physicalMiB(o,d,c,t)  = bytesPhysical / 2^20;

                fprintf("      ✅ %-40s in %.2f s, logical %.1f MiB, physical %.1f MiB\n", ...
                    strrep(sprintf('%s_%s_%s_%s',cfg.order{o,1},...
                    cfg.dtype{d,1},comp,tileMsg),'_',' | '), ...
                    tElapsed, logicalMiB(o,d,c,t), physicalMiB(o,d,c,t));
            end
        end
    end
end

%% Comparison table: STRIP vs TILE (Section B)
fprintf("\n   📊 STRIP vs TILE comparison (Time and Sizes, Section B):\n");
rows = {};
for o = 1:nLayouts
    for d = 1:nTypes
        for c = 1:nComps
            tStrip   = times(o,d,c,1);
            tTile    = times(o,d,c,2);
            sp       = tStrip / tTile;
            lStrip   = logicalMiB(o,d,c,1);
            lTile    = logicalMiB(o,d,c,2);
            pStrip   = physicalMiB(o,d,c,1);
            pTile    = physicalMiB(o,d,c,2);
            rows(end+1,:) = {cfg.order{o,1},cfg.dtype{d,1},cfg.comp{c}, tStrip, tTile, sp, lStrip, lTile, pStrip, pTile};
        end
    end
end
T = cell2table(rows, 'VariableNames', ...
    {'Layout','DataType','Compression', 'Time_STRIP_s', 'Time_TILE_s', 'Speedup', ...
     'Logical_STRIP_MiB','Logical_TILE_MiB','Physical_STRIP_MiB','Physical_TILE_MiB'});
disp(T);

%% ---------- C. Large block test: 100 big slices, compare strip vs tile ----------
szBig = [2048 2048 100];
Vbig = generateTestData(szBig, 'uint8');
files = arrayfun(@(k) fullfile(tmpRoot, sprintf('bigblock_%03d.tif',k)), 1:szBig(3), 'Uni', false);

fprintf('\n   🏁 Saving 100 large slices (STRIP mode)...\n');
t0 = tic;
save_bl_tif(Vbig, files, false, 'deflate', [], false); % STRIP
tStrip = toc(t0);
for k = 1:szBig(3)
    data = readTiff(files{k});
    assert(isequal(data, Vbig(:,:,k)), 'Big block mismatch at slice %d (strip mode)', k);
end
fprintf('      ✅ 100 large slices (STRIP mode) ok (%.2f s)\n', tStrip);

fprintf('\n   🏁 Saving 100 large slices (TILE mode)...\n');
t0 = tic;
save_bl_tif(Vbig, files, false, 'deflate', [], true); % TILE
tTile = toc(t0);
for k = 1:szBig(3)
    data = readTiff(files{k});
    assert(isequal(data, Vbig(:,:,k)), 'Big block mismatch at slice %d (tile mode)', k);
end
fprintf('      ✅ 100 large slices (TILE mode) ok (%.2f s)\n', tTile);

fprintf('\n   🚦  [Performance] Tiles vs Strips (100x %dx%d slices):\n', szBig(1), szBig(2));
fprintf('         STRIP: %.2f s\n', tStrip);
fprintf('         TILE : %.2f s\n', tTile);

if tTile < tStrip
    fprintf('      🟢 Tiles are FASTER (%.1fx speedup)\n', tStrip/tTile);
else
    fprintf('      🟡 Strips are FASTER (%.1fx speedup)\n', tTile/tStrip);
end

%% ---------- D. guard-clause checks ----------
fprintf("\n   🛡  guard-clause checks\n");
try
    save_bl_tif(uint8(0), {'/no/way/out.tif'}, false,'lzw',[],false);
    error('invalid-path accepted');
catch, fprintf('      ✅ invalid path rejected\n'); end

roFile = fullfile(tmpRoot,'readonly.tif');
imwrite(uint8(1),roFile);  fileattrib(roFile,'-w');
cRO = onCleanup(@() restore_rw(roFile));
try
    save_bl_tif(uint8(0), {roFile}, false,'none',[],false);
    error('read-only overwrite accepted');
catch, fprintf('      ✅ read-only overwrite rejected\n'); end

fprintf("\n🎉  all save_bl_tif tests passed (TILES + STRIPS)\n");

end

% ------- Helper functions as local functions below --------
function data = readTiff(fname)
    % Try MATLAB Tiff class (cross-platform)
    try
        t = Tiff(fname,'r');
        data = read(t);
        t.close();
        return;
    catch
        % fallback
        data = imread(fname);
    end
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
