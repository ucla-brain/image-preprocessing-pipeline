function save_bl_tif_test
% Extended regression + benchmark for save_bl_tif MEX

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

%% ---------- B. full matrix: {layout × dtype × compression} ----------
cfg.order = {'YXZ',false; 'XYZ',true};
cfg.dtype = {'uint8',@uint8; 'uint16',@uint16};
cfg.comp  = {'none','lzw','deflate'};
sz        = [2048 1024 4];                      % ≥ 2 MiB slice

for o = 1:size(cfg.order,1)
    if o>1, fprintf("\n"); end                  % blank line between YXZ / XYZ
    for d = 1:size(cfg.dtype,1)
        for c = 1:numel(cfg.comp)
            A = cfg.dtype{d,2}(randi(intmax(cfg.dtype{d,1}),sz));
            if cfg.order{o,2}, A = permute(A,[2 1 3]); end

            tag     = sprintf('%s_%s_%s',cfg.order{o,1},cfg.dtype{d,1},cfg.comp{c});
            tagSafe = regexprep(tag,'[^A-Za-z0-9]','_');
            files   = arrayfun(@(k) fullfile(tmpRoot, ...
                       sprintf('t_%s_%02d.tif',tagSafe,k)), ...
                       1:sz(3),'uni',0);

            save_bl_tif(A,files,cfg.order{o,2},cfg.comp{c});
            for k = 1:sz(3)
                ref = A(:,:,k); if cfg.order{o,2}, ref = ref.'; end
                assert(isequal(imread(files{k}),ref),"%s slice %d mismatch",tag,k);
            end
            fprintf("   ✅ %-30s\n", strrep(tag,'_',' | '));
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

%% ---------- D. benchmark: save_bl_tif vs async-imwrite ----------
benchSize = [512 512 64];                     % 256 MiB uint16 stack
benchVol  = uint16(randi(65535, benchSize));
mexFiles  = arrayfun(@(k) fullfile(tmpRoot,sprintf('mex_%03d.tif',k)), ...
                     1:benchSize(3),'uni',0);
asyncFiles = strrep(mexFiles,'mex_','async_');

% Benchmark save_bl_tif
fprintf("\n   🏁 benchmark save_bl_tif (uint16 %dx%dx%d)…\n", benchSize, benchSize(3));
tic
save_bl_tif(benchVol, mexFiles, false, 'none');
tMex = toc;

% Benchmark async imwrite using backgroundPool
if ~isempty(ver('parallel')) && exist('backgroundPool','builtin')
    fprintf("   🏁 benchmark async imwrite (%d tasks)…\n", benchSize(3));
    futures = parallel.FevalFuture.empty(0,benchSize(3));
    tic
    for k = 1:benchSize(3)
        futures(k) = parfeval(backgroundPool, @imwrite, 0, benchVol(:,:,k), asyncFiles{k});
    end
    wait(futures);
    tAsync = toc;

    bytesMiB = numel(benchVol)*2 / 2^20;
    spdMex   = bytesMiB / tMex;
    spdAsync = bytesMiB / tAsync;
    speedup  = tAsync / tMex;

    fprintf("      save_bl_tif : %.2f s  (%.1f MiB/s)\n", tMex, spdMex);
    fprintf("      async-imwrite: %.2f s  (%.1f MiB/s)\n", tAsync, spdAsync);
    fprintf("      speed-up     : %.2f× (async → save_bl_tif)\n", speedup);
else
    fprintf("   ⚠️  async-imwrite benchmark skipped: backgroundPool unavailable.\n");
end

fprintf("\n🎉  all save_bl_tif tests passed\n");
end

%% ---------- helpers --------------------------------------------------------
function sandbox_cleanup(dirPath)
fclose('all');
safe_rmdir(dirPath);
end

function restore_rw(p)
if exist(p,'file'), fileattrib(p,'+w'); end
end

function safe_rmdir(p)
if exist(p,'dir')
    try, rmdir(p,'s'); catch, pause(0.1); if exist(p,'dir'), rmdir(p,'s'); end; end
end
end
