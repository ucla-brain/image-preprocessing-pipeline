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

%% ---------- B. full matrix: {layout × dtype × compression} + timing ----------
cfg.order = {'YXZ',false; 'XYZ',true};
cfg.dtype = {'uint8',@uint8; 'uint16',@uint16};
cfg.comp  = {'none','lzw','deflate'};
sz        = [2048 1024 4];  % ≥ 2 MiB slice

layoutTimes = zeros(size(cfg.order,1),1);
for o = 1:size(cfg.order,1)
    fprintf("\n   🏁 Testing layout: %s\n", cfg.order{o,1});
    tStart = tic;
    for d = 1:size(cfg.dtype,1)
        for c = 1:numel(cfg.comp)
            % generate random volume
            A = cfg.dtype{d,2}(randi(intmax(cfg.dtype{d,1}), sz));
            if cfg.order{o,2}
                A = permute(A,[2 1 3]);
            end

            tag     = sprintf('%s_%s_%s',cfg.order{o,1},...
                              cfg.dtype{d,1},cfg.comp{c});
            tagSafe = regexprep(tag,'[^A-Za-z0-9]','_');
            files   = arrayfun(@(k) fullfile(tmpRoot, ...
                       sprintf('t_%s_%02d.tif',tagSafe,k)), 1:sz(3),'uni',0);

            % write & verify
            save_bl_tif(A, files, cfg.order{o,2}, cfg.comp{c});
            for k = 1:sz(3)
                ref = A(:,:,k);
                if cfg.order{o,2}, ref = ref.'; end
                assert(isequal(imread(files{k}), ref), ...
                       "%s slice %d mismatch", tag, k);
            end
            fprintf("      ✅ %-30s\n", strrep(tag,'_',' | '));
        end
    end
    layoutTimes(o) = toc(tStart);
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

%% ---------- D. benchmark: YXZ vs XYZ save_bl_tif (uint16 none) ----------
benchSize = [512 512 64];  % 256 MiB uint16 stack
benchVol  = uint16(randi(65535, benchSize));
benchLayouts = {'YXZ',false; 'XYZ',true};
benchResults = struct('Layout',{},'Time_s',{},'MiB_s',{});

for b = 1:size(benchLayouts,1)
    layoutName = benchLayouts{b,1};
    isXYZ = benchLayouts{b,2};

    fprintf("\n   🏁 benchmark %s save_bl_tif (uint16 %dx%dx%d)…\n", ...
            layoutName, benchSize(1), benchSize(2), benchSize(3));

    % prepare volume and file list
    if isXYZ
        V = permute(benchVol,[2 1 3]);
    else
        V = benchVol;
    end
    files = arrayfun(@(k) fullfile(tmpRoot, sprintf('%s_mex_%03d.tif',lower(layoutName),k)), ...
                     1:benchSize(3),'uni',0);

    % run benchmark
    tStart = tic;
    save_bl_tif(V, files, isXYZ, 'none');
    tElapsed = toc(tStart);

    % record
    benchResults(b).Layout = layoutName;
    benchResults(b).Time_s = tElapsed;
    benchResults(b).MiB_s  = prod(benchSize)*2/2^20 / tElapsed;

    fprintf("      Time: %.2f s  (%.1f MiB/s)\n", ...
            tElapsed, benchResults(b).MiB_s);
end

%% ---------- E. layout performance table ----------
fprintf("\n   📊 Layout performance comparison:\n");
T_layout = table({benchResults.Layout}.' , [benchResults.Time_s].', [benchResults.MiB_s].', ...
    'VariableNames', {'Layout','Time_s','MiB_s'});
disp(T_layout);

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
        try
            rmdir(p,'s');
        catch
            pause(0.1);
            if exist(p,'dir'), rmdir(p,'s'); end
        end
    end
end
