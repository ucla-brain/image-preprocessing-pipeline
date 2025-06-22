function save_bl_tif_test()
% Extended regression + benchmark for save_bl_tif MEX.
rng(42);
fprintf("🧪  save_bl_tif extended test-suite\n");

%% ----------  global sandbox  ------------------------------------------------
tmpRoot = tempname;           % one private temp directory
mkdir(tmpRoot);
cleanup = onCleanup(@() safe_rmdir(tmpRoot));

%% ---------- A.  basic 2-D and 3-D-singleton checks -------------------------
vol2d  = uint8(randi(255,[256 256]));
p2d    = fullfile(tmpRoot,'basic_2d.tif');
save_bl_tif(vol2d,{p2d},false,'none');
assert(isequal(imread(p2d),vol2d),"2-D round-trip failed");

vol3d1 = reshape(vol2d,256,256,1);
p3d1   = fullfile(tmpRoot,'basic_3d.tif');
save_bl_tif(vol3d1,{p3d1},false,'none');
assert(isequal(imread(p3d1),vol3d1(:,:,1)),"3-D singleton failed");

fprintf("   ✅ basic 2-D / 3-D paths ok\n");

%% ---------- B.  full matrix: {layout × dtype × compression} ----------------
cfg.order   = {'YXZ',false; 'XYZ',true};
cfg.dtype   = {'uint8',@uint8; 'uint16',@uint16};
cfg.comp    = {'none','lzw','deflate'};
stackSize   = [2048 1024 4];          % ≥2 MiB slice ⇒ hugepage branch

for o = 1:size(cfg.order,1)
  for d = 1:size(cfg.dtype,1)
    for c = 1:numel(cfg.comp)
        tag = sprintf('%s | %s | %s', ...
                      cfg.order{o,1},cfg.dtype{d,1},cfg.comp{c});
        try
            A = cfg.dtype{d,2}(randi(intmax(cfg.dtype{d,1}),stackSize));
            if cfg.order{o,2}, A = permute(A,[2 1 3]); end

            files = cellfun(@(k) fullfile(tmpRoot, ...
                            sprintf('test_%s_%03d.tif',tag,k)), ...
                            num2cell(1:stackSize(3)), 'uni',0);

            if mod(c,2)  % alternate explicit / implicit thread arg
                 save_bl_tif(A,files,cfg.order{o,2},cfg.comp{c});
            else save_bl_tif(A,files,cfg.order{o,2},cfg.comp{c}, ...
                             feature('numCores'));
            end

            for k = 1:stackSize(3)
               ref = A(:,:,k); if cfg.order{o,2}, ref = ref.'; end
               assert(isequal(imread(files{k}), ref), ...
                     "%s slice %d mismatch", tag,k);
            end
            fprintf("   ✅ %-28s\n", tag);
        catch ME
            fprintf("   ❌ %-28s – %s\n", tag, ME.message);
            rethrow(ME);
        end
    end
  end
end

%% ---------- C.  guard-clause checks ---------------------------------------
fprintf("   🛡  guard-clause checks\n");
try
    save_bl_tif(uint8(0), {'/does/not/exist/foo.tif'}, false,'lzw');
    error("invalid path accepted");
catch, fprintf("      ✅ invalid path rejected\n"); end

roFile = fullfile(tmpRoot,'readonly.tif');
imwrite(uint8(1),roFile); fileattrib(roFile,'-w');
cleanupRO = onCleanup(@() fileattrib(roFile,'+w')); %#ok

try
    save_bl_tif(uint8(0), {roFile}, false,'none');
    error("read-only overwrite accepted");
catch, fprintf("      ✅ read-only overwrite rejected\n"); end

%% ---------- D.  micro-benchmark  ------------------------------------------
benchSize = [512 512 64];                % 256 MiB uint16 stack
benchVol  = uint16(randi(65535, benchSize));
mexFiles  = arrayfun(@(k) fullfile(tmpRoot,sprintf('mex_%03d.tif',k)), ...
                     1:benchSize(3), 'uni',0);
matFiles  = strrep(mexFiles,'mex_','mat_');

fprintf("   🏁 benchmark (uint16 %dx%dx%d)…\n", benchSize);
tMex  = timeit(@() save_bl_tif(benchVol,mexFiles,false,'none'));
tLoop = timeit(@() mat_write_loop(benchVol,matFiles));

bytes = prod(benchSize)*2;
fprintf("      save_bl_tif : %6.2f s  (%.2f MB/s)\n", ...
        tMex , bytes/tMex /1e6);
fprintf("      MATLAB loop : %6.2f s  (%.2f MB/s)\n", ...
        tLoop, bytes/tLoop/1e6);

%% ---------- done ----------------------------------------------------------
fprintf("🎉  all save_bl_tif tests passed\n");

end  % main

% ---------------------------------------------------------------------------
function mat_write_loop(V, paths)
for k = 1:size(V,3), imwrite(V(:,:,k), paths{k}); end
end

function safe_rmdir(p)
if exist(p,'dir')
    try,  rmdir(p,'s');  catch, pause(0.1); if exist(p,'dir'), rmdir(p,'s'); end, end
end
end
