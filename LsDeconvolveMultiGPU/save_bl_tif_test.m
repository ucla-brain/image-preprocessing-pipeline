function save_bl_tif_test()
% Extended regression + benchmark for save_bl_tif MEX (crash-safe)
fprintf("🧪  save_bl_tif extended test-suite\n");

%% sandbox -------------------------------------------------------------------
tmpRoot = tempname;  mkdir(tmpRoot);
cSandbox = onCleanup(@() safe_rmdir(tmpRoot));

%% A. basic sanity -----------------------------------------------------------
rng(42);
vol2d  = uint8(randi(255,[256 256]));
fn2d   = fullfile(tmpRoot,'basic_2d.tif');
save_bl_tif(vol2d,{fn2d},false,'none');
assert(isequal(imread(fn2d),vol2d));

vol3d1 = reshape(vol2d,256,256,1);
fn3d1  = fullfile(tmpRoot,'basic_3d.tif');
save_bl_tif(vol3d1,{fn3d1},false,'none');
assert(isequal(imread(fn3d1),vol3d1(:,:,1)));

fprintf("   ✅ basic 2-D / 3-D paths ok\n");

%% B. matrix of {layout × dtype × compression} ------------------------------
cfg.order = {'YXZ',false; 'XYZ',true};
cfg.dtype = {'uint8',@uint8; 'uint16',@uint16};
cfg.comp  = {'none','lzw','deflate'};
sz        = [2048 1024 4];   % ≥2 MiB slice → huge-page path

for o = 1:2
  for d = 1:2
    for c = 1:3
        tag = sprintf('%s | %s | %s',cfg.order{o,1},cfg.dtype{d,1},cfg.comp{c});
        A   = cfg.dtype{d,2}(randi(intmax(cfg.dtype{d,1}),sz));
        if cfg.order{o,2}, A = permute(A,[2 1 3]); end

        files = cellfun(@(k) fullfile(tmpRoot,sprintf('t_%s_%02d.tif',tag,k)), ...
                        num2cell(1:sz(3)),'uni',0);

        save_bl_tif(A,files,cfg.order{o,2},cfg.comp{c});
        for k=1:sz(3)
            ref=A(:,:,k); if cfg.order{o,2}, ref = ref.'; end
            assert(isequal(imread(files{k}),ref),"%s slice %d mismatch",tag,k);
        end
        fprintf("   ✅ %-28s\n",tag);
    end
  end
end

%% C. guard-clause checks ----------------------------------------------------
fprintf("   🛡  guard-clauses\n");
try
    save_bl_tif(uint8(0),{'/no/way/out.tif'},false,'lzw');
    error("invalid path accepted");
catch, fprintf("      ✅ invalid path rejected\n"); end

ro = fullfile(tmpRoot,'ro.tif');
imwrite(uint8(1),ro);                            % create the file
fileattrib(ro,'-w');                             % make it read-only
cRO = onCleanup(@() ( exist(ro,'file') && fileattrib(ro,'+w') ));

try
    save_bl_tif(uint8(0),{ro},false,'none');
    error("read-only overwrite accepted");
catch, fprintf("      ✅ read-only overwrite rejected\n"); end

%% D. micro-benchmark (tic/toc) ---------------------------------------------
bench = uint16(randi(65535,[512 512 64]));    % 64 MiB
mexF  = arrayfun(@(k) fullfile(tmpRoot,sprintf('m_%03d.tif',k)),1:64,'uni',0);
matF  = strrep(mexF,'m_','p_');

fprintf("   🏁 benchmark   (uint16 512×512×64)\n");
tic, save_bl_tif(bench,mexF,false,'none'); tMex = toc;
clear mex;                         % free scratch buffers before MATLAB loop
tic, mat_write_loop(bench,matF);   tLoop = toc;

bytes = numel(bench)*2/1e6;
fprintf("      save_bl_tif : %.2f s  (%.1f MB/s)\n",tMex ,bytes/tMex );
fprintf("      MATLAB loop : %.2f s  (%.1f MB/s)\n",tLoop, bytes/tLoop);

fprintf("🎉  all tests passed\n");
end

%% helper functions ----------------------------------------------------------
function mat_write_loop(V,paths)
for k=1:size(V,3), imwrite(V(:,:,k),paths{k}); end
end
function safe_rmdir(p)
if exist(p,'dir'), try rmdir(p,'s'); catch, end, end
end
