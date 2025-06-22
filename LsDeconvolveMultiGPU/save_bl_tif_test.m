function save_bl_tif_test()
% Extended test-bench for save_bl_tif MEX
rng(42);
fprintf("🧪 save_bl_tif extended test-suite\n");

%% --------------------------------------------------------------------------
%%  housekeeping helpers
%% --------------------------------------------------------------------------
tmpRoot = tempname;   % one unique sandbox for all artefacts
mkdir(tmpRoot);
cleanup = onCleanup(@() safe_rmdir(tmpRoot));

%% --------------------------------------------------------------------------
%%  A. basic 2-D / 3-D singleton sanity
%% --------------------------------------------------------------------------
vol2d  = uint8(randi(255,[256 256]));
p2d    = fullfile(tmpRoot,'basic_2d.tif');
save_bl_tif(vol2d,{p2d},false,'none');
assert(isequal(imread(p2d),vol2d),"2-D write/read failed");

vol3d1 = reshape(vol2d,256,256,1);
p3d1   = fullfile(tmpRoot,'basic_3d.tif');
save_bl_tif(vol3d1,{p3d1},false,'none');
assert(isequal(imread(p3d1),vol3d1(:,:,1)),"3-D singleton failed");

fprintf("   ✅ basic 2-D / 3-D paths OK\n");

%% --------------------------------------------------------------------------
%%  B. full matrix of {layout × dtype × compression}
%% --------------------------------------------------------------------------
cfg.order   = {'YXZ',false;'XYZ',true};
cfg.dtype   = {'uint8',@uint8;'uint16',@uint16};
cfg.comp    = {'none','lzw','deflate'};

% size chosen so that a single slice of uint16 (2048×1024) ≈ 4 MiB → triggers
% huge-page scratch allocation branch
stackSize   = [2048 1024 4];

for o = 1:size(cfg.order,1)
  for d = 1:size(cfg.dtype,1)
    for c = 1:numel(cfg.comp)
        tag = sprintf('%s | %s | %s',...
                      cfg.order{o,1},cfg.dtype{d,1},cfg.comp{c});
        try
            % generate random stack
            A = cfg.dtype{d,2}(randi(intmax(cfg.dtype{d,1}),stackSize));
            if cfg.order{o,2}      % transpose to XYZ memory layout
                A = permute(A,[2 1 3]);
            end

            % build path list
            fList = cell(1,stackSize(3));
            for k = 1:stackSize(3)
                fList{k} = fullfile(tmpRoot,...
                              sprintf('test_%s_%03d.tif',tag,k));
            end

            % toggle threads arg every other run to hit both code paths
            if mod(c,2)
                save_bl_tif(A,fList,cfg.order{o,2},cfg.comp{c});
            else
                save_bl_tif(A,fList,cfg.order{o,2},cfg.comp{c},...
                            feature('numCores'));
            end

            % verify round-trip
            for k = 1:stackSize(3)
                B = imread(fList{k});
                ref = A(:,:,k);
                if cfg.order{o,2}, ref = ref.'; end
                assert(isequal(B,ref), ...
                       "%s slice %d mismatch", tag,k);
            end
            fprintf("   ✅ %-28s\n",tag);
        catch ME
            fprintf("   ❌ %-28s - %s\n",tag,ME.message);
            rethrow(ME);
        end
    end
  end
end

%% --------------------------------------------------------------------------
%%  C. guard clauses: invalid path & read-only overwrite
%% --------------------------------------------------------------------------
fprintf("   🛡  guard-clause checks\n");

try
    save_bl_tif(uint8(zeros(32,32,1)), {'/does/not/exist/foo.tif'},...
                false,'lzw');
    error("invalid-path accepted unexpectedly");
catch, fprintf("      ✅ invalid path rejected\n"); end

roFile = fullfile(tmpRoot,'readonly.tif');
imwrite(uint8(1),roFile);
fileattrib(roFile,'-w');   % make read-only
cleanupRO = onCleanup(@() fileattrib(roFile,'+w'));

try
    save_bl_tif(uint8(zeros(32,32,1)), {roFile}, false,'none');
    error("read-only overwrite accepted");
catch, fprintf("      ✅ read-only overwrite rejected\n"); end

%% --------------------------------------------------------------------------
%%  D. micro-benchmark: MATLAB loop vs save_bl_tif
%% --------------------------------------------------------------------------
benchSize = [512 512 64];           % 256 MiB uint16 stack
volBench  = uint16(randi(65535,benchSize));

% paths for MEX and MATLAB series
mexFiles = arrayfun(@(k) fullfile(tmpRoot,sprintf('mex_%03d.tif',k)),...
                    1:benchSize(3),'uni',0);
matFiles = strrep(mexFiles,'mex_','mat_');

fprintf("   🏁 benchmark (uint16 %dx%dx%d)…\n",benchSize);
tMex  = timeit(@() save_bl_tif(volBench,mexFiles,false,'none')) ;
tLoop = timeit(@() for k=1:benchSize(3)
                           imwrite(volBench(:,:,k),matFiles{k});
                       end) ;

fprintf("      save_bl_tif : %6.2f s  (%.2f MB/s)\n",...
        tMex,  prod(benchSize)*2/tMex/1e6);
fprintf("      MATLAB loop : %6.2f s  (%.2f MB/s)\n",...
        tLoop, prod(benchSize)*2/tLoop/1e6);

%% --------------------------------------------------------------------------
fprintf("🎉  All tests completed without error.\n");

end

%% --------------------------------------------------------------------------
function safe_rmdir(p)
% recursive delete that silences "in use" warnings on Windows
if exist(p,'dir')
    try
        rmdir(p,'s');
    catch
        pause(0.1);  % let OS close handles
        if exist(p,'dir'), rmdir(p,'s'); end
    end
end
end
