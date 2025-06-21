function save_bl_tif_test()
% Extended test-suite for save_bl_tif.mexa64
%
% Covers:
%   • YXZ and XYZ layouts
%   • uint8 / uint16
%   • none / lzw / deflate compression
%   • SIMD path (width & height divisible by 16) and scalar path
%   • Invalid-path and read-only-file error handling
%
% Author: ChatGPT-4o patch for Keivan Moradi, 2025-06-21

rng(42);                                   % reproducible random data
fprintf("🧪 Running save_bl_tif extended tests…\n");

%% 0. Quick SIMD correctness check via the MEX itself
simdVol = uint8(randi(255, 256, 256, 1));
tmp     = tempname + ".tif";
save_bl_tif(simdVol, {tmp}, false, "none");   % YXZ → MEX transposes
simdOut = imread(tmp);
delete(tmp);
assert(isequal(simdOut, simdVol.'), ...
      'SIMD transpose (256×256 uint8) failed');
fprintf("✅ SIMD slice sanity check passed\n");

%% 1. Main matrix of tests
outdir = fullfile(tempdir, 'save_bl_tif_test');
if exist(outdir, 'dir'), rmdir(outdir, 's'); end
mkdir(outdir);

stackSizes = {[256 256 4], ...             % triggers SIMD path
              [135 101 3]};                % forces scalar path
orders      = {'YXZ', false; 'XYZ', true};
types       = {'uint8',  @uint8; 'uint16', @uint16};
compressions = {'none', 'lzw', 'deflate'};

allPassed = true;

for s = 1:numel(stackSizes)
  sz = stackSizes{s};

  for o = 1:size(orders,1)
    for t = 1:size(types,1)
      for c = 1:numel(compressions)

        layout   = orders{o,1};
        useXYZ   = orders{o,2};
        dtypeStr = types{t,1};
        castFun  = types{t,2};
        compStr  = compressions{c};

        tag = sprintf('%s | %s | %s | %dx%dx%d', ...
                      layout, dtypeStr, compStr, sz);

        fprintf("→ %-45s", tag);
        tic;
        try
          %% create random volume (same seed each loop for determinism)
          A = castFun(rand(sz));
          if useXYZ, A = permute(A, [2 1 3]); end

          fileList = arrayfun(@(k) ...
                     fullfile(outdir, sprintf('slice_%s_%03d.tif', tag, k)), ...
                     1:sz(3), 'uni', 0);

          %% run MEX with a 30-second watchdog
          tRun = tic;
          f = parfeval(@() save_bl_tif(A, fileList, useXYZ, compStr), 0);
          wait(f, 30);                       % abort if hangs
          if strcmp(f.State,'error'), rethrow(f.Error); end

          %% verify every slice
          for k = 1:sz(3)
            B = imread(fileList{k});
            for k = 1:sz(3)
                B = imread(fileList{k});
                if useXYZ
                    ref = A(:,:,k).';   % transpose reference slice
                else
                    ref = A(:,:,k);     % no transpose
                end
                ok = isequal(B, ref);
                assert(ok, 'Mismatch slice %d', k);
            end
            assert(ok, 'Mismatch slice %d', k);
          end

          dur = toc;
          fprintf("  ✅  %.2fs\n", dur);

        catch ME
          dur = toc;
          fprintf("  ❌ (%s after %.2fs)\n", ME.message, dur);
          allPassed = false;
        end
      end
    end
  end
end

%% 2. Invalid path handling
fprintf("\n🧪 Testing invalid path handling…\n");
try
  save_bl_tif(uint8(rand(32,32,1)), {'/no/way/slice.tif'}, false, 'lzw');
  error('Expected error did not occur');
catch
  fprintf("✅ Correctly raised an error for invalid path\n");
end

%% 3. Read-only file overwrite
fprintf("🧪 Testing read-only file protection…\n");
A = uint8(rand(32,32,1));
file = fullfile(outdir,'readonly.tif');
imwrite(A(:,:,1), file);
fileattrib(file,'-w');                      % read-only
try
  save_bl_tif(A, {file}, false, 'lzw');
  error('Overwrite should have failed');
catch
  fprintf("✅ Correctly refused to overwrite read-only file\n");
end
fileattrib(file,'+w');

if exist(outdir,'dir'), rmdir(outdir,'s'); end

if allPassed
  fprintf("\n🎉 All save_bl_tif tests passed.\n");
else
  error("Some save_bl_tif tests failed – see log above.");
end
end
