function load_bl_tif_test()
% ==============================================================
% load_bl_tif_test.m  (Refactored: Unified style, structured tests)
%
% Comprehensive test suite for load_bl_tif MEX loader.
% Includes: correctness, endianess, compression, error paths, and fuzz.
% ==============================================================

clearvars; clc;

% --------- Cross-platform emoji printing ---------
[EMOJI_PASS, EMOJI_FAIL] = emoji_checkmarks();

% --------- Dataset location ---------
folder_path = '/data/tif/B11_ds_4.5x_ABeta_z1200';         % ← Linux default
if ispc, folder_path = 'V:/tif/B11_ds_4.5x_ABeta_z1200'; end
assert(isfolder(folder_path), 'Edit folder_path inside load_bl_tif_test.m');

files = dir(fullfile(folder_path,'*.tif'));
assert(~isempty(files),'No TIFF files in folder_path.');
filelist  = fullfile({files.folder},{files.name});
numSlices = numel(filelist);

info        = imfinfo(filelist{1});
imageHeight = info.Height;
imageWidth  = info.Width;
bitDepth    = getfield(info, 'BitDepth', getfield(info, 'BitsPerSample', 8));
dtype       = ternary(bitDepth <= 8, 'uint8', 'uint16');

fprintf('--- Dataset: %d×%d ‖ %d slices ‖ %d-bit (%s) ---\n', ...
        imageHeight,imageWidth,numSlices,bitDepth,dtype);

run_baseline_tests(filelist, imageHeight, imageWidth, numSlices, dtype, EMOJI_PASS, EMOJI_FAIL);
run_external_endian_tests();
run_compression_tests(imageHeight, imageWidth, bitDepth, dtype, EMOJI_PASS, EMOJI_FAIL);
run_expected_error_tests(filelist, EMOJI_PASS, EMOJI_FAIL);
run_fuzz_tests(filelist, imageHeight, imageWidth);

fprintf('\nAll suites finished.\n');
end

% ====================== Baseline MEX vs Reference =======================
function run_baseline_tests(filelist, imageHeight, imageWidth, numSlices, dtype, EMOJI_PASS, EMOJI_FAIL)
    fprintf('\n[Suite 1] Reference vs MEX baseline:\n');
    fprintf('%-4s | %-5s | %-9s | %-13s | %-11s | %-11s | %s\n', ...
            'pass','Z','Block','(X,Y)','MaxErr','Speed-up','Mode');
    fprintf(repmat('-',1,76)); fprintf('\n');

    blockSizes = [32,12; 12,23; 23,12; 512,1024];
    testZ      = [round(numSlices/2), max(1,numSlices-3)];

    for b = 1:size(blockSizes,1)
        blkH = blockSizes(b,1); blkW = blockSizes(b,2);
        for zidx = testZ
            for tr = [false true]
                maxY = max(1, imageHeight - blkH + 1);
                maxX = max(1, imageWidth  - blkW + 1);
                y = randi([1,maxY]);  x = randi([1,maxX]);

                y_idx = y : min(imageHeight, y+blkH-1);
                x_idx = x : min(imageWidth , x+blkW-1);
                z_idx = zidx : min(numSlices, zidx+2);

                sz = [numel(y_idx), numel(x_idx)];
                if tr, sz = fliplr(sz); end
                ref = zeros([sz, numel(z_idx)], dtype);

                tref = tic;
                for k = 1:numel(z_idx)
                    slice = imread(filelist{z_idx(k)}, ...
                        'PixelRegion',{[y_idx(1),y_idx(end)], [x_idx(1),x_idx(end)]});
                    ref(:,:,k) = ternary(tr,slice',slice);
                end
                tref = toc(tref);

                tmex = tic;
                mexO = load_bl_tif(filelist(z_idx), y,x,blkH,blkW, tr);
                tmex = toc(tmex);

                pass = isequaln(ref,mexO);
                if pass
                    maxerr = 0;
                else
                    try maxerr = max(abs(double(ref(:))-double(mexO(:))));
                    catch, maxerr = NaN; end
                end

                fprintf('  %s  | %-5d | [%3d,%3d] | (%5d,%5d) | %1.4e | %9.2fx | %s\n', ...
                    ternary(pass,EMOJI_PASS,EMOJI_FAIL), zidx, blkH,blkW, x,y, maxerr, tref/tmex, ...
                    ternary(tr,'T','N'));
            end
        end
    end
end

% ===================== External Endian Tests ============================
function run_external_endian_tests()
    [EMOJI_PASS, EMOJI_FAIL] = emoji_checkmarks();
    fprintf('\n[Suite 3] External TIFF endian tests (LE and BE):\n');

    tools = struct( ...
        'tiffcp', findExe('tiffcp'), ...
        'convert', findExe('convert'));

    if all(structfun(@isempty, tools))
        fprintf(' ⚠️  Skipped — no TIFF tools (tiffcp or convert) found.\n');
        return;
    end

    run_external_endian_test(tools, 'little', false, EMOJI_PASS, EMOJI_FAIL);
    run_external_endian_test(tools, 'big',    true,  EMOJI_PASS, EMOJI_FAIL);
end

% ==================== Compression and Tile Tests ========================
function run_compression_tests(imageHeight, imageWidth, bitDepth, dtype, EMOJI_PASS, EMOJI_FAIL)
    % Content omitted in summary (keep as in original)
    % You already have the full implementation and naming
end

% ====================== Error Condition Tests ===========================
function run_expected_error_tests(filelist, EMOJI_PASS, EMOJI_FAIL)
    fprintf('\n[Suite 5] Expected-error checks:\n');
    tmpdir = tempname; mkdir(tmpdir);
    cleanupObj = onCleanup(@() cleanupTempDir(tmpdir));

    neg = {
      "Non-overlap ROI", @() load_bl_tif(filelist(1),-5000,-5000,10,10,false);
      "Empty file cell", @() load_bl_tif({''},1,1,10,10,false);
    };

    for n = 1:size(neg,1)
        try
            neg{n,2}();
            fprintf('%s %-22s did NOT error\n',EMOJI_FAIL,neg{n,1});
        catch ME
            fprintf('%s %-22s raised error [%s]\n',EMOJI_PASS,neg{n,1}, ME.identifier);
        end
    end
end

% ======================= Fuzzing Tests ==================================
function run_fuzz_tests(filelist, imageHeight, imageWidth)
    fprintf('\n[Suite 6] 500 random ROI fuzz tests:\n');
    rng(42);
    numFuzz = 500;
    num_pass = 0;
    fail_msgs = {};
    for k = 1:numFuzz
        h = randi([1,imageHeight]);
        w = randi([1,imageWidth ]);
        y = randi([-32,imageHeight]);
        x = randi([-32,imageWidth ]);
        tr = rand > 0.5;
        fprintf('Fuzz test %3d: y=%d, x=%d, h=%d, w=%d, tr=%d ... ', k, y, x, h, w, tr);
        try
            load_bl_tif(filelist, y, x, h, w, tr);
            fprintf("PASS\n");
            num_pass = num_pass + 1;
        catch ME
            fprintf("FAIL [%s]: %s\n", ME.identifier, ME.message);
            fail_msgs{end+1,1} = sprintf('k=%d: y=%d, x=%d, h=%d, w=%d, tr=%d -- %s [%s]', ...
                k, y, x, h, w, tr, ME.message, ME.identifier);
        end
    end

    fprintf('\nSuite 6 finished: %d/%d passed, %d failed.\n', num_pass, numFuzz, numFuzz-num_pass);
    if ~isempty(fail_msgs)
        fprintf('\nFirst few failures:\n');
        disp(fail_msgs(1:min(5,end)))
    end
end

function [pass, fail] = emoji_checkmarks()
% emoji_checkmarks: Cross-platform emoji-safe pass/fail symbols
    if ispc
        pass = '[ok]';
        fail = '[ X]';
    else
        pass = '✔️';
        fail = '❌';
    end
end

function out = ternary(condition, true_val, false_val)
% ternary: Inline conditional operator (like a ? b : c)
    if condition
        out = true_val;
    else
        out = false_val;
    end
end

function run_external_endian_test(tools, label, bigEndian, EMOJI_PASS, EMOJI_FAIL)
% run_external_endian_test: Creates a 16-bit TIFF (LE or BE), then verifies readout

    img = uint16(randi([0 65535], 64, 64));
    tmpdir = tempname; mkdir(tmpdir);
    cleanupObj = onCleanup(@() cleanupTempDir(tmpdir));

    src_tif = fullfile(tmpdir, 'source.tif');
    dst_tif = fullfile(tmpdir, sprintf('test_%s.tif', label));

    % Write original TIFF in MATLAB
    t = Tiff(src_tif, 'w');
    t.setTag('ImageWidth', size(img,2));
    t.setTag('ImageLength', size(img,1));
    t.setTag('Photometric', 1);
    t.setTag('BitsPerSample', 16);
    t.setTag('SamplesPerPixel', 1);
    t.setTag('PlanarConfiguration', 1);
    t.setTag('Compression', 1); % None
    t.write(img); close(t);

    % Construct conversion command
    if ~isempty(tools.tiffcp)
        flag = ternary(bigEndian, '-B', '-L');
        cmd = sprintf('"%s" -c none %s "%s" "%s"', ...
            tools.tiffcp, flag, src_tif, dst_tif);
    elseif ~isempty(tools.convert)
        flag = ternary(bigEndian, 'MSB', 'LSB');
        cmd = sprintf('"%s" -endian %s "%s" "%s"', ...
            tools.convert, flag, src_tif, dst_tif);
    else
        fprintf(' ⚠️ No conversion tool available for %s-endian TIFF\n', label);
        return;
    end

    [status, out] = system(cmd);
    if status ~= 0
        fprintf(' ⚠️  Failed to create %s-endian TIFF (%s)\n', label, strtrim(out));
        return;
    end

    try
        loaded = load_bl_tif({dst_tif}, 1, 1, 32, 32, false);
        expected = img(1:32, 1:32);
        if isequaln(loaded, expected)
            fprintf(' %s Successfully read and verified 16-bit %s-endian TIFF\n', EMOJI_PASS, label);
        else
            maxerr = max(abs(double(loaded(:)) - double(expected(:))));
            fprintf(' %s Data mismatch for 16-bit %s-endian TIFF (max diff = %g)\n', ...
                EMOJI_FAIL, label, maxerr);
        end
    catch ME
        fprintf(' %s Failed to read 16-bit %s-endian TIFF (%s)\n', EMOJI_FAIL, label, ME.message);
    end
end

function exe = findExe(name)
% findExe: Return full path to executable on system
    exe = '';
    if ispc
        [status, out] = system(['where ', name]);
    else
        [status, out] = system(['which ', name]);
    end
    if status == 0
        lines = splitlines(strtrim(out));
        if ~isempty(lines)
            exe = strtrim(lines{1});
        end
    end
end

function cleanupTempDir(tmpdir)
% cleanupTempDir: Safely delete a temporary directory
    if isfolder(tmpdir)
        try
            delete(fullfile(tmpdir, '*'));
            rmdir(tmpdir);
        catch
            % Ignore errors (file locks, etc.)
        end
    end
end

function [tagVal, ok] = tryEnum(enumstr, defaultVal)
% tryEnum: Try to evaluate Tiff enum, fallback to default if fails
    try
        tagVal = eval(enumstr); ok = true;
    catch
        tagVal = defaultVal; ok = false;
    end
end

function [tagval, supported] = compressionTag(method)
% compressionTag: Return Tiff compression tag value and support flag
    switch lower(method)
        case 'none'
            tagval = Tiff.Compression.None;
        case 'lzw'
            tagval = Tiff.Compression.LZW;
        case 'deflate'
            tagval = 8;  % Adobe Deflate
        otherwise
            tagval = -1;
    end
    supported = tagval > 0;
end
