function load_bl_tif_test()
    % ================================
    % load_bl_tif_test.m
    % Usage: matlab -batch load_bl_tif_test
    % ================================

    folder_path = '/data/tif/B11_ds_4.5x_ABeta_z1200';
    if ispc
        folder_path = 'V:/tif/B11_ds_4.5x_ABeta_z1200';
    end
    if isempty(folder_path) || ~isfolder(folder_path)
        error('TIF test folder not found. Check that folder_path is valid.');
    end

    files = dir(fullfile(folder_path, '*.tif'));
    assert(~isempty(files), 'No TIF files found in the folder.');

    filelist = fullfile({files.folder}, {files.name});
    numSlices = numel(filelist);

    info = imfinfo(filelist{1});
    imageHeight = info.Height;
    imageWidth = info.Width;
    bitDepth = info.BitDepth;

    if bitDepth ~= 16
        error('Expected 16-bit grayscale TIFF images.');
    end

    blockSizes = [32, 12; 12, 23; 23, 12; 512, 1024];
    testZ = [round(numSlices / 2), max(1, numSlices - 3)];
    totalTests = size(blockSizes, 1) * numel(testZ) * 2;

    results = zeros(totalTests, 9); % [pass, z, blkH, blkW, x, y, maxerr, speedup, transposeFlag]
    testIdx = 1;

    fprintf('\n%-4s | %-6s | %-9s | %-13s | %-11s | %-12s | %s\n', ...
        'pass', 'Z', 'BlockSize', '(X,Y)', 'Max Error', 'Speedup', 'Mode');
    fprintf(repmat('-', 1, 74)); fprintf('\n');

    for b = 1:size(blockSizes, 1)
        blkH = blockSizes(b, 1);
        blkW = blockSizes(b, 2);

        for zidx = testZ
            for transposeFlag = [false, true]
                y = randi([1, imageHeight - blkH + 1]);
                x = randi([1, imageWidth  - blkW + 1]);
                y_indices = y : min(imageHeight, y + blkH - 1);
                x_indices = x : min(imageWidth,  x + blkW - 1);
                z_indices = zidx : min(numSlices, zidx + 2);

                % MATLAB reference
                if transposeFlag
                    bl_gt = zeros(numel(x_indices), numel(y_indices), numel(z_indices), 'uint16');
                else
                    bl_gt = zeros(numel(y_indices), numel(x_indices), numel(z_indices), 'uint16');
                end

                t1 = tic;
                for k = 1:numel(z_indices)
                    slice = imread(filelist{z_indices(k)}, ...
                        'PixelRegion', {[y_indices(1), y_indices(end)], [x_indices(1), x_indices(end)]});
                    if transposeFlag
                        bl_gt(:, :, k) = slice';  % Transpose MATLAB reference
                    else
                        bl_gt(:, :, k) = slice;   % No transpose
                    end
                end
                t_ref = toc(t1);

                % MEX call
                t2 = tic;
                bl_mex = load_bl_tif(filelist(z_indices), y, x, blkH, blkW, transposeFlag);
                t_mex = toc(t2);

                % Compare minimum overlap
                minH = min(size(bl_gt,1), size(bl_mex,1));
                minW = min(size(bl_gt,2), size(bl_mex,2));
                minZ = min(size(bl_gt,3), size(bl_mex,3));
                bl_gt_c = bl_gt(1:minH,1:minW,1:minZ);
                bl_mex_c = bl_mex(1:minH,1:minW,1:minZ);

                pass = isequal(bl_mex_c, bl_gt_c);
                if pass
                    maxerr = 0;
                    linearIdx = NaN;
                else
                    fprintf('Requesting block at y=%d..%d, x=%d..%d, size=[%d,%d]\n', ...
                        y_indices(1), y_indices(end), x_indices(1), x_indices(end), blkH, blkW);
                    disp('MATLAB slice(1:5,1:5,1):');
                    disp(bl_gt_c(1:min(5,minH),1:min(5,minW),1));
                    disp('MEX slice(1:5,1:5,1):');
                    disp(bl_mex_c(1:min(5,minH),1:min(5,minW),1));

                    diff = abs(bl_mex_c - bl_gt_c);
                    [maxerr, linearIdx] = max(diff(:));
                    [xi, yi, zi] = ind2sub(size(diff), linearIdx);
                    val_mex = bl_mex_c(xi, yi, zi);
                    val_gt = bl_gt_c(xi, yi, zi);
                    block_x = x + yi - 1;
                    block_y = y + xi - 1;
                    position = "middle";
                    if xi == 1 || xi == minH || yi == 1 || yi == minW
                        position = "edge";
                    end
                    fprintf("     ↳ First mismatch at (x=%d, y=%d, z=%d) → block [%d, %d] (%s): MEX=%d, GT=%d\n", ...
                        block_x, block_y, zidx + zi - 1, yi, xi, position, val_mex, val_gt);
                    disp('First 10 values in bl_mex:');
                    disp(bl_mex_c(1:min(10,numel(bl_mex_c))));
                end

                symbol = char(pass * 10003 + ~pass * 10007);  % ✓ or ✗
                modeStr = ternary(transposeFlag, 'T', 'N');

                fprintf('  %s  | %-6d | [%3d,%3d]  | (%5d,%5d) | %1.4e | %8.2fx |   %s\n', ...
                    symbol, zidx, blkH, blkW, x, y, maxerr, t_ref / t_mex, modeStr);

                results(testIdx, :) = [pass, zidx, blkH, blkW, x, y, maxerr, t_ref / t_mex, transposeFlag];
                testIdx = testIdx + 1;
            end
        end
    end

    if all(results(:,1))
        fprintf('\n🎉 All tests passed successfully.\n');
    else
        fprintf('\n❗ Some tests failed. Please review the table above.\n');
    end
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
