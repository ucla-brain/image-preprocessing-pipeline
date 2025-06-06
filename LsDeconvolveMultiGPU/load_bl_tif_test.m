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

    blockSizes = [64, 64; 128, 96; 240, 240];
    testZ = [round(numSlices / 2), max(1, numSlices - 3)];
    totalTests = size(blockSizes, 1) * numel(testZ);

    results = zeros(totalTests, 8); % [pass, z, blkH, blkW, x, y, maxerr, speedup]
    testIdx = 1;

    fprintf('\n%-4s | %-6s | %-9s | %-13s | %-11s | %-12s\n', ...
        'pass', 'Z', 'BlockSize', '(X,Y)', 'Max Error', 'Speedup');
    fprintf(repmat('-', 1, 64)); fprintf('\n');

    for b = 1:size(blockSizes, 1)
        blkH = blockSizes(b, 1);
        blkW = blockSizes(b, 2);

        for zidx = testZ
            y = randi([1, imageHeight - blkH + 1]);
            x = randi([1, imageWidth  - blkW + 1]);
            y_indices = y : y + blkH - 1;
            x_indices = x : x + blkW - 1;
            z_indices = zidx : min(numSlices, zidx + 2);

            % MATLAB reference
            t1 = tic;
            bl_gt = zeros(blkW, blkH, numel(z_indices), 'uint16');  % MATLAB expects [W, H, Z]
            for k = 1:numel(z_indices)
                slice = imread(filelist{z_indices(k)}, ...
                    'PixelRegion', {[y_indices(1), y_indices(end)], [x_indices(1), x_indices(end)]});
                bl_gt(:, :, k) = slice';
            end
            t_ref = toc(t1);

            % MEX output
            t2 = tic;
            bl_mex = load_bl_tif(filelist(z_indices), y, x, blkH, blkW);
            t_mex = toc(t2);

            % Comparison
            pass = isequal(bl_mex, bl_gt);
            if pass
                maxerr = 0;
                linearIdx = NaN;
            else
                diff = abs(bl_mex - bl_gt);
                [maxerr, linearIdx] = max(diff(:));
                [xi, yi, zi] = ind2sub(size(diff), linearIdx);
                val_mex = bl_mex(xi, yi, zi);
                val_gt = bl_gt(xi, yi, zi);
                block_x = x + xi - 1;
                block_y = y + yi - 1;
                position = "middle";
                if xi == 1 || xi == blkW || yi == 1 || yi == blkH
                    position = "edge";
                end
                fprintf("     ↳ First mismatch at (x=%d, y=%d, z=%d) → block [%d, %d] (%s): MEX=%d, GT=%d\n", ...
                    block_x, block_y, zidx + zi - 1, xi, yi, position, val_mex, val_gt);

                % Diagnostics: print boolean mask for all 6 faces
                mask = (bl_mex == bl_gt);

                squeeze(mask(1, :, :))        % x = 1 plane
                squeeze(mask(end, :, :))      % x = end plane
                squeeze(mask(:, 1, :))        % y = 1 plane
                squeeze(mask(:, end, :))      % y = end plane
                squeeze(mask(:, :, 1))        % z = 1 plane
                squeeze(mask(:, :, end))      % z = end plane
            end
            symbol = char(pass * 10003 + ~pass * 10007);  % ✓ or ✗

            fprintf('  %s  | %-6d | [%3d,%3d]  | (%5d,%5d) | %1.4e | %8.2fx\n', ...
                symbol, zidx, blkH, blkW, x, y, maxerr, t_ref / t_mex);
            results(testIdx, :) = [pass, zidx, blkH, blkW, x, y, maxerr, t_ref / t_mex];
            testIdx = testIdx + 1;
        end
    end

    if all(results(:,1))
        fprintf('\n🎉 All tests passed successfully.\n');
    else
        fprintf('\n❗ Some tests failed. Please review the table above.\n');
    end
end
