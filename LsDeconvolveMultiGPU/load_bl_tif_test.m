function load_bl_tif_test()
    % ================================
    % load_bl_tif_test.m
    % Usage: matlab -batch load_bl_tif_test
    % ================================

    folder_path = '/data/tif/B11_ds_4.5x_ABeta_z1200';
    if isempty(folder_path) || ~isfolder(folder_path)
        error('Please set the TIF_TEST_FOLDER environment variable to a folder containing .tif files.');
    end

    files = dir(fullfile(folder_path, '*.tif'));
    assert(~isempty(files), 'No TIF files found in the folder.');

    filelist = fullfile({files.folder}, {files.name});
    numSlices = numel(filelist);

    info = imfinfo(filelist{1});
    imageHeight = info.Height;
    imageWidth = info.Width;

    blockSizes = [64, 64; 128, 96; 240, 240];
    testZ = [round(numSlices / 2), max(1, numSlices - 3)];

    results = [];

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
            bl_gt = zeros(blkW, blkH, numel(z_indices), 'uint16');  % width x height x depth
            for k = 1:numel(z_indices)
                slice = imread(filelist{z_indices(k)}, ...
                    'PixelRegion', {[y_indices(1), y_indices(end)], [x_indices(1), x_indices(end)]});
                bl_gt(:, :, k) = slice';
            end
            t_ref = toc(t1);

            % MEX function
            t2 = tic;
            bl_mex = load_bl_tif(filelist(z_indices), y, x, blkH, blkW);
            t_mex = toc(t2);

            diff = abs(bl_mex - bl_gt);
            maxerr = max(diff(:));
            pass = maxerr == 0;

            % Print table row
            symbol = char(10003 * pass + 10007 * ~pass); % ✅ or ❌
            fprintf('  %s  | %-6d | [%3d,%3d]  | (%5d,%5d) | %1.4e | %8.2fx\n', ...
                symbol, zidx, blkH, blkW, x, y, maxerr, t_ref / t_mex);

            results = [results; pass, zidx, blkH, blkW, x, y, maxerr, t_ref / t_mex];
        end
    end

    if all(results(:,1))
        fprintf('\n🎉 All tests passed successfully.\n');
    else
        fprintf('\n❗ Some tests failed. Please review the table above.\n');
    end
end
