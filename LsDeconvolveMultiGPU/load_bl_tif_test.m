function load_bl_tif_test()
    % ================================
    % load_bl_tif_test.m
    % Usage: matlab -batch load_bl_tif_test
    % ================================

    %folder_path = getenv('TIF_TEST_FOLDER');
    folder_path = '/data/tif/B11_ds_4.5x_ABeta_z1200'
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

    for b = 1:size(blockSizes, 1)
        blkH = blockSizes(b, 1);
        blkW = blockSizes(b, 2);

        for zidx = testZ
            y = randi([1, imageHeight - blkH + 1]);
            x = randi([1, imageWidth  - blkW + 1]);
            y_indices = y : y + blkH - 1;
            x_indices = x : x + blkW - 1;
            z_indices = zidx : min(numSlices, zidx + 2);

            bl_gt = zeros(blkW, blkH, numel(z_indices), 'single');  % width x height x depth
            for k = 1:numel(z_indices)
                slice = imread(filelist{z_indices(k)}, ...
                    'PixelRegion', {[y_indices(1), y_indices(end)], [x_indices(1), x_indices(end)]});
                bl_gt(:, :, k) = im2single(slice)';
            end

            bl_mex = load_bl_tif(filelist(z_indices), y, x, blkH, blkW);

            disp("Size of bl_mex:"); disp(size(bl_mex));
            disp("Size of bl_gt:"); disp(size(bl_gt));


            diff = abs(im2single(bl_mex) - bl_gt);
            maxerr = max(diff(:));

            fprintf('Test z=%d, block=[%d,%d] at (x=%d,y=%d): max error = %.4g\n', ...
                zidx, blkH, blkW, x, y, maxerr);

            assert(maxerr < 1e-3, 'Mismatch between load_bl_tif and imread.');
        end
    end

    fprintf('All tests passed successfully.\n');
end
