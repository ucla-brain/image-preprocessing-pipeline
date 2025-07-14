function save_bl_tif_test
% Extended regression and performance benchmark for save_bl_tif MEX (tiles and strips)
%
% This test suite:
%   - Validates data integrity for basic and large TIFF volumes
%   - Benchmarks and compares STRIP and TILE mode for multiple layouts, types, compressions
%   - Measures both logical and physical disk file sizes
%   - Includes robust file/folder cleanup and guard-clause validation

fprintf("🧪  save_bl_tif extended test-suite (with tiles vs strips)\n");

% Temporary test folder (auto-removed)
temporaryTestRoot = tempname;
mkdir(temporaryTestRoot);
cleanupObj = onCleanup(@() sandbox_cleanup(temporaryTestRoot));

%% ========== A.1. Basic 2D/3D Single-Slice Validation ==========
rng(42);  % Reproducibility

singleSliceImage = uint8(randi(255,[256 256]));
singleSliceFilename = fullfile(temporaryTestRoot,'basic_2d.tif');
if ~exist(temporaryTestRoot, 'dir')
    error('Temporary directory does not exist: %s', temporaryTestRoot);
end
imwrite(uint8(zeros(10)), singleSliceFilename);
assert(exist(singleSliceFilename,'file')==2, 'imwrite failed to create file.');
delete(singleSliceFilename);
fprintf('Calling save_bl_tif for: %s\n', singleSliceFilename);
save_bl_tif(singleSliceImage,{singleSliceFilename},false,'none',[],false);
fprintf('Finished save_bl_tif for: %s\n', singleSliceFilename);
assert(isequal(readTiff(singleSliceFilename), singleSliceImage));

singleSliceVolume = reshape(singleSliceImage,256,256,1);
singleSliceVolumeFilename = fullfile(temporaryTestRoot,'basic_3d.tif');
save_bl_tif(singleSliceVolume,{singleSliceVolumeFilename},false,'none',[],false); % strip mode
assert(isequal(readTiff(singleSliceVolumeFilename), singleSliceVolume(:,:,1)));

fprintf("   ✅ basic 2D/3D single-slice paths OK\n");

%% ========== A.2. Edge Tile Zero-Padding Validation (Tile Mode, XYZ) ==========
edgeTileSize = [129 129 1];  % Not a multiple of default tile (128x128)
edgeTileValue = 100;
edgeTileVol = uint8(edgeTileValue * ones(edgeTileSize));
edgeTileFilename = fullfile(temporaryTestRoot, 'tile_edge_xyz.tif');
try
    save_bl_tif(edgeTileVol, {edgeTileFilename}, true, 'none', [], true); % XYZ, tile mode
    loaded = readTiff(edgeTileFilename);

    % Check main region
    if ~isequal(loaded(1:129,1:129), edgeTileVol(:,:,1))
        error('❌ Edge tile main region does not match input values!');
    end

    % Check padding rows/cols if present (should be zero)
    padding_ok = true;
    if size(loaded,1) > 129
        padding_rows = loaded(130:end, 1:129);
        if any(padding_rows(:) ~= 0)
            fprintf('❌ Edge tile test: Nonzero padding in bottom rows!\n');
            padding_ok = false;
        end
    end
    if size(loaded,2) > 129
        padding_cols = loaded(1:129, 130:end);
        if any(padding_cols(:) ~= 0)
            fprintf('❌ Edge tile test: Nonzero padding in right columns!\n');
            padding_ok = false;
        end
    end
    if size(loaded,1) > 129 && size(loaded,2) > 129
        padding_corner = loaded(130:end,130:end);
        if any(padding_corner(:) ~= 0)
            fprintf('❌ Edge tile test: Nonzero padding in corner!\n');
            padding_ok = false;
        end
    end

    if padding_ok
        fprintf('   ✅ tile edge zero-padding (tile mode, XYZ) OK\n');
    else
        error('❌ tile edge zero-padding (tile mode, XYZ) FAILED');
    end

catch ex
    fprintf('❌ tile edge zero-padding (tile mode, XYZ) error: %s\n', ex.message);
    rethrow(ex)
end

%% ========== B. Full Matrix: {layout × type × compression} + Strip/Tile Benchmark ==========

% Configurations for benchmarking
volumeLayouts = {'YXZ',false; 'XYZ',true};  % Name, isXYZ flag
volumeDataTypes = {'uint8',@uint8; 'uint16',@uint16};
compressionTypes = {'none','lzw','deflate'};
testVolumeSize = [3000 1500 4];    % [Height Width Depth]
tileModeFlags = [false true];
tileModeNames = {'STRIP','TILE'};

nLayouts   = size(volumeLayouts,1);
nTypes     = size(volumeDataTypes,1);
nComps     = numel(compressionTypes);
nTileModes = numel(tileModeFlags);

% Preallocate results
saveTimesSeconds     = zeros(nLayouts,nTypes,nComps,nTileModes);
logicalSizesMiB      = zeros(nLayouts,nTypes,nComps,nTileModes);
physicalSizesMiB     = zeros(nLayouts,nTypes,nComps,nTileModes);

% --- Main matrix benchmark: Each config is tested with strip and tile
for tileModeIndex = 1:nTileModes
    useTiles = tileModeFlags(tileModeIndex);
    tileModeDescription = tileModeNames{tileModeIndex};
    for layoutIndex = 1:nLayouts
        layoutName = volumeLayouts{layoutIndex,1};
        isXYZ = volumeLayouts{layoutIndex,2};
        fprintf("\n   🏁 Testing layout: %s (%s)\n", layoutName, tileModeDescription);
        for typeIndex = 1:nTypes
            dataTypeName = volumeDataTypes{typeIndex,1};
            dataTypeFunc = volumeDataTypes{typeIndex,2};
            for compIndex = 1:nComps
                compressionType = compressionTypes{compIndex};
                % --- Generate volume data
                testVolume = generateTestData(testVolumeSize, dataTypeName);
                if isXYZ
                    testVolume = permute(testVolume,[2 1 3]);
                end
                tagSafe = regexprep(sprintf('%s_%s_%s_%s', ...
                    layoutName,dataTypeName,compressionType,tileModeDescription), '[^A-Za-z0-9]','_');
                fileList = arrayfun(@(k) fullfile(temporaryTestRoot, ...
                    sprintf('t_%s_%02d.tif',tagSafe,k)), ...
                    1:testVolumeSize(3), 'UniformOutput',false);

                % --- Save and time
                ticID = tic;
                save_bl_tif(testVolume, fileList, isXYZ, compressionType, feature('numCores'), useTiles);
                elapsedSeconds = toc(ticID);

                % --- Size/accounting + data integrity
                totalLogicalBytes  = 0;
                totalPhysicalBytes = 0;
                for sliceIdx = 1:testVolumeSize(3)
                    referenceSlice = testVolume(:,:,sliceIdx);
                    if isXYZ, referenceSlice = referenceSlice.'; end
                    loadedSlice = readTiff(fileList{sliceIdx});
                    assert(isequal(loadedSlice, referenceSlice), ...
                        'Mismatch %s slice %d via Tiff', tagSafe, sliceIdx);
                    fileInfo = dir(fileList{sliceIdx});
                    totalLogicalBytes = totalLogicalBytes + fileInfo.bytes;
                    % Get physical size in bytes (platform: Linux, otherwise returns logical)
                    [~,blockCount] = system(sprintf('stat -c "%%b" "%s"', fileList{sliceIdx}));
                    [~,blockSize ] = system(sprintf('stat -c "%%B" "%s"', fileList{sliceIdx}));
                    totalPhysicalBytes = totalPhysicalBytes + ...
                        str2double(strtrim(blockCount)) * str2double(strtrim(blockSize));
                end

                saveTimesSeconds(layoutIndex,typeIndex,compIndex,tileModeIndex) = elapsedSeconds;
                logicalSizesMiB(layoutIndex,typeIndex,compIndex,tileModeIndex) = totalLogicalBytes  / 2^20;
                physicalSizesMiB(layoutIndex,typeIndex,compIndex,tileModeIndex)= totalPhysicalBytes / 2^20;

                fprintf("      ✅ %-40s in %.2f s, logical %.1f MiB, physical %.1f MiB\n", ...
                    strrep(sprintf('%s_%s_%s_%s',layoutName,dataTypeName,compressionType,tileModeDescription),'_',' | '), ...
                    elapsedSeconds, logicalSizesMiB(layoutIndex,typeIndex,compIndex,tileModeIndex), physicalSizesMiB(layoutIndex,typeIndex,compIndex,tileModeIndex));
            end
        end
    end
end

% ---- Print STRIP vs TILE summary comparison table ----
fprintf("\n   📊 STRIP vs TILE comparison (Speedup, Section B):\n");
summaryRows = {};
for typeIndex = 1:nTypes
    for compIndex = 1:nComps
        % Find row indices for YXZ (layoutIndex=1) and XYZ (layoutIndex=2)
        % Strip vs Tile for YXZ:
        tstrip_yxz = saveTimesSeconds(1,typeIndex,compIndex,1);
        ttile_yxz  = saveTimesSeconds(1,typeIndex,compIndex,2);
        % Strip vs Tile for XYZ:
        tstrip_xyz = saveTimesSeconds(2,typeIndex,compIndex,1);
        ttile_xyz  = saveTimesSeconds(2,typeIndex,compIndex,2);
        % Speedup: Strip vs Tile for YXZ and XYZ
        speedup_strip_tile_yxz = tstrip_yxz / ttile_yxz;
        speedup_strip_tile_xyz = tstrip_xyz / ttile_xyz;
        % Speedup: XYZ vs YXZ (strip mode)
        speedup_xyz_vs_yxz_strip = tstrip_yxz / tstrip_xyz;
        % Speedup: XYZ vs YXZ (tile mode)
        speedup_xyz_vs_yxz_tile  = ttile_yxz  / ttile_xyz;
        % Store table (you can show both, or just strip)
        summaryRows(end+1,:) = {volumeDataTypes{typeIndex,1}, compressionTypes{compIndex}, ...
            tstrip_yxz,   tstrip_xyz,   ttile_yxz,   ttile_xyz, ...
            speedup_strip_tile_yxz, speedup_strip_tile_xyz, ...
            speedup_xyz_vs_yxz_strip, speedup_xyz_vs_yxz_tile};
    end
end
comparisonTable = cell2table(summaryRows, 'VariableNames', ...
    {'DataType','Compression', ...
     'Time_STRIP_YXZ_s', 'Time_STRIP_XYZ_s', ...
     'Time_TILE_YXZ_s',  'Time_TILE_XYZ_s', ...
     'Speedup_StripVsTile_YXZ', 'Speedup_StripVsTile_XYZ', ...
     'Speedup_XYZvsYXZ_Strip', 'Speedup_XYZvsYXZ_Tile'});
disp(comparisonTable);


%% ========== C. Large Block Test: 100 Big Slices, Compare Strip vs Tile (XYZ) ==========

largeBlockSize = [25555 16531 200];
%largeBlockVolume = uint16(randi([0 65535], largeBlockSize));
largeBlockVolume = generateTestData(largeBlockSize, 'uint16');
largeBlockFileList = arrayfun(@(k) fullfile(temporaryTestRoot, sprintf('bigblock_%03d.tif',k)), 1:largeBlockSize(3), 'UniformOutput', false);

% --- STRIP mode (XYZ)
fprintf('\n   🏁 Saving %d large slices (STRIP mode, XYZ)...\n', largeBlockSize(3));
stripSaveTimeSec = tic;
save_bl_tif(largeBlockVolume, largeBlockFileList, true, 'deflate', feature('numCores'), false);  % isXYZ = true
stripElapsedSec = toc(stripSaveTimeSec);
fprintf('      ✅ %d large slices (STRIP mode, XYZ) ok (%.2f s)\n', largeBlockSize(3), stripElapsedSec);

% --- TILE mode (XYZ)
fprintf('\n   🏁 Saving %d large slices (TILE mode, XYZ)...\n', largeBlockSize(3));
tileSaveTimeSec = tic;
save_bl_tif(largeBlockVolume, largeBlockFileList, true, 'deflate', feature('numCores'), true);  % isXYZ = true
tileElapsedSec = toc(tileSaveTimeSec);
fprintf('      ✅ %d large slices (TILE mode, XYZ) ok (%.2f s)\n', largeBlockSize(3), tileElapsedSec);

% --- Print block test summary
fprintf('\n   🚦  [Performance] Tiles vs Strips (%dx %dx%d slices, XYZ):\n', largeBlockSize(3), largeBlockSize(1), largeBlockSize(2));
fprintf('         STRIP: %.2f s\n', stripElapsedSec);
fprintf('         TILE : %.2f s\n', tileElapsedSec);
if tileElapsedSec < stripElapsedSec
    fprintf('      🟢 Tiles are FASTER (%.1fx speedup)\n', stripElapsedSec/tileElapsedSec);
else
    fprintf('      🟡 Strips are FASTER (%.1fx speedup)\n', tileElapsedSec/stripElapsedSec);
end

%% ========== D. Guard-Clause Error Handling ==========
fprintf("\n   🛡  guard-clause checks\n");
try
    save_bl_tif(uint8(0), {'/no/way/out.tif'}, false,'lzw',feature('numCores'),false);
    error('invalid-path accepted');
catch, fprintf('      ✅ invalid path rejected\n'); end

readOnlyFilename = fullfile(temporaryTestRoot,'readonly.tif');
imwrite(uint8(1),readOnlyFilename);
fileattrib(readOnlyFilename,'-w');
readOnlyCleanupObj = onCleanup(@() restore_rw(readOnlyFilename));
try
    save_bl_tif(uint8(0), {readOnlyFilename}, false,'none',feature('numCores'),false);
    error('read-only overwrite accepted');
catch, fprintf('      ✅ read-only overwrite rejected\n'); end

fprintf("\n🎉  all save_bl_tif tests passed (TILES + STRIPS)\n");

end

% ------- Helper functions as local functions below --------

function data = readTiff(filename)
    % Robust TIFF reader, works with both 'Tiff' and 'imread'
    try
        tiffObj = Tiff(filename,'r');
        data = read(tiffObj);
        tiffObj.close();
    catch
        data = imread(filename);
    end
end

function outputVolume = generateTestData(targetShape, dataTypeName)
    %GENERATE_SPARSE_GAMMA_3D Generates a 3D matrix of specified size using
    %sparse gamma distributed 2D planes and broadcasts them into 3D.
    % The 6 edges are also multiplied by sparse gamma random vectors.

    assert(numel(targetShape) == 3, 'targetShape must be a vector of length 3');
    [a, b, c] = deal(targetShape(1), targetShape(2), targetShape(3));

    % === Parameters ===
    shapeParam = 2;
    scaleParam = 0.5;
    sparsity   = 0.5;

    % === Prepare scale factors for direct scaling ===
    switch dataTypeName
        case 'uint8'
            scaleFactor = 255;
            castFunc = @uint8;
        case 'uint16'
            scaleFactor = 65535;
            castFunc = @uint16;
        otherwise
            error("Unsupported dtype '%s'", dataTypeName);
    end

    % === Generate sparse gamma matrices, scaled & converted early ===
    A = castFunc(scaleFactor * ...
        (gamrnd(shapeParam, scaleParam, a, b) .* (rand(a, b) > sparsity)));
    B = castFunc(scaleFactor * ...
        (gamrnd(shapeParam, scaleParam, b, c) .* (rand(b, c) > sparsity)));

    % === Broadcasted outer product in integer ===
    A3 = repmat(A, 1, 1, c);
    B3 = permute(B, [3 1 2]);
    result = A3 .* B3;

    % === Generate edge vectors, scaled & converted early ===
    edgeX1 = castFunc(scaleFactor * ...
        (gamrnd(shapeParam, scaleParam, [a 1 1]) .* (rand(a,1,1) > sparsity)));
    edgeX2 = castFunc(scaleFactor * ...
        (gamrnd(shapeParam, scaleParam, [a 1 1]) .* (rand(a,1,1) > sparsity)));

    edgeY1 = castFunc(scaleFactor * ...
        (gamrnd(shapeParam, scaleParam, [1 b 1]) .* (rand(1,b,1) > sparsity)));
    edgeY2 = castFunc(scaleFactor * ...
        (gamrnd(shapeParam, scaleParam, [1 b 1]) .* (rand(1,b,1) > sparsity)));

    edgeZ1 = castFunc(scaleFactor * ...
        (gamrnd(shapeParam, scaleParam, [1 1 c]) .* (rand(1,1,c) > sparsity)));
    edgeZ2 = castFunc(scaleFactor * ...
        (gamrnd(shapeParam, scaleParam, [1 1 c]) .* (rand(1,1,c) > sparsity)));

    % === Apply edges multiplicatively ===
    result(1,:,:)   = result(1,:,:)   .* edgeX1(1,:,:);
    result(end,:,:) = result(end,:,:) .* edgeX2(end,:,:);

    result(:,1,:)   = result(:,1,:)   .* edgeY1(:,1,:);
    result(:,end,:) = result(:,end,:) .* edgeY2(:,end,:);

    result(:,:,1)   = result(:,:,1)   .* edgeZ1(:,:,1);
    result(:,:,end) = result(:,:,end) .* edgeZ2(:,:,end);

    % Clip in case of overflows
    maxVal = intmax(dataTypeName);
    result(result > maxVal) = maxVal;

    outputVolume = result;
end

function sandbox_cleanup(folderPath)
    % Cleanup function to close files and remove folder
    fclose('all'); safe_rmdir(folderPath);
end

function restore_rw(filePath)
    % Restore write permissions to a file
    if exist(filePath,'file'), fileattrib(filePath,'+w'); end
end

function safe_rmdir(folderPath)
    % Remove directory and its contents robustly
    if exist(folderPath,'dir')
        try rmdir(folderPath,'s');
        catch, pause(0.1);
            if exist(folderPath,'dir'), rmdir(folderPath,'s'); end
        end
    end
end
