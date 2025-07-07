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

%% ========== A. Basic 2D/3D Single-Slice Validation ==========
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

%% ========== B. Full Matrix: {layout × type × compression} + Strip/Tile Benchmark ==========

% Configurations for benchmarking
volumeLayouts = {'YXZ',false; 'XYZ',true};  % Name, isXYZ flag
volumeDataTypes = {'uint8',@uint8; 'uint16',@uint16};
compressionTypes = {'none','lzw','deflate'};
testVolumeSize = [2048 1024 4];    % [Height Width Depth]
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
                save_bl_tif(testVolume, fileList, isXYZ, compressionType, [], useTiles);
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
fprintf("\n   📊 STRIP vs TILE comparison (Time and Sizes, Section B):\n");
summaryRows = {};
for layoutIndex = 1:nLayouts
    for typeIndex = 1:nTypes
        for compIndex = 1:nComps
            timeStrip   = saveTimesSeconds(layoutIndex,typeIndex,compIndex,1);
            timeTile    = saveTimesSeconds(layoutIndex,typeIndex,compIndex,2);
            speedup     = timeStrip / timeTile;
            logicStrip  = logicalSizesMiB(layoutIndex,typeIndex,compIndex,1);
            logicTile   = logicalSizesMiB(layoutIndex,typeIndex,compIndex,2);
            physStrip   = physicalSizesMiB(layoutIndex,typeIndex,compIndex,1);
            physTile    = physicalSizesMiB(layoutIndex,typeIndex,compIndex,2);
            summaryRows(end+1,:) = {volumeLayouts{layoutIndex,1}, volumeDataTypes{typeIndex,1}, ...
                compressionTypes{compIndex}, timeStrip, timeTile, speedup, logicStrip, logicTile, physStrip, physTile};
        end
    end
end
comparisonTable = cell2table(summaryRows, 'VariableNames', ...
    {'Layout','DataType','Compression', 'Time_STRIP_s', 'Time_TILE_s', 'Speedup', ...
     'Logical_STRIP_MiB','Logical_TILE_MiB','Physical_STRIP_MiB','Physical_TILE_MiB'});
disp(comparisonTable);

%% ========== C. Large Block Test: 100 Big Slices, Compare Strip vs Tile ==========
largeBlockSize = [20480 20480 100];
largeBlockVolume = generateTestData(largeBlockSize, 'uint8');
largeBlockFileList = arrayfun(@(k) fullfile(temporaryTestRoot, sprintf('bigblock_%03d.tif',k)), 1:largeBlockSize(3), 'UniformOutput', false);

% --- STRIP mode
fprintf('\n   🏁 Saving 100 large slices (STRIP mode)...\n');
stripSaveTimeSec = tic;
save_bl_tif(largeBlockVolume, largeBlockFileList, false, 'deflate', [], false);
stripElapsedSec = toc(stripSaveTimeSec);
for sliceIdx = 1:largeBlockSize(3)
    data = readTiff(largeBlockFileList{sliceIdx});
    assert(isequal(data, largeBlockVolume(:,:,sliceIdx)), ...
        'Big block mismatch at slice %d (strip mode)', sliceIdx);
end
fprintf('      ✅ 100 large slices (STRIP mode) ok (%.2f s)\n', stripElapsedSec);

% --- TILE mode
fprintf('\n   🏁 Saving 100 large slices (TILE mode)...\n');
tileSaveTimeSec = tic;
save_bl_tif(largeBlockVolume, largeBlockFileList, false, 'deflate', [], true);
tileElapsedSec = toc(tileSaveTimeSec);
for sliceIdx = 1:largeBlockSize(3)
    data = readTiff(largeBlockFileList{sliceIdx});
    assert(isequal(data, largeBlockVolume(:,:,sliceIdx)), ...
        'Big block mismatch at slice %d (tile mode)', sliceIdx);
end
fprintf('      ✅ 100 large slices (TILE mode) ok (%.2f s)\n', tileElapsedSec);

% --- Print block test summary
fprintf('\n   🚦  [Performance] Tiles vs Strips (100x %dx%d slices):\n', largeBlockSize(1), largeBlockSize(2));
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
    save_bl_tif(uint8(0), {'/no/way/out.tif'}, false,'lzw',[],false);
    error('invalid-path accepted');
catch, fprintf('      ✅ invalid path rejected\n'); end

readOnlyFilename = fullfile(temporaryTestRoot,'readonly.tif');
imwrite(uint8(1),readOnlyFilename);
fileattrib(readOnlyFilename,'-w');
readOnlyCleanupObj = onCleanup(@() restore_rw(readOnlyFilename));
try
    save_bl_tif(uint8(0), {readOnlyFilename}, false,'none',[],false);
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

function outputVolume = generateTestData(volumeSize, dataTypeName)
    % Generate synthetic gamma-distributed, sparse 3D/2D test volume
    alpha = 2; beta = 50;
    randomData = gamrnd(alpha, beta, volumeSize);
    mask = rand(volumeSize) > 0.10;
    randomData(~mask) = 0;
    randomData = randomData / max(randomData(:));
    switch dataTypeName
        case 'uint8',  outputVolume = uint8(randomData * 255);
        case 'uint16', outputVolume = uint16(randomData * 65535);
        otherwise,     error("Unsupported dtype '%s'",dataTypeName);
    end
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
