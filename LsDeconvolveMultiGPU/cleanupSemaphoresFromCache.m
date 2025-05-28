function cleanupSemaphoresFromCache()
    OFFSET = 1e5;
    cacheDir = getCachePath();
    if ~isfolder(cacheDir)
        fprintf('Cache directory not found: %s\n', cacheDir);
        return;
    end

    % Get both CPU and GPU cache files
    files = [ ...
        dir(fullfile(cacheDir, 'key_*_gpu.bin')); ...
        dir(fullfile(cacheDir, 'key_*_cpu.bin')) ...
    ];

    for i = 1:numel(files)
        [~, stem, ~] = fileparts(files(i).name);  % Extract 'key_[...]_gpu' or 'key_[...]_cpu'
        key = string2hash(stem) + OFFSET;
        try
            semaphore('d', key);
            fprintf('Destroyed semaphore with key %d (from file %s)\n', key, files(i).name);
        catch
            warning('Failed to destroy semaphore with key %d from file %s', key, files(i).name);
        end
    end
end
