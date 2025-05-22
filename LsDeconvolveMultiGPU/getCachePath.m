function cache_path = getCachePath()
    cache_path  = fullfile(tempdir, 'otf_cache');
    % cache_path  = fullfile('/data', 'otf_cache');
    if ~exist(cache_path , 'dir')
        mkdir(cache_path);
    end
end