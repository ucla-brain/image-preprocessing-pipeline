% ===============================
% lz4_save_load_test.m (standalone, speedup columns, no parallel/worker code)
% ===============================

disp('Running save/load LZ4 test and benchmark with integrity checks ...');

% --- Check for save_lz4_mex and load_lz4_mex visibility ---
disp('Checking MEX visibility ...');
mexfiles = {'save_lz4_mex', 'load_lz4_mex'};
for m = 1:numel(mexfiles)
    mf = mexfiles{m};
    loc = which(mf);
    if isempty(loc)
        error('MEX file "%s" not found on MATLAB path. Please add its folder with addpath().', mf);
    else
        fprintf('Found "%s" at: %s\n', mf, loc);
    end
end

% =========== MAIN TEST LOGIC FOLLOWS ===========

% --- Ask for cache/test directory ---
if usejava('desktop')
    cache_dir = uigetdir(pwd, 'Select a folder for temp/cache test files (choose a fast drive if possible)');
    if isnumeric(cache_dir) && cache_dir == 0
        disp('User canceled; aborting test.');
        return
    end
else
    % Try using env var or fallback
    cache_dir = getenv('LZ4_TEST_DIR');
    if isempty(cache_dir) || ~isfolder(cache_dir)
        warning('No valid interactive input possible. Using tempdir() as fallback.');
        cache_dir = fullfile(tempdir, 'lz4_test_tmp');
        if ~isfolder(cache_dir), mkdir(cache_dir); end
    end
end
fprintf('Using folder for cache/test files: %s\n', cache_dir);

rng(42); % For reproducibility
filetypes = {'double', 'single', 'uint16'};
filenames = {'test_dbl.lz4', 'test_sgl.lz4', 'test_u16.lz4'};
shapes = { [100, 200], [20, 30, 10], [512, 512, 8] };
bench_results = {};

% --- Test moderate-sized arrays ---
for k = 1:numel(filetypes)
    t = filetypes{k};
    fname = fullfile(cache_dir, filenames{k});
    sz = shapes{k};
    fprintf('Testing %s ...\n', t);

    switch t
        case 'double'
            arr = rand(sz);
        case 'single'
            arr = rand(sz, 'single');
        case 'uint16'
            arr = uint16(65535 * rand(sz));
    end

    % ========== LZ4 ==========
    tic;
    save_lz4_mex(fname, arr);
    t_save_lz4 = toc;

    % --- DIAGNOSTIC FILE CHECK ---
    if ~isfile(fname)
        error('File not found after save_lz4_mex: %s', fname);
    end
    finfo = dir(fname);
    if finfo.bytes == 0
        error('File is zero bytes after save_lz4_mex: %s', fname);
    end
    [fid, errmsg] = fopen(fname, 'rb');
    if fid < 0
        error('Cannot open saved file for reading: %s. Error: %s', fname, errmsg);
    end
    fclose(fid);
    try
        arr2 = load_lz4_mex(fname);
    catch ME
        error('load_lz4_mex failed immediately after save_lz4_mex for %s: %s', fname, ME.message);
    end

    t_load_lz4 = toc;

    eq_type = isa(arr2, t);
    eq_shape = isequal(size(arr), size(arr2));
    eq_content = isequal(arr, arr2);
    res_lz4 = eq_type && eq_shape && eq_content;
    if ~eq_content
        warning('%s LZ4 content mismatch', t);
    end

    % ========== LZ4 Backmasking Integrity Check ==========
    fname2 = [fname '.copy.lz4'];
    save_lz4_mex(fname2, arr2);
    arr3 = load_lz4_mex(fname2);
    eq_content_backmask = isequal(arr, arr3) && isequal(arr2, arr3);
    if ~eq_content_backmask
        warning('%s LZ4 backmasking mismatch (original vs. save->load->save->load)', t);
    end
    if isfile(fname2), delete(fname2); end

    % ========== MATLAB save/load ==========
    fname_mat = fullfile(cache_dir, [filenames{k}(1:end-4) '_matlab.mat']);
    tic;
    save(fname_mat, 'arr', '-v7.3');
    t_save_mat = toc;
    clear arr4
    tic;
    S = load(fname_mat, 'arr');
    arr4 = S.arr;
    t_load_mat = toc;

    % ========== MATLAB matfile ==========
    tic;
    mf = matfile(fname_mat);
    arr5 = mf.arr;
    t_load_matfile = toc;

    % --- Store results + speedup ratios ---
    bench_results(end+1,:) = {t, ...
        t_save_lz4, t_load_lz4, res_lz4, ...
        eq_content_backmask, ...
        t_save_mat, t_load_mat, ...
        t_save_mat/t_save_lz4, t_load_mat/t_load_lz4, ...
        t_load_matfile};

    % Clean up files
    if isfile(fname), delete(fname); end
    if isfile(fname_mat), delete(fname_mat); end
end

% --- Test >2GB array for each type ---
big_types = {'double', 'single', 'uint16'};
big_fnames = {'test_big_dbl.lz4', 'test_big_sgl.lz4', 'test_big_u16.lz4'};
big_sz = { [340*1024*1024,1], [700*1024*1024,1], [1500*1024*1024,1] };

fprintf('\nTesting >2GB arrays:\n');
for k = 1:numel(big_types)
    t = big_types{k};
    fname = fullfile(cache_dir, big_fnames{k});
    sz = big_sz{k};
    fprintf('Testing %s, >2GB ...\n', t);

    switch t
        case 'double'
            arr = rand(sz);
        case 'single'
            arr = rand(sz, 'single');
        case 'uint16'
            arr = uint16(65535 * rand(sz));
    end

    % ========== LZ4 ==========
    tic;
    save_lz4_mex(fname, arr);
    t_save_lz4 = toc;

    if ~isfile(fname)
        error('File not found after save_lz4_mex: %s', fname);
    end
    finfo = dir(fname);
    if finfo.bytes == 0
        error('File is zero bytes after save_lz4_mex: %s', fname);
    end
    [fid, errmsg] = fopen(fname, 'rb');
    if fid < 0
        error('Cannot open saved file for reading: %s. Error: %s', fname, errmsg);
    end
    fclose(fid);
    try
        arr2 = load_lz4_mex(fname);
    catch ME
        error('load_lz4_mex failed immediately after save_lz4_mex for %s: %s', fname, ME.message);
    end

    t_load_lz4 = toc;

    eq_type = isa(arr2, t);
    eq_shape = isequal(size(arr), size(arr2));
    eq_content = isequal(arr, arr2);
    res_lz4 = eq_type && eq_shape && eq_content;
    if ~eq_content
        warning('%s LZ4 content mismatch (>2GB)', t);
    end

    % ========== LZ4 Backmasking Integrity Check ==========
    fname2 = [fname '.copy.lz4'];
    save_lz4_mex(fname2, arr2);
    arr3 = load_lz4_mex(fname2);
    eq_content_backmask = isequal(arr, arr3) && isequal(arr2, arr3);
    if ~eq_content_backmask
        warning('%s LZ4 backmasking mismatch (>2GB, original vs. save->load->save->load)', t);
    end
    if isfile(fname2), delete(fname2); end

    % ========== MATLAB save/load ==========
    fname_mat = fullfile(cache_dir, [big_fnames{k}(1:end-4) '_matlab.mat']);
    tic;
    save(fname_mat, 'arr', '-v7.3');
    t_save_mat = toc;
    clear arr4
    tic;
    S = load(fname_mat, 'arr');
    arr4 = S.arr;
    t_load_mat = toc;

    % ========== MATLAB matfile ==========
    tic;
    mf = matfile(fname_mat);
    arr5 = mf.arr;
    t_load_matfile = toc;

    % --- Store results + speedup ratios ---
    bench_results(end+1,:) = {['big-' t], ...
        t_save_lz4, t_load_lz4, res_lz4, ...
        eq_content_backmask, ...
        t_save_mat, t_load_mat, ...
        t_save_mat/t_save_lz4, t_load_mat/t_load_lz4, ...
        t_load_matfile};

    if isfile(fname), delete(fname); end
    if isfile(fname_mat), delete(fname_mat); end
end

% --- Display Table ---
headers = { ...
    'Type', ...
    'LZ4_Save', 'LZ4_Load', 'LZ4_Ok', ...
    'LZ4_Backmask_Ok', ...
    'MAT_Save', 'MAT_Load', ...
    'LZ4_Save_vs_MAT', 'LZ4_Load_vs_MAT', ...
    'Matfile_Load' ...
};
tbl = cell2table(bench_results, 'VariableNames', headers);
disp(tbl);

fprintf('\n--- LZ4 vs MAT Save/Load Speedup Ratios ---\n');
disp(tbl(:, {'Type','LZ4_Save_vs_MAT','LZ4_Load_vs_MAT'}));

if all([tbl.LZ4_Ok; tbl.LZ4_Backmask_Ok])
    disp('All LZ4 save/load and integrity (backmask) tests PASSED.');
else
    disp('Some LZ4 save/load or integrity (backmask) tests FAILED.');
end
