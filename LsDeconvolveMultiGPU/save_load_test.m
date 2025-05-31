% ===============================
% save_load_test.m (with cache location prompt and integrity backmasking)
% ===============================
disp('Running save/load LZ4 test and benchmark with integrity checks ...');

% --- Ask for cache/test directory ---
if usejava('desktop')
    cache_dir = uigetdir(pwd, 'Select a folder for temp/cache test files (choose a fast drive if possible)');
    if isnumeric(cache_dir) && cache_dir == 0
        disp('User canceled; aborting test.');
        return
    end
else
    prompt = 'Enter a folder for temp/cache test files (choose a fast drive if possible):\n> ';
    cache_dir = input(prompt, 's');
    if isempty(cache_dir) || ~isfolder(cache_dir)
        disp('No valid folder specified; aborting test.');
        return
    end
end
fprintf('Using folder for cache/test files: %s\n', cache_dir);

rng(42); % For reproducibility
filetypes = {'double', 'single', 'uint16'};
filenames = {'test_dbl.lz4', 'test_sgl.lz4', 'test_u16.lz4'};
shapes = { [100, 200], [20, 30, 10], [512, 512, 8] };
all_ok = true;
bench_results = {};

do_raw_bin = false; % Set true to include raw binary (optional)

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
    tic;
    arr2 = load_lz4_mex(fname);
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
    % Clean up extra file
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
    eq_type_mat = isa(arr4, t);
    eq_shape_mat = isequal(size(arr), size(arr4));
    eq_content_mat = isequal(arr, arr4);
    res_mat = eq_type_mat && eq_shape_mat && eq_content_mat;

    % ========== MATLAB matfile ==========
    tic;
    mf = matfile(fname_mat);
    arr5 = mf.arr;
    t_load_matfile = toc;
    eq_type_mf = isa(arr5, t);
    eq_shape_mf = isequal(size(arr), size(arr5));
    eq_content_mf = isequal(arr, arr5);
    res_mf = eq_type_mf && eq_shape_mf && eq_content_mf;

    t_save_raw = NaN; t_load_raw = NaN; res_raw = false;
    if do_raw_bin
        fname_raw = fullfile(cache_dir, [filenames{k}(1:end-4) '.rawbin']);
        tic;
        fid = fopen(fname_raw, 'wb');
        fwrite(fid, arr, class(arr));
        fclose(fid);
        t_save_raw = toc;
        clear arr6
        tic;
        fid = fopen(fname_raw, 'rb');
        arr6 = fread(fid, numel(arr), ['*' class(arr)]);
        fclose(fid);
        arr6 = reshape(arr6, size(arr));
        t_load_raw = toc;
        res_raw = isequal(arr, arr6);
        if isfile(fname_raw), delete(fname_raw); end
    end

    % --- Store results ---
    bench_results(end+1,:) = {t, ...
        t_save_lz4, t_load_lz4, res_lz4, ...
        eq_content_backmask, ...
        t_save_mat, t_load_mat, res_mat, ...
        t_load_matfile, res_mf, ...
        t_save_raw, t_load_raw, res_raw};

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
    tic;
    arr2 = load_lz4_mex(fname);
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
    eq_type_mat = isa(arr4, t);
    eq_shape_mat = isequal(size(arr), size(arr4));
    eq_content_mat = isequal(arr, arr4);
    res_mat = eq_type_mat && eq_shape_mat && eq_content_mat;

    % ========== MATLAB matfile ==========
    tic;
    mf = matfile(fname_mat);
    arr5 = mf.arr;
    t_load_matfile = toc;
    eq_type_mf = isa(arr5, t);
    eq_shape_mf = isequal(size(arr), size(arr5));
    eq_content_mf = isequal(arr, arr5);
    res_mf = eq_type_mf && eq_shape_mf && eq_content_mf;

    t_save_raw = NaN; t_load_raw = NaN; res_raw = false;
    % (skip raw binary for >2GB)

    % --- Store results ---
    bench_results(end+1,:) = {['big-' t], ...
        t_save_lz4, t_load_lz4, res_lz4, ...
        eq_content_backmask, ...
        t_save_mat, t_load_mat, res_mat, ...
        t_load_matfile, res_mf, ...
        t_save_raw, t_load_raw, res_raw};

    if isfile(fname), delete(fname); end
    if isfile(fname_mat), delete(fname_mat); end
end

% --- Display Table ---
headers = { ...
    'Type', ...
    'LZ4_Save', 'LZ4_Load', 'LZ4_Ok', ...
    'LZ4_Backmask_Ok', ...
    'MAT_Save', 'MAT_Load', 'MAT_Ok', ...
    'Matfile_Load', 'Matfile_Ok', ...
    'Raw_Save', 'Raw_Load', 'Raw_Ok' ...
};
tbl = cell2table(bench_results, 'VariableNames', headers);
disp(tbl);

if all([tbl.LZ4_Ok; tbl.LZ4_Backmask_Ok; tbl.MAT_Ok; tbl.Matfile_Ok])
    disp('All LZ4/MAT/matfile save/load and integrity (backmask) tests PASSED.');
else
    disp('Some LZ4/MAT/matfile save/load or integrity (backmask) tests FAILED.');
end
