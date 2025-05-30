% ===============================
% save_load_test.m (with benchmarks)
% ===============================
disp('Running save/load LZ4 and MATLAB benchmark test ...');

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
    fname = filenames{k};
    sz = shapes{k};
    fprintf('Testing %s ...\n', t);

    % Create a random array of the given type and shape
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

    % ========== MATLAB save/load ==========
    fname_mat = [fname(1:end-4) '_matlab.mat'];
    tic;
    save(fname_mat, 'arr', '-v7.3');
    t_save_mat = toc;
    clear arr3
    tic;
    S = load(fname_mat, 'arr');
    arr3 = S.arr;
    t_load_mat = toc;
    eq_type_mat = isa(arr3, t);
    eq_shape_mat = isequal(size(arr), size(arr3));
    eq_content_mat = isequal(arr, arr3);
    res_mat = eq_type_mat && eq_shape_mat && eq_content_mat;

    % ========== MATLAB matfile ==========
    tic;
    mf = matfile(fname_mat);
    arr4 = mf.arr;
    t_load_matfile = toc;
    eq_type_mf = isa(arr4, t);
    eq_shape_mf = isequal(size(arr), size(arr4));
    eq_content_mf = isequal(arr, arr4);
    res_mf = eq_type_mf && eq_shape_mf && eq_content_mf;

    % ========== RAW BINARY (optional, only if do_raw_bin true) ==========
    t_save_raw = NaN; t_load_raw = NaN; res_raw = false;
    if do_raw_bin
        fname_raw = [fname(1:end-4) '.rawbin'];
        tic;
        fid = fopen(fname_raw, 'wb');
        fwrite(fid, arr, class(arr));
        fclose(fid);
        t_save_raw = toc;
        clear arr5
        tic;
        fid = fopen(fname_raw, 'rb');
        arr5 = fread(fid, numel(arr), ['*' class(arr)]);
        fclose(fid);
        arr5 = reshape(arr5, size(arr));
        t_load_raw = toc;
        res_raw = isequal(arr, arr5);
        if isfile(fname_raw), delete(fname_raw); end
    end

    % --- Store results ---
    bench_results(end+1,:) = {t, ...
        t_save_lz4, t_load_lz4, res_lz4, ...
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
big_sz = { [340*1024*1024,1], [700*1024*1024,1], [1500*1024*1024,1] }; % doubles: 2.5GB, singles: ~2.7GB, uint16: ~3GB

fprintf('\nTesting >2GB arrays:\n');
for k = 1:numel(big_types)
    t = big_types{k};
    fname = big_fnames{k};
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

    % ========== MATLAB save/load ==========
    fname_mat = [fname(1:end-4) '_matlab.mat'];
    tic;
    save(fname_mat, 'arr', '-v7.3');
    t_save_mat = toc;
    clear arr3
    tic;
    S = load(fname_mat, 'arr');
    arr3 = S.arr;
    t_load_mat = toc;
    eq_type_mat = isa(arr3, t);
    eq_shape_mat = isequal(size(arr), size(arr3));
    eq_content_mat = isequal(arr, arr3);
    res_mat = eq_type_mat && eq_shape_mat && eq_content_mat;

    % ========== MATLAB matfile ==========
    tic;
    mf = matfile(fname_mat);
    arr4 = mf.arr;
    t_load_matfile = toc;
    eq_type_mf = isa(arr4, t);
    eq_shape_mf = isequal(size(arr), size(arr4));
    eq_content_mf = isequal(arr, arr4);
    res_mf = eq_type_mf && eq_shape_mf && eq_content_mf;

    t_save_raw = NaN; t_load_raw = NaN; res_raw = false;
    % (skip raw binary for >2GB)

    % --- Store results ---
    bench_results(end+1,:) = {['big-' t], ...
        t_save_lz4, t_load_lz4, res_lz4, ...
        t_save_mat, t_load_mat, res_mat, ...
        t_load_matfile, res_mf, ...
        t_save_raw, t_load_raw, res_raw};

    if isfile(fname), delete(fname); end
    if isfile(fname_mat), delete(fname_mat); end
end

% --- Display Table ---
headers = { ...
    'Type', ...
    'LZ4 Save', 'LZ4 Load', 'LZ4 Ok', ...
    'MAT Save', 'MAT Load', 'MAT Ok', ...
    'Matfile Load', 'Matfile Ok', ...
    'Raw Save', 'Raw Load', 'Raw Ok' ...
};
tbl = cell2table(bench_results, 'VariableNames', headers);
disp(tbl);

if all([tbl.LZ4_Ok; tbl.MAT_Ok; tbl.Matfile_Ok])
    disp('All LZ4/MAT/matfile save/load tests PASSED.');
else
    disp('Some LZ4/MAT/matfile save/load tests FAILED.');
end
