function save_bl_tif_test()
    rng(42);
    fprintf("🧪 Running save_bl_tif extended tests…\n");

    %% (A) single-slice volume  — accepted as 2-D or 3-D [ dim3 == 1 ]
    vol1  = uint8(randi(255, [256 256]));      % plain 2-D matrix
    out1  = [tempname '.tif'];
    save_bl_tif(vol1, {out1}, false, 'none', feature('numCores')); % YXZ, no transpose needed
    assert(isequal(imread(out1), vol1),  "2-D path failed (YXZ)");
    delete(out1);

    vol1b = reshape(vol1, 256,256,1);          % explicit 3-D singleton
    out1b = [tempname '.tif'];
    save_bl_tif(vol1b, {out1b}, false, 'none');
    assert(isequal(imread(out1b), vol1b(:,:,1)), "3-D-singleton path failed");
    delete(out1b);

    %% (B) two-slice volume — true 3-D
    vol2 = uint8(randi(255, [256 256 2]));     % 2 slices
    files = { [tempname '_0.tif'], [tempname '_1.tif'] };

    save_bl_tif(vol2, files, false, 'none');   % YXZ → no transpose

    for k = 1:2
        data = imread(files{k});
        assert(isequal(data, vol2(:,:,k)), ...
            "Multi-slice check failed on slice %d", k);
        delete(files{k});
    end

    fprintf("✅ 2-D and 3-D slice sanity checks passed\n");

    outdir = fullfile(tempdir, 'save_bl_tif_test');
    if exist(outdir, 'dir'), rmdir(outdir, 's'); end
    mkdir(outdir);

    sz = [128, 96, 4];  % slightly larger stack
    orders = {'YXZ', false; 'XYZ', true};
    types = {'uint8', @uint8; 'uint16', @uint16};
    compressions = {'none', 'lzw', 'deflate'};

    all_passed = true;

    for o = 1:size(orders, 1)
        for t = 1:size(types, 1)
            for c = 1:numel(compressions)
                try
                    fprintf("→ %s | %s | %s\n", orders{o,1}, types{t,1}, compressions{c});

                    A = types{t,2}(rand(sz));
                    if orders{o,2}, A = permute(A, [2 1 3]); end

                    fileList = cell(1, sz(3));
                    for k = 1:sz(3)
                        fileList{k} = fullfile(outdir, sprintf('test_%s_%s_%s_%03d.tif', ...
                            orders{o,1}, types{t,1}, compressions{c}, k));
                    end

                    tic;
                    save_bl_tif(A, fileList, orders{o,2}, compressions{c});
                    dur = toc;

                    for k = 1:sz(3)
                        B = imread(fileList{k});
                        if orders{o,2}   % XYZ layout → MEX transposes
                            ok = isequal(B, A(:,:,k).');    % compare to transposed slice
                        else             % YXZ layout → no transpose
                            ok = isequal(B, A(:,:,k));
                        end
                        if ~ok
                            error("Mismatch in slice %d", k);
                        end
                    end

                    fprintf("✅ Passed in %.2fs\n", dur);
                catch ME
                    fprintf("❌ Failed: %s\n", ME.message);
                    all_passed = false;
                end
            end
        end
    end

    %% Simulated corrupted path test
    fprintf("🧪 Testing invalid path handling...\n");
    A = randi(255, [32 32 1], 'uint8');
    try
        save_bl_tif(A, {'/invalid/path/slice001.tif'}, false, 'lzw');
        fprintf("❌ Expected error for invalid path was not raised.\n");
        all_passed = false;
    catch
        fprintf("✅ Correctly handled invalid path error.\n");
    end

    %% Simulated file overwrite protection test
    fprintf("🧪 Testing read-only file protection...\n");
    A = randi(255, [32 32 1], 'uint8');
    file = fullfile(outdir, 'readonly_slice.tif');
    imwrite(A(:,:,1), file);  % pre-create
    fileattrib(file, '-w');   % make read-only

    try
        save_bl_tif(A, {file}, false, 'lzw');
        fprintf("❌ Should have failed to overwrite read-only file.\n");
        all_passed = false;
    catch
        fprintf("✅ Correctly failed to overwrite read-only file.\n");
    end
    fileattrib(file, '+w');

    if all_passed
        fprintf("🎉 All save_bl_tif extended tests passed.\n");
    else
        error("Some tests failed. See log above.");
    end

    if exist(outdir, 'dir'), rmdir(outdir, 's'); end
end
