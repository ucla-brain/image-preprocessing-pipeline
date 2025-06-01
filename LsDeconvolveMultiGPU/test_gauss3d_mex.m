function test_gauss3d_mex_features()
% Robust + colorful test harness for gauss3d_mex vs imgaussfilt3 (GPU/CPU fallback).
% Now also tests half precision ('half' mode). Features: accuracy, type/class checks,
% perf timing, OOM/fallback, output/precision summary, color output.

    g = gpuDevice(1);
    reset(g);

    hasCprintf = exist('cprintf','file') == 2;
    col = @(c,str) colored_str(c,str,hasCprintf);

    szs = {
        [32, 64, 32], ...
        [512, 512, 512], ...
        [48, 17, 119]  % Oddball non-cube size
    };
    types = {@single, @double};
    sigma_tests = {2.5, [2.5 2.5 2.5], [1.5 2.0 2.5], 0.25, 12}; % add extreme sigmas
    ksize_tests = {'auto', 9, [9 11 15], 3, 41};  % 3=smallest, 41=large (still < MAX_KERNEL_SIZE)
    disable_warning = getenv('GAUSS3D_WARN_KSIZE');
    if isempty(disable_warning)
        disp(col('cyan','  (Kernel size warnings are ENABLED: set GAUSS3D_WARN_KSIZE=0 to disable.)'));
    elseif strcmp(disable_warning, '0')
        disp(col('cyan','  (Kernel size warnings are DISABLED.)'));
    end

    % Define error thresholds for test pass/fail
    SINGLE_THRESH = 5e-5;
    DOUBLE_THRESH = 1e-7;
    HALF_THRESH   = 1e-2; % Acceptable for half

    for ityp = 1:numel(types)
        for isz = 1:numel(szs)
            sz = szs{isz};
            T = types{ityp};
            type_str = func2str(T);

            for isig = 1:numel(sigma_tests)
                sigma = sigma_tests{isig};
                for iksz = 1:numel(ksize_tests)
                    ksz = ksize_tests{iksz};
                    if ischar(ksz) || isstring(ksz)
                        kdesc = char(ksz);
                    else
                        kdesc = mat2str(ksz);
                    end

                    fprintf('\n%s\n',col('yellow',repmat('=',1,80)));
                    fprintf('%s\n',col('magenta',sprintf('Testing %s %s, sigma=%s, ksize=%s ...', ...
                        type_str, mat2str(sz), mat2str(sigma), kdesc)));

                    % Determine kernel size for padding
                    if ischar(ksz) || isstring(ksz)
                        kernel_sz = odd_kernel_size(sigma);
                    else
                        kernel_sz = ksz;
                        if isscalar(kernel_sz)
                            kernel_sz = repmat(kernel_sz,1,3);
                        end
                    end
                    pad_amt = floor(kernel_sz / 2);

                    % Prepare input
                    rng(0);
                    x = rand(sz, type_str);

                    % Pad input for both methods (mimic 'replicate')
                    x_pad = padarray(x, pad_amt, 'replicate', 'both');
                    x_pad_gpu = gpuArray(x_pad);

                    opts = {'Padding', 'replicate', 'FilterSize', kernel_sz, 'FilterDomain', 'spatial'};

                    try
                        % --- Reference (imgaussfilt3, GPU or CPU fallback) ---
                        t1 = tic;
                        try
                            y_ref_gpu = imgaussfilt3(x_pad_gpu, sigma, opts{:});
                        catch ME1
                            % If FFT plan error or OOM, fallback to CPU reference
                            if contains(ME1.message, 'fftn') || ...
                               contains(ME1.message, 'FFT') || ...
                               contains(ME1.message, 'out of memory') || ...
                               contains(ME1.identifier, 'parallel:gpu:array:OOM')
                                warning(col('blue','imgaussfilt3 GPU failed (FFT or OOM), using CPU reference for this size.'));
                                y_ref_gpu = gpuArray(imgaussfilt3(gather(x_pad_gpu), sigma, opts{:}));
                            else
                                rethrow(ME1);
                            end
                        end
                        t_ref = toc(t1);

                        % --- gauss3d_mex (single/double as baseline) ---
                        t2 = tic;
                        if ischar(ksz) || isstring(ksz)
                            y_mex_gpu = gauss3d_mex(x_pad_gpu, sigma);
                        else
                            y_mex_gpu = gauss3d_mex(x_pad_gpu, sigma, kernel_sz);
                        end
                        t_mex = toc(t2);

                        % --- Type, class, shape checks ---
                        assert(isequal(size(y_mex_gpu), size(x_pad_gpu)), 'Output size mismatch.');
                        assert(strcmp(class(y_mex_gpu), class(x_pad_gpu)), 'Output class mismatch.');
                        assert(strcmp(classUnderlying(y_mex_gpu), classUnderlying(x_pad_gpu)), 'Output underlying class mismatch.');
                        fprintf('%s\n',col('green','  Output matches input type/class/shape.'));

                        % --- Unpad both for comparison ---
                        idx1 = (1+pad_amt(1)):(size(x_pad,1)-pad_amt(1));
                        idx2 = (1+pad_amt(2)):(size(x_pad,2)-pad_amt(2));
                        idx3 = (1+pad_amt(3)):(size(x_pad,3)-pad_amt(3));
                        y_ref_unpad = y_ref_gpu(idx1,idx2,idx3);
                        y_mex_unpad = y_mex_gpu(idx1,idx2,idx3);

                        % --- Compute error ---
                        err = max(abs(y_mex_unpad(:) - y_ref_unpad(:)));
                        rms_err = sqrt(mean((y_mex_unpad(:) - y_ref_unpad(:)).^2));
                        rel_err = rms_err / (max(abs(y_ref_unpad(:))) + eps);

                        fprintf('%s\n',col('cyan',sprintf('  3D validation (unpadded): max = %.2e, RMS = %.2e, rel RMS = %.2e', ...
                            gather(err), gather(rms_err), gather(rel_err))));

                        % --- Pass/Fail ---
                        if strcmp(type_str, 'single')
                            pass = gather(err) < SINGLE_THRESH;
                            report_passfail(pass, 'single', hasCprintf);
                        else
                            pass = gather(err) < DOUBLE_THRESH;
                            report_passfail(pass, 'double', hasCprintf);
                        end

                        % --- Timing ---
                        speedup = t_ref / t_mex;
                        fprintf('%s\n',col('blue',sprintf('  Reference time: %.3fs | gauss3d_mex time: %.3fs | Speedup: %.2fx', t_ref, t_mex, speedup)));

                        % -- HISTOGRAM for quick outlier/precision insight --
                        show_hist(y_mex_unpad, 'Output histogram (single/double)', hasCprintf);

                        % --- Half precision test: only for single input ---
                        if strcmp(type_str, 'single')
                            t3 = tic;
                            if ischar(ksz) || isstring(ksz)
                                y_half_gpu = gauss3d_mex(x_pad_gpu, sigma, 'half');
                            else
                                y_half_gpu = gauss3d_mex(x_pad_gpu, sigma, kernel_sz, 'half');
                            end
                            t_half = toc(t3);

                            assert(isequal(size(y_half_gpu), size(x_pad_gpu)), 'Half output size mismatch.');
                            assert(strcmp(class(y_half_gpu), class(x_pad_gpu)), 'Half output class mismatch.');
                            assert(strcmp(classUnderlying(y_half_gpu), classUnderlying(x_pad_gpu)), 'Half output underlying class mismatch.');
                            fprintf('%s\n',col('yellow','  [Half precision] Output matches input type/class/shape.'));

                            y_half_unpad = y_half_gpu(idx1,idx2,idx3);

                            err_half = max(abs(y_half_unpad(:) - y_ref_unpad(:)));
                            rms_half = sqrt(mean((y_half_unpad(:) - y_ref_unpad(:)).^2));
                            rel_half = rms_half / (max(abs(y_ref_unpad(:))) + eps);

                            fprintf('%s\n',col('yellow',sprintf('  [Half] max = %.2e, RMS = %.2e, rel RMS = %.2e', ...
                                gather(err_half), gather(rms_half), gather(rel_half))));

                            pass_half = gather(err_half) < HALF_THRESH;
                            report_passfail(pass_half, 'half', hasCprintf);

                            speedup_half = t_ref / t_half;
                            fprintf('%s\n',col('blue',sprintf('  [Half] gauss3d_mex time: %.3fs | Speedup: %.2fx', t_half, speedup_half)));

                            show_hist(y_half_unpad, '[Half] Output histogram', hasCprintf);

                            % Print a comparison summary for a random voxel
                            linear_idx = randi(numel(y_half_unpad));
                            fprintf('%s\n', col('magenta', sprintf('    Voxel comparison [ref/single/half]: %g / %g / %g\n', ...
                                gather(y_ref_unpad(linear_idx)), gather(y_mex_unpad(linear_idx)), gather(y_half_unpad(linear_idx)))));
                        end

                    catch ME
                        if contains(ME.message, 'out of memory') || ...
                           contains(ME.identifier, 'parallel:gpu:array:OOM')
                            warning(col('red',sprintf('OOM: Skipping test for %s %s, sigma=%s, ksize=%s due to GPU memory.', ...
                                type_str, mat2str(sz), mat2str(sigma), kdesc)));
                            reset(gpuDevice); % Try to recover memory for next test
                            continue;
                        else
                            fprintf('%s\n',col('red',sprintf('  ERROR during test: %s', ME.message)));
                            rethrow(ME);
                        end
                    end

                    clear x_pad_gpu y_mex_gpu y_ref_gpu y_mex_unpad y_ref_unpad y_half_gpu y_half_unpad
                end
            end
        end
    end
end

function sz = odd_kernel_size(sigma)
    if isscalar(sigma)
        sigma = [sigma sigma sigma];
    end
    sz = 2*ceil(3*sigma) + 1;
    sz = max(sz, 3);
end

function report_passfail(pass, precision_str, hasCprintf)
    if pass
        s = sprintf('    PASS: %s precision\n', precision_str);
        print_color(s, 'blue', hasCprintf);
    else
        s = sprintf('    FAIL: %s precision\n', precision_str);
        print_color(s, 'red', hasCprintf);
    end
end

function out = colored_str(color, str, hasCprintf)
    % Return color-coded string if cprintf available, else normal string
    if hasCprintf
        out = evalc(['cprintf(''', color, ''','' ', 'str', ')']);
        out = out(1:end-1); % remove trailing newline from evalc
    else
        out = str;
    end
end

function print_color(str, color, hasCprintf)
    if hasCprintf
        cprintf(color, str);
    else
        fprintf('%s', str);
    end
end

function show_hist(arr, msg, hasCprintf)
    arr = gather(arr(:));
    if ~isempty(arr)
        fprintf('%s\n', colored_str('cyan', [msg, sprintf(': min=%g max=%g mean=%g', min(arr), max(arr), mean(arr))], hasCprintf));
        edges = linspace(min(arr), max(arr), 16);
        counts = histcounts(arr, edges);
        fprintf('%s\n', colored_str('cyan', ['      hist: ', num2str(counts)] , hasCprintf));
    end
end
