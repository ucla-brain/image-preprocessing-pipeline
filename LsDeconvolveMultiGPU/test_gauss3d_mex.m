function test_gauss3d_mex_features()
% Robust & colorful test harness for gauss3d_mex vs imgaussfilt3 (GPU/CPU fallback).
% Single-precision only. Shows performance, speedup, pass/fail, and summary.

    g = gpuDevice(1);
    reset(g);

    % Color helper
    hasCprintf = exist('cprintf','file') == 2;
    col = @(c,str) colored_str(c,str,hasCprintf);

    szs = {[32, 64, 32], [512, 512, 512]};
    types = {@single};   % Single only!
    sigma_tests = {2.5, [2.5 2.5 2.5], [0.5 0.5 2.5], 0.25, 8};
    ksize_tests = {'auto', 9, [9 11 15], 3, 41};
    SINGLE_THRESH = 5e-5;

    % Summary counters
    total = 0; pass = 0; fail = 0; skip = 0; oom = 0; ksizeerr = 0;

    % Print summary header
    fprintf('%-4s %-5s %-11s %-20s %-14s %-16s %-8s %-8s %-8s %-8s %-10s\n', ...
        'PF', 'Test', 'Type', 'Size', 'Sigma', 'Kernel', 'maxErr', 'RMS', 'relErr', 'mex(s)', 'Speedup');

    for ityp = 1:numel(types)
        for isz = 1:numel(szs)
            sz = szs{isz};
            T = types{ityp};
            type_str = func2str(T);

            for isig = 1:numel(sigma_tests)
                sigma = sigma_tests{isig};
                for iksz = 1:numel(ksize_tests)
                    total = total + 1;
                    ksz = ksize_tests{iksz};
                    if ischar(ksz) || isstring(ksz)
                        kdesc = char(ksz);
                    else
                        kdesc = mat2str(ksz);
                    end

                    % Kernel size for padding
                    if ischar(ksz) || isstring(ksz)
                        kernel_sz = odd_kernel_size(sigma);
                    else
                        kernel_sz = ksz;
                        if isscalar(kernel_sz)
                            kernel_sz = repmat(kernel_sz,1,3);
                        end
                    end
                    pad_amt = floor(kernel_sz / 2);

                    rng(0);
                    x = rand(sz, type_str);
                    x = x ./ max(x(:));
                    x_pad = padarray(x, pad_amt, 'replicate', 'both');
                    x_pad_gpu = gpuArray(x_pad);
                    opts = {'Padding', 'replicate', 'FilterSize', kernel_sz, 'FilterDomain', 'spatial'};

                    try
                        % Reference (imgaussfilt3, fallback to CPU)
                        t1 = tic;
                        try
                            y_ref_gpu = imgaussfilt3(x_pad_gpu, sigma, opts{:});
                        catch ME1
                            if contains(ME1.message, 'fftn') || ...
                               contains(ME1.message, 'FFT') || ...
                               contains(ME1.message, 'out of memory') || ...
                               contains(ME1.identifier, 'parallel:gpu:array:OOM')
                                warning(col('blue','    imgaussfilt3 GPU failed (FFT or OOM), using CPU reference.'));
                                y_ref_gpu = gpuArray(imgaussfilt3(gather(x_pad_gpu), sigma, opts{:}));
                            else
                                rethrow(ME1);
                            end
                        end
                        t_ref = toc(t1);

                        % gauss3d_mex single (standard)
                        t2 = tic;
                        y_mex_gpu = gauss3d_mex(x_pad_gpu, sigma, kernel_sz);
                        t_mex = toc(t2);

                        % Output checks and stats
                        assert(isequal(size(y_mex_gpu), size(x_pad_gpu)), 'Output size mismatch.');
                        assert(strcmp(class(y_mex_gpu), class(x_pad_gpu)), 'Output class mismatch.');
                        assert(strcmp(classUnderlying(y_mex_gpu), classUnderlying(x_pad_gpu)), 'Output underlying class mismatch.');

                        % Unpad for fair comparison
                        [idx1, idx2, idx3] = unpad_indices(size(x_pad), pad_amt);
                        y_ref_unpad = y_ref_gpu(idx1,idx2,idx3);
                        y_mex_unpad = y_mex_gpu(idx1,idx2,idx3);

                        % Error metrics
                        err = max(abs(y_mex_unpad(:) - y_ref_unpad(:)));
                        rms_err = sqrt(mean((y_mex_unpad(:) - y_ref_unpad(:)).^2));
                        rel_err = rms_err / (max(abs(y_ref_unpad(:))) + eps);

                        % Speedup (relative to reference)
                        speedup = t_ref / t_mex;
                        if isfinite(speedup)
                            if speedup > 1
                                speed_str = sprintf('+%.0f%%', 100*(speedup-1));
                            else
                                speed_str = sprintf('-%.0f%%', 100*(1-speedup));
                            end
                        else
                            speed_str = 'N/A';
                        end

                        % Pass/Fail
                        p = (strcmp(type_str,'single') && gather(err) < SINGLE_THRESH);

                        % Print single-line result (left-aligned)
                        if p
                            pf = col('green', '✔️');
                            pass = pass+1;
                        else
                            pf = col('red', '❌');
                            fail = fail+1;
                        end

                        fprintf('%-4s %-5d %-11s %-20s %-14s %-16s %-8.2e %-8.2e %-8.2e %-8.3f %-10s\n', ...
                            pf, total, type_str, mat2str(sz), mat2str(sigma), kdesc, err, rms_err, rel_err, t_mex, speed_str);

                    catch ME
                        % OOM or kernel size errors
                        if contains(ME.message, 'out of memory') || ...
                           contains(ME.identifier, 'parallel:gpu:array:OOM')
                            print_fail('OOM: Skipping due to GPU memory.', col);
                            oom = oom+1; reset(gpuDevice); continue;
                        elseif contains(ME.message, 'Kernel size exceeds')
                            print_fail('Kernel size exceeds device limit.', col);
                            ksizeerr = ksizeerr + 1;
                            continue;
                        else
                            print_fail(['ERROR: ' ME.message], col);
                            fail = fail+1;
                            continue;
                        end
                    end

                    clear x_pad_gpu y_mex_gpu y_ref_gpu y_mex_unpad y_ref_unpad
                end
            end
        end
    end

    % --- Suite summary ---
    fprintf('\n%s\n',col('yellow', repmat('=',1,80)));
    fprintf('%s\n', col('magenta', 'TEST SUITE SUMMARY'));
    fprintf('%-24s %-8d\n', col('green', 'Total tests:'), total);
    fprintf('%-24s %-8d\n', col('green', 'Passed:'), pass);
    fprintf('%-24s %-8d\n', col('red',   'Failed:'), fail);
    fprintf('%-24s %-8d\n', col('blue',  'OOM/Skipped:'), oom);
    fprintf('%-24s %-8d\n', col('red',   'Kernel size skip:'), ksizeerr);
    fprintf('%s\n', col('yellow', repmat('=',1,80)));
end

function [idx1, idx2, idx3] = unpad_indices(sz, pad_amt)
    idx1 = (1+pad_amt(1)):(sz(1)-pad_amt(1));
    idx2 = (1+pad_amt(2)):(sz(2)-pad_amt(2));
    idx3 = (1+pad_amt(3)):(sz(3)-pad_amt(3));
end

function sz = odd_kernel_size(sigma)
    if isscalar(sigma)
        sigma = [sigma sigma sigma];
    end
    sz = 2*ceil(3*sigma) + 1;
    sz = max(sz, 3);
end

function print_fail(str, col)
    fprintf('%s %s\n', col('red', '  ❌'), str);
end

function out = colored_str(color, str, hasCprintf)
    if hasCprintf
        out = evalc(['cprintf(''', color, ''','' ', 'str', ')']);
        out = out(1:end-1);
    else
        out = str;
    end
end
