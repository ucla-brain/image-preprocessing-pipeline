function test_gauss3d_mex_features()
% Robust & colorful test harness for gauss3d_mex vs imgaussfilt3 (GPU/CPU fallback).
% Features: accuracy, type/class checks, perf timing, OOM/fallback, color output, and summary.

    g = gpuDevice(1);
    reset(g);

    % Color helper
    hasCprintf = exist('cprintf','file') == 2;
    col = @(c,str) colored_str(c,str,hasCprintf);

    szs = {[32, 64, 32], [512, 512, 512]};
    types = {@single, @double};
    sigma_tests = {2.5, [2.5 2.5 2.5], [0.5 0.5 2.5], 0.25, 7};
    ksize_tests = {'auto', 9, [9 11 15], 3, 41};
    SINGLE_THRESH = 5e-5;
    DOUBLE_THRESH = 1e-7;

    % Summary counters
    total = 0; pass = 0; fail = 0; skip = 0; oom = 0; ksizeerr = 0; pass_half = 0; fail_half = 0;

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

                    % Section header
                    print_section_header(total, type_str, sz, sigma, kdesc, col);

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

                        % gauss3d_mex single/double (standard)
                        t2 = tic;
                        y_mex_gpu = gauss3d_mex(x_pad_gpu, sigma, kernel_sz);
                        t_mex = toc(t2);

                        % Output checks and stats
                        assert(isequal(size(y_mex_gpu), size(x_pad_gpu)), 'Output size mismatch.');
                        assert(strcmp(class(y_mex_gpu), class(x_pad_gpu)), 'Output class mismatch.');
                        assert(strcmp(classUnderlying(y_mex_gpu), classUnderlying(x_pad_gpu)), 'Output underlying class mismatch.');
                        print_ok('Output matches input type/class/shape.', col);

                        % Unpad for fair comparison
                        [idx1, idx2, idx3] = unpad_indices(size(x_pad), pad_amt);
                        y_ref_unpad = y_ref_gpu(idx1,idx2,idx3);
                        y_mex_unpad = y_mex_gpu(idx1,idx2,idx3);

                        % Error metrics
                        err = max(abs(y_mex_unpad(:) - y_ref_unpad(:)));
                        rms_err = sqrt(mean((y_mex_unpad(:) - y_ref_unpad(:)).^2));
                        rel_err = rms_err / (max(abs(y_ref_unpad(:))) + eps);

                        % Histogram/stats
                        print_histogram(y_mex_unpad, y_ref_unpad, col, false);

                        % Pass/Fail
                        p = (strcmp(type_str,'single') && gather(err) < SINGLE_THRESH) || ...
                            (strcmp(type_str,'double') && gather(err) < DOUBLE_THRESH);
                        print_result(p, type_str, err, rms_err, rel_err, t_ref, t_mex, col);
                        if p, pass = pass+1; else, fail = fail+1; end

                        % ========== HALF PRECISION TEST =============
                        if strcmp(type_str,'single')
                            try
                                t_half = tic;
                                y_half_gpu = gauss3d_mex(x_pad_gpu, sigma, kernel_sz, 'half');
                                t_half_mex = toc(t_half);

                                assert(isequal(size(y_half_gpu), size(x_pad_gpu)), 'Half: Output size mismatch.');
                                assert(strcmp(class(y_half_gpu), class(x_pad_gpu)), 'Half: Output class mismatch.');

                                print_ok('[Half precision] Output matches input type/class/shape.', col);

                                y_half_unpad = y_half_gpu(idx1,idx2,idx3);
                                err_half = max(abs(y_half_unpad(:) - y_ref_unpad(:)));
                                rms_err_half = sqrt(mean((y_half_unpad(:) - y_ref_unpad(:)).^2));
                                rel_err_half = rms_err_half / (max(abs(y_ref_unpad(:))) + eps);

                                % Histogram/stats
                                print_histogram(y_half_unpad, y_ref_unpad, col, true);

                                p_half = gather(err_half) < SINGLE_THRESH;
                                print_result(p_half, 'half', err_half, rms_err_half, rel_err_half, [], t_half_mex, col, true);

                                if p_half, pass_half = pass_half+1; else, fail_half = fail_half+1; end
                            catch MEhalf
                                if contains(MEhalf.message, 'Kernel size exceeds')
                                    print_fail('[Half precision] Kernel size exceeds device limit.', col);
                                    ksizeerr = ksizeerr + 1;
                                else
                                    print_fail(['[Half precision] ERROR: ' MEhalf.message], col);
                                    fail_half = fail_half + 1;
                                end
                            end
                        end

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
    fprintf('\n%s\n',col('yellow', repmat('=',1,60)));
    fprintf('%s\n', col('magenta', 'TEST SUITE SUMMARY'));
    fprintf('%s %d\n', col('green', '    Total tests:      '), total);
    fprintf('%s %d\n', col('green', '    Passed:           '), pass);
    fprintf('%s %d\n', col('red',   '    Failed:           '), fail);
    fprintf('%s %d\n', col('blue',  '    OOM/Skipped:      '), oom);
    fprintf('%s %d\n', col('red',   '    Kernel size skip: '), ksizeerr);
    fprintf('%s %d\n', col('green', '    Half pass:        '), pass_half);
    fprintf('%s %d\n', col('red',   '    Half fail:        '), fail_half);
    fprintf('%s\n', col('yellow', repmat('=',1,60)));
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

function print_section_header(test_idx, type_str, sz, sigma, kdesc, col)
    fprintf('\n%s\n', col('yellow', repmat('=',1,60)));
    fprintf('%s\n', col('magenta', sprintf('Test %d: %s [%s]', test_idx, type_str, mat2str(sz))));
    fprintf('    %s %s\n', col('cyan','Sigma:   '), mat2str(sigma));
    fprintf('    %s %s\n', col('cyan','Kernel:  '), kdesc);
end

function print_ok(str, col)
    fprintf('%s %s\n', col('green', '  ✔️'), str);
end

function print_fail(str, col)
    fprintf('%s %s\n', col('red', '  ❌'), str);
end

function print_result(pass, type_str, err, rms_err, rel_err, t_ref, t_mex, col, is_half)
    if nargin < 9, is_half = false; end
    if pass
        mark = col('green', '✔️');
    else
        mark = col('red', '❌');
    end
    label = upper(type_str); if is_half, label = ['HALF (' label ')']; end
    fprintf('    %s %s: max=%.2e, RMS=%.2e, rel=%.2e\n', mark, label, err, rms_err, rel_err);
    if ~isempty(t_ref)
        fprintf('        Time: ref=%.3fs | mex=%.3fs\n', t_ref, t_mex);
    else
        fprintf('        Time: mex=%.3fs\n', t_mex);
    end
end

function print_histogram(y, yref, col, is_half)
    y = gather(y); yref = gather(yref);
    minv = min(y(:)); maxv = max(y(:)); meanv = mean(y(:));
    if is_half
        fprintf('      [Half] Output: min=%.4f max=%.4f mean=%.4f\n', minv, maxv, meanv);
    else
        fprintf('    Output: min=%.4f max=%.4f mean=%.4f\n', minv, maxv, meanv);
    end
    % Optionally, show a small histogram or a voxel sample
    idx = numel(y)/2 + 1;
    fprintf('        Voxel sample [ref/this]: %.6f / %.6f\n', yref(idx), y(idx));
end

function out = colored_str(color, str, hasCprintf)
    if hasCprintf
        out = evalc(['cprintf(''', color, ''','' ', 'str', ')']);
        out = out(1:end-1);
    else
        out = str;
    end
end
