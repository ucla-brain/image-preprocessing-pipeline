function test_otf_gpu_mex_debug
fprintf('\n=== DEBUGGING otf_gpu_mex PADDING ===\n');
results = {};

% 1. Small fingerprint pattern, shape 2x2x2
psf = reshape(single(1:8), [2 2 2]);
fft_shape = [4 4 4];
psf_shifted_gpu = ifftshift(psf);
results{end+1} = run_one_otf_debug_test('Fingerprint 2x2x2→4x4x4', psf_shifted_gpu, fft_shape);

% 2. Asymmetric shape with unique numbers
psf = reshape(single(1:30), [5 3 2]);
fft_shape = [7 5 4];
psf_shifted_gpu = ifftshift(psf);
results{end+1} = run_one_otf_debug_test('Asym 5x3x2→7x5x4', psf_shifted_gpu, fft_shape);

% 3. Random
psf = rand(4,3,5, 'single', 'gpuArray');
fft_shape = [8 8 8];
psf_shifted_gpu = ifftshift(psf);
results{end+1} = run_one_otf_debug_test('Random 4x3x5→8x8x8', psf_shifted_gpu, fft_shape);

% 4. Identity: shape-matched, no padding
psf = reshape(single(1:60), [5 4 3]);
fft_shape = [5 4 3];
psf_shifted_gpu = ifftshift(psf);
results{end+1} = run_one_otf_debug_test('Identity 5x4x3', psf_shifted_gpu, fft_shape);

% 5. All-zeros
psf = zeros(3,2,4, 'single', 'gpuArray');
fft_shape = [5 5 5];
psf_shifted_gpu = ifftshift(psf);
results{end+1} = run_one_otf_debug_test('All zeros', psf_shifted_gpu, fft_shape);

results = [results{:}];

fprintf('\n======== DEBUG PAD SUMMARY: otf_gpu_mex ========\n');
all_passed = all([results.passed]);
for k = 1:numel(results)
    r = results(k);
    sym = pass_symbol(r.passed, true);
    fprintf('%-22s %s  (pad rel.err: %.2g)\n', r.name, sym, r.rel_error);
end
if all_passed
    fprintf('\n%s ALL PADDING TESTS PASSED for otf_gpu_mex!\n\n', pass_symbol(true, true));
else
    fprintf('\n%s SOME PADDING TESTS FAILED. Investigate above.\n\n', pass_symbol(false, true));
end

end

function result = run_one_otf_debug_test(name, psf_shifted_gpu, fft_shape)
try
    % Run the MEX with debug mode (3 outputs: otf_mex, otf_conj_mex, pad_mex)
    [~, ~, pad_mex] = otf_gpu_mex(psf_shifted_gpu, fft_shape);

    % MATLAB reference padding (identical to your main workflow)
    psf_sz = size(psf_shifted_gpu);
    padsize = max(fft_shape - psf_sz, 0);
    prepad = floor(padsize/2);
    postpad = padsize - prepad;
    psf_pad = padarray(psf_shifted_gpu, prepad, 0, 'pre');
    psf_pad = padarray(psf_pad, postpad, 0, 'post');
    psf_pad = psf_pad(1:fft_shape(1), 1:fft_shape(2), 1:fft_shape(3));

    % Compare: normed error
    pad_mex_cpu = gather(pad_mex);
    psf_pad_cpu = gather(psf_pad);

    rel_err = norm(psf_pad_cpu(:) - pad_mex_cpu(:)) / max(norm(psf_pad_cpu(:)), eps('single'));
    % Elementwise check for nonzeros (useful for zeros test)
    passed = (rel_err < 1e-6) || (all(psf_pad_cpu(:) == 0) && all(pad_mex_cpu(:) == 0));

    if ~passed
        fprintf('\nDEBUG: mismatch in "%s"\n', name);
        fprintf('MATLAB pad slice(:,:,1):\n'); disp(psf_pad_cpu(:,:,1));
        fprintf('CUDA pad slice(:,:,1):\n'); disp(pad_mex_cpu(:,:,1));
    end

    fprintf('Debug %-20s %s  pad rel.err %.2g\n', name, pass_symbol(passed, false), rel_err);

    result = struct('name', name, 'rel_error', rel_err, 'passed', passed);
catch err
    fprintf('Debug %-20s %s  ERROR: %s\n', name, pass_symbol(false, false), err.message);
    result = struct('name', name, 'rel_error', nan, 'passed', false);
end
end

function sym = pass_symbol(passed, big)
if passed, sym = char([11035 65039 10004 65039]*(big) + [10004]*(~big)); % ✅ or ✔
else,      sym = char([10060 10060 10060 10060]*(big) + [10060]*(~big)); % ❌❌❌❌ or ❌
end
end
