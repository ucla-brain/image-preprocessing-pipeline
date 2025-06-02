function test_otf_gpu_mex
fprintf('\n=== Testing otf_gpu_mex ===\n');
results = [];

%% 1. Synthetic Gaussian Test (main)
fft_shape = [96 88 80];  % nonsymmetric, large
psf_sz = [47 33 25];
sigma = [6 11 3.5];
center = (psf_sz+1)/2;
[x,y,z] = ndgrid(1:psf_sz(1), 1:psf_sz(2), 1:psf_sz(3));
psf = exp(...
    -0.5*((x-center(1))/sigma(1)).^2 ...
    -0.5*((y-center(2))/sigma(2)).^2 ...
    -0.5*((z-center(3))/sigma(3)).^2 );
psf = psf / sum(psf(:));
psf_shifted = ifftshift(psf);
psf_shifted_gpu = gpuArray(single(psf_shifted));
results(end+1) = run_one_otf_test('Asym Gaussian', psf_shifted_gpu, fft_shape);

%% 2. Edge test: All-ones input, minimal size (should yield sum everywhere)
psf = ones(2,2,2,'single','gpuArray');
fft_shape = [4 4 4];
psf_shifted_gpu = ifftshift(psf); % shift still applied
results(end+1) = run_one_otf_test('All-ones 2x2x2→4x4x4', psf_shifted_gpu, fft_shape);

%% 3. Random noise, odd shape
psf = rand(7,9,5,'single','gpuArray');
fft_shape = [11 13 7];
psf_shifted_gpu = ifftshift(psf);
results(end+1) = run_one_otf_test('Rand noise 7x9x5→11x13x7', psf_shifted_gpu, fft_shape);

%% 4. Zero input (output must be all zeros)
psf = zeros(5,7,3,'single','gpuArray');
fft_shape = [8 8 8];
psf_shifted_gpu = ifftshift(psf);
results(end+1) = run_one_otf_test('Zero PSF', psf_shifted_gpu, fft_shape);

%% 5. Identity: shape-matched, no padding
psf = rand(9,8,6,'single','gpuArray');
fft_shape = size(psf);
psf_shifted_gpu = ifftshift(psf);
results(end+1) = run_one_otf_test('Identity shape', psf_shifted_gpu, fft_shape);

%% Summary
fprintf('\n');
fprintf('======== SUMMARY: otf_gpu_mex ========\n');
all_passed = all([results.passed]);
for k = 1:numel(results)
    r = results(k);
    sym = pass_symbol(r.passed, true);
    fprintf('%-20s %s  (rel.err: %.2g, time gain: %.1f%%)\n', ...
        r.name, sym, r.rel_error, r.perf_gain);
end
if all_passed
    fprintf('\n%s ALL TESTS PASSED for otf_gpu_mex!\n\n', pass_symbol(true, true));
else
    fprintf('\n%s SOME TESTS FAILED for otf_gpu_mex. See details above.\n\n', pass_symbol(false, true));
end
end

function result = run_one_otf_test(name, psf_shifted_gpu, fft_shape)
rel_tol = 1e-6;
try
    gpuDevice;
    t_mex = tic;
    [otf_mex, otf_conj_mex] = otf_gpu_mex(psf_shifted_gpu, fft_shape);
    otf_mex = arrayfun(@(r, i) complex(r, i), real(otf_mex), imag(otf_mex));
    otf_conj_mex = arrayfun(@(r, i) complex(r, i), real(otf_conj_mex), imag(otf_conj_mex));
    mex_time = toc(t_mex);

    t_matlab = tic;
    psf_sz = size(psf_shifted_gpu);
    padsize = max(fft_shape - psf_sz, 0);
    prepad = floor(padsize/2);
    postpad = padsize - prepad;
    psf_pad = padarray(psf_shifted_gpu, prepad, 0, 'pre');
    psf_pad = padarray(psf_pad, postpad, 0, 'post');
    psf_pad = psf_pad(1:fft_shape(1), 1:fft_shape(2), 1:fft_shape(3));
    otf_matlab = fftn(psf_pad);
    otf_matlab = arrayfun(@(r, i) complex(r, i), real(otf_matlab), imag(otf_matlab));
    otf_conj_matlab = conj(otf_matlab);
    otf_conj_matlab = arrayfun(@(r, i) complex(r, i), real(otf_conj_matlab), imag(otf_conj_matlab));
    matlab_time = toc(t_matlab);

    % Main test: normed relative error, all outputs finite, not all zeros
    err_otf = gather(norm(otf_matlab(:)-otf_mex(:)) / norm(otf_matlab(:)));
    err_conj = gather(norm(otf_conj_matlab(:)-otf_conj_mex(:)) / norm(otf_conj_matlab(:)));
    out_ok = all(isfinite(gather(otf_mex(:)))) && all(isfinite(gather(otf_conj_mex(:))));
    not_all_zeros = (norm(gather(otf_mex(:)))>0) || (norm(gather(otf_conj_mex(:)))>0) || (norm(gather(psf_shifted_gpu(:)))==0);
    passed = (err_otf < rel_tol) && (err_conj < rel_tol) && out_ok && not_all_zeros;

    perf_gain = (matlab_time - mex_time)/matlab_time*100;
    fprintf('Test %-25s %s  OTF rel.err %.2g | conj %.2g | perf gain: %+5.1f%%\n', ...
        name, pass_symbol(passed,false), err_otf, err_conj, perf_gain);

    result = struct('name',name, 'rel_error', max(err_otf,err_conj), 'perf_gain', perf_gain, 'passed', passed);
catch err
    fprintf('Test %-25s %s  ERROR: %s\n', name, pass_symbol(false,false), err.message);
    result = struct('name',name, 'rel_error', nan, 'perf_gain', nan, 'passed', false);
end
end

function sym = pass_symbol(passed, big)
if passed, sym = char([11035 65039 10004 65039]*(big) + [10004]*(~big)); % ✅ or ✔
else,      sym = char([10060 10060 10060 10060]*(big) + [10060]*(~big)); % ❌❌❌❌ or ❌
end
end
