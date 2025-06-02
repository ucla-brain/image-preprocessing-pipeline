function test_otf_gpu_mex_concise
fprintf('\n=== Testing otf_gpu_mex ===\n');
assert(exist('otf_gpu_mex', 'file') == 3, ...
    'otf_gpu_mex MEX not found on path. Please compile it first.');
try, g = gpuDevice; reset(g); end

results = {};

% Define concise tests
testcases = {
    {'Asym Gaussian',  [47 33 25], [96 88 80], [6 11 3.5], 'gaussian'}
    {'All-ones',       [2 2 2],    [4 4 4],    1,           'ones'}
    {'Rand noise',     [7 9 5],    [11 13 7],  [],          'rand'}
    {'Zero PSF',       [5 7 3],    [8 8 8],    0,           'zeros'}
    {'Identity shape', [9 8 6],    [9 8 6],    [],          'rand'}
    {'Permute Axes',   [6 8 7],    [6 8 7],    [],          'permute'}
};

for k = 1:numel(testcases)
    t = testcases{k};
    name = t{1}; psf_sz = t{2}; fft_shape = t{3}; par = t{4}; mode = t{5};
    switch mode
        case 'gaussian'
            sigma = par;
            center = (psf_sz+1)/2;
            [x,y,z] = ndgrid(1:psf_sz(1), 1:psf_sz(2), 1:psf_sz(3));
            psf = exp(-0.5*((x-center(1))/sigma(1)).^2 ...
                      -0.5*((y-center(2))/sigma(2)).^2 ...
                      -0.5*((z-center(3))/sigma(3)).^2 );
            psf = psf / sum(psf(:));
        case 'ones'
            psf = ones(psf_sz, 'single');
        case 'zeros'
            psf = zeros(psf_sz, 'single');
        case 'rand'
            rng(42); psf = rand(psf_sz, 'single');
        case 'permute'
            rng(99); psf = rand(psf_sz, 'single');
            psf = permute(psf, [2 3 1]);
        otherwise
            error('Unknown mode: %s', mode);
    end
    psf_shifted = ifftshift(psf);
    psf_shifted_gpu = gpuArray(single(psf_shifted));
    results{end+1} = run_one_otf_test(name, psf_shifted_gpu, fft_shape, psf_shifted);
    try, g = gpuDevice; reset(g); end
end
results = [results{:}];

fprintf('\n======= otf_gpu_mex summary =======\n');
for k = 1:numel(results)
    r = results(k);
    sym = pass_symbol(r.passed, true);
    fprintf('%-18s %s  rel.err: %.2g\n', r.name, sym, r.rel_error);
    if ~r.passed
        fprintf('   Max |real diff|: %.3g,  Max |imag diff|: %.3g\n', r.max_real, r.max_imag);
    end
end
if all([results.passed])
    fprintf('\n%s ALL TESTS PASSED for otf_gpu_mex!\n\n', pass_symbol(true, true));
else
    fprintf('\n%s SOME TESTS FAILED for otf_gpu_mex.\n\n', pass_symbol(false, true));
end
end

function result = run_one_otf_test(name, psf_shifted_gpu, fft_shape, psf_shifted_cpu)
rel_tol = 1e-6;
try
    [otf_mex, otf_conj_mex] = otf_gpu_mex(psf_shifted_gpu, fft_shape);
    otf_mex = arrayfun(@(r, i) complex(r, i), real(otf_mex), imag(otf_mex));
    otf_conj_mex = arrayfun(@(r, i) complex(r, i), real(otf_conj_mex), imag(otf_conj_mex));

    padsize = max(fft_shape - size(psf_shifted_cpu), 0);
    prepad = floor(padsize/2);
    postpad = padsize - prepad;
    psf_pad = padarray(single(psf_shifted_cpu), prepad, 0, 'pre');
    psf_pad = padarray(psf_pad, postpad, 0, 'post');
    psf_pad = psf_pad(1:fft_shape(1), 1:fft_shape(2), 1:fft_shape(3));
    otf_matlab = fftn(psf_pad);
    otf_matlab = arrayfun(@(r, i) complex(r, i), real(otf_matlab), imag(otf_matlab));
    otf_conj_matlab = conj(otf_matlab);
    otf_conj_matlab = arrayfun(@(r, i) complex(r, i), real(otf_conj_matlab), imag(otf_conj_matlab));

    err_otf = gather(norm(otf_matlab(:)-otf_mex(:)) / norm(otf_matlab(:)));
    err_conj = gather(norm(otf_conj_matlab(:)-otf_conj_mex(:)) / norm(otf_conj_matlab(:)));
    out_ok = all(isfinite(gather(otf_mex(:)))) && all(isfinite(gather(otf_conj_mex(:))));
    not_all_zeros = (norm(gather(otf_mex(:)))>0) || (norm(gather(otf_conj_mex(:)))>0) || (norm(gather(psf_shifted_gpu(:)))==0);
    passed = (err_otf < rel_tol) && (err_conj < rel_tol) && out_ok && not_all_zeros;

    max_real = max(abs(gather(real(otf_mex(:)) - real(otf_matlab(:)))));
    max_imag = max(abs(gather(imag(otf_mex(:)) - imag(otf_matlab(:)))));
    result = struct('name',name, 'rel_error', max(err_otf,err_conj), 'passed', passed, ...
        'max_real', max_real, 'max_imag', max_imag);
catch err
    fprintf('Test %-18s %s ERROR: %s\n', name, pass_symbol(false,false), err.message);
    result = struct('name',name, 'rel_error', nan, 'passed', false, ...
        'max_real', nan, 'max_imag', nan);
end
end

function sym = pass_symbol(passed, big)
if passed, sym = char([11035 65039 10004 65039]*(big) + [10004]*(~big)); % ✅ or ✔
else,      sym = char([10060 10060 10060 10060]*(big) + [10060]*(~big)); % ❌❌❌❌ or ❌
end
end
