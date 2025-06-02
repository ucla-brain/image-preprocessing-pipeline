%% Test otf_gpu_mex (headless CLI)

fft_shape = [96 88 80];  % Nonsymmetric, fairly large
psf_sz = [47 33 25];
sigma = [6 11 3.5];

center = (psf_sz+1)/2;
[x,y,z] = ndgrid(1:psf_sz(1), 1:psf_sz(2), 1:psf_sz(3));
psf = exp(...
    -0.5*((x-center(1))/sigma(1)).^2 ...
    -0.5*((y-center(2))/sigma(2)).^2 ...
    -0.5*((z-center(3))/sigma(3)).^2 ...
    );
psf = psf / sum(psf(:));
psf_shifted = ifftshift(psf);
psf_shifted_gpu = gpuArray(single(psf_shifted));

% Run MEX version
gpuDevice;
t_mex = tic;
[otf_mex, otf_conj_mex] = otf_gpu_mex(psf_shifted_gpu, fft_shape);
otf_mex = arrayfun(@(r, i) complex(r, i), real(otf_mex), imag(otf_mex));
otf_conj_mex = arrayfun(@(r, i) complex(r, i), real(otf_conj_mex), imag(otf_conj_mex));

mex_time = toc(t_mex);

% Run MATLAB version
t_matlab = tic;
padsize = max(fft_shape - psf_sz, 0);
prepad = floor(padsize/2);
postpad = padsize - prepad;
psf_pad = padarray(psf_shifted_gpu, prepad, 0, 'pre');
psf_pad = padarray(psf_pad, postpad, 0, 'post');
psf_pad = psf_pad(1:fft_shape(1), 1:fft_shape(2), 1:fft_shape(3)); % crop if needed
otf_matlab = fftn(psf_pad);
otf_matlab = arrayfun(@(r, i) complex(r, i), real(otf_matlab), imag(otf_matlab));
otf_conj_matlab = conj(otf_matlab);
otf_conj_matlab = arrayfun(@(r, i) complex(r, i), real(otf_conj_matlab), imag(otf_conj_matlab));
matlab_time = toc(t_matlab);

% Accuracy checks
rel_tol = 1e-6;
err_otf = gather(norm(otf_matlab(:)-otf_mex(:)) / norm(otf_matlab(:)));
err_conj = gather(norm(otf_conj_matlab(:)-otf_conj_mex(:)) / norm(otf_conj_matlab(:)));

pass_otf = err_otf < rel_tol;
pass_conj = err_conj < rel_tol;
check = @(b) char([11035 65039 10004 65039]*(b) + [10060]*(~b)); % ✅ or ❌

% Performance gain
perf_gain = (matlab_time - mex_time)/matlab_time*100;

% CLI Table report
fprintf('\n');
fprintf('┌───────────────────────┬──────────────┬──────────┬────────┐\n');
fprintf('│ Test                  │  MEX value   │ MATLAB   │ Pass?  │\n');
fprintf('├───────────────────────┼──────────────┼──────────┼────────┤\n');
fprintf('│ OTF rel. error        │ %10.3g   │    -     │  %s   │\n', err_otf, check(pass_otf));
fprintf('│ OTF_conj rel. error   │ %10.3g   │    -     │  %s   │\n', err_conj, check(pass_conj));
fprintf('│ Runtime (seconds)     │ %10.4f   │ %7.4f  │    -   │\n', mex_time, matlab_time);
fprintf('│ Perf. gain (vs MATLAB)│ %10.2f %%  │    -     │    -   │\n', perf_gain);
fprintf('└───────────────────────┴──────────────┴──────────┴────────┘\n');
if pass_otf && pass_conj
    fprintf('\n%s Test passed: outputs match within tolerance (%.0e)\n', check(true), rel_tol);
else
    fprintf('\n%s Test failed: see above errors.\n', check(false));
end
