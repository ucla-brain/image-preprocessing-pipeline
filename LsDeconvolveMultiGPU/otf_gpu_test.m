%% Test otf_gpu_mex vs. MATLAB/gpuArray for 3D OTF
fft_shape = [96 88 80];  % Nonsymmetric, large enough for GPU but fits in VRAM

% Asymmetric Gaussian PSF parameters
psf_sz = [47 33 25];
sigma = [6 11 3.5];
center = (psf_sz+1)/2;

% Make grid
[x,y,z] = ndgrid(1:psf_sz(1), 1:psf_sz(2), 1:psf_sz(3));
psf = exp(...
    -0.5*((x-center(1))/sigma(1)).^2 ...
    -0.5*((y-center(2))/sigma(2)).^2 ...
    -0.5*((z-center(3))/sigma(3)).^2 ...
    );
psf = psf / sum(psf(:));

% Center/shift PSF so its peak is at (1,1,1)
psf_shifted = ifftshift(psf);    % Now ready for direct FFT

psf_shifted_gpu = gpuArray(single(psf_shifted)); % Ensure single/gpuArray

fprintf('Running otf_gpu_mex (MEX CUDA)...\n');
gpuDevice; % clear/activate device
tic;
[otf_mex, otf_conj_mex] = otf_gpu_mex(psf_shifted_gpu, fft_shape);
mex_time = toc;

fprintf('Running MATLAB gpuArray version...\n');
tic;
% Pad/crop on GPU
padsize = max(fft_shape - psf_sz, 0);
prepad = floor(padsize/2);
postpad = padsize - prepad;
psf_pad = padarray(psf_shifted_gpu, prepad, 0, 'pre');
psf_pad = padarray(psf_pad, postpad, 0, 'post');
psf_pad = psf_pad(1:fft_shape(1), 1:fft_shape(2), 1:fft_shape(3)); % crop if needed
otf_matlab = fftn(psf_pad);
otf_conj_matlab = conj(otf_matlab);
matlab_time = toc;

%% Compare accuracy (relative error)
% MATLAB and MEX should agree to machine precision (single)
err_otf = gather(norm(otf_matlab(:)-otf_mex(:)) / norm(otf_matlab(:)));
err_conj = gather(norm(otf_conj_matlab(:)-otf_conj_mex(:)) / norm(otf_conj_matlab(:)));

fprintf('MEX time:   %.4f s\n', mex_time);
fprintf('MATLAB time: %.4f s\n', matlab_time);
fprintf('Relative error OTF:       %.3g\n', err_otf);
fprintf('Relative error OTF_conj:  %.3g\n', err_conj);

% Show a slice to confirm visually
figure;
subplot(1,2,1); imagesc(abs(gather(otf_mex(:,:,round(end/2))))); axis image; colorbar; title('OTF_{MEX} abs (mid-slice)');
subplot(1,2,2); imagesc(abs(gather(otf_matlab(:,:,round(end/2))))); axis image; colorbar; title('OTF_{MATLAB} abs (mid-slice)');
