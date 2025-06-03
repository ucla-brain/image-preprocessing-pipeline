% Test edge taper: MATLAB edgetaper (CPU 2D) vs. GPU CUDA (3D)

% --- Make test image and PSF ---
sz = [64, 64, 32];
A = rand(sz, 'single');
PSF = fspecial('gaussian', 15, 2);           % 2D PSF
PSF3 = zeros(15, 15, 7, 'single');           % Make 3D PSF for GPU
for z = 1:7
    PSF3(:,:,z) = fspecial('gaussian', 15, 2) * exp(-((z-4).^2)/6);
end
PSF3 = PSF3 / sum(PSF3(:));

% --- 2D edge taper on CPU (middle slice) ---
A2D = A(:,:,round(sz(3)/2));
et_cpu = edgetaper(A2D, PSF);

% --- 3D edge taper with CUDA on GPU ---
Ag = gpuArray(A);
PSFg = gpuArray(PSF3);
et_gpu = edge_taper_auto(Ag, PSFg);
et_gpu_cpu = gather(et_gpu);

% --- Show center slice: CPU vs. GPU ---
figure;
subplot(1,3,1);
imagesc(A2D); axis image; colorbar; title('Original slice');
subplot(1,3,2);
imagesc(et_cpu); axis image; colorbar; title('CPU edgetaper (2D)');
subplot(1,3,3);
imagesc(et_gpu_cpu(:,:,round(sz(3)/2))); axis image; colorbar; title('GPU edge taper (3D)');

% --- Quantitative difference ---
diff = et_cpu - et_gpu_cpu(:,:,round(sz(3)/2));
fprintf('Max abs difference in central slice: %g\n', max(abs(diff(:))));
fprintf('Mean abs difference in central slice: %g\n', mean(abs(diff(:))));
