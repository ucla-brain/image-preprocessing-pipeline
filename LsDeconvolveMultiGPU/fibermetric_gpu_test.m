% benchmark_fibermetric_gpu_gpuonly.m
% Compare MATLAB's fibermetric (CPU) and your fibermetric_gpu (gpuArray only)
clear; clc;

fprintf('\n==== Benchmark: fibermetric (CPU) vs fibermetric\\_gpu (gpuArray only) ====\n');

% --- Generate a 3D test volume (already single, normalized, blurred)
sz = [128 128 32];
rng(42);
vol = imgaussfilt3(rand(sz, 'single'), 2); % smooth random texture
vol = single(mat2gray(vol));                       % normalized [0,1]

% --- Set vesselness thickness/scale parameters to match MATLAB defaults
sigma_from = 1; sigma_to = 4; sigma_step = 1;

% --- 1. MATLAB fibermetric (CPU, reference)
t1 = tic;
fm_cpu = fibermetric(vol, sigma_from:sigma_step:sigma_to);
tcpu = toc(t1);

% --- 2. fibermetric_gpu with gpuArray input ONLY
gpu = gpuDevice();
gvol = gpuArray(vol);
t2 = tic;
fm_gpu_gpu = fibermetric_gpu(gvol, sigma_from, sigma_to, sigma_step, 0.1, 5, 3.5e5, 'bright');
tgpu_gpu = toc(t2);             % time for GPU kernel only
wait(gpu);
fm_gpu_gpu = gather(fm_gpu_gpu);

% --- Quantitative comparison
diff_gpu = abs(fm_cpu - fm_gpu_gpu);

% --- Print timings and accuracy
fprintf('  %-36s : %8.4f s\n', 'MATLAB fibermetric (CPU)', tcpu);
fprintf('  %-36s : %8.4f s\n', 'fibermetric_gpu (gpuArray input)', tgpu_gpu);
fprintf('  %-36s : %8.2fx\n', 'Speedup (GPU vs CPU)', tcpu / tgpu_gpu);

fprintf('\nAccuracy (vs MATLAB CPU):\n');
fprintf('  %-36s : max %.3g, mean %.3g\n', 'gpuArray input', max(diff_gpu(:)), mean(diff_gpu(:)));

% --- Summarize in a table
resTable = table( ...
    [tcpu; tgpu_gpu], ...
    [NaN; max(diff_gpu(:))], ...
    [NaN; mean(diff_gpu(:))], ...
    'VariableNames', {'Time_sec', 'MaxAbsDiff_vsCPU', 'MeanAbsDiff_vsCPU'}, ...
    'RowNames', {'fibermetric_CPU', 'fibermetric_gpu_GPUin'});
disp(resTable);

fprintf('fm_gpu_gpu min: %.5g, max: %.5g, nnz: %d\n', min(fm_gpu_gpu(:)), max(fm_gpu_gpu(:)), nnz(fm_gpu_gpu));

fprintf('\nDone!\n');

% --- Visualization (center slice)
midz = round(sz(3)/2);
figure;
subplot(1,3,1); imagesc(fm_cpu(:,:,midz)); axis image; colorbar; title('MATLAB fibermetric');
subplot(1,3,2); imagesc(fm_gpu_gpu(:,:,midz)); axis image; colorbar; title('fibermetric\_gpu (GPU)');
subplot(1,3,3); imagesc(abs(fm_cpu(:,:,midz)-fm_gpu_gpu(:,:,midz))); axis image; colorbar; title('Abs diff (CPU vs GPU)');
