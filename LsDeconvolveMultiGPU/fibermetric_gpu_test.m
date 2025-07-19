% benchmark_fibermetric_gpu.m
% Compare MATLAB's fibermetric and your fibermetric_gpu (CPU and GPU modes)
clear; clc;

fprintf('\n==== Benchmark: fibermetric vs fibermetric\\_gpu (CPU & GPU) ====\n');

% --- Generate a 3D test volume (already single, normalized, blurred)
sz = [128 128 32];
rng(42);
vol = imgaussfilt3(rand(sz, 'single'), 2); % smooth random texture
vol = single(mat2gray(vol));                       % normalized [0,1]

% --- Set vesselness thickness/scale parameters to match MATLAB defaults
sigma_from = 1; sigma_to = 4; sigma_step = 1;

% --- 1. MATLAB fibermetric (CPU)
t1 = tic;
fm_cpu = fibermetric(vol, sigma_from:sigma_step:sigma_to);
tcpu = toc(t1);

% --- 2. fibermetric_gpu with CPU input
t2 = tic;
fm_gpu_cpu = fibermetric_gpu(vol, sigma_from, sigma_to, sigma_step);
tgpu_cpu = toc(t2);

% --- 3. fibermetric_gpu with gpuArray input
gvol = gpuArray(vol);
t3 = tic;
fm_gpu_gpu = fibermetric_gpu(gvol, sigma_from, sigma_to, sigma_step);
tgpu_gpu = toc(t3);        % time includes GPU computation only
fm_gpu_gpu = gather(fm_gpu_gpu); % gather after timing for pure GPU kernel time

% --- Quantitative comparison
diff_cpu_vs_gpu = abs(fm_cpu - fm_gpu_cpu);
diff_cpu_vs_gpuarray = abs(fm_cpu - fm_gpu_gpu);

% --- Print timings and accuracy
fprintf('  %-32s : %8.4f s\n', 'MATLAB fibermetric (CPU)', tcpu);
fprintf('  %-32s : %8.4f s\n', 'fibermetric_gpu (CPU input)', tgpu_cpu);
fprintf('  %-32s : %8.4f s\n', 'fibermetric_gpu (gpuArray input)', tgpu_gpu);
fprintf('  %-32s : %8.2fx\n', 'Speedup (GPU vs CPU)', tcpu / tgpu_gpu);

fprintf('\nAccuracy (vs MATLAB):\n');
fprintf('  %-32s : max %.3g, mean %.3g\n', 'CPU input', max(diff_cpu_vs_gpu(:)), mean(diff_cpu_vs_gpu(:)));
fprintf('  %-32s : max %.3g, mean %.3g\n', 'gpuArray input', max(diff_cpu_vs_gpuarray(:)), mean(diff_cpu_vs_gpuarray(:)));

% --- Summarize in a table
resTable = table( ...
    [tcpu; tgpu_cpu; tgpu_gpu], ...
    [NaN; max(diff_cpu_vs_gpu(:)); max(diff_cpu_vs_gpuarray(:))], ...
    [NaN; mean(diff_cpu_vs_gpu(:)); mean(diff_cpu_vs_gpuarray(:))], ...
    'VariableNames', {'Time_sec', 'MaxAbsDiff_vsCPU', 'MeanAbsDiff_vsCPU'}, ...
    'RowNames', {'fibermetric_CPU', 'fibermetric_gpu_CPUin', 'fibermetric_gpu_GPUin'});
disp(resTable);

fprintf('\nDone!\n');

% --- Visualization (center slice)
midz = round(sz(3)/2);
figure;
subplot(2,2,1); imagesc(fm_cpu(:,:,midz)); axis image; colorbar; title('MATLAB fibermetric');
subplot(2,2,2); imagesc(fm_gpu_cpu(:,:,midz)); axis image; colorbar; title('fibermetric\_gpu (CPU)');
subplot(2,2,3); imagesc(fm_gpu_gpu(:,:,midz)); axis image; colorbar; title('fibermetric\_gpu (GPU)');
subplot(2,2,4); imagesc(abs(fm_cpu(:,:,midz)-fm_gpu_gpu(:,:,midz))); axis image; colorbar; title('Abs diff (CPU vs GPU)');
