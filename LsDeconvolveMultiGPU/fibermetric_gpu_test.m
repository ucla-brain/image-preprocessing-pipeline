% benchmark_fibermetric_gpu_gpuonly.m
% Compare MATLAB's fibermetric (CPU) and your fibermetric_gpu (gpuArray only), both 'bright' and 'dark'
%clear; clc;

fprintf('\n==== Benchmark: fibermetric (CPU) vs fibermetric_gpu (gpuArray only) [bright/dark] ====\n');

% --- Generate a 3D test volume (already single, normalized, blurred)
sz = [128 128 32];
rng(42);
vol = imgaussfilt3(rand(sz, 'single'), 2); % smooth random texture
vol = single(mat2gray(vol));               % normalized [0,1]

% --- Set vesselness thickness/scale parameters to match MATLAB defaults
% alpha – weight for "blobness" suppression (Ra): typically 0.5 or 0.5²
% beta – weight for "plate-like" suppression (Rb): typically 0.5 or 0.5²
% gamma – normalization for the "second order structureness" (S): often set high, e.g., 15² or 500
sigma_from = 1; sigma_to = 4; sigma_step = 1;
alpha = 0.5; beta = 0.5; gamma = 500;



polarities = {'bright', 'dark'};

for i = 1:2
    pol = polarities{i};
    fprintf('\n--- Testing Polarity: %s ---\n', pol);

    % --- 1. MATLAB fibermetric (CPU, reference)
    t1 = tic;
    fm_cpu = fibermetric(vol, sigma_from:sigma_step:sigma_to, 'ObjectPolarity', pol);
    tcpu = toc(t1);

    % --- 2. fibermetric_gpu with gpuArray input ONLY
    gpu = gpuDevice(2);
    gvol = gpuArray(vol);
    t2 = tic;
    fm_gpu = fibermetric_gpu(gvol, sigma_from, sigma_to, sigma_step, alpha, beta, gamma, pol);
    wait(gpu);
    tgpu_gpu = toc(t2);             
    fm_gpu = gather(fm_gpu);

    % --- Quantitative comparison
    diff_gpu = abs(fm_cpu - fm_gpu);

    % --- Print timings and accuracy
    fprintf('  %-36s : %8.4f s\n', 'MATLAB fibermetric (CPU)', tcpu);
    fprintf('  %-36s : %8.4f s\n', 'fibermetric_gpu (gpuArray input)', tgpu_gpu);
    fprintf('  %-36s : %8.2fx\n', 'Speedup (GPU vs CPU)', tcpu / tgpu_gpu);

    fprintf('  %-36s : max %.3g, mean %.3g\n', 'Abs diff (CPU-GPU)', max(diff_gpu(:)), mean(diff_gpu(:)));
    fprintf('  %-36s : min %.5g, mean %.5g, max %.5g, nnz: %d\n', 'CPU output', min(fm_cpu(:)), mean(fm_cpu(:)), max(fm_cpu(:)), nnz(fm_cpu));
    fprintf('  %-36s : min %.5g, mean %.5g, max %.5g, nnz: %d\n', 'GPU output', min(fm_gpu(:)), mean(fm_gpu(:)), max(fm_gpu(:)), nnz(fm_gpu));

    % --- Summarize in a table
    resTable = table( ...
        [tcpu; tgpu_gpu], ...
        [NaN; max(diff_gpu(:))], ...
        [NaN; mean(diff_gpu(:))], ...
        'VariableNames', {'Time_sec', 'MaxAbsDiff_vsCPU', 'MeanAbsDiff_vsCPU'}, ...
        'RowNames', {'fibermetric_CPU', 'fibermetric_gpu_GPUin'});
    disp(resTable);

    % --- Visualization (center slice)
    midz = round(sz(3)/2);
    figure('Name',['Polarity: ', pol]);
    subplot(1,3,1); imagesc(fm_cpu(:,:,midz)); axis image; colorbar; title(['CPU (',pol,')']);
    subplot(1,3,2); imagesc(fm_gpu(:,:,midz)); axis image; colorbar; title(['GPU (',pol,')']);
    subplot(1,3,3); imagesc(diff_gpu(:,:,midz)); axis image; colorbar; title('Abs diff (CPU-GPU)');
end

fprintf('\nDone!\n');
