% benchmark_fibermetric_gpu_gpuonly.m
% Compare MATLAB's fibermetric (CPU) and your fibermetric_gpu (gpuArray only), both 'bright' and 'dark'
%clear; clc;

fprintf('\n==== Benchmark: fibermetric (CPU) vs fibermetric_gpu (gpuArray only) [bright/dark] ====\n');

sz = [128 128 32];
rng(42);
vol = imgaussfilt3(rand(sz, 'single'), 2);
vol = single(mat2gray(vol));
polarities = {'dark', 'bright'};

sigma_from = 1; sigma_to = 4; sigma_step = 1;
alpha = 0.5; beta = 0.5; gamma = 15;
pol = 'bright'; % or 'dark'—do both if you want
structureSensitivity = eps('single');

% Find alpha, beta, and gamma with optimization
%fm_cpu = fibermetric(vol, sigma_from:sigma_step:sigma_to, 'ObjectPolarity', pol, 'StructureSensitivity',structureSensitivity);
%gpu = gpuDevice(2);
%gvol = gpuArray(vol);
%loss_fun = @(params) vesselness_param_loss(params, gvol, sigma_from, sigma_to, sigma_step, pol, structureSensitivity, fm_cpu);
%nvars = 3; % [alpha, beta, gamma]
%fprintf('\nOptimizing alpha, beta, gamma for best match (particleswarm)...\n');
%opts = optimoptions('particleswarm', ...
%    'Display','iter', ...
%    'MaxIterations', Inf, ...     % Increase for better results
%    'SwarmSize', 10, ...          % Increase for thorough search
%    'UseParallel', false);        % If you have Parallel Toolbox
%lb = [0, 0, 0]; % Lower bounds for alpha, beta, gamma
%ub = [1, 1, 1000]; % Upper bounds for alpha, beta, gamma
%[xopt, fval, exitflag, output] = particleswarm(loss_fun, nvars, lb, ub, opts);
%alpha = xopt(1);
%beta  = xopt(2);
%gamma = xopt(3);
%fprintf('\nOptimal params: alpha=%.4f, beta=%.4f, gamma=%.2f (mean diff=%.5g)\n', alpha, beta, gamma, fval);

for i = 1:2
    pol = polarities{i};
    fprintf('\n--- Testing Polarity: %s ---\n', pol);

    % --- 1. MATLAB fibermetric (CPU, reference)
    t1 = tic;
    fm_cpu = fibermetric(vol, sigma_from:sigma_step:sigma_to, 'ObjectPolarity', pol);
    tcpu = toc(t1);

    % --- 2. fibermetric_gpu with gpuArray input ONLY
    gvol = gpuArray(vol);
    t2 = tic;
    fm_gpu = fibermetric_gpu(gvol, sigma_from, sigma_to, sigma_step, alpha, beta, gamma, pol, structureSensitivity);
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

function loss = vesselness_param_loss(params, gvol, sigma_from, sigma_to, sigma_step, pol, structureSensitivity, fm_cpu)
    alpha = params(1); beta = params(2); gamma = params(3);
    try
        fm_gpu = fibermetric_gpu(gvol, sigma_from, sigma_to, sigma_step, alpha, beta, gamma, pol, structureSensitivity);
        %fm_gpu = fm_gpu / max(fm_gpu, [], 'all');
        fm_gpu = gather(fm_gpu);
        loss = mean(abs(fm_cpu - fm_gpu), 'all');
        if isnan(loss) || isinf(loss)
            loss = 1e6;
        end
    catch
        loss = 1e6;
    end
end
