gpu = gpuDevice(2);
reset(gpu);
fprintf('\n==== Benchmark: fibermetric (CPU) vs fibermetric_gpu (gpuArray only) [bright/dark, frangi/sato/meijering/jerman] ====\n');

% ===== Set your reference method here ('frangi' or 'sato') =====
referenceMethod = 'frangi'; % 'sato' or 'frangi'
targetMethods = setdiff({'frangi','sato'}, referenceMethod, 'stable');
targetMethod = targetMethods{1};

volOrig = im2single(tiffreadVolume("test_volume.tif"));
polarities = {'bright', 'dark'};

sigma_from = 1; sigma_to = 7; sigma_step = 1;
if strcmp(referenceMethod, 'frangi')
    alpha_init = 0.7144561; beta_init = 0.8752375; structureSensitivity_init = 0.0995564;
else
    alpha_init = 1; beta_init = 0.01; structureSensitivity_init = 0.5;
end

options = optimoptions('particleswarm', 'Display', 'off', 'MaxIterations', 1e9, 'SwarmSize', 200, ...
    'MaxStallIterations', 50, 'InertiaRange', [0.2 0.9], ...
    'SelfAdjustmentWeight', 1.5, 'SocialAdjustmentWeight', 1.2, ...
    'FunctionTolerance', 1e-3, 'HybridFcn', @fmincon); 

benchmarks = [];
results = struct();

for i = 1:numel(polarities)
    pol = polarities{i};
    if strcmp(pol, 'dark')
        vol = 1 - volOrig;
    else
        vol = volOrig;
    end

    gvol = gpuArray(vol);

    % --- Run Reference Method (frangi or sato) ---
    t = tic;
    fm_gpu_ref = fibermetric_gpu(gvol, sigma_from, sigma_to, sigma_step, alpha_init, beta_init, structureSensitivity_init, pol, referenceMethod);
    wait(gpu);
    tgpu_ref = toc(t);
    fm_gpu_ref_optim = normalize_gpu(fm_gpu_ref);

    % --- Optimize targetMethod using reference as reference ---
    if i == 1
        if strcmp(targetMethod, 'frangi')
            objfun = @(x) gather(double(norm(normalize_gpu( ...
                fibermetric_gpu(gvol, sigma_from, sigma_to, sigma_step, x(1), x(2), x(3), pol, 'frangi')) ...
                - fm_gpu_ref_optim, 'fro')));
            options.InitialSwarmMatrix = [0.7144561, 0.8752375, 0.0995564];
            lb = [0, 0,    0];
            ub = [2, 1, 9999];
            [x_opt, ~] = particleswarm(objfun, 3, lb, ub, options);
            alpha_target = x_opt(1); beta_target = x_opt(2); structureSensitivity_target = x_opt(3);
        else % targetMethod == 'sato'
            objfun = @(x) gather(double(norm(normalize_gpu( ...
                fibermetric_gpu(gvol, sigma_from, sigma_to, sigma_step, x(1), x(2), 0.5, pol, 'sato')) ...
                - fm_gpu_ref_optim, 'fro')));
            options.InitialSwarmMatrix = [1.0, 0.01];
            lb = [0, 0];
            ub = [2, 1];
            [x_opt, ~] = particleswarm(objfun, 2, lb, ub, options);
            alpha_target = x_opt(1); beta_target = x_opt(2); structureSensitivity_target = 0.5;
        end
        fprintf("%s (%s): alpha=%.7f, beta=%.7f, StructureSensitivity=%.7f \n", ...
                capitalizeFirst(targetMethod), pol, alpha_target, beta_target, structureSensitivity_target);
    end

    % --- Run TargetMethod with optimized params ---
    t = tic;
    fm_gpu_target = fibermetric_gpu(gvol, sigma_from, sigma_to, sigma_step, ...
        alpha_target, beta_target, structureSensitivity_target, pol, targetMethod);
    wait(gpu);
    tgpu_target = toc(t);
    fm_gpu_target = gather(fm_gpu_target);

    % --- MATLAB fibermetric (CPU), uses same structureSensitivity as optimized ---
    t = tic;
    fm_cpu = fibermetric(vol, sigma_from:sigma_step:sigma_to, ...
        'ObjectPolarity', pol, 'StructureSensitivity', structureSensitivity_target);
    tcpu = toc(t);

    % --- Meijering GPU (no tuning) ---
    t = tic;
    fm_gpu_meijering = fibermetric_gpu(gvol, sigma_from, sigma_to, sigma_step, ...
        alpha_init, beta_init, structureSensitivity_init, pol, 'meijering');
    wait(gpu);
    tgpu_meijering = toc(t);
    fm_gpu_meijering = gather(fm_gpu_meijering);

    % --- Jerman optimization ---
    if i == 1
        objfun = @(x) gather(double(norm(normalize_gpu( ...
            fibermetric_gpu(gvol, sigma_from, sigma_to, sigma_step, x(1), x(2), x(3), pol, 'jerman')) ...
            - fm_gpu_ref_optim, 'fro')));
        options.InitialSwarmMatrix = [1.4774389, 3.7746898, 0];
        lb = [0, 0,    0.0];
        ub = [9, 9,    0.5];
        [x_jerman, ~] = particleswarm(objfun, 3, lb, ub, options);
        alpha_jerman = x_jerman(1); beta_jerman = x_jerman(2); structureSensitivity_jerman = x_jerman(3);
        fprintf("Jerman (%s): alpha=%.7f, beta=%.7f, StructureSensitivity=%.7f \n", pol, alpha_jerman, beta_jerman, structureSensitivity_jerman);
    end

    t = tic;
    fm_gpu_jerman = fibermetric_gpu(gvol, sigma_from, sigma_to, sigma_step, ...
        alpha_jerman, beta_jerman, structureSensitivity_jerman, pol, 'jerman');
    wait(gpu);
    tgpu_jerman = toc(t);
    fm_gpu_jerman = gather(fm_gpu_jerman);

    % --- Store for summary and plotting ---
    results(i).vol                          = vol;
    results(i).fm_cpu                       = fm_cpu;
    results(i).(['fm_gpu_' referenceMethod]) = fm_gpu_ref;
    results(i).(['fm_gpu_' targetMethod])    = fm_gpu_target;
    results(i).fm_gpu_meijering             = fm_gpu_meijering;
    results(i).fm_gpu_jerman                = fm_gpu_jerman;
    results(i).pol                          = pol;
    results(i).(['alpha_' targetMethod])                = alpha_target;
    results(i).(['beta_' targetMethod])                 = beta_target;
    results(i).(['structureSensitivity_' targetMethod]) = structureSensitivity_target;
    results(i).alpha_jerman                 = alpha_jerman;
    results(i).beta_jerman                  = beta_jerman;
    results(i).structureSensitivity_jerman  = structureSensitivity_jerman;

    benchmarks = [benchmarks;
        {pol, referenceMethod, tcpu, tgpu_ref, tcpu/tgpu_ref};
        {pol, targetMethod,   tcpu, tgpu_target, tcpu/tgpu_target};
        {pol, 'meijering',    tcpu, tgpu_meijering, tcpu/tgpu_meijering};
        {pol, 'jerman',       tcpu, tgpu_jerman, tcpu/tgpu_jerman}];
end

% Print summary table
T = cell2table(benchmarks, ...
    'VariableNames', {'Polarity', 'Method', 'CPU_Time_sec', 'GPU_Time_sec', 'Speedup_vs_CPU'});

fprintf('\n=== Benchmark Results ===\n');
disp(T);

% --- Combined plot: 2 rows (polarity) x 6 columns, dynamically named ---
figure('Name', 'Max Projections: Both Polarities', 'Position', [100 100 2400 600]);
methodLabels = {'Original', 'fibermetric (CPU)', capitalizeFirst(referenceMethod), capitalizeFirst(targetMethod), 'meijering', 'jerman'};
vol_axis = 3;
for i = 1:numel(polarities)
    pol = results(i).pol;
    idxOffset = (i-1)*6;
    % Original
    if strcmp(pol, 'dark')
        subplot(2,6,idxOffset+1); imagesc(squeeze(min(results(i).vol,[],vol_axis))); axis image off; colorbar; title(['Original (',pol,')']);
    else
        subplot(2,6,idxOffset+1); imagesc(squeeze(max(results(i).vol,[],vol_axis))); axis image off; colorbar; title(['Original (',pol,')']);
    end
    % fibermetric (CPU)
    subplot(2,6,idxOffset+2); imagesc(squeeze(max(results(i).fm_cpu,[],vol_axis))); axis image off; colorbar; title('fibermetric (CPU)');
    % Reference
    subplot(2,6,idxOffset+3); imagesc(squeeze(max(results(i).(['fm_gpu_' referenceMethod]),[],vol_axis))); axis image off; colorbar; title(capitalizeFirst(referenceMethod));
    % Target
    subplot(2,6,idxOffset+4); imagesc(squeeze(max(results(i).(['fm_gpu_' targetMethod]),[],vol_axis))); axis image off; colorbar; title(capitalizeFirst(targetMethod));
    % Meijering
    subplot(2,6,idxOffset+5); imagesc(squeeze(max(results(i).fm_gpu_meijering,[],vol_axis))); axis image off; colorbar; title('meijering');
    % Jerman
    subplot(2,6,idxOffset+6); imagesc(squeeze(max(results(i).fm_gpu_jerman,[],vol_axis))); axis image off; colorbar; title('jerman');
end
colormap(gray);

fprintf('\nDone!\n');

function str = capitalizeFirst(str)
    if ~isempty(str), str(1) = upper(str(1)); end
end

function arr = normalize_gpu(arr)
    arrmax = max(arr, [], 'all', 'omitnan');
    if isempty(arrmax) || isnan(arrmax) || arrmax == 0
        arr = zeros(size(arr), 'like', arr);
    else
        arr = arr ./ arrmax;
    end
end
