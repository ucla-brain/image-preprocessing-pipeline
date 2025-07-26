gpu = gpuDevice(2);
reset(gpu); % Reset the GPU device to clear any previous state
fprintf('\n==== Benchmark: fibermetric (CPU) vs fibermetric_gpu (gpuArray only) [bright/dark, frangi/sato/meijering/jerman] ====\n');

% volOrig = im2single(load("ExampleVolumeStent.mat").V);
volOrig = im2single(tiffreadVolume("V:\tif\Glycin_MORF\crop_ds_it03_g2.0_crop.tif"));
polarities = {'bright', 'dark'};

sigma_from = 1; sigma_to = 7; sigma_step = 1;
alpha_init = 1; beta_init = 0.01; structureSensitivity_init = 0.5;
lb = [0, 0,    0];
ub = [2, 1, 9999];

methodNames = {'frangi', 'sato', 'meijering', 'jerman'};
nMethods = numel(methodNames);
options = optimoptions('particleswarm', 'Display', 'off', 'MaxIterations', 300, 'SwarmSize', 200, 'MaxStallIterations', 50, 'InertiaRange', [0.2 0.9], ...
    'SelfAdjustmentWeight', 1.5, 'SocialAdjustmentWeight', 1.2, 'FunctionTolerance', 1e-3, 'HybridFcn', @fmincon); 

benchmarks = [];
results = struct();

for i = 1:numel(polarities)
    pol = polarities{i};
    if strcmp(pol, 'dark')
        vol = 1 - volOrig; % robust inversion for floating-point
    else
        vol = volOrig;
    end

    % GPU: Prepare
    gvol = gpuArray(vol);

    % --- Run Sato (reference) ---
    t = tic;
    fm_gpu_sato = fibermetric_gpu(gvol, sigma_from, sigma_to, sigma_step, alpha_init, beta_init, structureSensitivity_init, pol, 'sato');
    wait(gpu);
    tgpu_sato = toc(t);
    fm_gpu_sato_optim = normalize_gpu(fm_gpu_sato);

    % --- Optimize alpha, beta, and structureSensitivity for frangi, using Sato as reference ---
    if i == 1
        objfun = @(x) gather(double(norm(normalize_gpu(fibermetric_gpu(gvol, sigma_from, sigma_to, sigma_step, x(1), x(2), x(3), pol, 'frangi')) - fm_gpu_sato_optim, 'fro')));
        options.InitialSwarmMatrix = [0.7144562, 0.8752373, 0.0995564];
        [x_frangi, ~] = particleswarm(objfun, 3, lb, ub, options);
        alpha_frangi = x_frangi(1); beta_frangi = x_frangi(2); structureSensitivity_frangi = x_frangi(3);
        fprintf("Ferengi (%s): alpha=%.7f, beta=%.7f, StructureSensitivity=%.7f \n", pol, alpha_frangi, beta_frangi, structureSensitivity_frangi);
    end

    % --- Frangi GPU with optimized params ---
    t = tic;
    fm_gpu_frangi = fibermetric_gpu(gvol, sigma_from, sigma_to, sigma_step, alpha_frangi, beta_frangi, structureSensitivity_frangi, pol, 'frangi');
    wait(gpu);
    tgpu_frangi = toc(t);
    fm_gpu_frangi = gather(fm_gpu_frangi);

    % MATLAB fibermetric with optimized params (CPU, Frangi)
    t = tic;
    fm_cpu = fibermetric(vol, sigma_from:sigma_step:sigma_to,  'ObjectPolarity', pol, 'StructureSensitivity', structureSensitivity_frangi);
    tcpu = toc(t);

    % --- Meijering GPU (no tuning) ---
    t = tic;
    fm_gpu_meijering = fibermetric_gpu(gvol, sigma_from, sigma_to, sigma_step, alpha_init, beta_init, structureSensitivity_init, pol, 'meijering');
    wait(gpu);
    tgpu_meijering = toc(t);
    fm_gpu_meijering = gather(fm_gpu_meijering);

    % --- Optimize alpha, beta, and structureSensitivity for Jerman using Sato as reference ---
    if i == 1
        objfun = @(x) gather(double(norm(normalize_gpu( fibermetric_gpu(gvol, sigma_from, sigma_to, sigma_step, x(1), x(2), x(3), pol, 'jerman') ) - fm_gpu_sato_optim , 'fro')));
        options.InitialSwarmMatrix = [0, 0.002, 0];
        [x_jerman, ~] = particleswarm(objfun, 3, lb, ub, options);
        alpha_jerman = x_jerman(1); beta_jerman = x_jerman(2); structureSensitivity_jerman = x_jerman(3);
        fprintf("Jerman (%s): alpha=%.7f, beta=%.7f, StructureSensitivity=%.7f \n", pol, alpha_jerman, beta_jerman, structureSensitivity_jerman);
    end

    t = tic;
    fm_gpu_jerman = fibermetric_gpu(gvol, sigma_from, sigma_to, sigma_step, alpha_jerman, beta_jerman, structureSensitivity_jerman, pol, 'jerman');
    wait(gpu);
    tgpu_jerman = toc(t);
    fm_gpu_jerman = gather(fm_gpu_jerman);

    % --- Store for summary and plotting ---
    results(i).vol                         = vol;
    results(i).fm_cpu                      = fm_cpu;
    results(i).fm_gpu_frangi               = fm_gpu_frangi;
    results(i).fm_gpu_sato                 = fm_gpu_sato;
    results(i).fm_gpu_meijering            = fm_gpu_meijering;
    results(i).fm_gpu_jerman               = fm_gpu_jerman;
    results(i).pol                         = pol;
    results(i).alpha_frangi                = alpha_frangi;
    results(i).beta_frangi                 = beta_frangi;
    results(i).structureSensitivity_frangi = structureSensitivity_frangi;
    results(i).alpha_jerman                = alpha_jerman;
    results(i).beta_jerman                 = beta_jerman;
    results(i).structureSensitivity_jerman = structureSensitivity_jerman;

    benchmarks = [benchmarks;
        {pol, 'frangi', tcpu, tgpu_frangi, tcpu/tgpu_frangi};
        {pol, 'sato',   tcpu, tgpu_sato,   tcpu/tgpu_sato};
        {pol, 'meijering', tcpu, tgpu_meijering, tcpu/tgpu_meijering};
        {pol, 'jerman',    tcpu, tgpu_jerman, tcpu/tgpu_jerman}];
end

% Print summary table
T = cell2table(benchmarks, ...
    'VariableNames', {'Polarity', 'Method', 'CPU_Time_sec', 'GPU_Time_sec', 'Speedup_vs_CPU'});

fprintf('\n=== Benchmark Results ===\n');
disp(T);

% --- Combined plot: 2 rows (polarity) x 6 columns ---
figure('Name', 'Max Projections: Both Polarities', 'Position', [100 100 2400 600]);
methodLabels = {'Original', 'fibermetric (CPU)', 'frangi', 'sato', 'meijering', 'jerman'};
vol_axis = 3;
for i = 1:numel(polarities)
    pol = results(i).pol;
    if strcmp(pol, 'dark')
        subplot(2,6,(i-1)*6+1); imagesc(squeeze(min(results(i).vol         ,[],vol_axis))); axis image off; colorbar; title(['Original (',pol,')']);
    else
        subplot(2,6,(i-1)*6+1); imagesc(squeeze(max(results(i).vol         ,[],vol_axis))); axis image off; colorbar; title(['Original (',pol,')']);
    end
    subplot(2,6,(i-1)*6+2); imagesc(squeeze(max(results(i).fm_cpu          ,[],vol_axis))); axis image off; colorbar; title('fibermetric (CPU)');
    subplot(2,6,(i-1)*6+3); imagesc(squeeze(max(results(i).fm_gpu_frangi   ,[],vol_axis))); axis image off; colorbar; title('frangi');
    subplot(2,6,(i-1)*6+4); imagesc(squeeze(max(results(i).fm_gpu_sato     ,[],vol_axis))); axis image off; colorbar; title('sato');
    subplot(2,6,(i-1)*6+5); imagesc(squeeze(max(results(i).fm_gpu_meijering,[],vol_axis))); axis image off; colorbar; title('meijering');
    subplot(2,6,(i-1)*6+6); imagesc(squeeze(max(results(i).fm_gpu_jerman   ,[],vol_axis))); axis image off; colorbar; title('jerman');
end
colormap(gray);

fprintf('\nDone!\n');

function arr = normalize_gpu(arr)
    arrmax = max(arr, [], 'all', 'omitnan');
    if isempty(arrmax) || isnan(arrmax) || arrmax == 0
        arr = zeros(size(arr), 'like', arr);
    else
        arr = arr ./ arrmax;
    end
end

