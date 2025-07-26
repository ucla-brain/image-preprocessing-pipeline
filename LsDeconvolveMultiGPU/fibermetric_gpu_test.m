gpu = gpuDevice(2);
fprintf('\n==== Benchmark: fibermetric (CPU) vs fibermetric_gpu (gpuArray only) [bright/dark, frangi/sato/meijering/jerman] ====\n');

% volOrig = im2single(load("ExampleVolumeStent.mat").V);
volOrig = im2single(tiffreadVolume("V:\tif\Glycin_MORF\crop_ds_it03_g2.0_crop.tif"));
polarities = {'bright', 'dark'};

sigma_from = 1; sigma_to = 7; sigma_step = 1;
alpha_init = 1; beta_init = 0.01; structureSensitivity_init = 0.5;
lb = [   0, 0, 0];
ub = [ 999, 9, 1];

methodNames = {'frangi', 'sato', 'meijering', 'jerman'};
nMethods = numel(methodNames);
options = optimoptions('particleswarm','Display','off','MaxIterations',1e9);

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
    objfun = @(x) norm(normalize_gpu(fibermetric_gpu(gvol, sigma_from, sigma_to, sigma_step, x(1), x(2), x(3), pol, 'frangi')) - fm_gpu_sato_optim, 'fro');
    x0 = [alpha_init, beta_init, structureSensitivity_init];
    [x_frangi,~] = particleswarm(objfun, numel(x0), lb, ub, options);

    alpha_frangi = x_frangi(1); beta_frangi = x_frangi(2); structureSensitivity_frangi = x_frangi(3);
    fprintf("Ferengi (%s): alpha=%.3f, beta=%.3f, StructureSensitivity=%.3f\n", pol, alpha_frangi, beta_frangi, structureSensitivity_frangi);

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

    % --- Optimize alpha, beta for Jerman using Sato as reference ---
    objfunJ = @(x) norm(normalize_gpu( fibermetric_gpu(gvol, sigma_from, sigma_to, sigma_step, x(1), x(2), x(3), pol, 'jerman') ) - fm_gpu_sato_optim , 'fro');
    x0 = [alpha_init, beta_init, 0];
    [x_jerman,~] = particleswarm(objfun, numel(x0), lb, ub, options);

    alpha_jerman = x_jerman(1); beta_jerman = x_jerman(2); structureSensitivity_jerman = x_jerman(3);
    fprintf("Jerman (%s): alpha=%.3f, beta=%.3f, StructureSensitivity=%.3f \n", pol, alpha_jerman, beta_jerman, structureSensitivity_jerman);

    t = tic;
    fm_gpu_jerman = fibermetric_gpu(gvol, sigma_from, sigma_to, sigma_step, alpha_jerman, beta_jerman, structureSensitivity_jerman, pol, 'jerman');
    wait(gpu);
    tgpu_jerman = toc(t);
    fm_gpu_jerman = gather(fm_gpu_jerman);

    % --- Store for summary and plotting ---
    results(i).vol            = vol;
    results(i).fm_cpu         = fm_cpu;
    results(i).fm_gpu_frangi  = fm_gpu_frangi;
    results(i).fm_gpu_sato    = fm_gpu_sato;
    results(i).fm_gpu_meijering = fm_gpu_meijering;
    results(i).fm_gpu_jerman  = fm_gpu_jerman;
    results(i).pol            = pol;
    results(i).alpha_frangi   = alpha_frangi;
    results(i).beta_frangi    = beta_frangi;
    results(i).structureSensitivity_frangi = structureSensitivity_frangi;
    results(i).alpha_jerman  = alpha_jerman;
    results(i).beta_jerman   = beta_jerman;
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
for i = 1:numel(polarities)
    pol = results(i).pol;
    colormap(gray);
    if strcmp(pol, 'dark')
        subplot(2,6,(i-1)*6+1); imagesc(min(results(i).vol,[],3)); axis image off; colorbar; title(['Original (',pol,')']);
    else
        subplot(2,6,(i-1)*6+1); imagesc(max(results(i).vol,[],3)); axis image off; colorbar; title(['Original (',pol,')']);
    end
    subplot(2,6,(i-1)*6+2); imagesc(max(results(i).fm_cpu,[],3)); axis image off; colorbar; title('fibermetric (CPU)');
    subplot(2,6,(i-1)*6+3); imagesc(max(results(i).fm_gpu_frangi,[],3)); axis image off; colorbar; title('frangi');
    subplot(2,6,(i-1)*6+4); imagesc(max(results(i).fm_gpu_sato,[],3)); axis image off; colorbar; title('sato');
    subplot(2,6,(i-1)*6+5); imagesc(max(results(i).fm_gpu_meijering,[],3)); axis image off; colorbar; title('meijering');
    subplot(2,6,(i-1)*6+6); imagesc(max(results(i).fm_gpu_jerman,[],3)); axis image off; colorbar; title('jerman');
end
colormap(gray);

fprintf('\nDone!\n');

% --- Utility: fminsearch with bounds ---
function [x,fval] = fminsearchbnd(fun, x0, LB, UB, options)
    % fminsearch, but clamp variables to [LB,UB] each iteration
    funwrap = @(x) fun(max(LB, min(UB, x)));
    [x, fval] = fminsearch(funwrap, x0, options);
end

function normed = normalize_gpu(arr)
    normed = arr ./ max(arr, [], 'all');
end
