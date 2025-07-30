gpu = gpuDevice(1);
reset(gpu);
fprintf('\n==== Modular Benchmark: fibermetric_gpu (GPU only) with PSO tuning ====\n');

% ----------------------- Configuration Section ------------------------
sigma_from = 1; sigma_step = 2; sigma_to = 9; 
referenceMethod             = 'original';         % Options: 'original', 'frangi', 'sato', 'ferengi', 'jerman'
methodsToOptimize           = {'frangi', 'sato'}; % Methods to optimize (cell)
methodsToPlot               = {'frangi', 'sato'}; % Methods to plot, subset of optimized or not
polaritiesToTest            = {'bright'};         % Choose at least one: {'bright'}, {'dark'}, or both
vol_axes                    = [3 2];              % Projection axes for plotting
showParameters              = true;               % Print optimized parameters on plot and console
options = optimoptions('particleswarm', ...
    'Display'               , 'off', ...
    'MaxIterations'         , 1, ...
    'SwarmSize'             , 2, ...
    'MaxStallIterations'    , 50, ...
    'InertiaRange'          , [0.2 0.9], ...
    'SelfAdjustmentWeight'  , 1.5, ...
    'SocialAdjustmentWeight', 1.2, ...
    'FunctionTolerance'     , 1e-7, ...
    'HybridFcn'             , @patternsearch);
% ----------------------------------------------------------------------

assert(~isempty(polaritiesToTest), 'At least one polarity must be tested.');

volOrig = im2single(tiffreadVolume("test_volume.tif"));
gvol = gpuArray(volOrig);
gvol_normal = gvol ./ max(gvol, [], 'all', 'omitnan');

fields = [{['fm_gpu_' referenceMethod]}, ...
          strcat('fm_gpu_', methodsToPlot), ...
          strcat('time_gpu_', methodsToPlot), ...
          {'pol','vol','opt_params'}];
Npol = numel(polaritiesToTest);
results = repmat(cell2struct(cell(size(fields)), fields, 2), Npol, 1);

for polIdx = 1:Npol
    pol = polaritiesToTest{polIdx};
    if strcmp(pol, 'dark')
        vol = 1 - volOrig;
        gvol_used = 1 - gvol_normal;
    else
        vol = volOrig;
        gvol_used = gvol_normal;
    end

    % ------------------ Compute reference volume --------------------
    refVol = [];
    refMethodLC = lower(referenceMethod);
    switch refMethodLC
        case 'original'
            refVol = gvol_used;
        case {'frangi','sato','jerman','ferengi'}
            [alpha_init, beta_init, ss_init] = getDefaultInitParams(referenceMethod);
            refVol = fibermetric_gpu(gvol_used, sigma_from, sigma_to, sigma_step, ...
                alpha_init, beta_init, ss_init, pol, referenceMethod);
        otherwise
            error('Unknown referenceMethod "%s"', referenceMethod);
    end
    refVol = normalize_gpu(refVol);

    % ------------------- PSO Optimization for selected methods -------------------
    optParams = struct();
    for m = 1:numel(methodsToOptimize)
        method = lower(methodsToOptimize{m});
        [alpha_init, beta_init, ss_init] = getDefaultInitParams(method);
        switch method
            case 'frangi'
                options.InitialSwarmMatrix = [alpha_init, beta_init, ss_init];
                lb = [0, 0, 0];
                ub = [2, 1, 9999];
                objfun = @(x) double(norm(normalize_gpu(...
                    fibermetric_gpu(gvol_used, sigma_from, sigma_to, sigma_step, x(1), x(2), x(3), pol, 'frangi')) - refVol, 'fro'));
                [x_opt, ~] = particleswarm(objfun, 3, lb, ub, options);
                optParams.frangi = struct('alpha', x_opt(1), 'beta', x_opt(2), 'structureSensitivity', x_opt(3));
            case 'sato'
                options.InitialSwarmMatrix = [alpha_init, beta_init];
                lb = [0, 0];
                ub = [2, 1];
                objfun = @(x) double(norm(normalize_gpu(...
                    fibermetric_gpu(gvol_used, sigma_from, sigma_to, sigma_step, x(1), x(2), 0.5, pol, 'sato')) - refVol, 'fro'));
                [x_opt, ~] = particleswarm(objfun, 2, lb, ub, options);
                optParams.sato = struct('alpha', x_opt(1), 'beta', x_opt(2), 'structureSensitivity', 0.5);
            case 'jerman'
                options.InitialSwarmMatrix = [alpha_init, beta_init, ss_init];
                lb = [0.5, 0, 0.0];
                ub = [1.0, 9, 1.0];
                objfun = @(x) double(norm(normalize_gpu(...
                    fibermetric_gpu(gvol_used, sigma_from, sigma_to, sigma_step, x(1), x(2), x(3), pol, 'jerman')) - refVol, 'fro'));
                [x_opt, ~] = particleswarm(objfun, 3, lb, ub, options);
                optParams.jerman = struct('tau', x_opt(1), 'C', x_opt(2), 'structureSensitivity', x_opt(3));
            otherwise
                warning('Unknown method "%s" for optimization', method);
        end
    end

    % Print optimized params in console
    if showParameters
        fprintf('\nPolarity: %s\n', pol);
        fn = fieldnames(optParams);
        for k = 1:numel(fn)
            fprintf('  %s: \n', fn{k});
            disp(optParams.(fn{k}));
        end
    end

    % ------------------ Run/Store all methods for plotting --------------------
    thisResult = cell2struct(cell(size(fields)), fields, 2);
    thisResult.pol = pol;
    thisResult.vol = vol;
    thisResult.opt_params = optParams;

    % Reference always included
    thisResult.(['fm_gpu_' referenceMethod]) = gather(refVol);

    for m = 1:numel(methodsToPlot)
        method = lower(methodsToPlot{m});
        [alpha_init, beta_init, ss_init] = getDefaultInitParams(method);
        switch method
            case 'frangi'
                if isfield(optParams, 'frangi'), p = optParams.frangi;
                else, p = struct('alpha', alpha_init, 'beta', beta_init, 'structureSensitivity', ss_init); end
                t = tic;
                fm = fibermetric_gpu(gvol_used, sigma_from, sigma_to, sigma_step, p.alpha, p.beta, p.structureSensitivity, pol, 'frangi');
                wait(gpu); tgpu = toc(t); fm = gather(fm);
            case 'sato'
                if isfield(optParams, 'sato'), p = optParams.sato;
                else, p = struct('alpha', alpha_init, 'beta', beta_init, 'structureSensitivity', 0.5); end
                t = tic;
                fm = fibermetric_gpu(gvol_used, sigma_from, sigma_to, sigma_step, p.alpha, p.beta, 0.5, pol, 'sato');
                wait(gpu); tgpu = toc(t); fm = gather(fm);
            case 'jerman'
                if isfield(optParams, 'jerman'), p = optParams.jerman;
                else, p = struct('tau', alpha_init, 'C', beta_init, 'structureSensitivity', ss_init); end
                t = tic;
                fm = fibermetric_gpu(gvol_used, sigma_from, sigma_to, sigma_step, p.tau, p.C, p.structureSensitivity, pol, 'jerman');
                wait(gpu); tgpu = toc(t); fm = gather(fm);
            case 'meijering'
                t = tic;
                fm = fibermetric_gpu(gvol_used, sigma_from, sigma_to, sigma_step, 0, 0, 0, pol, 'meijering');
                wait(gpu); tgpu = toc(t); fm = gather(fm);
            otherwise
                warning('Unknown method "%s" for plotting', method);
                continue;
        end
        thisResult.(['fm_gpu_' method]) = fm;
        thisResult.(['time_gpu_' method]) = tgpu;
    end
    results(polIdx) = thisResult;
end

% =========== Plotting Section (vol_axis=3 and vol_axis=2 in columns) ==============
nPol = numel(polaritiesToTest);
nMethod = numel(methodsToPlot) + 1; % +1 for reference
nAxis = numel(vol_axes);
figure('Name', 'Max Projections: All Methods, Both Axes', 'Position', [100 100 2500 1200]);
for a = 1:nAxis
    axis_val = vol_axes(a);
    for i = 1:nPol
        res = results(i);
        pol = res.pol;
        idxOffset = (a-1)*nPol*nMethod + (i-1)*nMethod;
        % Reference (Input)
        subplot(nAxis*nPol, nMethod, idxOffset+1);
        if strcmp(pol, 'dark')
            imagesc(squeeze(min(res.vol, [], axis_val)));
            tit = sprintf('Input (%s,axis=%d)', pol, axis_val);
        else
            imagesc(squeeze(max(res.vol, [], axis_val)));
            tit = sprintf('Input (%s,axis=%d)', pol, axis_val);
        end
        axis image off; colorbar; title(tit, 'Interpreter','none');
        % Under-plot parameters (none for input)
        if showParameters, text(2,2,'','Color','k'); end

        for m = 1:numel(methodsToPlot)
            method = methodsToPlot{m};
            f = res.(['fm_gpu_' method]);
            subplot(nAxis*nPol, nMethod, idxOffset+1+m);
            imagesc(squeeze(max(f, [], axis_val)));
            axis image off; colorbar;
            tstr = capitalizeFirst(method);
            title(sprintf('%s (axis=%d)', tstr, axis_val), 'Interpreter','none');
            % Print optimized values under image
            if showParameters && isfield(res.opt_params, method)
                p = res.opt_params.(method);
                fieldsP = fieldnames(p);
                vals = cellfun(@(fn) p.(fn), fieldsP, 'UniformOutput', false);
                txt = strjoin(cellfun(@(fn,v) sprintf('%s=%.4g',fn,v), fieldsP, vals, 'UniformOutput', false), ', ');
                ylabel(txt, 'Interpreter','none', 'FontSize',10, 'FontWeight','bold', 'Color','b');
            end
        end
    end
end
colormap(gray);

fprintf('\nDone!\n');

% --- Helper: Default params ---
function [alpha_init, beta_init, ss_init] = getDefaultInitParams(method)
    switch lower(method)
        case 'frangi'
            alpha_init = 0.0165; beta_init = 0.6507; ss_init = 0.0958;
        case {'sato'}
            alpha_init = 0.0380; beta_init = 0;      ss_init = 0.5;
        case 'jerman'
            alpha_init = 0.75;   beta_init = 2;      ss_init = 1e-4;
        otherwise % meijering or fallback
            alpha_init = 0;      beta_init = 0;      ss_init = 0;
    end
end

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
