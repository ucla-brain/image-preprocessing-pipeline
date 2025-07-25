% clear; clc;
gpu = gpuDevice(2);

fprintf('\n==== Benchmark: fibermetric (CPU) vs fibermetric_gpu (gpuArray only) [bright/dark, frangi/sato] ====\n');

% volOrig = im2single(load("ExampleVolumeStent.mat").V);
%volOrig = im2single(tiffreadVolume("V:\tif\Glycin_MORF\crop_ds_it03_g2.0_crop.tif"));
polarities = {'dark', 'bright'};

sigma_from = 1; sigma_to = 16; sigma_step = 1;
alpha = 2; beta = 0.01; structureSensitivity = 15;

benchmarks = [];
results = struct();

for i = 1:numel(polarities)
    pol = polarities{i};
    % Use a copy of the original volume for each polarity!
    if strcmp(pol, 'dark')
        vol = 1 - volOrig; % robust inversion for floating-point
    else
        vol = volOrig;
    end

    % MATLAB fibermetric (CPU, Frangi)
    t = tic;
    fm_cpu = fibermetric(vol, sigma_from:sigma_step:sigma_to,  'ObjectPolarity', pol, 'StructureSensitivity', structureSensitivity);
    tcpu = toc(t);

    % GPU Ferengi/Frangi
    gvol = gpuArray(vol);
    t = tic;
    fm_gpu_frangi = fibermetric_gpu(gvol, sigma_from, sigma_to, sigma_step, alpha, beta, structureSensitivity, pol, 'frangi');
    wait(gpu);
    tgpu_frangi = toc(t);
    fm_gpu_frangi = gather(fm_gpu_frangi);

    % GPU Sato
    t = tic;
    fm_gpu_sato = fibermetric_gpu(gvol, sigma_from, sigma_to, sigma_step, alpha, beta, structureSensitivity, pol, 'sato');
    wait(gpu);
    tgpu_sato = toc(t);
    fm_gpu_sato = gather(fm_gpu_sato);

    % Store for summary and plotting
    results(i).vol           = vol;
    results(i).fm_cpu        = fm_cpu;
    results(i).fm_gpu_frangi = fm_gpu_frangi;
    results(i).fm_gpu_sato   = fm_gpu_sato;
    results(i).pol           = pol;

    benchmarks = [benchmarks;
        {pol, 'frangi', tcpu, tgpu_frangi, tcpu/tgpu_frangi};
        {pol, 'sato',   tcpu, tgpu_sato,   tcpu/tgpu_sato}];
end

% Print summary table
T = cell2table(benchmarks, ...
    'VariableNames', {'Polarity', 'Method', 'CPU_Time_sec', 'GPU_Time_sec', 'Speedup_vs_CPU'});

fprintf('\n=== Benchmark Results ===\n');
disp(T);

% --- Combined plot: 2 rows (polarity) x 4 columns ---
figure('Name', 'Max Projections: Both Polarities', 'Position', [100 100 1600 600]);
for i = 1:numel(polarities)
    pol = results(i).pol;
    colormap(gray);
    if strcmp(pol, 'dark')
    subplot(2,4,(i-1)*4+1); imagesc(min(results(i).vol          ,[],3)); axis image off; colorbar; title(['Original (',pol,')']);
    else
    subplot(2,4,(i-1)*4+1); imagesc(max(results(i).vol          ,[],3)); axis image off; colorbar; title(['Original (',pol,')']);
    end
    subplot(2,4,(i-1)*4+2); imagesc(max(results(i).fm_cpu       ,[],3)); axis image off; colorbar; title('fibermetric (CPU)');
    subplot(2,4,(i-1)*4+3); imagesc(max(results(i).fm_gpu_frangi,[],3)); axis image off; colorbar; title('fibermetric\_gpu (frangi)');
    subplot(2,4,(i-1)*4+4); imagesc(max(results(i).fm_gpu_sato  ,[],3)); axis image off; colorbar; title('fibermetric\_gpu (sato)');
end
colormap(gray);


% % --- ScaleNorm Estimation Section ---
% fprintf('\n=== scaleNorm Estimation Experiments ===\n');
% scalenorms = [];
% randSeedList = [42 123 2025];
% szList = [32 32 32; 64 32 16; 32 32 4; 48 48 8];
% sigma_for_norm = 1; % Only sigma==1 is meaningful for MATLAB norm matching
% 
% % For both random volumes and the stent volume
% volumes = {volOrig, []}; % Second will be filled in loop with randoms
% volLabels = {'ExampleVolumeStent', 'Random'};
% 
% for v=1:2
%     for si = 1:size(szList,1)
%         sz = szList(si,:);
%         for r = 1:numel(randSeedList)
%             seed = randSeedList(r);
%             if v==1
%                 vol = imresize3(volOrig, sz, 'nearest');
%             else
%                 rng(seed);
%                 vol = rand(sz, 'single');
%             end
%             gvol = gpuArray(vol);
% 
%             fm_cpu = fibermetric(vol, sigma_for_norm, ...
%                 'ObjectPolarity', 'bright', ...
%                 'StructureSensitivity', structureSensitivity);
% 
%             fm_gpu = fibermetric_gpu(gvol, sigma_for_norm, sigma_for_norm, 1, alpha, beta, structureSensitivity, 'bright', 'frangi');
%             fm_gpu = gather(fm_gpu);
% 
%             % Mask for regions where both are nonzero
%             mask = (fm_cpu > 0 & fm_gpu > 0);
%             ratio = fm_cpu(mask) ./ fm_gpu(mask);
%             scaleNorm_median = median(ratio(:));
%             scaleNorm_mean   = mean(ratio(:));
%             scaleNorm_std    = std(ratio(:));
%             scalenorms = [scalenorms;
%                 {volLabels{v}, sz, seed, scaleNorm_median, scaleNorm_mean, scaleNorm_std}];
%         end
%     end
% end
% 
% scaleTable = cell2table(scalenorms, ...
%     'VariableNames', {'VolumeType', 'Size', 'Seed', 'MedianRatio', 'MeanRatio', 'StdRatio'});
% 
% disp(scaleTable);
% 
% % Optional: plot histogram of scaleNorm ratios for the last case
% figure('Name','scaleNorm ratio distribution (last case)');
% histogram(ratio,50);
% xlabel('fibermetric(MATLAB) ./ fibermetric\_gpu'), ylabel('Voxel Count');
% title('Distribution of scaleNorm ratio (last test)');
% grid on;
% 
% % Print summary
% fprintf('Median scaleNorm ratio across all tests: %.5f ± %.5f\n', ...
%     mean(scaleTable.MedianRatio), std(scaleTable.MedianRatio));
% 
% 
% 
% % --- fibermetric_gpu Output Distribution Across Sigma ---
% fprintf('\n=== fibermetric\\_gpu Output Distribution for Different Sigmas ===\n');
% test_sigmas = [0.5 1 2 4 8 16];
% inputs = {volOrig, rand(size(volOrig), 'single')};
% input_labels = {'ExampleVolumeStent', 'Random'};
% 
% gpu = gpuDevice; % Make sure GPU is ready
% 
% statRows = {};
% figure('Name', 'fibermetric\_gpu Value Distribution (various sigmas)', 'Position', [100 200 1400 500]);
% 
% for vi = 1:numel(inputs)
%     test_vol = inputs{vi};
%     gvol = gpuArray(test_vol);
%     for si = 1:numel(test_sigmas)
%         sigma = test_sigmas(si);
%         out_gpu = gather(fibermetric_gpu(gvol, sigma, sigma, 1, alpha, beta, structureSensitivity, 'bright', 'frangi'));
%         mask = out_gpu > 0;
%         vals = out_gpu(mask);
% 
%         statRows{end+1,1} = input_labels{vi};
%         statRows{end,2} = sigma;
%         statRows{end,3} = mean(vals);
%         statRows{end,4} = std(vals);
%         statRows{end,5} = min(vals);
%         statRows{end,6} = max(vals);
%         statRows{end,7} = prctile(vals,1);
%         statRows{end,8} = prctile(vals,50);
%         statRows{end,9} = prctile(vals,99);
% 
%         subplot(numel(inputs), numel(test_sigmas), (vi-1)*numel(test_sigmas)+si);
%         histogram(vals,50, 'EdgeColor','none');
%         title(sprintf('%s\n\\sigma=%.2f', input_labels{vi}, sigma));
%         xlabel('Value'); ylabel('Count');
%         set(gca,'yscale','log');
%         grid on;
%     end
% end
% 
% statTable = cell2table(statRows, ...
%     'VariableNames', {'InputType','Sigma','Mean','Std','Min','Max','P01','P50','P99'});
% 
% disp(statTable);



fprintf('\nDone!\n');
