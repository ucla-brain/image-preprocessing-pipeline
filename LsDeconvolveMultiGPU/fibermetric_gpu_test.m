%clear; clc;
%gpu=gpuDevice(2);
%
%% --- Compare Hessian outputs ---
%sz = [128 128 32];
%sigma = 4;
%rng(42);
%vol = rand(sz, 'single');
%
%% Padding for CPU side (to replicate what GPU will process without boundary issues)
%padSize = ceil(3 * sigma);
%padShape = [padSize padSize padSize];
%vol_padded = padarray(vol, padShape, 'replicate', 'both');
%
%% === Run MATLAB version on padded volume ===
%[Dxx_cpu, Dyy_cpu, Dzz_cpu, Dxy_cpu, Dxz_cpu, Dyz_cpu] = Hessian3D(vol, sigma);
%
%% === Run GPU MEX version ===
%gvol = gpuArray(vol_padded);
%[Dxx_gpu, Dyy_gpu, Dzz_gpu, Dxy_gpu, Dxz_gpu, Dyz_gpu, eig1_gpu, eig2_gpu, eig3_gpu] = ...
%    fibermetric_gpu(gvol, sigma, sigma, 1, 0, 0, 0, 'bright', 1);
%
%% Gather & crop to match CPU unpadded result
%cropX = padShape(1)+1 : padShape(1)+sz(1);
%cropY = padShape(2)+1 : padShape(2)+sz(2);
%cropZ = padShape(3)+1 : padShape(3)+sz(3);
%Dxx_gpu = gather(Dxx_gpu(cropX, cropY, cropZ));
%Dyy_gpu = gather(Dyy_gpu(cropX, cropY, cropZ));
%Dzz_gpu = gather(Dzz_gpu(cropX, cropY, cropZ));
%Dxy_gpu = gather(Dxy_gpu(cropX, cropY, cropZ));
%Dxz_gpu = gather(Dxz_gpu(cropX, cropY, cropZ));
%Dyz_gpu = gather(Dyz_gpu(cropX, cropY, cropZ));
%eig1_gpu = gather(eig1_gpu(cropX, cropY, cropZ));
%eig2_gpu = gather(eig2_gpu(cropX, cropY, cropZ));
%eig3_gpu = gather(eig3_gpu(cropX, cropY, cropZ));
%
%% === Run eig3volume.c (CPU reference) ===
%[e1_ref, e2_ref, e3_ref] = eig3volume(Dxx_cpu, Dxy_cpu, Dxz_cpu, Dyy_cpu, Dyz_cpu, Dzz_cpu);
%
%% === Compare Hessian derivatives ===
%fprintf('\n== HESSIAN COMPARISON ==\n');
%fprintf('Max abs diff (Dxx): %g\n', max(abs(Dxx_cpu(:) - Dxx_gpu(:))));
%fprintf('Max abs diff (Dyy): %g\n', max(abs(Dyy_cpu(:) - Dyy_gpu(:))));
%fprintf('Max abs diff (Dzz): %g\n', max(abs(Dzz_cpu(:) - Dzz_gpu(:))));
%fprintf('Max abs diff (Dxy): %g\n', max(abs(Dxy_cpu(:) - Dxy_gpu(:))));
%fprintf('Max abs diff (Dxz): %g\n', max(abs(Dxz_cpu(:) - Dxz_gpu(:))));
%fprintf('Max abs diff (Dyz): %g\n', max(abs(Dyz_cpu(:) - Dyz_gpu(:))));
%
%% === Compare eigenvalues ===
%% Sort eigenvalues by abs value ascending
%% eig_ref = sort(abs(cat(4, e1_ref, e2_ref, e3_ref)), 4);
%% eig_gpu = sort(abs(cat(4, eig1_gpu, eig2_gpu, eig3_gpu)), 4);
%% 
%% e1_ref = eig_ref(:,:,:,1);  e2_ref = eig_ref(:,:,:,2);  e3_ref = eig_ref(:,:,:,3);
%% eig1_gpu = eig_gpu(:,:,:,1); eig2_gpu = eig_gpu(:,:,:,2); eig3_gpu = eig_gpu(:,:,:,3);
%
%
%fprintf('\n== EIGENVALUE COMPARISON (GPU vs eig3volume.c) ==\n');
%fprintf('Max abs diff (eig1): %g\n', max(abs(e1_ref(:) - eig1_gpu(:))));
%fprintf('Max abs diff (eig2): %g\n', max(abs(e2_ref(:) - eig2_gpu(:))));
%fprintf('Max abs diff (eig3): %g\n', max(abs(e3_ref(:) - eig3_gpu(:))));
%
%% === Optional visualization ===
%midz = round(sz(3)/2);
%figure('Name', 'Hessian Comparison: Dxx');
%subplot(1,2,1); imagesc(Dxx_cpu(:,:,midz)); axis image; title('Dxx CPU');
%subplot(1,2,2); imagesc(Dxx_gpu(:,:,midz)); axis image; title('Dxx GPU');
%
%figure('Name', 'Eigenvalue Comparison: eig1');
%subplot(1,2,1); imagesc(e1_ref(:,:,midz)); axis image; title('eig1 CPU');
%subplot(1,2,2); imagesc(eig1_gpu(:,:,midz)); axis image; title('eig1 GPU');

% benchmark_fibermetric_gpu_gpuonly.m
% Compare MATLAB's fibermetric (CPU) and your fibermetric_gpu (gpuArray only), both 'bright' and 'dark'

fprintf('\n==== Benchmark: fibermetric (CPU) vs fibermetric_gpu (gpuArray only) [bright/dark] ====\n');

sz = [128 128 32];
rng(42);
vol = imgaussfilt3(rand(sz, 'single'), 2);
vol = single(mat2gray(vol));
polarities = {'dark', 'bright'};

sigma_from = 4; sigma_to = 4; sigma_step = 1;
alpha = 0.5; beta = 0.5; structureSensitivity = 0.5;
pol = 'bright'; % or 'dark'—do both if you want

for i = 1:2
    pol = polarities{i};
    fprintf('\n--- Testing Polarity: %s ---\n', pol);

    % --- 1. MATLAB fibermetric (CPU, reference)
    t1 = tic;
    fm_cpu = fibermetric(vol, sigma_from:sigma_step:sigma_to, 'ObjectPolarity', pol, 'StructureSensitivity', structureSensitivity);
    tcpu = toc(t1);

    % --- 2. fibermetric_gpu with gpuArray input ONLY
    gvol = gpuArray(vol);
    t2 = tic;
    fm_gpu = fibermetric_gpu(gvol, sigma_from, sigma_to, sigma_step, alpha, beta, pol, structureSensitivity);
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
