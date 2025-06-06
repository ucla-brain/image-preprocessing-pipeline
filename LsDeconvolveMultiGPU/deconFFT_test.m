function deconFFT_test
% Comprehensive test & benchmark for deconFFT_mex
% Compares MATLAB CPU, MATLAB GPU, and custom CUDA MEX results.

tests = {
    'Small array, lambda=0',      [64,64,32],    0.0
    'Medium array, lambda=0',     [128,128,64],  0.0
    'Large array, lambda=0',      [512,512,128], 0.0
    'Large array, lambda=0.05',   [512,512,128], 0.05
    'Non-cube, lambda=0',         [32,48,96],    0.0
    'Non-cube, lambda=0.1',       [32,48,96],    0.1
};

tol = 2e-4; % accuracy threshold (adjust for your kernel if needed)
gpu = gpuDevice;
fprintf('Using GPU: %s\n', gpu.Name);

results = cell(size(tests,1), 12);

for k = 1:size(tests,1)
    label = tests{k,1};
    sz = tests{k,2};
    lambda = tests{k,3};
    % Generate input
    bl = rand(sz, 'single');
    psf = fspecial3('gaussian', [15 15 9], 3);
    psf = single(psf ./ sum(psf(:)));
    otf = fftn(psf, sz);
    otf_conj = conj(otf);

    bl_gpu = gpuArray(bl);
    otf_gpu = gpuArray(single(otf));
    otf_conj_gpu = gpuArray(single(otf_conj));

    % -- MATLAB CPU Reference
    cpu_tic = tic;
    buf = convFFT_matlab(bl, otf);
    buf = max(buf, eps('single'));
    buf = bl ./ buf;
    buf = convFFT_matlab(buf, otf_conj);
    if lambda > 0
        R = fspecial3('laplacian', [3 3 3]);
        reg = convFFT_matlab(bl, fftn(R, sz));
        result_ref = bl .* buf .* (1-lambda) + reg .* lambda;
    else
        result_ref = bl .* buf;
    end
    result_ref = abs(result_ref);
    cpu_time = toc(cpu_tic);

    % -- MATLAB GPU Reference
    gpu_ref_tic = tic;
    buf_gpu = convFFT_matlab(bl_gpu, otf_gpu);
    buf_gpu = max(buf_gpu, eps('single'));
    buf_gpu = bl_gpu ./ buf_gpu;
    buf_gpu = convFFT_matlab(buf_gpu, otf_conj_gpu);
    if lambda > 0
        Rg = gpuArray(fspecial3('laplacian', [3 3 3]));
        reg_gpu = convFFT_matlab(bl_gpu, fftn(Rg, sz));
        result_ref_gpu = bl_gpu .* buf_gpu .* (1-lambda) + reg_gpu .* lambda;
    else
        result_ref_gpu = bl_gpu .* buf_gpu;
    end
    result_ref_gpu = abs(result_ref_gpu);
    wait(gpu);
    gpu_ref_time = toc(gpu_ref_tic);
    result_ref_gpu_host = gather(result_ref_gpu);

    % -- GPU MEX
    mex_tic = tic;
    result_gpu = deconFFT_mex(bl_gpu, otf_gpu, otf_conj_gpu, single(lambda));
    wait(gpu);
    mex_time = toc(mex_tic);
    result_gpu_host = gather(result_gpu);

    % -- Accuracy (vs CPU and vs MATLAB GPU)
    diff_cpu = result_ref - result_gpu_host;
    max_diff_cpu = max(abs(diff_cpu(:)));
    mean_diff_cpu = mean(abs(diff_cpu(:)));
    rel_norm_cpu = norm(diff_cpu(:)) / norm(result_ref(:));
    pass_cpu = (rel_norm_cpu < tol) || (max_diff_cpu < tol);

    diff_gpu = result_ref_gpu_host - result_gpu_host;
    max_diff_gpu = max(abs(diff_gpu(:)));
    mean_diff_gpu = mean(abs(diff_gpu(:)));
    rel_norm_gpu = norm(diff_gpu(:)) / norm(result_ref_gpu_host(:));
    pass_gpu = (rel_norm_gpu < tol) || (max_diff_gpu < tol);

    % -- Perf gain
    speedup_cpu = (cpu_time/mex_time - 1)*100;
    speedup_gpu = (gpu_ref_time/mex_time - 1)*100;

    % Store results
    results{k,1}  = pass_cpu;
    results{k,2}  = pass_gpu;
    results{k,3}  = label;
    results{k,4}  = sprintf('%dx%dx%d', sz);
    results{k,5}  = lambda;
    results{k,6}  = max_diff_cpu;
    results{k,7}  = mean_diff_cpu;
    results{k,8}  = rel_norm_cpu;
    results{k,9}  = max_diff_gpu;
    results{k,10} = mean_diff_gpu;
    results{k,11} = rel_norm_gpu;
    results{k,12} = [cpu_time, gpu_ref_time, mex_time, speedup_cpu, speedup_gpu];
end

% --- Print Table ---
header = ["Pass(CPU)", "Pass(GPU)", "Test", "Input size", "lambda", ...
          "Max abs diff (CPU)", "Mean abs diff (CPU)", "Rel norm (CPU)", ...
          "Max abs diff (GPU)", "Mean abs diff (GPU)", "Rel norm (GPU)", ...
          "Perf (s) [CPU, GPUref, MEX] / Speedup [% CPU, % GPUref]"];
fmt = ['%-10s  %-10s  %-22s  %-13s  %-8s  %-17s  %-19s  %-14s  %-17s  %-19s  %-14s  %-38s\n'];
fprintf('\n');
fprintf(fmt, header{:});
fprintf('%s\n', repmat('-',1,180));
for k = 1:size(results,1)
    % Unicode + color
    if results{k,1}
        pass_str_cpu = [char(10003), ' ']; cstr_cpu = '\033[32m';
    else
        pass_str_cpu = [char(10007), ' ']; cstr_cpu = '\033[31m';
    end
    if results{k,2}
        pass_str_gpu = [char(10003), ' ']; cstr_gpu = '\033[32m';
    else
        pass_str_gpu = [char(10007), ' ']; cstr_gpu = '\033[31m';
    end
    cpu_time = results{k,12}(1);
    gpu_time = results{k,12}(2);
    mex_time = results{k,12}(3);
    speedup_cpu = results{k,12}(4);
    speedup_gpu = results{k,12}(5);

    fprintf(['%s%-9s  %s%-9s\033[0m  %-22s  %-13s  %-8s  %-17.3g  %-19.3g  %-14.3g  %-17.3g  %-19.3g  %-14.3g  [%.2fs, %.2fs, %.2fs] / [%.1f%%, %.1f%%]\n'], ...
        cstr_cpu, pass_str_cpu, cstr_gpu, pass_str_gpu, ...
        results{k,3}, results{k,4}, num2str(results{k,5}), ...
        results{k,6}, results{k,7}, results{k,8}, ...
        results{k,9}, results{k,10}, results{k,11}, ...
        cpu_time, gpu_time, mex_time, speedup_cpu, speedup_gpu);
end

end

function out = convFFT_matlab(x, otf)
    fx = fftn(x);
    fx = fx .* otf;
    out = real(ifftn(fx));
end

function h = fspecial3(type, sz, sigma)
% Minimal 3D gaussian/laplacian generator for testing
if isscalar(sz)
    sz = repmat(sz,1,3);
end
if nargin < 3, sigma = 1; end
switch lower(type)
    case 'gaussian'
        [x, y, z] = ndgrid(-(sz(1)-1)/2:(sz(1)-1)/2, ...
                           -(sz(2)-1)/2:(sz(2)-1)/2, ...
                           -(sz(3)-1)/2:(sz(3)-1)/2);
        h = exp(-(x.^2 + y.^2 + z.^2)/(2*sigma^2));
        h = h/sum(h(:));
    case 'laplacian'
        h = zeros(sz, 'single');
        c = floor(sz/2)+1;
        h(c(1),c(2),c(3)) = -6;
        h([c(1)-1,c(1)+1],c(2),c(3)) = 1;
        h(c(1),[c(2)-1,c(2)+1],c(3)) = 1;
        h(c(1),c(2),[c(3)-1,c(3)+1]) = 1;
    otherwise
        error('Unknown kernel');
end
end