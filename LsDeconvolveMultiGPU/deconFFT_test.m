function deconFFT_test
% Comprehensive test & benchmark for deconFFT_mex
% Reports results in a pretty, colored table

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

results = cell(size(tests,1), 8);

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

    % -- MATLAB Reference
    cpu_tic = tic;
    buf = convFFT_matlab(bl, otf);
    buf = max(buf, eps('single'));
    buf = bl ./ buf;
    buf = convFFT_matlab(buf, otf_conj);
    if lambda > 0
        % Simple Tikhonov regularization: 3D Laplacian kernel
        R = fspecial3('laplacian', [3 3 3]); % now robust
        reg = convFFT_matlab(bl, fftn(R, sz));
        result_ref = bl .* buf .* (1-lambda) + reg .* lambda;
    else
        result_ref = bl .* buf;
    end
    result_ref = abs(result_ref);
    cpu_time = toc(cpu_tic);

    % -- GPU MEX
    gpu_tic = tic;
    result_gpu = deconFFT_mex(bl_gpu, otf_gpu, otf_conj_gpu, single(lambda));
    wait(gpu);
    gpu_time = toc(gpu_tic);
    result_gpu_host = gather(result_gpu);

    % -- Accuracy
    diff = result_ref - result_gpu_host;
    max_diff = max(abs(diff(:)));
    mean_diff = mean(abs(diff(:)));
    rel_norm = norm(diff(:)) / norm(result_ref(:));
    pass = (rel_norm < tol) || (max_diff < tol);

    % -- Perf gain
    speedup = (cpu_time/gpu_time - 1)*100;

    results{k,1} = pass;
    results{k,2} = label;
    results{k,3} = sprintf('%dx%dx%d', sz);
    results{k,4} = lambda;
    results{k,5} = max_diff;
    results{k,6} = mean_diff;
    results{k,7} = rel_norm;
    results{k,8} = speedup;
end

% --- Print Table ---
header = ["Pass", "Test", "Input size", "lambda", "Max abs diff", "Mean abs diff", "Rel norm diff", "GPU speedup (%)"];
fmt = ['%-8s  %-22s  %-13s  %-8s  %-14s  %-14s  %-15s  %-15s\n'];
fprintf('\n');
fprintf(fmt, header{:});
fprintf('%s\n', repmat('-',1,100));
for k = 1:size(results,1)
    if results{k,1}
        pass_str = [char(10003), ' ']; % ✓
        cstr = '\033[32m'; % Green
    else
        pass_str = [char(10007), ' ']; % ✗
        cstr = '\033[31m'; % Red
    end
    fprintf([cstr fmt '\033[0m'], ...
        pass_str, results{k,2}, results{k,3}, ...
        num2str(results{k,4}), ...
        sprintf('%.2g', results{k,5}), ...
        sprintf('%.2g', results{k,6}), ...
        sprintf('%.2g', results{k,7}), ...
        sprintf('%.1f', results{k,8}));
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
