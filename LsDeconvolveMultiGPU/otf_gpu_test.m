function otf_gpu_test
fprintf('\n');
fprintf('PF   Test  Type    Size              Sigma         Kernel          maxErr    RMS       relErr    mex(s)   Speedup\n');
fprintf('---------------------------------------------------------------------------------------------------------------\n');

Nrep = 5; % <--- Number of repeats for timing (ignore first run for warmup)

sz = [120 120 150];
kernels = {'auto', 9, [9 9 21], 3, 41};
sigmas = {2.5, [2.5 2.5 2.5], [0.5 0.5 2.5], 0.25, 8};
results = [];

test_id = 1;
for s = 1:length(sigmas)
    sigma = sigmas{s};
    for k = 1:length(kernels)
        kernel = kernels{k};
        sigma_disp = fmtvec(sigma);
        kernel_disp = fmtvec(kernel);

        % Build PSF (as before)
        if isequal(kernel, 'auto')
            kernsz = sz;
        elseif isnumeric(kernel) && isscalar(kernel)
            kernsz = min(sz, kernel*ones(1,3));
        elseif isnumeric(kernel) && isvector(kernel)
            kernsz = min(sz, kernel(:)');
            if numel(kernsz)==1, kernsz = kernsz*ones(1,3); end
        else
            kernsz = sz;
        end
        center = (kernsz+1)/2;
        [x,y,z] = ndgrid(1:kernsz(1), 1:kernsz(2), 1:kernsz(3));
        if numel(sigma)==1
            psf = exp(-0.5*(((x-center(1))/sigma).^2 + ((y-center(2))/sigma).^2 + ((z-center(3))/sigma).^2));
        else
            psf = exp(-0.5*((x-center(1))/sigma(1)).^2 -0.5*((y-center(2))/sigma(2)).^2 -0.5*((z-center(3))/sigma(3)).^2);
        end
        psf = psf / sum(psf(:));
        psf = single(psf); % always single

        % ---- GPU: pad and reference calculation ALL on GPU ----
        psf_gpu = gpuArray(psf);   % Move PSF to GPU as single
        [psf_pad, pad_pre, pad_post] = pad_block_to_fft_shape(psf_gpu, sz, 0);

        % --- MATLAB reference: pad, ifftshift, fftn, all on GPU, single
        otf_mat = fftn(ifftshift(psf_pad));   % On GPU
        otf_mat = single(otf_mat);            % Enforce single (should already be, but safe)

        % --- MEX timing (repeat, ignore first run) ---
        t_mex_all = zeros(1, Nrep, 'gpuArray'); % <--- Keep as gpuArray for true GPU timing
        for rr = 1:Nrep
            g = gpuDevice; wait(g); % make sure GPU is idle before timing
            t0 = tic;
            otf_mex = otf_gpu_mex(psf_gpu, sz); % test mex function (gpuArray in, out)
            wait(g); % ensure finished
            t_mex_all(rr) = toc(t0);
        end
        t_mex = double(gather(mean(t_mex_all(2:end)))); % average, ignore first

        % --- MATLAB timing (repeat, ignore first run) ---
        t_mat_all = zeros(1, Nrep, 'gpuArray');
        for rr = 1:Nrep
            g = gpuDevice; wait(g);
            t0 = tic;
            otf_mat_ref = fftn(ifftshift(psf_pad));
            wait(g);
            t_mat_all(rr) = toc(t0);
        end
        t_mat = double(gather(mean(t_mat_all(2:end)))); % average, ignore first

        % --- Error metrics: compare gathered outputs
        otf_mat_gather = gather(otf_mat);
        otf_mex_gather = gather(otf_mex);
        maxErr = double(max(abs(otf_mat_gather(:) - otf_mex_gather(:))));
        rmsErr = double(rms(otf_mat_gather(:) - otf_mex_gather(:)));
        relErr = double(norm(otf_mat_gather(:) - otf_mex_gather(:)) / max(norm(otf_mat_gather(:)), eps('single')));

        pf = relErr < 2e-6; % relax to 2e-6 for roundoff
        pfmark = pass_symbol(pf);
        speedup = 100*(t_mat-t_mex)/t_mat;

        % Print table row
        fprintf('%-3s  %-4d  %-7s %-18s %-13s %-15s %8.2e %8.2e %8.2e %7.3f %+7.0f%%\n', ...
            pfmark, test_id, 'single', fmtvec(sz), sigma_disp, kernel_disp, maxErr, rmsErr, relErr, t_mex, speedup);

        results = [results; pf];
        test_id = test_id+1;
    end
end

fprintf('---------------------------------------------------------------------------------------------------------------\n');
fprintf('Total: %d, Passed: %d, Failed: %d\n', length(results), sum(results), sum(~results));
end

function str = fmtvec(v)
if ischar(v), str = v; return; end
if isscalar(v), str = sprintf('%.2g',v); return; end
str = sprintf('[%s]', strjoin(arrayfun(@(x) sprintf('%.2g',x), v, 'uni',0),' '));
end

function s = pass_symbol(pf)
if pf, s = char([11035 65039 10004 65039]); else, s = char([10060 10060 10060 10060]); end % ✅ or ❌
end

function out = rms(x)
out = sqrt(mean(abs(x).^2));
end

function [bl, pad_pre, pad_post] = pad_block_to_fft_shape(bl, fft_shape, mode)
    sz = size(bl);
    sz = [sz, ones(1, 3-numel(sz))];      % pad size to 3 elements if needed
    fft_shape = [fft_shape(:)', ones(1, 3-numel(fft_shape))]; % ensure row vector, 3 elements
    assert(all(fft_shape >= sz), ...
        sprintf('pad_block_to_fft_shape: bl [%s] is larger than FFT shape [%s], cannot pad', ...
        num2str(sz), num2str(fft_shape)))
    missing = max(fft_shape - sz, 0);
    pad_pre = floor(missing/2);
    pad_post = ceil(missing/2);
    if any(pad_pre > 0 | pad_post > 0)
        bl = padarray(bl, pad_pre, mode, 'pre');
        bl = padarray(bl, pad_post, mode, 'post');
    end
end
