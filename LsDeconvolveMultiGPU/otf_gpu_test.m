function otf_gpu_test
fprintf('\n');
fprintf('PF   Test  Type    Size              Sigma         Kernel          maxErr    RMS       relErr    mex(s)   Speedup\n');
fprintf('---------------------------------------------------------------------------------------------------------------\n');

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
        psf = single(psf); % <--- unshifted

        % GPU: Pad and reference calculation ALL on GPU in single precision
        psf_gpu = gpuArray(psf);   % Move PSF to GPU as single
        [psf_pad, pad_pre, pad_post] = pad_block_to_fft_shape(psf_gpu, sz, 0);

        % Warm up
        otf_mat = zeros(sz, 'single', 'gpuArray');
        otf_mat = complex(otf_mat, otf_mat);
        otf_mat = otf_gpu(gpuArray(psf), sz, otf_mat);

        % --- MATLAB reference: pad, ifftshift, fftn, all on GPU, single
        otf_mat = fftn(ifftshift(psf_pad));

        % --- MEX timing (repeat, ignore first run) ---
        t_mex_all = zeros(1,Nrep);
        for rr = 1:Nrep
            t0 = tic;
            otf_mex = otf_gpu(gpuArray(psf), sz);
            t_mex_all(rr) = toc(t0);
        end
        t_mex = mean(t_mex_all(2:end)); % average, ignore first

        % --- MATLAB timing (repeat, ignore first run) ---
        t_mat_all = zeros(1,Nrep);
        for rr = 1:Nrep
            t0 = tic;
            otf_mat = fftn(ifftshift(psf_pad));
            t_mat_all(rr) = toc(t0);
        end
        t_mat = mean(t_mat_all(2:end)); % average, ignore first

        otf_mat = single(otf_mat);  % force single precision, but should already be

        % gather for error calculation
        maxErr = double(max(abs(gather(otf_mat(:))-gather(otf_mex(:)))));
        rmsErr = double(rms(gather(otf_mat(:))-gather(otf_mex(:))));
        relErr = double(norm(gather(otf_mat(:))-gather(otf_mex(:))) / max(norm(gather(otf_mat(:))),eps('single')));

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
    % Ensure 3 dimensions for both bl and fft_shape
    sz = size(bl);
    sz = [sz, ones(1, 3-numel(sz))];      % pad size to 3 elements if needed
    fft_shape = [fft_shape(:)', ones(1, 3-numel(fft_shape))]; % ensure row vector, 3 elements

    % Compute missing for each dimension
    assert(all(fft_shape >= sz), ...
        sprintf('pad_block_to_fft_shape: bl [%s] is larger than FFT shape [%s], cannot pad', ...
        num2str(sz), num2str(fft_shape)))
    missing = max(fft_shape - sz, 0);

    % Vectorized pad pre and post calculation
    pad_pre = floor(missing/2);
    pad_post = ceil(missing/2);

    % Only pad if needed
    if any(pad_pre > 0 | pad_post > 0)
        bl = padarray(bl, pad_pre, mode, 'pre');
        bl = padarray(bl, pad_post, mode, 'post');
    end
end