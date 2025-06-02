function test_otf_gpu_mex_tabular
fprintf('\n');
fprintf('PF   Test  Type    Size              Sigma         Kernel          maxErr    RMS       relErr    mex(s)   Speedup\n');
fprintf('---------------------------------------------------------------------------------------------------------------\n');

sz = [1200 1200 1200];
kernels = {'auto', 9, [9 11 15], 3, 41};
sigmas = {2.5, [2.5 2.5 2.5], [0.5 0.5 2.5], 0.25, 8};
results = [];

test_id = 1;
for s = 1:length(sigmas)
    sigma = sigmas{s};
    for k = 1:length(kernels)
        kernel = kernels{k};

        % For reporting
        sigma_disp = fmtvec(sigma);
        kernel_disp = fmtvec(kernel);

        % Build PSF (Gaussian with custom kernel size or auto, always single)
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
        psf_shifted = ifftshift(single(psf));
        psf_shifted_gpu = gpuArray(psf_shifted);

        % Zero pad to output size
        padsize = max(sz - kernsz, 0);
        prepad = floor(padsize/2);
        postpad = padsize - prepad;
        psf_pad = padarray(psf_shifted, prepad, 0, 'pre');
        psf_pad = padarray(psf_pad, postpad, 0, 'post');
        psf_pad = psf_pad(1:sz(1), 1:sz(2), 1:sz(3));

        % -- Run MEX
                Nrep = 10;   % Repeat for robustness

        % --- Warm up GPU and code paths
        [~,~] = otf_gpu_mex(psf_shifted_gpu, sz);
        otf_mat = fftn(psf_pad);

        % --- MEX timing (repeat, ignore first run) ---
        t_mex_all = zeros(1,Nrep);
        for rr = 1:Nrep
            t0 = tic;
            [otf_mex, otf_conj_mex] = otf_gpu_mex(psf_shifted_gpu, sz);
            t_mex_all(rr) = toc(t0);
        end
        t_mex = mean(t_mex_all(2:end)); % average, ignore first

        % --- MATLAB timing (repeat, ignore first run) ---
        t_mat_all = zeros(1,Nrep);
        for rr = 1:Nrep
            t0 = tic;
            otf_mat = fftn(psf_pad);
            t_mat_all(rr) = toc(t0);
        end
        t_mat = mean(t_mat_all(2:end)); % average, ignore first

        otf_mat = single(otf_mat);  % force single precision for fair diff

        maxErr = double(max(abs(otf_mat(:)-gather(otf_mex(:)))));
        rmsErr = double(rms(otf_mat(:)-gather(otf_mex(:))));
        relErr = double(norm(otf_mat(:)-gather(otf_mex(:))) / max(norm(otf_mat(:)),eps('single')));

        pf = relErr < 2e-6; % relax to 2e-6 for roundoff
        pfmark = pass_symbol(pf);

        % Speedup
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
if pf, s = char([11035 65039 10004 65039]); else, s = char([10060 10060 10060 10060]); end % ✅ or ❌❌❌❌
end

function out = rms(x)
out = sqrt(mean(abs(x).^2));
end
