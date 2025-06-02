function bl = decon(bl, psf, niter, lambda, stop_criterion, regularize_interval, device_id, use_fft, fft_shape)
    % Performs Richardson-Lucy or blind deconvolution (with optional Tikhonov regularization).
    %
    % Inputs:
    % - bl: 3D observed volume or initial estimate (single or gpuArray)
    % - psf: 3D point spread function (must match type and device of bl)
    % - niter: number of iterations
    % - lambda: Tikhonov regularization weight (applies during Gaussian regularization steps)
    % - stop_criterion: early stopping threshold in % change (0 disables)
    % - regularize_interval: enables blind mode when > 0 (with PSF updates + smoothing)
    % - device_id: int, 0 = cpu, >0 = GPU
    % - use_fft: true = use FFT-based convolution (faster, more memory), false = use convn (slower, low-memory)

    if use_fft
        bl = deconFFT(bl, psf.shifted, fft_shape, niter, lambda, stop_criterion, regularize_interval, device_id);
    else
        bl = deconSpatial(bl, psf.psf, psf.inv  , niter, lambda, stop_criterion, regularize_interval, device_id);
    end
end

% === Spatial-domain version ===
function bl = deconSpatial(bl, psf, psf_inv, niter, lambda, stop_criterion, regularize_interval, device_id)

    if ~isa(bl, 'single'), bl = single(bl); end
    if ~isa(psf, 'single'), psf = single(psf); end
    if ~isa(psf_inv, 'single'), psf_inv = single(psf_inv); end

    use_gpu = isgpuarray(bl);

    if use_gpu
        psf = gpuArray(psf);
        psf_inv = gpuArray(psf_inv);
    end

    % Apply smoothing and optional Tikhonov every N iterations (except final iteration)

    if regularize_interval < niter && lambda > 0
        R = single(1/26 * ones(3,3,3)); R(2,2,2) = 0;
        if use_gpu, R = gpuArray(R); end
    end

    if stop_criterion > 0
        delta_prev = norm(bl(:));
    end

    for i = 1:niter
        start_time = tic;

        apply_regularization = (regularize_interval > 0) && (regularize_interval < niter);
        is_regularization_time = apply_regularization && (i > 1) && (i < niter) && (mod(i, regularize_interval) == 0);

        if is_regularization_time
            if device_id > 0
                bl = gauss3d_mex(bl, buf, 0.5);
            else
                bl = imgaussfilt3(bl, 0.5);
            end
        end

        buf = convn(bl, psf, 'same');
        buf = max(buf, eps('single'));
        buf = bl ./ buf;
        buf = convn(buf, psf_inv, 'same');

        % Apply smoothing and optional Tikhonov every N iterations (except final iteration)
        if is_regularization_time

            if lambda > 0
                reg = convn(bl, R, 'same');
                bl = bl .* buf .* (1 - lambda) + reg .* lambda;
            else
                bl = bl .* buf;
            end
        else
            bl = bl .* buf;
        end

        bl = abs(bl);

        if stop_criterion > 0
            delta_current = norm(bl(:));
            delta_rel = abs(delta_prev - delta_current) / delta_prev * 100;
            delta_prev = delta_current;
            disp([device_name(device_id) ': Iter ' num2str(i) ...
                  ', ΔD: ' num2str(delta_rel,3) ...
                  ', ΔT: ' num2str(round(toc(start_time),1)) 's']);
            if i > 1 && delta_rel <= stop_criterion
                disp('Stop criterion reached. Finishing iterations.');
                break
            end
        else
            disp([device_name(device_id) ': Iter ' num2str(i) ...
                  ', ΔT: ' num2str(round(toc(start_time),1)) 's']);
        end
    end
end

% === Frequency-domain version with cached OTFs ===
function bl = deconFFT(bl, psf, fft_shape, niter, lambda, stop_criterion, regularize_interval, device_id)
    [otf, otf_conj] = calculate_otf(psf, fft_shape, device_id);

    if regularize_interval < niter && lambda > 0
        R = single(1/26 * ones(3,3,3)); R(2,2,2) = 0;
        if device_id > 0, R = gpuArray(R); end
    end

    [bl, pad_pre, pad_post] = pad_block_to_fft_shape(bl, fft_shape, 'replicate');

    if stop_criterion > 0
        delta_prev = norm(bl(:));
    end

    for i = 1:niter
        start_time = tic;

        apply_regularization = (regularize_interval > 0) && (regularize_interval < niter);
        is_regularization_time = apply_regularization && (i > 1) && (i < niter) && (mod(i, regularize_interval) == 0);

        if is_regularization_time
            if device_id > 0
                bl = gauss3d_mex(bl, buf, 0.5);
            else
                bl = imgaussfilt3(bl, 0.5);
            end
        end

        buf = convFFT(bl, otf);
        buf = max(buf, eps('single'));
        buf = bl ./ buf;
        buf = convFFT(buf, otf_conj);

        if is_regularization_time
            if lambda > 0
                reg = convn(bl, R, 'same');
                bl = bl .* buf .* (1 - lambda) + reg .* lambda;
            else
                bl = bl .* buf;
            end
        else
            bl = bl .* buf;
        end

        bl = abs(bl);

        if stop_criterion > 0
            delta_current = norm(bl(:));
            delta_rel = abs(delta_prev - delta_current) / delta_prev * 100;
            delta_prev = delta_current;
            disp([device_name(device_id) ': Iter ' num2str(i) ...
                  ', ΔD: ' num2str(delta_rel,3) ...
                  ', ΔT: ' num2str(round(toc(start_time),1)) 's']);
            if i > 1 && delta_rel <= stop_criterion
                disp('Stop criterion reached. Finishing iterations.');
                break
            end
        else
            disp([device_name(device_id) ': Iter ' num2str(i) ...
                  ', ΔT: ' num2str(round(toc(start_time),2)) 's']);
        end
    end

    bl = unpad_block(bl, pad_pre, pad_post);
end

function x = convFFT(x, otf)
    x = fftn(x);              % x now holds fft(x)
    x = x .* otf;             % x now holds fft(x) .* otf
    x = ifftn(x);
    x = real(x);              % final output
end

function device = device_name(id)
    device = 'CPU ';
    if id > 0
        device = ['GPU' num2str(id)];
    end
end

function [otf, otf_conj] = calculate_otf(psf_shifted, fft_shape, device_id)
    t_compute = tic;
    if ~isa(psf_shifted, 'single'), psf_shifted = single(psf_shifted); end
    if device_id > 0, psf_shifted = gpuArray(psf_shifted); end
    [otf, ~, ~] = pad_block_to_fft_shape(psf_shifted, fft_shape, 0);
    otf = fftn(otf);
    % if device_id > 0, otf = arrayfun(@(r, i) complex(r, i), real(otf), imag(otf)); end
    otf_conj = conj(otf);
    % if device_id > 0, otf_conj = arrayfun(@(r, i) complex(r, i), real(otf_conj), imag(otf_conj)); end
    fprintf('%s: OTF computed for size %s in %.2fs\n', ...
        device_name(device_id), mat2str(fft_shape), toc(t_compute));
end

function [bl, pad_pre, pad_post] = pad_block_to_fft_shape(bl, fft_shape, mode)
    % Ensure 3 dimensions for both bl and fft_shape
    sz = size(bl);
    sz = [sz, ones(1, 3-numel(sz))];      % pad size to 3 elements if needed
    fft_shape = [fft_shape(:)', ones(1, 3-numel(fft_shape))]; % ensure row vector, 3 elements

    % Compute missing for each dimension
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

function bl = unpad_block(bl, pad_pre, pad_post)
    % Ensure pad vectors are 3 elements
    pad_pre  = [pad_pre(:)'  zeros(1,3-numel(pad_pre))];
    pad_post = [pad_post(:)' zeros(1,3-numel(pad_post))];
    sz = size(bl);

    idx = arrayfun(@(dim) ...
        (pad_pre(dim)+1):(sz(dim)-pad_post(dim)), ...
        1:ndims(bl), 'UniformOutput', false);

    bl = bl(idx{:});
end