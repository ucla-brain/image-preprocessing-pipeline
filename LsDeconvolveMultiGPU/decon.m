function bl = decon(bl, psf, niter, lambda, stop_criterion, regularize_interval, device_id, use_fft, fft_shape, adaptive_psf)
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
        if adaptive_psf
            bl = deconFFT_Wiener(bl, psf.psf, fft_shape, niter, lambda, stop_criterion, regularize_interval, device_id);
        else
            bl = deconFFT       (bl, psf.psf, fft_shape, niter, lambda, stop_criterion, regularize_interval, device_id);
        end
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

    bl = edgetaper_3d(bl, psf);
    for i = 1:niter
        % start_time = tic;

        apply_regularization = (regularize_interval > 0) && (regularize_interval < niter);
        is_regularization_time = apply_regularization && (i > 1) && (i < niter) && (mod(i, regularize_interval) == 0);

        if is_regularization_time
            bl = imgaussfilt3(bl,0.5,'FilterDomain', 'spatial', 'Padding', 'symmetric');
        end

        buf = convn(bl, psf, 'same');
        buf = max(buf, single(eps('single')));
        buf = bl ./ buf;
        buf = convn(buf, psf_inv, 'same');

        % Apply smoothing and optional Tikhonov every N iterations (except final iteration)
        if is_regularization_time

            if lambda > 0
                reg = convn(bl, R, 'same');
                buf = bl .* buf .* (1 - lambda) + reg .* lambda;
            else
                buf = bl .* buf;
            end
        else
            buf = bl .* buf;
        end

        bl = abs(buf);

        if stop_criterion > 0
            delta_current = norm(bl(:));
            delta_rel = abs(delta_prev - delta_current) / delta_prev * 100;
            delta_prev = delta_current;
            % disp([current_device(device_id) ': Iter ' num2str(i) ...
            %       ', ΔD: ' num2str(delta_rel,3) ...
            %       ', ΔT: ' num2str(round(toc(start_time),1)) 's']);
            if i > 1 && delta_rel <= stop_criterion
                disp('Stop criterion reached. Finishing iterations.');
                break
            end
        % else
        %     disp([current_device(device_id) ': Iter ' num2str(i) ...
        %           ', ΔT: ' num2str(round(toc(start_time),1)) 's']);
        end
    end
end

%=== Frequency-domain version ===
function bl = deconFFT(bl, psf, fft_shape, niter, lambda, stop_criterion, regularize_interval, device_id)
    use_gpu = isgpuarray(bl);

    if use_gpu, psf = gpuArray(psf); end
    [buf_otf, ~, ~] = pad_block_to_fft_shape(psf, fft_shape, 0);
    buf_otf = ifftshift(buf_otf);
    buf_otf = fftn(buf_otf);

    if regularize_interval < niter && lambda > 0
        if use_gpu
            R = single(1/26) * gpuArray.ones(3,3,3, 'single');
        else
            R = single(1/26) *          ones(3,3,3, 'single');
        end
        R(2,2,2) = 0;
    end
    bl = edgetaper_3d(bl, psf);
    [bl, pad_pre, pad_post] = pad_block_to_fft_shape(bl, fft_shape, 0); % 'symmetric'
    if stop_criterion > 0
        delta_prev = norm(bl(:));
    end

    if use_gpu
        buf = gpuArray.zeros(fft_shape, 'single') + 0i;
    else
        buf =          zeros(fft_shape, 'single') + 0i;
    end
    epsilon = single(eps('single'));
    for i = 1:niter
        % start_time = tic;
        apply_regularization = (regularize_interval > 0) && (regularize_interval < niter);
        is_regularization_time = apply_regularization && (i > 1) && (i < niter) && (mod(i, regularize_interval) == 0);
        if is_regularization_time
            bl = imgaussfilt3(bl,0.5,'FilterDomain', 'spatial', 'Padding', 'symmetric');
        end
        buf = fftn(bl);                                                        % x now holds fft(x)             complex
        buf = buf .* buf_otf;                                                  % x now holds fft(x) .* otf      complex
        buf = ifftn(buf);                                                      % inverse fft                    complex
        buf = real(buf);                                                       % convFFT                        real
        buf = max(buf, epsilon);                                               % remove zeros                   real
        buf = bl ./ buf;                                                       %                                real
        buf_otf = conj(buf_otf);                                               % otf_conj                       complex
        buf = fftn(buf);                                                       % x now holds fft(buff)          complex
        buf = buf .* buf_otf;                                                  % x now holds fft(x) .* otf      complex
        buf = ifftn(buf);                                                      % inverse fft                    complex
        buf = real(buf);                                                       % convFFT                        real
        if i<niter
            buf_otf = conj(buf_otf);                                           % otf                            complex
        end
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
            % disp([current_device(device_id) ': Iter ' num2str(i) ...
            %       ', ΔD: ' num2str(delta_rel,3) ...
            %       ', ΔT: ' num2str(round(toc(start_time),1)) 's']);
            if i > 1 && delta_rel <= stop_criterion
                disp('Stop criterion reached. Finishing iterations.');
                break
            end
        % else
        %     disp([current_device(device_id) ': Iter ' num2str(i) ...
        %           ', ΔT: ' num2str(round(toc(start_time),2)) 's']);
        end
    end
    bl = unpad_block(bl, pad_pre, pad_post);
end

function bl = deconFFT_Wiener(bl, psf, fft_shape, niter, lambda, stop_criterion, regularize_interval, device_id)

    % Richardson–Lucy + on-the-fly Wiener PSF refinement
    % RAM-minimal version

    use_gpu = isgpuarray(bl);

    bl = edgetaper_3d(bl, psf);
    [bl, pad_pre, pad_post] = pad_block_to_fft_shape(bl, fft_shape, 0);

    if stop_criterion>0, delta_prev = norm(bl(:)); end

    if use_gpu
        psf = gpuArray(psf);                                             % transfer to GPU
        buff2 = gpuArray.zeros(fft_shape, 'single');                     % allocate directly on GPU
        % Laplacian-like regulariser (only allocated if used)
        if regularize_interval < niter && lambda > 0
            R = single(1/26) * gpuArray.ones(3,3,3, 'single'); R(2,2,2) = 0;       % allocate directly on GPU
        end
    else
        buff2 = zeros(fft_shape, 'single');
        if regularize_interval < niter && lambda > 0
            R = single(1/26) *          ones(3,3,3, 'single'); R(2,2,2) = 0;       % allocate on CPU
        end
    end
    buff1    = complex(buff2, buff2);  % complex(single) zeros
    buff3    = complex(buff2, buff2);
    otf_buff = complex(buff2, buff2);
    bl_previous  = bl;
    G_km2 = buff2;

    epsilon = single(eps('single'));
    psf_sz = size(psf);
    center = floor((fft_shape - psf_sz) / 2) + 1;
    for i = 1:niter
        % ----------- Richardson–Lucy core ------------
        % Y: observed image (blurred, bl)
        % X: current object estimate (sharpened)

        % prepare OTF from PSF
        [otf_buff, ~, ~] = pad_block_to_fft_shape(psf, fft_shape, 0);
        otf_buff = ifftshift(otf_buff);
        otf_buff = fftn(otf_buff);

        % only recompute F{Y} initially or after regularization
        if i == 1
            buff1 = fftn(bl);
        elseif regularize_interval>0 && mod(i, regularize_interval)==0
            bl = imgaussfilt3(bl,0.5,'FilterDomain', 'spatial', 'Padding', 'symmetric');
            buff1 = fftn(bl);                                            % F{Y}                                 complex
        end
        % apply PSF: H{Y}
        buff3 = buff1 .* otf_buff;                                       % H{F{Y}}                              complex
        buff3 = ifftn(buff3);                                            % H{Y}                                 complex
        buff2 = real(buff3);                                             % H{Y}                                 real

        buff2 = max(buff2, epsilon);                                     % H{Y} + epsilon                       real
        buff2 = bl ./ buff2;                                             % Y/H{Y}                               real

        % apply adjoint filter: H'{Y/H{Y}}
        buff3 = fftn(buff2);                                             % F{Y/H{Y}}                            complex
        buff3 = buff3 .* conj(otf_buff);                                 % H'{F{Y/H{Y}}}                        complex
        buff3 = ifftn(buff3);                                            % X/Y                                  complex
        buff2 = real(buff3);                                             % X/Y                                  real

        % optional regularization
        if regularize_interval>0 && mod(i,regularize_interval)==0 && lambda>0 && i<niter
            buff3 = convn(bl, R, 'same');                                % Laplacian                            real
            buff2 = bl .* buff2 .* (1-lambda) + buff3 .* lambda;         % re-equalized X                       real
        else
            buff2 = bl .* buff2;                                         % X                                    real
        end

        bl = abs(buff2);                                                 % X                                    real

        % -------------- Acceleration step ----------------------
        if i > 1
            G_km1 = bl - bl_previous;    % current change
            buff2 = G_km1 .* G_km2;
            accel_lambda = sum(buff2, 'all', 'double');
            buff2 = G_km2 .* G_km2;
            accel_lambda = accel_lambda / (sum(buff2, 'all', 'double') + eps('double'));
            accel_lambda = single(max(0, min(1, accel_lambda))); % clamp for stability
            % ensure λ lives where bl lives and stays single precision:
            if use_gpu,  accel_lambda = gpuArray(single(accel_lambda));
            else,        accel_lambda =          single(accel_lambda);
            end
            buff2 = G_km1 * accel_lambda;
            bl = bl + buff2;
            bl = abs(bl); % store back into bl, enforce positivity
        end

        % Update previous iterates
        G_km2 = G_km1;
        bl_previous  = bl;
        % -------------------------------------------------------

        % ----------- Wiener PSF update ---------------
        % Wiener update: improves PSF estimate based on current object (bl) and blurred image spectrum
        % otf_new = (F{Y} . conj(F{X})) ./ (F{X} . conj(F{X}) + epsilon)
        if i < niter
            buff3 = fftn(bl);                                            % F{X}                                 complex
            otf_buff = conj(buff3);                                      % conj(F{X})                           complex
            buff2 = buff3 .* otf_buff;                                   % |F{X}|^2                             real
            buff2 = max(buff2, epsilon);                                 % |F{X}|^2 + epsilon                   real
            otf_buff = buff1 .* otf_buff;                                % F{Y} .* conj(F{X})                   complex
            buff1 = buff3;                                               % store F{X} for next iteration
            otf_buff = otf_buff ./ buff2;                                % otf_new                              complex

            % inverse FFT to get new PSF
            buff3 = ifftn(otf_buff);                                     % psf                                  complex
            buff2 = real(buff3);                                         % psf                                  real
            psf = buff2(center(1):center(1)+psf_sz(1)-1, ...
                        center(2):center(2)+psf_sz(2)-1, ...
                        center(3):center(3)+psf_sz(3)-1);
            psf = max(psf, 0);                                           % clamp negatives
            sum_psf = sum(psf(:));
            if sum_psf > 0
                psf = psf / sum(psf(:));                                 % normalize to unit energy
            end
            if use_gpu && ~isgpuarray(psf)
                psf = gpuArray(psf);
            end
        end

        % ------------- stopping test -----------------
        if stop_criterion > 0
            delta_cur = norm(bl(:));
            if abs(delta_prev - delta_cur)/delta_prev*100 <= stop_criterion
                fprintf('Stop criterion reached in %d iterations.\n', i);
                break;
            end
            delta_prev = delta_cur;
        end
    end

    bl = unpad_block(bl, pad_pre, pad_post);
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

function bl = unpad_block(bl, pad_pre, pad_post)
    % Ensure pad vectors are 3 elements
    pad_pre  = [pad_pre(:)'  zeros(1,3-numel(pad_pre))];
    pad_post = [pad_post(:)' zeros(1,3-numel(pad_post))];
    sz = size(bl);
    sz = [sz, ones(1, 3-numel(sz))];  % Ensure length 3 for safety

    % Compute the index range for each dimension and check validity
    idx = cell(1, 3);
    for dim = 1:3
        start_idx = pad_pre(dim) + 1;
        end_idx   = sz(dim) - pad_post(dim);
        if start_idx > end_idx || end_idx < 1
            error(['unpad_block: Attempted to unpad dimension %d with invalid indices: [%d:%d], ' ...
                'input size: [%s], pad_pre: [%s], pad_post: [%s]'], ...
                dim, start_idx, end_idx, num2str(sz), num2str(pad_pre), num2str(pad_post));
        end
        idx{dim} = start_idx:end_idx;
    end

    % Actually unpad
    bl = bl(idx{1}, idx{2}, idx{3});

    % Optional: Check if result has at least one element in every dim
    if any(size(bl) < 1)
        error(['unpad_block: Output block size is empty in at least one dimension! ' ...
            'Resulting size: [%s]'], num2str(size(bl)));
    end
end
