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
        bl = deconFFT(bl, psf.psf, fft_shape, niter, lambda, stop_criterion, regularize_interval, device_id);
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
            if use_gpu, clear buf; bl = gauss3d_gpu(bl, 0.5); else, bl = imgaussfilt3(bl, 0.5); end
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

% === Frequency-domain version with cached OTFs ===
% function bl = deconFFT(bl, psf, fft_shape, niter, lambda, stop_criterion, regularize_interval, device_id)
%     use_gpu = isgpuarray(bl);
%
%     otf = calculate_otf(psf, fft_shape, device_id);
%     if use_gpu, otf_conj = conj_gpu(otf); else, otf_conj = conj(otf); end
%
%     if regularize_interval < niter && lambda > 0
%         R = single(1/26 * ones(3,3,3)); R(2,2,2) = 0;
%         if use_gpu, R = gpuArray(R); end
%     end
%
%     bl = edgetaper_3d(bl, psf);
%
%     [bl, pad_pre, pad_post] = pad_block_to_fft_shape(bl, fft_shape, 0); % 'symmetric'
%
%     if stop_criterion > 0
%         delta_prev = norm(bl(:));
%     end
%
%     for i = 1:niter
%         % start_time = tic;
%
%         apply_regularization = (regularize_interval > 0) && (regularize_interval < niter);
%         is_regularization_time = apply_regularization && (i > 1) && (i < niter) && (mod(i, regularize_interval) == 0);
%
%         if is_regularization_time
%             if use_gpu, clear buf; bl = gauss3d_gpu(bl, 0.5); else, bl = imgaussfilt3(bl, 0.5); end
%         end
%
%         buf = convFFT(bl, otf);
%         buf = max(buf, eps('single'));
%         buf = bl ./ buf;
%         buf = convFFT(buf, otf_conj);
%
%         if is_regularization_time
%             if lambda > 0
%                 reg = convn(bl, R, 'same');
%                 bl = bl .* buf .* (1 - lambda) + reg .* lambda;
%             else
%                 bl = bl .* buf;
%             end
%         else
%             bl = bl .* buf;
%         end
%
%         bl = abs(bl);
%
%         if stop_criterion > 0
%             delta_current = norm(bl(:));
%             delta_rel = abs(delta_prev - delta_current) / delta_prev * 100;
%             delta_prev = delta_current;
%             % disp([current_device(device_id) ': Iter ' num2str(i) ...
%             %       ', ΔD: ' num2str(delta_rel,3) ...
%             %       ', ΔT: ' num2str(round(toc(start_time),1)) 's']);
%             if i > 1 && delta_rel <= stop_criterion
%                 disp('Stop criterion reached. Finishing iterations.');
%                 break
%             end
%         % else
%         %     disp([current_device(device_id) ': Iter ' num2str(i) ...
%         %           ', ΔT: ' num2str(round(toc(start_time),2)) 's']);
%         end
%     end
%
%     bl = unpad_block(bl, pad_pre, pad_post);
% end

function bl = deconFFT(bl, psf, fft_shape, niter, lambda, stop_criterion, ...
                        regularize_interval, device_id)

    % Richardson–Lucy + on-the-fly Wiener PSF refinement
    % RAM-minimal version: **only buff1 & buff2 are ever allocated**

    use_gpu = isgpuarray(bl);
    dtype   = classUnderlying(bl);

    otf = calculate_otf(psf, fft_shape, device_id);

    % Laplacian-like regulariser (only allocated if used)
    if regularize_interval < niter && lambda>0
        R = single(1/26*ones(3,3,3)); R(2,2,2)=0;
        if use_gpu, R = gpuArray(R); end
    end

    bl = edgetaper_3d(bl, psf);
    [bl, pad_pre, pad_post] = pad_block_to_fft_shape(bl, fft_shape, 0);

    if stop_criterion>0, delta_prev = norm(bl(:)); end

    for i = 1:niter
        apply_regularization = (regularize_interval > 0) && (regularize_interval < niter);
        is_regularization_time = apply_regularization && (i > 1) && (i < niter) && (mod(i, regularize_interval) == 0);

        if is_regularization_time
            if use_gpu, bl = gauss3d_gpu(bl, 0.5); else, bl = imgaussfilt3(bl, 0.5); end
        end

        % ----------- Richardson–Lucy core ------------
        buff1 = convFFT(bl, otf);                                        % buff1: H⊗x
        buff1 = max(buff1, eps(dtype));                                  % avoid /0
        buff2 = bl ./ buff1;                                             % buff2: ratio
        if use_gpu, buff1 = conj_gpu(otf); else, buff1 = conj(otf); end  % buff1: otf_conj
        buff2 = convFFT(buff2, buff1);                                   % buff2: correction

        if regularize_interval>0 && mod(i,regularize_interval)==0 && lambda>0 && i<niter
            buff1 = convn(bl, R, 'same');         % buff1 reused as Laplacian
            bl    = bl .* buff2 .* (1-lambda) + buff1 .* lambda;
        else
            bl    = bl .* buff2;
        end

        bl = abs(bl);

        % ----------- Wiener PSF update ---------------
        if i<niter
            buff1 = fftn(bl, fft_shape);          % buff1: F{X}
            buff2 = convFFT(bl, otf);
            buff2 = fftn(buff2, fft_shape);       % buff2: F{Y}

            buff2 = buff2 .* conj(buff1);         % buff2: numerator  F{Y}·F{X}*
            buff1 = abs(buff1);
            buff1 = buff1.^2;
            buff1 = buff1 + eps(dtype);           % buff1: denom |F{X}|^2+ε
            buff2 = buff2 ./ buff1;               % buff2: OTF_new
            buff1 = ifftn(buff2);
            buff1 = real(buff1);                  % buff1: PSF estimate

            % crop --> psf, normalise, regenerate OTFs
            sz  = size(psf);
            c   = floor((fft_shape - sz)/2) + 1;
            psf = buff1(c(1):c(1)+sz(1)-1, c(2):c(2)+sz(2)-1, c(3):c(3)+sz(3)-1);
            psf = max(psf,0);
            psf = psf / sum(psf(:));
            otf = calculate_otf(psf, fft_shape, device_id);
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

function x = convFFT(x, otf)
    %CONVFFT  Frequency–domain convolution with fixed precision.
    %
    %   y = convFFT(x, otf)
    %
    %   Arguments
    %   ---------
    %     x   – real/complex volume (single or gpuArray/single)
    %     otf – pre-computed OTF with the *same* class as x
    %
    %   The result y is real(single) and the same “gather-state” (CPU vs GPU)
    %   as the input.

    x =  fftn(x);   % x now holds fft(x)
    x = x .* otf;                  % x now holds fft(x) .* otf
    x = ifftn(x);
    x = real(x);                   % final output
end

function otf = calculate_otf(psf, fft_shape, device_id)
    %CALCULATE_OTF  Build an OTF in single precision (CPU or GPU).
    %
    %   otf = calculate_otf(psf, fft_shape, device_id)
    %
    %   • psf         – real PSF, any numeric class
    %   • fft_shape   – 3-element vector, power-of-2 friendly
    %   • device_id   – >0 ⇒ use GPU via otf_gpu
    %
    %   Both outputs are single (CPU) or gpuArray/single (GPU).

    % t_compute = tic;
    if ~isa(psf, 'single'), psf = single(psf); end
    if device_id > 0
        psf = gpuArray(psf);
        otf = otf_gpu(psf, fft_shape);
    else
        [otf, ~, ~] = pad_block_to_fft_shape(psf, fft_shape, 0);
        otf = ifftshift(otf);
        otf = fftn(otf);
    end
    % fprintf('%s: OTF computed for size %s in %.2fs\n', ...
    %     current_device(device_id), mat2str(fft_shape), toc(t_compute));
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
