function bl = decon(bl, psf, niter, lambda, stop_criterion, regularize_interval, device_id, use_fft)
    % Performs Richardson-Lucy or blind deconvolution (with optional Tikhonov regularization).
    %
    % Inputs:
    % - bl: 3D observed volume or initial estimate (single or gpuArray)
    % - psf: 3D point spread function (must match type and device of bl)
    % - niter: number of iterations
    % - lambda: Tikhonov regularization weight (applies only in blind mode)
    % - stop_criterion: early stopping threshold in % change (0 disables)
    % - regularize_interval: enables blind mode when > 0 (with PSF updates + smoothing)
    % - device_id: string label for logging (e.g. 'GPU0', 'CPU')
    % - use_fft: true = use FFT-based convolution (faster, more memory), false = use convn (slower, low-memory)

    % === Device detection and type consistency ===
    use_gpu = isa(bl, 'gpuArray');

    if ~isa(bl, 'single'), bl = single(bl); end
    if ~isa(psf, 'single'), psf = single(psf); end

    % Ensure psf matches bl's device
    if use_gpu && ~isa(psf, 'gpuArray'), psf = gpuArray(psf); end
    if ~use_gpu && isa(psf, 'gpuArray'), psf = gather(psf); end
    if use_gpu && ~isa(bl, 'gpuArray'), bl = gpuArray(bl); end

    imsize = size(bl);

    % === Precompute FFT of padded PSF if needed ===
    if use_fft
        otf = fftn(padPSF(psf, imsize));
        otf_conj = conj(otf);  % flipped PSF in Fourier domain
    end

    % === Tikhonov kernel: 3D Laplacian-like stencil ===
    R = single(1/26 * ones(3,3,3)); R(2,2,2) = 0;
    if use_gpu, R = gpuArray(R); end

    delta_prev = [];

    for i = 1:niter
        start_time = tic;

        % === Blind update path (PSF + smoothing + optional Tikhonov) ===
        if regularize_interval > 0 && mod(i, regularize_interval) == 0 && i > regularize_interval
            % === Regularize image ===
            bl = imgaussfilt3(bl, 0.5);

            % === buf: ratio = bl / conv(bl, psf) ===
            if use_fft
                buf = convFFT(bl, otf);                        % forward convolution (FFT)
            else
                buf = convn(bl, psf, 'same');                 % forward convolution (spatial)
            end
            buf = max(buf, eps('single'));
            buf = bl ./ buf;                                 % RL ratio

            % === Blind PSF update ===
            psf = psf .* convn(flipall(bl), buf, 'same');    % spatial update (always convn)
            psf = max(psf, 0);
            psf = psf / sum(psf(:));
            psf = imgaussfilt3(psf, 0.5);                    % smooth PSF
            psf = psf / sum(psf(:));                         % normalize again

            % === Recompute OTF after PSF change ===
            if use_fft
                otf = fftn(padPSF(psf, imsize));
                otf_conj = conj(otf);
                buf = convFFT(buf, otf_conj);                % backward convolution (FFT)
            else
                buf = convn(buf, flipall(psf), 'same');      % backward convolution (spatial)
            end

            % === Apply RL update with optional Tikhonov ===
            if lambda > 0
                reg = convn(bl, R, 'same');                  % regularization term
                bl = bl .* buf .* (1 - lambda) + reg .* lambda;
            else
                bl = bl .* buf;
            end

        else
            % === Standard RL update ===
            if use_fft
                buf = convFFT(bl, otf);                      % forward blur
                buf = max(buf, eps('single'));
                buf = bl ./ buf;                             % RL ratio
                buf = convFFT(buf, otf_conj);                % deblur
            else
                buf = convn(bl, psf, 'same');                % forward blur
                buf = max(buf, eps('single'));
                buf = bl ./ buf;                             % RL ratio
                buf = convn(buf, flipall(psf), 'same');      % deblur
            end
            bl = bl .* buf;
        end

        % === Cleanup numerical noise ===
        bl = abs(bl);

        % === Early stopping (Δ norm) ===
        if stop_criterion > 0
            delta_current = norm(bl(:));
            if isempty(delta_prev)
                delta_rel = 0;
            else
                delta_rel = abs(delta_prev - delta_current) / delta_prev * 100;
            end
            delta_prev = delta_current;

            disp([device_name(device_id) ': Iter ' num2str(i) ...
                  ', Δ: ' num2str(delta_rel,3) ...
                  ', Time: ' num2str(round(toc(start_time),1)) 's']);

            if i > 1 && delta_rel <= stop_criterion
                disp('Stop criterion reached. Finishing iterations.');
                break
            end
        else
            disp([device_name(device_id) ': Iter ' num2str(i) ...
                  ', Time: ' num2str(round(toc(start_time),1)) 's']);
        end
    end
end

function y = convFFT(x, otf)
    % Optimized frequency-domain convolution for low VRAM usage.
    % Performs: y = real(ifftn(fftn(x) .* otf))
    %
    % Key: avoids holding both fft(x) and ifftn result simultaneously.

    % Compute FFT of input
    x = fftn(x);              % x now holds fft(x)

    % Multiply with OTF (in-place)
    x = x .* otf;             % x now holds fft(x) .* otf

    % Compute inverse FFT, overwrite x with result
    x = ifftn(x);
    y = real(x);       % final output
end

function x = flipall(x)
    % Flips array along all dimensions (for PSF symmetry)
    for d = 1:ndims(x)
        x = flip(x, d);
    end
end

function otf = padPSF(psf, imsize)
    % Pads and centers a PSF to the full image size before FFT
    psf_padded = zeros(imsize, 'like', psf);
    center = floor((imsize - size(psf)) / 2);
    idx = arrayfun(@(c, s) c+(1:s), center, size(psf), 'UniformOutput', false);
    psf_padded(idx{:}) = psf;
    otf = fftn(psf_padded);
end

function device = device_name(id)
    device = 'CPU ';
    if id > 0
        device = ['GPU' num2str(id)];
    end
end
