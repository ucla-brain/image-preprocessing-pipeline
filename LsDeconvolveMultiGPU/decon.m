function bl = decon(bl, psf, niter, lambda, stop_criterion, regularize_interval, device_id, use_fft)
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
        bl = deconFFT(bl, psf.psf, niter, lambda, stop_criterion, regularize_interval, device_id);
    else
        bl = deconSpatial(bl, psf.psf, psf.inv, niter, lambda, stop_criterion, regularize_interval, device_id);
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

    if regularize_interval < niter && lambda > 0
        R = single(1/26 * ones(3,3,3)); R(2,2,2) = 0;
        if use_gpu, R = gpuArray(R); end
    end

    if stop_criterion > 0
        delta_prev = norm(bl(:));
    end

    for i = 1:niter
        start_time = tic;

        buf = convn(bl, psf, 'same');
        buf = max(buf, eps('single'));
        buf = bl ./ buf;
        buf = convn(buf, psf_inv, 'same');

        % Apply smoothing and optional Tikhonov every N iterations (except final iteration)
        if regularize_interval < niter && mod(i, regularize_interval) == 0
            bl = imgaussfilt3(bl, 0.5);
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
function bl = deconFFT(bl, psf, niter, lambda, stop_criterion, regularize_interval, device_id)
    imsize = size(bl);
    use_gpu = isgpuarray(bl);

    [otf, otf_conj] = getCachedOTF(psf, imsize, use_gpu);  % Always CPU
    if use_gpu
        otf = gpuArray(otf);
        otf_conj = gpuArray(otf_conj);
    end

    if regularize_interval < niter && lambda > 0
        R = single(1/26 * ones(3,3,3)); R(2,2,2) = 0;
        if use_gpu, R = gpuArray(R); end
    end

    if stop_criterion > 0
        delta_prev = norm(bl(:));
    end

    for i = 1:niter
        start_time = tic;

        if i < niter && mod(i, regularize_interval) == 0
            bl = imgaussfilt3(bl, 0.5);
        end

        buf = convFFT(bl, otf);
        buf = max(buf, eps('single'));
        buf = bl ./ buf;
        buf = convFFT(buf, otf_conj);

        if i < niter && mod(i, regularize_interval) == 0
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

function [otf, otf_conj] = getCachedOTF(psf, imsize, use_gpu)
    cache_dir = getCachePath();
    key = ['key_' strrep(mat2str(imsize), ' ', '_')];
    cache_file = fullfile(cache_dir, key);
    lock_file = [cache_file, '.lock'];

    % Wait if locked
    while isfile(lock_file)
        pause(0.05);
    end

    if isfile([cache_file, '.bin']) && isfile([cache_file, '.meta'])
        try
            [otf, otf_conj] = loadOTFCacheBinary(cache_file);
            return;
        catch
            warning('Failed to read binary cache. Recomputing.');
        end
    end

    % Create lock BEFORE writing
    fid = fopen(lock_file, 'w'); fclose(fid);

    try
        % Recompute OTF
        otf = psf;
        if use_gpu, otf = gpuArray(otf); end
        otf = padPSF(otf, imsize);
        otf = fftn(otf);
        otf_conj = conj(otf);
        if use_gpu
            otf = gather(otf);
            otf_conj = gather(otf_conj);
        end

        % Save BOTH .bin and .meta while locked
        saveOTFCacheBinary(cache_file, otf);

    catch e
        warning("Failed to write cache: %s", e.message);
    end

    delete(lock_file);  % Release lock
end

function cache_path = getCachePath()
    % cache_path  = fullfile(tempdir, 'otf_cache');
    cache_path  = fullfile('/data', 'otf_cache');
    if ~exist(cache_path , 'dir')
        mkdir(cache_path);
    end
end

function saveOTFCacheBinary(filename, otf)
    % Split into real and imaginary parts
    otf_real = single(real(otf));
    otf_imag = single(imag(otf));
    shape = size(otf_real);

    % Write binary data
    fid = fopen([filename, '.bin'], 'w');
    fwrite(fid, otf_real(:), 'single');
    fwrite(fid, otf_imag(:), 'single');
    fclose(fid);

    % Save metadata as .mat file
    meta.shape = shape;
    meta.class = 'single';
    save([filename, '.meta'], '-struct', 'meta');
end

function [otf, otf_conj] = loadOTFCacheBinary(filename)
    lock_file = [filename, '.lock'];

    % Wait if a writer is active
    while isfile(lock_file)
        pause(0.05);
    end

    meta = load([filename, '.meta']);
    shape = meta.shape;

    fid = fopen([filename, '.bin'], 'r');
    count = prod(shape);
    real_part = fread(fid, count, 'single');
    imag_part = fread(fid, count, 'single');
    fclose(fid);

    if numel(real_part) < count || numel(imag_part) < count
        error('Cache file is incomplete or corrupted: %s.bin', filename);
    end

    otf = complex(real_part, imag_part);
    otf = reshape(otf, shape);
    otf_conj = conj(otf);
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

function psf_padded = padPSF(psf, imsize)
    psf_padded = zeros(imsize, 'like', psf);
    center = floor((imsize - size(psf)) / 2);
    idx = arrayfun(@(c, s) c+(1:s), center, size(psf), 'UniformOutput', false);
    psf_padded(idx{:}) = psf;
end

function device = device_name(id)
    device = 'CPU ';
    if id > 0
        device = ['GPU' num2str(id)];
    end
end
