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
            bl = imgaussfilt3(bl, 0.5);
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
function bl = deconFFT(bl, psf, niter, lambda, stop_criterion, regularize_interval, device_id)
    imsize = size(bl);
    use_gpu = isgpuarray(bl);

    [otf, otf_conj] = getCachedOTF(psf, imsize, use_gpu);
    if use_gpu
        if ~isgpuarray(otf), otf = gpuArray(otf); end
        if ~isgpuarray(otf_conj), otf_conj = gpuArray(otf_conj); end
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

        apply_regularization = (regularize_interval > 0) && (regularize_interval < niter);
        is_regularization_time = apply_regularization && (i > 1) && (i < niter) && (mod(i, regularize_interval) == 0);

        if is_regularization_time
            bl = imgaussfilt3(bl, 0.5);
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
                  ', ΔT: ' num2str(round(toc(start_time),1)) 's']);
        end
    end
end

function [otf, otf_conj] = getCachedOTF(psf, imsize, use_gpu)
    cache_dir = getCachePath();
    key_str = ['key_' strrep(mat2str(imsize), ' ', '_')];
    base = fullfile(cache_dir, key_str);
    OFFSET = 1e5;
    sem_key = double(string2hash(key_str)) + OFFSET;
    registerSemaphoreKey(sem_key);  % record for cleanup

    % === Try to load cache ===
    if isfile([base, '.bin']) && isfile([base, '.meta'])
        try
            [otf, otf_conj] = loadOTFCacheMapped(base);
            disp(['Loaded cached OTF for size ' mat2str(imsize)]);
            return;
        catch e
            warnNoBacktrace('getCachedOTF:CacheReadFailed', 'Failed to read binary cache. %s', e.message);
        end
    end

    % === Compute OTF ===
    try
        disp(['Recomputing OTF for size ' mat2str(imsize)]);
        otf = padPSF(psf, imsize);
        if use_gpu, otf = gpuArray(otf); end
        otf = fftn(otf);
        otf_conj = conj(otf);
    catch e
        error('getCachedOTF:ComputationFailed', 'Failed to compute OTF: %s', e.message);
    end

    % === Save to cache (async, thread-safe)
    try
        if use_gpu
            otf_cpu = gather(otf);
            otf_conj_cpu = gather(otf_conj);
        else
            otf_cpu = otf;
            otf_conj_cpu = otf_conj;
        end
        % f = parfeval(backgroundPool, @saveOTFCacheMapped, 0, base, otf_cpu, otf_conj_cpu, sem_key);
        % afterAll(f, @(fut) checkFutureError(fut), 0);
        disp(['Caching OTF for size ' mat2str(imsize)]);
        saveOTFCacheMapped(base, otf_cpu, otf_conj_cpu, sem_key);
    catch e
        warnNoBacktrace('getCachedOTF:SaveCacheFailed', 'OTF computed but failed to save: %s', e.message);
    end
end

function checkFutureError(fut)
    if ~isempty(fut.Error)
        warning('Async OTF save failed: %s', fut.Error.message);
    end
end

function warnNoBacktrace(id, msg, varargin)
    % Validate and sanitize warning ID
    if ~ischar(id) && ~isstring(id)
        id = "warnNoBacktrace:InvalidID";
    end
    id = char(id);

    % Validate and format message
    if nargin > 2
        try
            msg = sprintf(msg, varargin{:});
        catch
            msg = 'Warning formatting failed.';
        end
    end

    if ~ischar(msg) && ~isstring(msg)
        msg = 'Unknown warning message';
    end
    msg = char(msg);

    % Suppress backtrace
    st = warning('query', 'backtrace');
    warning('off', 'backtrace');
    warning(id, msg);
    warning(st.state, 'backtrace');
end

function saveOTFCacheMapped(base, otf, otf_conj, sem_key)
     % === Skip if file already exists
    if isfile([base, '.bin']) && isfile([base, '.meta'])
        return;
    end

    % === Lock
    semaphore('w', sem_key);
    cleanup_sem = onCleanup(@() semaphore('p', sem_key));

    % === Skip if file already exists after wait time was over
    if isfile([base, '.bin']) && isfile([base, '.meta'])
        return;
    end

    try
        otf_real   = single(real(otf));
        otf_imag   = single(imag(otf));
        conj_real  = single(real(otf_conj));
        conj_imag  = single(imag(otf_conj));
        shape      = size(otf_real);

        tmp_bin = [base, '.bin.tmp'];
        tmp_meta = [base, '.meta.tmp'];
        final_bin = [base, '.bin'];
        final_meta = [base, '.meta'];

        % Write .bin
        fid = fopen(tmp_bin, 'wb');
        if fid == -1
            error('Cannot open file for writing: %s', tmp_bin);
        end
        fwrite(fid, [otf_real(:); otf_imag(:); conj_real(:); conj_imag(:)], 'single');
        fclose(fid);
        fileattrib(tmp_bin, '+w', 'a');

        % Write .meta
        try
            % save(tmp_meta, '-struct', 'meta', '-v7');
            fid = fopen(tmp_meta, 'w');
            if fid == -1, error('Failed to open meta file for writing'); end

            fprintf(fid, 'shape %s\n', mat2str(shape));
            fprintf(fid, 'class single\n');
            fprintf(fid, 'version 2\n');
            fclose(fid);
        catch e
            error('Meta save failed: %s', e.message);
        end
        fileattrib(tmp_meta, '+w', 'a');

        movefile(tmp_bin, final_bin, 'f');
        movefile(tmp_meta, final_meta, 'f');
    catch e
        warnNoBacktrace('getCachedOTF:SaveCacheFailed', ...
                        'OTF computed but failed to save: %s', e.message);
    end
end

function [otf, otf_conj] = loadOTFCacheMapped(filename)
    % === Parse metadata
    meta = struct();
    fid = fopen([filename '.meta'], 'r');
    if fid == -1
        error('Cannot open meta file: %s.meta', filename);
    end

    while true
        line = fgetl(fid);
        if ~ischar(line), break; end  % EOF or invalid
        line = strtrim(line);
        if isempty(line), continue; end

        tokens = regexp(line, '^(\S+)\s+(.*)$', 'tokens', 'once');
        if isempty(tokens), continue; end

        key = lower(tokens{1});
        value = strtrim(tokens{2});

        switch key
            case 'shape'
                meta.shape = str2num(strrep(value, '[', ''));  %#ok<ST2NM>
            case 'class'
                meta.class = value;
            case 'version'
                meta.version = str2double(value);
        end
    end
    fclose(fid);

    % === Validate metadata
    if ~isfield(meta, 'shape') || ~isfield(meta, 'version') || meta.version ~= 2
        error('Invalid or missing metadata in %s.meta', filename);
    end
    shape = meta.shape;

    % === Load binary data
    fid = fopen([filename, '.bin'], 'rb');
    if fid == -1
        error('Cannot open binary cache file: %s.bin', filename);
    end

    count = prod(shape);
    total = 4 * count;
    data = fread(fid, total, 'single');
    fclose(fid);

    if numel(data) < total
        error('Incomplete or corrupted binary cache file: %s.bin', filename);
    end

    % === Reconstruct OTF and its conjugate
    otf      = reshape(complex(data(1:count), data(count+1:2*count)), shape);
    otf_conj = reshape(complex(data(2*count+1:3*count), data(3*count+1:end)), shape);
end


function registerSemaphoreKey(key)
    persistent used_keys cleanupObj
    if isempty(used_keys)
        used_keys = containers.Map('KeyType', 'double', 'ValueType', 'logical');
        cleanupObj = onCleanup(@destroyAllSemaphores);
    end
    if ~isKey(used_keys, key)
        semaphore('c', key, 1);  % create with count = 1
        used_keys(key) = true;
    end
end

function destroyAllSemaphores()
    persistent used_keys
    if isempty(used_keys), return; end
    keys = used_keys.keys;
    for i = 1:numel(keys)
        try
            semaphore('d', keys{i});
        catch
            warning('Failed to destroy semaphore %d: %s', keys{i}, lasterr);  %#ok<LERR>
        end
    end
end

function h = string2hash(str)
    str = double(str);
    h = 5381;
    for i = 1:length(str)
        h = mod(h * 33 + str(i), 2^31 - 1);  % DJB2
    end
end

function cache_path = getCachePath()
    % cache_path  = fullfile(tempdir, 'otf_cache');
    cache_path  = fullfile('/data', 'otf_cache');
    if ~exist(cache_path , 'dir')
        mkdir(cache_path);
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

function psf_padded = padPSF(psf, imsize)
    psf_padded = zeros(imsize, 'like', psf);
    center = floor((imsize - size(psf)) / 2);
    idx = arrayfun(@(c, s) c + (1:s), center, size(psf), 'UniformOutput', false);
    psf_padded(idx{:}) = psf;
end

function device = device_name(id)
    device = 'CPU ';
    if id > 0
        device = ['GPU' num2str(id)];
    end
end
