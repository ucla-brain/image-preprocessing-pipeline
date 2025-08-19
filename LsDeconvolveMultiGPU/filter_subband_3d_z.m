function bl = filter_subband_3d_z(bl, sigma, levels, wavelet)
    % Applies filter_subband to each XZ slice (along Y-axis)
    % In-place update version to avoid extra allocation
    start_time = tic;
    [X, Y, Z] = size(bl);
    device = "CPU";
    if isgpuarray(bl)
        % Use underlyingType to get the data type inside the gpuArray
        original_class = underlyingType(bl);
        dev = gpuDevice();
        device = sprintf("GPU%d", dev.Index);
    else
        original_class = class(bl);
    end
    if ~strcmp(original_class, 'single')
        bl = single(bl);
    end

    % Dynamic range compression
    bl = log1p(bl);

    % Apply filtering across Y axis
    for y = 1:Y
        slice = reshape(bl(:, y, :), [X, Z]);
        slice = filter_subband(slice, sigma, levels, wavelet, [2]);
        bl(:, y, :) = slice;
    end

    % Undo compression
    bl = expm1(bl);

    % Restore original data type
    % Restore original data type (remains on GPU if started as gpuArray)
    if isgpuarray(bl)
        bl_class = classUnderlying(bl);
    else
        bl_class = class(bl);
    end
    if ~strcmp(bl_class, original_class)
        bl = cast(bl, original_class);
    end
    fprintf("%s: destripe ΔT: %.1f s\n", device, toc(start_time));
end

function img = filter_subband(img, sigma, levels, wavelet, axes)
    wl = lower(string(wavelet));
    if startsWith(wl, ["coif","bior","rbio"])
        dwtmode('per','nodisp');
    else
        dwtmode('sym','nodisp');
    end

    if ~isa(img,'single'), img = single(img); end
    pad = [0 0];

    if levels == 0
        levels = wmaxlev(size(img), wavelet);
    end
    [C, S] = wavedec2(img, levels, wavelet);

    start_idx = prod(S(1, :));
    for n = 1:levels
        sz = prod(S(n + 1, :));

        idxH = start_idx + (1:sz);
        idxV = idxH(end) + (1:sz);

        if ismember(2, axes)
            H = reshape(C(idxH), S(n + 1, :));
            H = filter_coefficient(H, sigma / size(img, 1), 2);
            C(idxH) = H(:);
        end
        if ismember(1, axes)
            V = reshape(C(idxV), S(n + 1, :));
            V = filter_coefficient(V, sigma / size(img, 2), 1);
            C(idxV) = V(:);
        end

        start_idx = idxV(end) + sz;
    end

    img = waverec2(C, S, wavelet);

    idx = arrayfun(@(d) 1:(size(img, d) - pad(d)), 1:ndims(img), 'UniformOutput', false);
    img = img(idx{:});
end

function mat = filter_coefficient(mat, sigma, axis)
    sigma = max(sigma, eps('single'));
    n = size(mat, axis);
    sigma = (floor(n/2) + 1) * sigma;

    mat = fft(mat, n, axis);

    g = gaussian_notch_filter_1d(n, sigma, isgpuarray(mat), classUnderlying(mat));

    if axis == 1
        mat = mat .* g(:);
    elseif axis == 2
        mat = mat .* g;
    else
        error('Invalid axis');
    end

    mat = real(ifft(mat, n, axis));
end

function g = gaussian_notch_filter_1d(n, sigma, use_gpu, underlying_class)
    m = floor(n/2);

    if use_gpu
        x = gpuArray.colon(cast(0, underlying_class), cast(1, underlying_class), cast(m, underlying_class));
    else
        x = cast(0:m, underlying_class);
    end

    sigma = cast(sigma, 'like', x);
    gpos = 1 - exp(-x.^2 ./ (2 * sigma.^2));

    if use_gpu
        g = gpuArray.zeros(1, n, underlying_class);
    else
        g = zeros(1, n, underlying_class);
    end

    g(1:m+1) = gpos;
    if mod(n,2) == 0
        g(m+2:n) = gpos(m-1:-1:1);
    else
        g(m+2:n) = gpos(m:-1:1);
    end
end