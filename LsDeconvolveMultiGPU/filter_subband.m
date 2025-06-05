function img = filter_subband(img, sigma, levels, wavelet, axes)
    % Applies Gaussian notch filtering to wavelet subbands
    % axes: [1] for vertical filtering, [2] for horizontal filtering

    % Pad image to even dimensions
    pad = mod(size(img), 2);
    img = padarray(img, pad, 'post');

    % Wavelet decomposition
    if levels == 0
        levels = wmaxlev(size(img), wavelet);
    end
    [C, S] = wavedec2(img, levels, wavelet);

    % Track starting index in C (skip approximation part)
    start_idx = prod(S(1, :));
    for n = 1:levels
        sz = prod(S(n + 1, :));

        % Indices for detail coefficients at level n
        idxH = start_idx + (1:sz);
        idxV = idxH(end) + (1:sz);
        idxD = idxV(end) + (1:sz);

        % Apply filtering
        % if ismember(3, axes)
        %     % Diagonal filtering on D
        %     D = reshape(C(idxD), S(n + 1, :));
        %     D = filter_coefficient(D, sigma / mean(size(D)), 3);
        %     C(idxD) = D(:);
        % end
        if ismember(2, axes)
            % Horizontal filtering on H
            H = reshape(C(idxH), S(n + 1, :));
            H = filter_coefficient(H, sigma / size(H, 2), 2);
            C(idxH) = H(:);
        end
        if ismember(1, axes)
            % Vertical filtering on V
            V = reshape(C(idxV), S(n + 1, :));
            V = filter_coefficient(V, sigma / size(V, 1), 1);
            C(idxV) = V(:);
        end
        start_idx = idxD(end);  % Move to next level
    end

    % Wavelet reconstruction
    img = waverec2(C, S, wavelet);

    % Unpadding (generalized)
    idx = arrayfun(@(d) 1:(size(img, d) - pad(d)), 1:ndims(img), 'UniformOutput', false);
    img = img(idx{:});
end

function mat = filter_coefficient(mat, sigma, axis)
    % clamping sigma to avoid potential division by zero or numerical instability
    sigma = max(sigma, eps('single'));
    n = size(mat, axis);
    mat = fft(mat, n, axis);

    % Gaussian filter
    g = gaussian_notch_filter_1d(n, sigma, isgpuarray(mat));
    if axis == 1
        g = repmat(g(:), 1, size(mat, 2));
    elseif axis == 2
        g = repmat(g, size(mat, 1), 1);
    else
        error('Invalid axis');
    end

    % Convert filter to gpuArray if mat is gpuArray
    if isgpuarray(mat)
        g = gpuArray(g);
    end

    % Apply filter to complex spectrum
    mat = mat .* complex(g, g);
    mat = real(ifft(mat, n, axis));
end

function g = gaussian_notch_filter_1d(n, sigma, use_gpu)
    x = (0:n-1) - floor(n/2);
    x = single(x);
    if use_gpu, x = gpuArray(x); end
    g = 1 - exp(-(x .^ 2) / (2 * sigma ^ 2));
    g = fftshift(g);
end