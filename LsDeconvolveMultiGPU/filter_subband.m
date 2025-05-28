function img = filter_subband(img, sigma, levels, wavelet, axes)
    % Applies Gaussian notch filtering to wavelet subbands
    % axes: [1] for vertical filtering, [2] for horizontal filtering

    % original_class = class(img);
    % img = im2single(img);

    % Pad image to even dimensions
    pad_x = mod(-size(img,1), 2); % 0 if even, 1 if odd
    pad_y = mod(-size(img,2), 2);
    img = padarray(img, [pad_x, pad_y], 'post');

    % Dynamic range compression
    % img = log1p(img);

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

        % Reshape from C
        H = reshape(C(idxH), S(n + 1, :));
        V = reshape(C(idxV), S(n + 1, :));
        D = reshape(C(idxD), S(n + 1, :));

        % Apply filtering
        if ismember(2, axes)
            % Horizontal filtering on H
            H = filter_coefficient(H, sigma / size(H, 2), 2);
        end
        if ismember(1, axes)
            % Vertical filtering on V
            V = filter_coefficient(V, sigma / size(V, 1), 1);
        end

        % Overwrite filtered values in C
        C(idxH) = H(:);
        C(idxV) = V(:);

        start_idx = idxD(end);  % Move to next level
    end

    % Wavelet reconstruction
    img = waverec2(C, S, wavelet);
    % img = expm1(img);

    % Crop
    img = img(1:end - pad_x, 1:end - pad_y);

    % Restore class
    % switch original_class
    %     case 'uint8'
    %         img = im2uint8(img);
    %     case 'uint16'
    %         img = im2uint16(img);
    %     case 'double'
    %         img = im2double(img);
    %     otherwise
    %         img = max(min(img, 1), 0);
    % end
end

function mat = filter_coefficient(mat, sigma, axis)
    % clamping sigma to avoid potential division by zero or numerical instability
    sigma = max(sigma, 1e-5);
    n = size(mat, axis);
    mat = fft(mat, n, axis);

    % Gaussian filter
    g = gaussian_notch_filter_1d(n, sigma);
    if axis == 1
        g = repmat(g(:), 1, size(mat, 2));
    elseif axis == 2
        g = repmat(g, size(mat, 1), 1);
    else
        error('Invalid axis');
    end

    % Apply filter to complex spectrum
    mat = mat .* complex(g, g);
    mat = real(ifft(mat, n, axis));
end

function g = gaussian_notch_filter_1d(n, sigma)
    x = (0:n-1) - floor(n/2);
    g = 1 - exp(-(x .^ 2) / (2 * sigma ^ 2));
    g = fftshift(g);
end