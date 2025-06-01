function img3d = filter_subband_3d_z(img3d, sigma, levels, wavelet)
    % Applies filter_subband to each XZ slice (along Y-axis)
    % In-place update version to avoid extra allocation

    [X, Y, Z] = size(img3d);
    original_class = class(img3d);
    if ~isa(img3d, 'single')
        img3d = single(img3d);
    end

    % Dynamic range compression
    img3d = log1p(img3d);

    % Apply filtering across Y axis
    for y = 1:Y
        slice = reshape(img3d(:, y, :), [X, Z]);
        slice = filter_subband(slice, sigma, levels, wavelet, [2]);
        img3d(:, y, :) = slice;
    end

    % Undo compression
    img3d = expm1(img3d);

    % Restore original data type
    if ~isa(img3d, original_class)
        img3d = cast(img3d, original_class);
    end
end