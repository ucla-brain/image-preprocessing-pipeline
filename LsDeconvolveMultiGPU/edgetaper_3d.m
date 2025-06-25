function bl = edgetaper_3d(bl, psf)
% EDGE_TAPER_AUTO  Edge tapering for 3D (GPU: conv3d_gpu, CPU: imfilter)
%   bl  - image, 3D, single, cpu or gpuArray(single)
%   psf - psf, 3D, single, cpu or gpuArray(single)
%
% On GPU: expects both bl and psf to be 3D, single, gpuArray
% On CPU: expects 3D, single
%
% Requires conv3d_gpu for GPU 3D convolution.
% Requires make_taper.m in your path.

    % Normalize PSF, check positivity/finiteness
    assert(all(isfinite(psf(:))) && all(psf(:) >= 0), 'PSF must be non-negative and finite');
    psf = psf ./ sum(psf(:));

    if isa(bl, 'gpuArray')
        assert(strcmp(classUnderlying(bl), 'single') && ndims(bl) == 3, 'bl must be 3D gpuArray single');
        if ~isa(psf, 'gpuArray'), psf = gpuArray(psf); end
        assert(strcmp(classUnderlying(psf), 'single') && ndims(psf) == 3, 'psf must be 3D gpuArray single');
        bl_blur = conv3d_gpu(bl, psf);
    else
        assert(strcmp(class(bl), 'single') && ndims(bl) == 3, 'bl must be 3D single');
        if isa(psf, 'gpuArray'), psf = gather(psf); end
        if ~isa(psf, 'double' ), psf = double(psf); end
        bl_blur = imfilter(bl, psf, 'replicate', 'same', 'conv');
    end

    sz = size(bl);
    mask = ones(1, 'like', bl);  % Start as 'like', keeps class
    for d = 1:3
        dimsz = sz(d);
        taper_width = max(8, round(size(psf, d)/2));
        taper = make_taper(dimsz, taper_width);
        taper = cast(taper, 'like', bl);
        assert(numel(taper) == dimsz, ...
            'Taper length %d does not match dim %d', numel(taper), dimsz);
        shape = ones(1,3); shape(d) = dimsz;
        mask = mask .* reshape(taper, shape);
    end
    if isa(bl, 'gpuArray') && ~isa(mask,'gpuArray')
        mask = gpuArray(mask);
    end

    bl = mask .* bl + (1 - mask) .* bl_blur;
end
