function bl = edge_taper_auto(bl, psf)
% EDGE_TAPER_AUTO  Edge tapering for 2D/3D (GPU: conv3d_mex, CPU: edgetaper)
%   bl  - image, 2D or 3D, single, cpu or gpuArray(single)
%   psf - psf, 2D or 3D, single, cpu or gpuArray(single)
%
% On GPU: expects both bl and psf to be 3D, single, gpuArray (auto-promotes 2D)
% On CPU: uses MATLAB's edgetaper for each slice (2D only)
%
% Requires conv3d_mex for GPU 3D convolution.
% Requires make_taper.m in your path.

    % DEBUG: Print array types/sizes
    disp(['class(bl): ' class(bl) ', ndims(bl): ' num2str(ndims(bl)) ', size: ' mat2str(size(bl))]);
    disp(['class(psf): ' class(psf) ', ndims(psf): ' num2str(ndims(psf)) ', size: ' mat2str(size(psf))]);

    % Promote both bl and psf to 3D (guaranteed)
    sz_bl  = size(bl);
    sz_psf = size(psf);
    if numel(sz_bl) < 3,  sz_bl(3)  = 1; end
    if numel(sz_psf) < 3, sz_psf(3) = 1; end
    bl  = reshape(bl,  sz_bl(1), sz_bl(2), sz_bl(3));
    psf = reshape(psf, sz_psf(1), sz_psf(2), sz_psf(3));
    orig_2d = (sz_bl(3) == 1);

    % DEBUG: Print array types/sizes
    disp(['class(bl): ' class(bl) ', ndims(bl): ' num2str(ndims(bl)) ', size: ' mat2str(size(bl))]);
    disp(['class(psf): ' class(psf) ', ndims(psf): ' num2str(ndims(psf)) ', size: ' mat2str(size(psf))]);

    % Normalize PSF
    psf = psf ./ sum(psf(:));

    if isa(bl, 'gpuArray')
        assert(strcmp(classUnderlying(bl), 'single') && ndims(bl) == 3, ...
            'bl must be 3D gpuArray single');
        if ~isa(psf, 'gpuArray'), psf = gpuArray(psf); end
        assert(strcmp(classUnderlying(psf), 'single') && ndims(psf) == 3, ...
            'psf must be 3D gpuArray single');
        bl_blur = conv3d_mex(bl, psf);

        sz = size(bl);
        mask = 1;
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
        if ~isa(mask,'gpuArray'), mask = gpuArray(mask); end
        bl = mask .* bl + (1 - mask) .* bl_blur;

    else
        if isa(psf, 'gpuArray'), psf = gather(psf); end
        if ndims(bl) == 2
            bl = edgetaper(bl, psf);
        elseif ndims(bl) == 3
            for k = 1:size(bl,3)
                bl(:,:,k) = edgetaper(bl(:,:,k), psf(:,:,min(k,size(psf,3))));
            end
        else
            error('CPU path only supports 2D or 3D arrays');
        end
    end

    % Squeeze if original input was 2D
    if orig_2d
        bl = squeeze(bl);
    end
end