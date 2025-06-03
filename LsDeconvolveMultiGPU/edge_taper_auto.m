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

    % DEBUG: Print array types/sizes BEFORE promotion
    disp(['class(bl): ' class(bl) ', ndims(bl): ' num2str(ndims(bl)) ', size: ' mat2str(size(bl))]);
    disp(['class(psf): ' class(psf) ', ndims(psf): ' num2str(ndims(psf)) ', size: ' mat2str(size(psf))]);

    % Promote both bl and psf to 3D (safe allocation)
    bl  = promote_to_3d_safe(bl);
    psf = promote_to_3d_safe(psf);

    orig_2d = (size(bl,3) == 1);

    % DEBUG: Print array types/sizes AFTER promotion
    disp(['class(bl): ' class(bl) ', ndims(bl): ' num2str(ndims(bl)) ', size: ' mat2str(size(bl))]);
    disp(['class(psf): ' class(psf) ', ndims(psf): ' num2str(ndims(psf)) ', size: ' mat2str(size(psf))]);

    % Normalize PSF
    psf = psf ./ sum(psf(:));

    if isa(bl, 'gpuArray')
        assert(strcmp(classUnderlying(bl), 'single'), 'bl must be gpuArray single');
        if ~isa(psf, 'gpuArray'), psf = gpuArray(psf); end
        assert(strcmp(classUnderlying(psf), 'single'), 'psf must be gpuArray single');

        bl_blur = conv3d_mex(bl, psf); % your CUDA MEX

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
        if size(bl,3) == 1
            bl2d = edgetaper(bl(:,:,1), psf(:,:,1));
            % Safe assignment for 3D
            bl_tmp = zeros(size(bl,1), size(bl,2), 1, 'like', bl);
            bl_tmp(:,:,1) = bl2d;
            bl = bl_tmp;
        else
            for k = 1:size(bl,3)
                bl(:,:,k) = edgetaper(bl(:,:,k), psf(:,:,min(k,size(psf,3))));
            end
        end
    end

    % Squeeze to 2D if original input was 2D
    if orig_2d
        bl = squeeze(bl);
    end
end

function arr3 = promote_to_3d_safe(arr)
    sz = size(arr);
    if numel(sz) < 3
        sz(3) = 1;
    end
    if size(arr,3) == 1 && ndims(arr) == 2
        arr3 = zeros(sz(1), sz(2), 1, 'like', arr);
        arr3(:,:,1) = arr;
    else
        arr3 = arr; % already 3D
    end
end
