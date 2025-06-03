function bl = edge_taper_auto(bl, psf)
% EDGE_TAPER_AUTO  Edge tapering for 3D volumes (GPU: custom CUDA, CPU: edgetaper)
%   bl  - image, 3D, single, cpu or gpuArray(single)
%   psf - psf, 3D, single, cpu or gpuArray(single)
%
% On GPU: expects both bl and psf to be 3D, single, gpuArray
% On CPU: uses MATLAB's edgetaper for each slice (2D only)
%
% Requires conv3d_mex for GPU 3D convolution.

    % Normalize PSF
    psf = psf ./ sum(psf(:));

    if isa(bl, 'gpuArray')
        % --- GPU path (3D only) ---
        assert(strcmp(classUnderlying(bl), 'single') && ndims(bl) == 3, ...
            'bl must be 3D gpuArray single');
        if ~isa(psf, 'gpuArray'), psf = gpuArray(psf); end
        assert(strcmp(classUnderlying(psf), 'single') && ndims(psf) == 3, ...
            'psf must be 3D gpuArray single');

        bl_blur = conv3d_mex(bl, psf); % custom CUDA MEX, see earlier message

        sz = size(bl);
        mask = 1;
        for d = 1:3
            dimsz = sz(d);
            taper_width = max(8, round(size(psf, d)/2));
            taper = make_taper(dimsz, taper_width); % always length==dimsz!
            taper = cast(taper, 'like', bl);

            % Defensive: Guarantee shape matches exactly
            assert(numel(taper) == dimsz, ...
                'Taper length %d does not match dim %d', numel(taper), dimsz);

            shape = ones(1,3); shape(d) = dimsz;
            mask = mask .* reshape(taper, shape);
        end
        if ~isa(mask,'gpuArray'), mask = gpuArray(mask); end

        bl = mask .* bl + (1 - mask) .* bl_blur;

    else
        % --- CPU path (2D edgetaper) ---
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
end

function taper = make_taper(dimsz, taper_width)
    dimsz = double(dimsz);  % ensure numeric
    taper_width = double(taper_width);

    taper_width = min([taper_width, floor(dimsz/2)]);
    if taper_width <= 0
        taper = ones(dimsz,1,'single');
        return
    end
    ramp = linspace(0,1,round(taper_width)+1)';
    if 2*taper_width < dimsz
        plateau = ones(round(dimsz-2*taper_width),1,'single');
        ramp_down = flipud(ramp(1:end-1));
        taper = [ramp; plateau; ramp_down];
    else
        ramp_down = flipud(ramp(1:end-1));
        taper = [ramp; ramp_down];
    end
    % Defensive guarantee of correct length
    if numel(taper) > dimsz
        taper = taper(1:round(dimsz));
    elseif numel(taper) < dimsz
        taper = [taper; ones(round(dimsz - numel(taper)), 1, 'single')];
    end
end

