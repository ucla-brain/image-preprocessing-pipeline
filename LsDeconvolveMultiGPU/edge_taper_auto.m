function bl = edge_taper_auto(bl, psf)
    psf = psf / sum(psf(:));
    if isa(bl, 'gpuArray')
        if ~isa(psf, 'gpuArray'), psf = gpuArray(psf); end
        bl_blur = conv3d_mex(bl, psf);  % your mexcuda kernel (single precision)
        % Build and apply mask as in edge_taper_inplace (see below)
        sz = size(bl);
        nd = numel(sz);
        mask = 1;
        for d = 1:nd
            dimsz = sz(d);
            taper_width = max(8, round(size(psf, d)/2));
            if taper_width > 0 && 2*taper_width < dimsz
                x = linspace(0,1,taper_width+1)';
                mid = ones(dimsz-2*taper_width,1,'like',bl);
                taper = [x; mid; flipud(x)];
            else
                taper = ones(dimsz,1,'like',bl);
            end
            shape = ones(1,nd); shape(d) = dimsz;
            mask = mask .* reshape(taper, shape);
        end
        if ~isa(mask,'gpuArray'), mask = gpuArray(mask); end
        bl = mask .* bl + (1-mask) .* bl_blur;
    else
        if isa(psf, 'gpuArray'), psf = gather(psf); end
        bl = edgetaper(bl, psf);  % MATLAB built-in (for 2D)
        % For 3D, loop over slices:
        % for z = 1:size(bl,3), bl(:,:,z) = edgetaper(bl(:,:,z), psf(:,:,min(z,end))); end
    end
end
