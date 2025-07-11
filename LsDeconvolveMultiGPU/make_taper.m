function taper = make_taper(dimsz, taper_width)
%MAKE_TAPER Robustly constructs a 1D edge taper vector for a given dimension and taper width.
%   taper = make_taper(dimsz, taper_width)
%   - dimsz: integer, size of dimension to taper (e.g., 64)
%   - taper_width: integer, width of the taper region (e.g., 8)
%
%   Returns: taper [dimsz x 1] single, rising from 0 to 1 at the edge, 1 in the plateau

    dimsz = double(dimsz);
    taper_width = double(taper_width);

    % Clamp taper_width to valid range
    taper_width = min([taper_width, floor(dimsz/2)]);
    if taper_width <= 0
        taper = ones(round(dimsz), 1, 'single');
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
    % Guarantee exact length
    taper = single(taper(:));
    n_out = round(dimsz);
    if numel(taper) > n_out
        taper = taper(1:n_out);
    elseif numel(taper) < n_out
        taper = [taper; ones(n_out-numel(taper),1,'single')];
    end
end
