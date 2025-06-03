function taper = make_taper(dimsz, taper_width)
    % Robust 1D taper vector, always length==dimsz, regardless of input type
    dimsz = double(dimsz);
    taper_width = double(taper_width);
    taper_width = min([taper_width, floor(dimsz/2)]);
    if taper_width <= 0
        taper = ones(round(dimsz),1,'single');
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
    % Defensive: guarantee length matches
    if numel(taper) > round(dimsz)
        taper = taper(1:round(dimsz));
    elseif numel(taper) < round(dimsz)
        taper = [taper; ones(round(dimsz - numel(taper)), 1, 'single')];
    end
end