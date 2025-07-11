function arr = load_lz4(filename)
    % load_lz4: Robust wrapper for load_lz4_mex to accept char or string scalar
    if isstring(filename)
        if isscalar(filename)
            filename = char(filename);
        else
            error('Filename must be a string scalar or char row vector (not a string array).');
        end
    elseif ~ischar(filename)
        error('Filename must be a char row vector or string scalar.');
    end
    arr = load_lz4_mex(filename);
end