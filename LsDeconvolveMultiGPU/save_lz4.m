function save_lz4(fname, arr)
    if isstring(fname)
        if isscalar(fname)
            fname = char(fname);
        else
            error('Filename must be a scalar string, not an array.');
        end
    end
    save_lz4_mex(fname, arr);
end
