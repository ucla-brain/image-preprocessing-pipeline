function pad = pad_size(x, min_pad_size)
    target = findGoodFFTLength(x + min_pad_size);
    pad_total = target - x;
    pad = pad_total/2;
end

function tf = isfftgood(x)
    f = factor(x);
    tf = all(f <= 7);
end

function x = findGoodFFTLength(x)
    while ~isfftgood(x)
        x = x + 1;
    end
end