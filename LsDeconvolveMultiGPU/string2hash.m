function h = string2hash(str)
    str = double(str);
    h = 5381;
    for i = 1:length(str)
        h = mod(h * 33 + str(i), 2^31 - 1);  % DJB2 hash
    end
end
