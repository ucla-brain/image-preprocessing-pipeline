function memGB = free_GPU_vRAM(id, g)
% Returns free GPU memory in GB for device ID `id` (1-based)
% On Windows uses `nvidia-smi`; on Linux assumes `g` is already set

if ispc  % Windows
    [~, cmdout] = system('nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits');
    lines = strsplit(strtrim(cmdout), newline);
    mem = str2double(lines) / 1e3;  % convert MB to GB

    % Validate and index
    if id > numel(mem) || id < 1 || isnan(mem(id))
        error('Invalid GPU ID or unavailable memory for GPU %d.', id);
    end
    memGB = mem(id);

else  % Linux/macOS
    if nargin < 2 || g.DeviceIndex ~= id
        error('On non-Windows platforms, provide gpuDevice(id) as second argument.');
    end
    memGB = g.AvailableMemory / 1e9;  % scalar, no need to index
end
end
