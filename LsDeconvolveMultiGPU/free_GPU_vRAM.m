function memGB = free_GPU_vRAM(id, g)
% Returns free GPU memory in MB for device ID `id` (1-based)
% Uses gpuDevice on Linux, nvidia-smi on Windows

if ispc  % Windows
    [~, cmdout] = system('nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits');
    lines = strsplit(strtrim(cmdout), newline);
    mem = str2double(lines) / 1e3;  % convert to GB
else  % Linux
    % g = gpuDevice(id);
    mem = g.AvailableMemory / 1e9;  % convert to GB
end

% Validate ID and return requested
if id > numel(mem) || id < 1 || isnan(mem(id))
    error('Invalid GPU ID or unavailable memory for GPU %d.', id);
end

memGB = mem(id);
end
