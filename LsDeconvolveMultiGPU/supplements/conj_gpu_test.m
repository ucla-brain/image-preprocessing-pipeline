% Set target size: 8GB for complex single (8 bytes/element)
clear;
reset(gpuDevice);
target_GB = 8;
bytes_per_element = 8; % single complex: 4 bytes real + 4 bytes imag
numel_needed = target_GB * 2^30 / bytes_per_element;
dim = floor(sqrt(numel_needed)); % Use a square matrix for simplicity
fprintf('Creating %dx%d complex single gpuArray (%.2f GB)\n', dim, dim, dim^2 * bytes_per_element / 2^30);

% Generate large random gpuArray
A = gpuArray.rand(dim, dim, 'single') + 1i * gpuArray.rand(dim, dim, 'single');

% Warm-up (important for GPU timing)
tmp = conj(A); clear tmp; wait(gpuDevice);

% Benchmark
tic;
B = conj_gpu(A);
wait(gpuDevice);
elapsed = toc;

% Optional: verify correctness on a small sub-block
idx = 1:5;  % Just for display
fprintf('A(1:5,1:5):\n'); disp(gather(A(idx,idx)));
fprintf('conj(A)(1:5,1:5):\n'); disp(gather(B(idx,idx)));

fprintf('conj(A) on %.2f GB gpuArray took %.3f seconds.\n', dim^2 * bytes_per_element / 2^30, elapsed);
clear;
reset(gpuDevice);
