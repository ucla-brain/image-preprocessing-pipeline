% test_max_blocksize_gauss3d_mex_descend.m
% Descending search: finds largest N for gauss3d_mex(A, sigma) without OOM

N = 1290;       % starting max size
minN = 64;      % min size to try
step = 16;      % decrement step (try 32 or 64 for speed, then fine-tune)
sigma = 3;      % adjust as needed

gpuInfo = gpuDevice();
success = false;

while N >= minN
    try
        fprintf('Trying N = %d ... ', N);
        A = gpuArray.ones(N, N, N, 'single');
        out = gauss3d_mex(A, sigma);
        clear A out;
        reset(gpuDevice);
        fprintf('Success!\n');
        success = true;
        break;
    catch ME
        fprintf('OOM or error: %s\n', ME.message);
        clear A out;
        reset(gpuDevice);
        N = N - step;
    end
end

if success
    fprintf('\nLargest successful N: %d (%.2f GB)\n', ...
        N, 4*N^3/2^30);
else
    fprintf('\nNo valid N found in tested range.\n');
end

reset(gpuDevice);
