function benchmark_filter_subband_3d()
% ==============================================================
% benchmark_filter_subband_3d.m
%
% Compares CPU vs GPU performance and correctness for
% filter_subband_3d_z() over various 3D single-precision volumes.
% ==============================================================

sizes = [32, 64, 128, 256];  % cube sizes (i.e., 32x32x32 to 256³)
sigma = 1000;
levels = 0;
wavelet = 'db9';
num_trials = 3;

fprintf('Benchmarking filter_subband_3d_z (σ=%d, levels=%d, wavelet=%s)\n', sigma, levels, wavelet);
fprintf('-------------------------------------------------------------\n');
fprintf('%8s | %10s | %10s | %8s\n', 'Size³', 'CPU Time (s)', 'GPU Time (s)', 'Speedup');
fprintf('-------------------------------------------------------------\n');

for s = sizes
    sz = [s, s, s];
    cpu_times = zeros(1, num_trials);
    gpu_times = zeros(1, num_trials);
    max_diff = NaN;

    for t = 1:num_trials
        % Generate test volume
        rng(42 + t);  % Slight variation
        bl = rand(sz, 'single');

        % --- CPU ---
        bl_cpu = bl;  % fresh copy
        tic;
        bl_cpu_out = filter_subband_3d_z(bl_cpu, sigma, levels, wavelet);
        cpu_times(t) = toc;

        % --- GPU ---
        bl_gpu = gpuArray(bl);  % fresh copy
        wait(gpuDevice);
        tic;
        bl_gpu_out = filter_subband_3d_z(bl_gpu, sigma, levels, wavelet);
        wait(gpuDevice);
        gpu_times(t) = toc;

        % Compare output
        if t == 1
            bl_gpu_out_cpu = gather(bl_gpu_out);
            diff = abs(bl_cpu_out - bl_gpu_out_cpu);
            max_diff = max(diff(:));
        end
    end

    t_cpu = mean(cpu_times);
    t_gpu = mean(gpu_times);
    speedup = t_cpu / t_gpu;

    fprintf('%8d | %10.4f | %10.4f | %8.2fx\n', s, t_cpu, t_gpu, speedup);
    if max_diff < 1e-4
        fprintf('         ✅ max abs diff: %.2e (PASS)\n', max_diff);
    else
        fprintf('         ❌ max abs diff: %.2e (FAIL)\n', max_diff);
    end
end

fprintf('-------------------------------------------------------------\n');
end
