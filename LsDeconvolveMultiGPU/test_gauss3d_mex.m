function test_gauss3d_mex_large_gpu()
    % 3D block GPU Gaussian filtering test (interior validation, timing, 3GB test)
    szs = {
        [32, 128, 128], ...
        [256, 512, 512], ...
        [928, 928, 376]    % ~3.0 GB single
    };
    types = {@single, @double};
    sigma = 2.5;

    g = gpuDevice(1);  % Use first GPU
    reset(g);

    for ityp = 1:numel(types)
        for isz = 1:numel(szs)
            sz = szs{isz};
            T = types{ityp};
            type_str = func2str(T);
            bytes = prod(sz) * sizeof(T);
            fprintf('\nTesting %s %s (%.2f GB)...\n', type_str, mat2str(sz), bytes/2^30);

            % Always use the same input for both tests!
            rng(0);
            x_val = rand(sz, type_str);

            % --- GPU Gaussian Filtering (timed) ---
            tic;
            y_result = gauss3d_mex(x_val, sigma);
            t_gpu = toc;
            fprintf('  gauss3d_mex execution time: %.2f s\n', t_gpu);

            % --- CPU Validation with imgaussfilt3 (timed) ---
            k3d = odd_kernel_size(sigma);
            t_cpu_start = tic;
            y_ref = imgaussfilt3(x_val, sigma, ...
                'Padding', 'replicate', 'FilterSize', k3d); % 3D ground truth
            t_cpu = toc(t_cpu_start);
            fprintf('  imgaussfilt3 execution time: %.2f s\n', t_cpu);

            % --- Exclude edges: margin = ceil(4*sigma) voxels in each dim ---
            margin = max(ceil(4*sigma), 1); % at least 1 voxel
            x_rng = (1+margin):(sz(1)-margin);
            y_rng = (1+margin):(sz(2)-margin);
            z_rng = (1+margin):(sz(3)-margin);

            y_interior    = y_result(x_rng, y_rng, z_rng);
            y_ref_interior = y_ref(x_rng, y_rng, z_rng);

            % --- Error metrics ---
            err = max(abs(y_interior(:) - y_ref_interior(:)));
            rms_err = sqrt(mean((y_interior(:) - y_ref_interior(:)).^2));
            rel_err = rms_err / (max(abs(y_ref_interior(:))) + eps);

            fprintf('  3D validation (interior): max = %.2e, RMS = %.2e, rel RMS = %.2e\n', err, rms_err, rel_err);

            % Extra: print statistics for debugging
            fprintf('    mean(y_result) = %.6g, mean(y_ref) = %.6g, mean(diff) = %.6g\n', ...
                mean(y_result(:)), mean(y_ref(:)), mean(y_result(:)-y_ref(:)));
            fprintf('    min/max(y_result) = %.6g/%.6g, min/max(y_ref) = %.6g/%.6g\n', ...
                min(y_result(:)), max(y_result(:)), min(y_ref(:)), max(y_ref(:)));

            % --- Pass/fail threshold ---
            if strcmp(type_str, 'single')
                if err < 5e-5
                    fprintf('    PASS: single precision\n');
                else
                    fprintf('    FAIL: single precision\n');
                end
            else
                if err < 1e-8
                    fprintf('    PASS: double precision\n');
                else
                    fprintf('    FAIL: double precision\n');
                end
            end

            % --- Speedup reporting ---
            fprintf('  Speedup (gauss3d_mex/imgaussfilt3): %.2fx\n', t_cpu/t_gpu);

            clear y_result y_interior y_ref_interior y_ref

            % --- GPU memory reporting ---
            g = gpuDevice; % Update device info
            fprintf('  GPU free memory after test: %.2f GB\n', g.AvailableMemory/2^30);
            reset(g);
        end
    end
end

function sz = odd_kernel_size(sigma)
    % Returns a 3-element vector for 3D filter size
    if isscalar(sigma)
        sigma = [sigma sigma sigma];
    end
    sz = 2*ceil(3*sigma) + 1;
    sz = max(sz, 3);
end

function b = sizeof(T)
    switch func2str(T)
        case 'single', b = 4;
        case 'double', b = 8;
        otherwise, error('Unknown type');
    end
end
