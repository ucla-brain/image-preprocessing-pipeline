function test_gauss3d_mex_gpu_only()
    % Compare gauss3d_mex and imgaussfilt3 on *the same gpuArray* input

    szs = {
        [32, 128, 128], ...
        [256, 512, 512], ...
        [928, 928, 376]    % ~3.0 GB single
    };
    types = {@single, @double};
    sigma = 2.5;

    g = gpuDevice(1);

    for ityp = 1:numel(types)
        for isz = 1:numel(szs)
            sz = szs{isz};
            T = types{ityp};
            type_str = func2str(T);
            bytes = prod(sz) * sizeof(T);
            fprintf('\nTesting %s %s (%.2f GB)...\n', type_str, mat2str(sz), bytes/2^30);

            % Single gpuArray input used for both!
            rng(0);
            x_val_gpu = gpuArray.rand(sz, type_str);

            % --- gauss3d_mex ---
            reset(g);
            mem0 = g.AvailableMemory;
            tic;
            y_result_gpu = gauss3d_mex(x_val_gpu, sigma);
            wait(g);
            t_mex = toc;
            mem1 = g.AvailableMemory;
            vram_mex = mem0 - mem1;
            y_result = gather(y_result_gpu);

            % --- imgaussfilt3(gpuArray) ---
            reset(g);
            mem0 = g.AvailableMemory;
            k3d = odd_kernel_size(sigma);
            tic;
            y_ref_gpu = imgaussfilt3(x_val_gpu, sigma, ...
                'Padding', 'replicate', 'FilterSize', k3d);
            wait(g);
            t_matlab = toc;
            mem1 = g.AvailableMemory;
            vram_matlab = mem0 - mem1;
            y_ref = gather(y_ref_gpu);

            % --- Exclude edges for validation ---
            margin = max(ceil(4*sigma), 1);
            x_rng = (1+margin):(sz(1)-margin);
            y_rng = (1+margin):(sz(2)-margin);
            z_rng = (1+margin):(sz(3)-margin);

            y_interior    = y_result(x_rng, y_rng, z_rng);
            y_ref_interior = y_ref(x_rng, y_rng, z_rng);

            err = max(abs(y_interior(:) - y_ref_interior(:)));
            rms_err = sqrt(mean((y_interior(:) - y_ref_interior(:)).^2));
            rel_err = rms_err / (max(abs(y_ref_interior(:))) + eps);

            fprintf('  3D validation (interior): max = %.2e, RMS = %.2e, rel RMS = %.2e\n', err, rms_err, rel_err);
            fprintf('    mean(y_result) = %.6g, mean(y_ref) = %.6g, mean(diff) = %.6g\n', ...
                mean(y_result(:)), mean(y_ref(:)), mean(y_result(:)-y_ref(:)));
            fprintf('    min/max(y_result) = %.6g/%.6g, min/max(y_ref) = %.6g/%.6g\n', ...
                min(y_result(:)), max(y_result(:)), min(y_ref(:)), max(y_ref(:)));

            if strcmp(type_str, 'single')
                if err < 5e-5
                    fprintf('    PASS: single precision\n');
                else
                    fprintf('    FAIL: single precision\n');
                end
            else
                if err < 1e-7
                    fprintf('    PASS: double precision\n');
                else
                    fprintf('    FAIL: double precision\n');
                end
            end

            fprintf('  gauss3d_mex: %.2f s, vRAM used: %.2f MB\n', t_mex, vram_mex/2^20);
            fprintf('  imgaussfilt3(gpuArray): %.2f s, vRAM used: %.2f MB\n', t_matlab, vram_matlab/2^20);
            fprintf('  Speedup (gauss3d_mex/imgaussfilt3): %.2fx\n', t_matlab/t_mex);
            fprintf('  vRAM ratio (mex/Matlab): %.2f\n', vram_mex/max(1,vram_matlab));

            clear y_result y_interior y_ref_interior y_ref y_result_gpu y_ref_gpu
            reset(g);
        end
    end
end

function sz = odd_kernel_size(sigma)
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
