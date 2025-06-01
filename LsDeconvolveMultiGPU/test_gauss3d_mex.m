function test_gauss3d_mex_vs_imgaussfilt3_gpu()
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

            rng(0);
            x_val = rand(sz, type_str);

            % ---- gauss3d_mex test ----
            reset(g);
            mem0 = g.AvailableMemory;
            tic;
            y_result = gauss3d_mex(x_val, sigma);
            wait(g);
            t_mex = toc;
            mem1 = g.AvailableMemory;
            vram_mex = mem0 - mem1;
            fprintf('  gauss3d_mex: %.2f s, vRAM used: %.2f MB\n', t_mex, vram_mex/2^20);

            % ---- imgaussfilt3(gpuArray) test ----
            reset(g);
            mem0 = g.AvailableMemory;
            tic;
            y_ref_gpu = imgaussfilt3(gpuArray(x_val), sigma, ...
                'Padding', 'replicate', 'FilterSize', odd_kernel_size(sigma));
            wait(g);
            t_matlab_gpu = toc;
            mem1 = g.AvailableMemory;
            vram_matlab = mem0 - mem1;
            y_ref = gather(y_ref_gpu);
            clear y_ref_gpu;
            fprintf('  imgaussfilt3(gpuArray): %.2f s, vRAM used: %.2f MB\n', t_matlab_gpu, vram_matlab/2^20);

            % ---- Exclude edges, validate ----
            margin = max(ceil(4*sigma), 1);
            x_rng = (1+margin):(sz(1)-margin);
            y_rng = (1+margin):(sz(2)-margin);
            z_rng = (1+margin):(sz(3)-margin);

            y_interior    = y_result(x_rng, y_rng, z_rng);
            y_ref_interior = y_ref(x_rng, y_rng, z_rng);

            err = max(abs(y_interior(:) - y_ref_interior(:)));
            rms_err = sqrt(mean((y_interior(:) - y_ref_interior(:)).^2));
            rel_err = rms_err / (max(abs(y_ref_interior(:))) + eps);

            fprintf('  Validation (interior): max = %.2e, RMS = %.2e, rel RMS = %.2e\n', err, rms_err, rel_err);

            % Extra: print mean, min, max for debugging
            fprintf('    mean(y_result) = %.6g, mean(y_ref) = %.6g, mean(diff) = %.6g\n', ...
                mean(y_result(:)), mean(y_ref(:)), mean(y_result(:)-y_ref(:)));
            fprintf('    min/max(y_result) = %.6g/%.6g, min/max(y_ref) = %.6g/%.6g\n', ...
                min(y_result(:)), max(y_result(:)), min(y_ref(:)), max(y_ref(:)));

            fprintf('  Speedup (gauss3d_mex / imgaussfilt3(gpuArray)): %.2fx\n', t_matlab_gpu/t_mex);
            fprintf('  vRAM ratio (mex/Matlab): %.2f\n', vram_mex/max(1,vram_matlab)); % avoid division by zero

            clear y_result y_interior y_ref_interior y_ref
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
