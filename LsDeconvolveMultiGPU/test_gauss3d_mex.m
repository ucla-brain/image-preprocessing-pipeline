function test_gauss3d_mex_large_gpu()
    % 3D block GPU Gaussian filtering test (with interior-only validation)
    szs = {
        [32, 128, 128], ...
        [256, 512, 512], ...
        [768, 768, 768]    % ~1.8 GB single, ~3.6 GB double (fits 3GB in single)
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
            fprintf('Testing %s %s (%.2f GB)...\n', type_str, mat2str(sz), bytes/2^30);

            max_gpu_bytes = 9 * 2^30;  % Max per-chunk (safe)
            slice_bytes = prod(sz(1:2)) * sizeof(T);
            slices_per_chunk = max(floor(max_gpu_bytes / slice_bytes), 1);

            y_result = zeros(sz, type_str);

            tic;
            for z1 = 1:slices_per_chunk:sz(3)
                z2 = min(z1 + slices_per_chunk - 1, sz(3));
                this_chunk = double(z2 - z1 + 1);

                x_chunk_gpu = gpuArray.rand(sz(1), sz(2), this_chunk, type_str);
                y_chunk_gpu = gauss3d_mex(x_chunk_gpu, sigma);
                y_result(:, :, z1:z2) = gather(y_chunk_gpu);

                clear x_chunk_gpu y_chunk_gpu
            end
            t_gpu = toc;
            fprintf('Completed GPU Gaussian filtering in %.2f seconds\n', t_gpu);

            % Validation: only test the interior region (exclude edges)
            rng(0);
            x_val = rand(sz, type_str);
            k3d = odd_kernel_size(sigma);
            y_ref = imgaussfilt3(x_val, sigma, ...
                'Padding', 'replicate', 'FilterSize', k3d); % 3D ground truth

            r = floor(k3d/2);  % margin in each dimension
            x_rng = (1+r(1)):(sz(1)-r(1));
            y_rng = (1+r(2)):(sz(2)-r(2));
            z_rng = (1+r(3)):(sz(3)-r(3));
            y_interior    = y_result(x_rng, y_rng, z_rng);
            y_ref_interior = y_ref(x_rng, y_rng, z_rng);

            err = max(abs(y_interior(:) - y_ref_interior(:)));
            rms_err = sqrt(mean((y_interior(:) - y_ref_interior(:)).^2));
            rel_err = rms_err / (max(abs(y_ref_interior(:))) + eps);

            fprintf('  3D validation (interior): max = %.2e, RMS = %.2e, rel RMS = %.2e\n', err, rms_err, rel_err);

            % Pass/fail threshold
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

            clear y_result y_interior y_ref_interior

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
