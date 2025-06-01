function test_gauss3d_mex_large_gpu()
    % 3D block GPU Gaussian filtering test
    szs = {
        [32, 128, 128], ...
        [256, 512, 512], ...
        [512, 1024, 1024] % Large test
    };
    types = {@single, @double};
    sigma = 2.5;
    g = gpuDevice(1);
    reset(g);
    for ityp = 1:numel(types)
        for isz = 1:numel(szs)
            sz = szs{isz};
            T = types{ityp};
            type_str = func2str(T);
            bytes = prod(sz) * sizeof(T);
            fprintf('Testing %s %s (%.2f GB)...\n', type_str, mat2str(sz), bytes/2^30);

            max_gpu_bytes = 9 * 2^30;
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

            % 3D validation
            rng(0);
            x_val = rand(sz, type_str);
            k3d = odd_kernel_size(sigma);
            y_ref = imgaussfilt3(x_val, sigma, ...
                'Padding', 'replicate', 'FilterSize', k3d); % 3D ground truth

            % Compare the entire output (random slice check is less meaningful in 3D)
            err = max(abs(y_result(:) - y_ref(:)));
            fprintf('  3D validation: max error = %.2e\n', err);

            clear y_result
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
