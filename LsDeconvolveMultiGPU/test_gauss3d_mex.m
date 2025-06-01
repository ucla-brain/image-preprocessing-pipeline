function test_gauss3d_mex_large_gpu()
    % Large-array GPU Gaussian filtering with gpuArray support
    szs = {
        [32, 128, 128], ...
        [256, 512, 512], ...
        [2048, 1024, 1024] % Larger example (~8GB single)
    };
    types = {@single, @double};
    sigma = 2.5;

    for ityp = 1:numel(types)
        for isz = 1:numel(szs)
            sz = szs{isz};
            T = types{ityp};
            bytes = prod(sz) * sizeof(T);
            fprintf('Testing %s %s (%.2f GB)...\n', func2str(T), mat2str(sz), bytes/2^30);

            % Determine chunk size along third dimension to fit GPU
            max_gpu_bytes = 9 * 2^30; % Safe margin, 9 GB
            slice_bytes = prod(sz(1:2)) * sizeof(T);
            slices_per_chunk = max(floor(max_gpu_bytes / slice_bytes), 1);

            % Allocate result on CPU
            y_gpu = zeros(sz, func2str(T));

            tic;
            for z = 1:slices_per_chunk:sz(3)
                z_end = min(z + slices_per_chunk - 1, sz(3));
                chunk_sz = [sz(1), sz(2), z_end - z + 1];

                % FIX: Unpack dimensions as separate arguments
                x_chunk_gpu = gpuArray.rand(chunk_sz(1), chunk_sz(2), chunk_sz(3), 'like', feval(T, 0));

                % Process chunk directly on GPU
                y_chunk_gpu = gauss3d_mex(x_chunk_gpu, sigma);

                % Retrieve chunk from GPU
                y_gpu(:, :, z:z_end) = gather(y_chunk_gpu);

                clear x_chunk_gpu y_chunk_gpu; % Free GPU memory
            end
            t_gpu = toc;
            fprintf('Completed GPU Gaussian filtering in %.2f seconds\n', t_gpu);

            % Basic validation (random slice)
            rng(0); % For reproducibility
            slice = round(sz(3)/2);
            x_validation = rand(sz(1:2), T);
            y_ref = imgaussfilt(x_validation, sigma, 'Padding', 'replicate', ...
                'FilterSize', odd_kernel_size(sigma));

            % Compare to the same slice in the processed GPU array
            maxerr = max(abs(y_gpu(:, :, slice) - y_ref), [], 'all');
            fprintf('  Validation slice: max error = %.2e\n', maxerr);

            clear y_gpu;
        end
    end
end

function sz = odd_kernel_size(sigma)
    if numel(sigma)==1, sigma = [sigma sigma sigma]; end
    sz = 2*ceil(3*sigma) + 1;
    sz = max(sz, 3);
end

function bytes = sizeof(T)
    switch func2str(T)
        case 'single', bytes = 4;
        case 'double', bytes = 8;
        otherwise, error('Unknown type');
    end
end
