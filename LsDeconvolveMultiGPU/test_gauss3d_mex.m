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
            sz = szs{isz};            % Ensure sz is numeric vector
            T = types{ityp};          % @single or @double
            bytes = prod(sz) * sizeof(T);
            fprintf('Testing %s %s (%.2f GB)...\n', func2str(T), mat2str(sz), bytes/2^30);

            % Chunk along the third (Z) dimension for GPU RAM safety
            max_gpu_bytes = 9 * 2^30; % 9GB
            slice_bytes = prod(sz(1:2)) * sizeof(T);
            slices_per_chunk = max(floor(max_gpu_bytes / slice_bytes), 1);

            % Preallocate result on CPU with proper type
            y_result = zeros(sz, 'like', feval(T, 0));

            tic;
            for z1 = 1:slices_per_chunk:sz(3)
                z2 = min(z1 + slices_per_chunk - 1, sz(3));
                this_chunk = z2 - z1 + 1;

                % Allocate chunk on GPU, with type
                x_chunk_gpu = gpuArray.rand(sz(1), sz(2), this_chunk, 'like', feval(T, 0));
                % Process chunk (assume gauss3d_mex supports gpuArray)
                y_chunk_gpu = gauss3d_mex(x_chunk_gpu, sigma);
                y_result(:, :, z1:z2) = gather(y_chunk_gpu);

                clear x_chunk_gpu y_chunk_gpu
            end
            t_gpu = toc;
            fprintf('Completed GPU Gaussian filtering in %.2f seconds\n', t_gpu);

            % Validation: compare middle slice to imgaussfilt
            rng(0); % Reproducibility
            midz = round(sz(3)/2);
            x_val = rand(sz(1), sz(2), T);
            y_ref = imgaussfilt(x_val, sigma, ...
                'Padding', 'replicate', 'FilterSize', odd_kernel_size(sigma));
            err = max(abs(y_result(:,:,midz) - y_ref), [], 'all');
            fprintf('  Validation slice: max error = %.2e\n', err);

            clear y_result
        end
    end
end

function sz = odd_kernel_size(sigma)
    % Minimal valid kernel size for given sigma
    if isscalar(sigma)
        sigma = [sigma sigma sigma];
    end
    sz = 2*ceil(3*sigma) + 1;
    sz = max(sz, 3);
end

function b = sizeof(T)
    % Returns bytes for MATLAB type
    switch func2str(T)
        case 'single', b = 4;
        case 'double', b = 8;
        otherwise, error('Unknown type');
    end
end
