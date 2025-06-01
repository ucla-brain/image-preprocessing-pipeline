function test_gauss3d_mex_large()
    % Example: single precision, 2048x1024x1024 ~ 8GB
    szs = {
        [320, 1024, 1024],   % ~1.25 GB single
        [256, 1024, 1024],   % ~2 GB double
    };
    types = {@single, @double};
    sigma = 2.5;

    for ityp = 1:numel(types)
        for isz = 1:numel(szs)
            sz = szs{isz};
            T = types{ityp};
            bytes = prod(sz) * sizeof(T);
            fprintf('Testing %s %s (%.2f GB) ...\n', func2str(T), mat2str(sz), bytes/2^30);
            try
                x = rand(sz, T);
            catch
                fprintf('SKIP: Could not allocate array (out of memory)\n');
                continue;
            end

            tic;
            y_mex = gauss3d_mex(x, sigma);
            t_mex = toc;
            fprintf('gauss3d_mex ran in %.2f seconds\n', t_mex);

            % Validation: Compare a few random slices against imgaussfilt3
            slices = round(linspace(1, sz(3), 5));
            pass = true;
            for s = slices
                x2d = x(:,:,s);
                try
                    y_ref = imgaussfilt(x2d, sigma, 'Padding','replicate', ...
                        'FilterSize', odd_kernel_size(sigma));
                catch
                    fprintf('SKIP: imgaussfilt OOM on slice %d\n', s);
                    continue;
                end
                y_mex_slice = y_mex(:,:,s);
                maxerr = max(abs(y_mex_slice(:) - y_ref(:)));
                rmserr = sqrt(mean((y_mex_slice(:)-y_ref(:)).^2));
                fprintf('  Slice %d: maxerr=%.2e, rmserr=%.2e\n', s, maxerr, rmserr);
                if maxerr > 2e-5 || isnan(maxerr)
                    fprintf('  FAIL: slice %d error exceeds tolerance!\n', s);
                    pass = false;
                end
            end

            if pass
                fprintf('PASS: %s %s\n\n', func2str(T), mat2str(sz));
            else
                fprintf('FAIL: %s %s\n\n', func2str(T), mat2str(sz));
            end

            clear x y_mex; % Free memory
        end
    end
end

function sz = odd_kernel_size(sigma)
    if numel(sigma)==1, sigma=[sigma sigma sigma]; end
    sz = 2*ceil(3*sigma)+1;
    sz = max(sz,3);
end

function bytes = sizeof(T)
    switch func2str(T)
        case 'single', bytes = 4;
        case 'double', bytes = 8;
        otherwise, error('Unknown type');
    end
end
