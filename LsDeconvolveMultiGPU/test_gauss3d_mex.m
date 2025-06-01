function test_gauss3d_mex_gpu_inplace()
    % Test gauss3d_mex (in-place) against imgaussfilt3 with all parameter variants

    szs = {
        [32, 128, 128], ...
        [256, 512, 512], ...
        [512, 512, 512]
    };
    types = {@single, @double};
    sigma_tests = {2.5, [2.5 3 4]};
    ksize_tests = {'auto', 13, [13 19 25]};

    g = gpuDevice(1);

    for ityp = 1:numel(types)
        for isz = 1:numel(szs)
            sz = szs{isz};
            T = types{ityp};
            type_str = func2str(T);

            for isig = 1:numel(sigma_tests)
                sigma = sigma_tests{isig};
                for iksz = 1:numel(ksize_tests)
                    ksz = ksize_tests{iksz};
                    % Description string for printout
                    kdesc = ischar(ksz) ? 'auto' : mat2str(ksz);
                    fprintf('\nTesting %s %s, sigma=%s, ksize=%s ...\n', ...
                        type_str, mat2str(sz), mat2str(sigma), kdesc);

                    rng(0);
                    x_val_gpu = gpuArray.rand(sz, type_str);

                    % --- imgaussfilt3 ---
                    opts = {'Padding', 'replicate'};
                    if ~ischar(ksz)
                        opts = [opts, {'FilterSize', ksz}];
                    else
                        opts = [opts, {'FilterSize', odd_kernel_size(sigma)}];
                    end
                    if exist('imgaussfilt3', 'file') && any(strcmp('FilterDomain', ...
                            methods('imgaussfilt3')))
                        opts = [opts, {'FilterDomain', 'spatial'}];
                    end

                    y_ref_gpu = imgaussfilt3(x_val_gpu, sigma, opts{:});

                    % --- gauss3d_mex (in-place) ---
                    if ischar(ksz)
                        y_result_gpu = gauss3d_mex(x_val_gpu, sigma); % default kernel size
                    else
                        y_result_gpu = gauss3d_mex(x_val_gpu, sigma, ksz);
                    end

                    % --- Validation (all on GPU) ---
                    margin = max(ceil(4*max(sigma)), 1); % conservative margin
                    x_rng = (1+margin):(sz(1)-margin);
                    y_rng = (1+margin):(sz(2)-margin);
                    z_rng = (1+margin):(sz(3)-margin);

                    y_interior_gpu    = y_result_gpu(x_rng, y_rng, z_rng);
                    y_ref_interior_gpu = y_ref_gpu(x_rng, y_rng, z_rng);

                    err = max(abs(y_interior_gpu(:) - y_ref_interior_gpu(:)));
                    rms_err = sqrt(mean((y_interior_gpu(:) - y_ref_interior_gpu(:)).^2));
                    rel_err = rms_err / (max(abs(y_ref_interior_gpu(:))) + eps);

                    fprintf('  3D validation (interior): max = %.2e, RMS = %.2e, rel RMS = %.2e\n', ...
                        gather(err), gather(rms_err), gather(rel_err));
                    fprintf('    mean(y_result) = %.6g, mean(y_ref) = %.6g, mean(diff) = %.6g\n', ...
                        gather(mean(y_result_gpu(:))), gather(mean(y_ref_gpu(:))), ...
                        gather(mean(y_result_gpu(:) - y_ref_gpu(:))));
                    fprintf('    min/max(y_result) = %.6g/%.6g, min/max(y_ref) = %.6g/%.6g\n', ...
                        gather(min(y_result_gpu(:))), gather(max(y_result_gpu(:))), ...
                        gather(min(y_ref_gpu(:))), gather(max(y_ref_gpu(:))));

                    if strcmp(type_str, 'single')
                        if gather(err) < 5e-5
                            fprintf('    PASS: single precision\n');
                        else
                            fprintf('    FAIL: single precision\n');
                        end
                    else
                        if gather(err) < 1e-7
                            fprintf('    PASS: double precision\n');
                        else
                            fprintf('    FAIL: double precision\n');
                        end
                    end

                    clear x_val_gpu y_result_gpu y_interior_gpu y_ref_interior_gpu y_ref_gpu
                    reset(g);
                end
            end
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
