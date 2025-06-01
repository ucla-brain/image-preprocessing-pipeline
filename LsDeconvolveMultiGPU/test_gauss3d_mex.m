function test_gauss3d_mex_vs_imgaussfilt3_padded()
    % Test gauss3d_mex against imgaussfilt3 for sigma <= 2.5, with pre-padding

    szs = {
        [32, 128, 128], ...
        [256, 512, 512]
    };
    types = {@single, @double};
    sigma_tests = {2.5, [2.5 2.5 2.5]}; % Only up to 2.5
    ksize_tests = {'auto', 13, [13 13 13]}; % Only up to 13

    for ityp = 1:numel(types)
        for isz = 1:numel(szs)
            sz = szs{isz};
            T = types{ityp};
            type_str = func2str(T);

            for isig = 1:numel(sigma_tests)
                sigma = sigma_tests{isig};
                for iksz = 1:numel(ksize_tests)
                    ksz = ksize_tests{iksz};
                    if ischar(ksz)
                        kdesc = 'auto';
                    else
                        kdesc = mat2str(ksz);
                    end
                    fprintf('\nTesting %s %s, sigma=%s, ksize=%s ...\n', ...
                        type_str, mat2str(sz), mat2str(sigma), kdesc);

                    % Determine kernel size for padding
                    if ischar(ksz)
                        kernel_sz = odd_kernel_size(sigma);
                    else
                        kernel_sz = ksz;
                        if isscalar(kernel_sz)
                            kernel_sz = repmat(kernel_sz,1,3);
                        end
                    end
                    pad_amt = floor(kernel_sz / 2);

                    % Prepare input
                    rng(0);
                    x = rand(sz, type_str);

                    % Pad input for both methods
                    x_pad = padarray(x, pad_amt, 'replicate', 'both');
                    x_pad_gpu = gpuArray(x_pad);

                    % imgaussfilt3 (on GPU, padding 'replicate')
                    opts = {'Padding', 'replicate', 'FilterSize', kernel_sz};
                    if exist('imgaussfilt3', 'file') && ...
                            any(strcmp('FilterDomain', methods('imgaussfilt3')))
                        opts = [opts, {'FilterDomain', 'spatial'}];
                    end
                    y_ref_gpu = imgaussfilt3(x_pad_gpu, sigma, opts{:});

                    % gauss3d_mex (no padding, in-place)
                    if ischar(ksz)
                        y_mex_gpu = gauss3d_mex(x_pad_gpu, sigma); % default kernel size
                    else
                        y_mex_gpu = gauss3d_mex(x_pad_gpu, sigma, kernel_sz);
                    end

                    % Unpad both results for comparison
                    idx1 = (1+pad_amt(1)):(size(x_pad,1)-pad_amt(1));
                    idx2 = (1+pad_amt(2)):(size(x_pad,2)-pad_amt(2));
                    idx3 = (1+pad_amt(3)):(size(x_pad,3)-pad_amt(3));
                    y_ref_unpad = y_ref_gpu(idx1,idx2,idx3);
                    y_mex_unpad = y_mex_gpu(idx1,idx2,idx3);

                    % Compute error
                    err = max(abs(y_mex_unpad(:) - y_ref_unpad(:)));
                    rms_err = sqrt(mean((y_mex_unpad(:) - y_ref_unpad(:)).^2));
                    rel_err = rms_err / (max(abs(y_ref_unpad(:))) + eps);

                    fprintf('  3D validation (unpadded): max = %.2e, RMS = %.2e, rel RMS = %.2e\n', ...
                        gather(err), gather(rms_err), gather(rel_err));

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

                    clear x_pad_gpu y_mex_gpu y_ref_gpu y_mex_unpad y_ref_unpad
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
