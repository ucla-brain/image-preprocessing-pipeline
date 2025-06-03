try
    % Parameters
    pass_thresh = 0.2;     % Acceptable max diff (for 3D vs 2D)
    mean_thresh = 0.03;    % Acceptable mean diff

    % 1. Standard 3D test
    sz = [64, 64, 32];
    A = rand(sz, 'single');
    PSF = fspecial('gaussian', 15, 2);
    PSF3 = zeros(15, 15, 7, 'single');
    for z = 1:7
        PSF3(:,:,z) = fspecial('gaussian', 15, 2) * exp(-((z-4).^2)/6);
    end
    PSF3 = PSF3 / sum(PSF3(:));

    % CPU edgetaper on central slice
    A2D = A(:,:,round(sz(3)/2));
    et_cpu = edgetaper(A2D, PSF);

    % GPU 3D edge taper
    Ag = gpuArray(A);
    PSFg = gpuArray(PSF3);
    et_gpu = edge_taper_auto(Ag, PSFg);
    et_gpu_cpu = gather(et_gpu);

    % Central slice compare
    diff_central = abs(et_cpu - et_gpu_cpu(:,:,round(sz(3)/2)));
    maxdiff = max(diff_central(:));
    meandiff = mean(diff_central(:));
    if maxdiff < pass_thresh && meandiff < mean_thresh
        print_pass(sprintf('Central slice: max diff %.4g, mean diff %.4g', maxdiff, meandiff));
    else
        print_fail(sprintf('Central slice: max diff %.4g, mean diff %.4g', maxdiff, meandiff));
    end

    % 2. Full 3D array self-consistency (should match original except at boundaries)
    if max(et_gpu_cpu(:)) <= 1 && min(et_gpu_cpu(:)) >= 0
        print_pass('GPU-tapered 3D volume values are within expected [0,1] range');
    else
        print_fail('GPU-tapered 3D volume values out of expected range!');
    end

    % 3. Small array edge case (robustness)
    small_sz = [7,6,5];
    Asmall = rand(small_sz, 'single');
    PSFsmall = rand(3,3,3, 'single'); PSFsmall = PSFsmall/sum(PSFsmall(:));
    try
        Asmall_gpu = gpuArray(Asmall);
        PSFsmall_gpu = gpuArray(PSFsmall);
        edge_taper_auto(Asmall_gpu, PSFsmall_gpu); % <-- No ~=
        print_pass('Small 3D GPU array processed without reshape error');
    catch ME
        print_fail(sprintf('Small 3D GPU array failed: %s', ME.message));
    end

    % 4. Taper vector length matches dimension (test all dims/tapers)
    all_ok = true;
    for n = 1:20
        for t = 0:n
            taper = make_taper(n, t);
            if numel(taper) ~= n
                all_ok = false;
                print_fail(sprintf('make_taper(%d, %d) returns %d elements', n, t, numel(taper)));
            end
        end
    end
    if all_ok
        print_pass('make_taper robust for all dims/taper_width');
    end

    % 5. Output size matches input
    if isequal(size(et_gpu_cpu), sz)
        print_pass('Output size matches input size for 3D GPU test');
    else
        print_fail('Output size does not match input size for 3D GPU test');
        all_ok = false;
    end

    % 6. Test conv3d_mex against MATLAB's CPU reference (convn + replicate padding)
    conv_ok = false;
    try
        img_sz = [24, 25, 22];
        ker_sz = [7, 5, 3];
        I = rand(img_sz, 'single');
        K = rand(ker_sz, 'single');
        K = K / sum(K(:));
        Ig = gpuArray(I);
        Kg = gpuArray(K);

        % GPU result
        O_gpu = conv3d_mex(Ig, Kg);
        O_gpu_cpu = gather(O_gpu);

        % CPU reference using convn with replicate boundary (pad & crop)
        pad = floor(ker_sz/2);
        Ipad = padarray(I, pad, 'replicate', 'both');
        O_cpu_full = convn(Ipad, K, 'valid');  % valid convolution after padding gives same as 'same' with replicate
        % Shape should match
        if isequal(size(O_gpu_cpu), size(O_cpu_full))
            diff_conv = abs(O_cpu_full - O_gpu_cpu);
            max_conv_diff = max(diff_conv(:));
            mean_conv_diff = mean(diff_conv(:));
            if max_conv_diff < 5e-4 && mean_conv_diff < 1e-4
                print_pass(sprintf('conv3d_mex matches convn+replicate (max %.2g, mean %.2g)', max_conv_diff, mean_conv_diff));
                conv_ok = true;
            else
                print_fail(sprintf('conv3d_mex vs convn+replicate: max diff %.2g, mean %.2g', max_conv_diff, mean_conv_diff));
            end
        else
            print_fail('conv3d_mex output shape does not match CPU reference!');
        end
    catch ME
        print_fail(['conv3d_mex test failed: ' ME.message]);
    end

    % 7. Trivial all-ones kernel/input test
    try
        I = ones(5,5,5,'single');
        K = ones(3,3,3,'single');
        Ig = gpuArray(I);
        Kg = gpuArray(K);
        O_gpu = conv3d_mex(Ig, Kg);
        O_gpu_cpu = gather(O_gpu);
        pad = floor(size(K)/2);
        Ipad = padarray(I, pad, 'replicate', 'both');
        O_cpu_full = convn(Ipad, K, 'valid');
        center_val_gpu = O_gpu_cpu(3,3,3);
        center_val_cpu = O_cpu_full(3,3,3);
        if abs(center_val_gpu - 27) < 1e-4 && abs(center_val_cpu - 27) < 1e-4
            print_pass('Trivial all-ones kernel/input test: center value is correct (27)');
        else
            print_fail(sprintf('Trivial all-ones test: center values gpu: %.4g, cpu: %.4g', center_val_gpu, center_val_cpu));
            all_ok = false;
        end
    catch ME
        print_fail(['Trivial all-ones test failed: ' ME.message]);
    end

    % 8. Large 3D block edge_taper_auto GPU performance/stress test
    large_block_ok = true;
    try
        bl_large_sz = [256, 256, 96];      % ~1.5GB, adjust for your GPU RAM
        psf_large_sz = [21, 21, 11];
        bl_large = rand(bl_large_sz, 'single');
        psf_large = rand(psf_large_sz, 'single'); psf_large = psf_large / sum(psf_large(:));
        bl_large_g = gpuArray(bl_large);
        psf_large_g = gpuArray(psf_large);

        % GPU timing
        gpu_time = [];
        for k = 1:3
            tic;
            bl_large_tapered = edge_taper_auto(bl_large_g, psf_large_g);
            wait(gpuDevice);  % Make sure operation completes
            gpu_time(k) = toc;
        end
        t_gpu = min(gpu_time);

        % CPU timing (optional for very large arrays, may take much longer)
        cpu_time = [];
        cpu_test_possible = all(bl_large_sz < 300); % avoid OOM on huge arrays
        if cpu_test_possible
            for k = 1:2
                tic;
                bl_cpu = bl_large;
                % *** PATCH: Cast filter kernel to double for imfilter! ***
                bl_blur = imfilter(bl_cpu, double(psf_large), 'replicate', 'same', 'conv');
                mask = ones(bl_large_sz, 'single');
                for d = 1:3
                    dimsz = bl_large_sz(d);
                    taper_width = max(8, round(psf_large_sz(d)/2));
                    taper = make_taper(dimsz, taper_width);
                    shape = ones(1,3); shape(d) = dimsz;
                    mask = mask .* reshape(taper, shape);
                end
                bl_large_cpu = mask .* bl_cpu + (1-mask) .* bl_blur;
                cpu_time(k) = toc;
            end
            t_cpu = min(cpu_time);
        else
            t_cpu = NaN;
        end

        bl_large_tapered_cpu = gather(bl_large_tapered);

        if all(isfinite(bl_large_tapered_cpu(:))) && ...
           max(bl_large_tapered_cpu(:)) <= 1 && ...
           min(bl_large_tapered_cpu(:)) >= 0
            if ~isnan(t_cpu)
                perf_ratio = t_cpu / t_gpu * 100;
                print_pass(sprintf('Large 3D edge_taper_auto: GPU %.3fs, CPU %.3fs (GPU is %.1f%% faster)', ...
                    t_gpu, t_cpu, perf_ratio));
            else
                print_pass(sprintf('Large 3D edge_taper_auto: GPU %.3fs, CPU timing N/A (too big)', t_gpu));
            end
        else
            print_fail('Large 3D block: edge_taper_auto output contains non-finite or out-of-bounds values!');
            large_block_ok = false;
        end
    catch ME
        print_fail(['Large 3D block edge_taper_auto test failed: ' ME.message]);
        large_block_ok = false;
    end

    summary_ok = all_ok && maxdiff < pass_thresh && meandiff < mean_thresh && ...
        max(et_gpu_cpu(:)) <= 1 && min(et_gpu_cpu(:)) >= 0 && isequal(size(et_gpu_cpu), sz) && conv_ok && large_block_ok;
    if summary_ok
        print_pass('ALL TESTS PASSED 🎉');
    end

catch ME
    print_fail(['Test script crashed: ' ME.message]);
end

function print_pass(msg)
    fprintf('\x1b[1;32m✅ PASS:\x1b[0m %s\n', msg);
end

function print_fail(msg)
    fprintf('\x1b[1;31m❌ FAIL:\x1b[0m %s\n', msg);
end
