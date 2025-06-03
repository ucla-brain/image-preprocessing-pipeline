try
    pass_thresh = 0.2;     % Acceptable max diff (for 3D vs 2D)
    mean_thresh = 0.03;    % Acceptable mean diff

    %% 0. 2D edgetaper vs edge_taper_auto (CPU/GPU) test
    try
        img_sz = [128, 128];
        psf_sz = [11, 11];

        rng(1);
        img = rand(img_sz, 'single');
        psf = fspecial('gaussian', psf_sz, 2);
        psf = single(psf / sum(psf(:)));

        edgetaper_ref = edgetaper(img, psf);
        out_cpu = edge_taper_auto(img, psf);
        out_gpu = edge_taper_auto(gpuArray(img), gpuArray(psf));
        out_gpu_cpu = gather(out_gpu);

        diff_cpu = abs(edgetaper_ref - out_cpu);
        maxdiff_cpu = max(diff_cpu(:));
        meandiff_cpu = mean(diff_cpu(:));
        fprintf('2D CPU: max diff %.3g, mean diff %.3g\n', maxdiff_cpu, meandiff_cpu);
        if maxdiff_cpu < 5e-4 && meandiff_cpu < 1e-4
            print_pass('CPU edge_taper_auto matches MATLAB edgetaper (2D)');
        else
            print_fail('CPU edge_taper_auto does NOT match MATLAB edgetaper (2D)');
        end

        diff_gpu = abs(edgetaper_ref - out_gpu_cpu);
        maxdiff_gpu = max(diff_gpu(:));
        meandiff_gpu = mean(diff_gpu(:));
        fprintf('2D GPU: max diff %.3g, mean diff %.3g\n', maxdiff_gpu, meandiff_gpu);
        if maxdiff_gpu < 5e-4 && meandiff_gpu < 1e-4
            print_pass('GPU edge_taper_auto matches MATLAB edgetaper (2D)');
        else
            print_fail('GPU edge_taper_auto does NOT match MATLAB edgetaper (2D)');
        end

        twod_ok = maxdiff_cpu < 5e-4 && meandiff_cpu < 1e-4 && maxdiff_gpu < 5e-4 && meandiff_gpu < 1e-4;
    catch ME
        print_fail(['2D edgetaper vs edge_taper_auto test failed: ' ME.message]);
        twod_ok = false;
    end

    %% 1. Standard 3D test
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

    diff_central = abs(et_cpu - et_gpu_cpu(:,:,round(sz(3)/2)));
    maxdiff = max(diff_central(:));
    meandiff = mean(diff_central(:));
    if maxdiff < pass_thresh && meandiff < mean_thresh
        print_pass(sprintf('Central slice: max diff %.4g, mean diff %.4g', maxdiff, meandiff));
    else
        print_fail(sprintf('Central slice: max diff %.4g, mean diff %.4g', maxdiff, meandiff));
    end

    %% 2. Full 3D value range
    if max(et_gpu_cpu(:)) <= 1 && min(et_gpu_cpu(:)) >= 0
        print_pass('GPU-tapered 3D volume values are within expected [0,1] range');
    else
        print_fail('GPU-tapered 3D volume values out of expected range!');
    end

    %% 3. Small array edge case
    small_sz = [7,6,5];
    Asmall = rand(small_sz, 'single');
    PSFsmall = rand(3,3,3, 'single'); PSFsmall = PSFsmall/sum(PSFsmall(:));
    try
        Asmall_gpu = gpuArray(Asmall);
        PSFsmall_gpu = gpuArray(PSFsmall);
        edge_taper_auto(Asmall_gpu, PSFsmall_gpu);
        print_pass('Small 3D GPU array processed without reshape error');
    catch ME
        print_fail(sprintf('Small 3D GPU array failed: %s', ME.message));
    end

    %% 4. Taper vector length matches dimension
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

    %% 5. Output size matches input
    if isequal(size(et_gpu_cpu), sz)
        print_pass('Output size matches input size for 3D GPU test');
    else
        print_fail('Output size does not match input size for 3D GPU test');
        all_ok = false;
    end

    %% 6. Test conv3d_mex against MATLAB's CPU reference (convn + replicate padding)
    conv_ok = false;
    try
        img_sz = [24, 25, 22];
        ker_sz = [7, 5, 3];
        I = rand(img_sz, 'single');
        K = rand(ker_sz, 'single');
        K = K / sum(K(:));
        Ig = gpuArray(I);
        Kg = gpuArray(K);

        O_gpu = conv3d_mex(Ig, Kg);
        O_gpu_cpu = gather(O_gpu);

        pad = floor(ker_sz/2);
        Ipad = padarray(I, pad, 'replicate', 'both');
        O_cpu_full = convn(Ipad, K, 'valid');
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

    %% 7. Trivial all-ones kernel/input test
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

    %% 8. Large 3D block edge_taper_auto GPU performance/stress test
    large_block_ok = true;
    try
        bl_large_sz = [512, 512, 512];
        psf_large_sz = [9, 9, 21];
        bl_large = rand(bl_large_sz, 'single');
        psf_large = rand(psf_large_sz, 'single'); psf_large = psf_large / sum(psf_large(:));
        bl_large_g = gpuArray(bl_large);
        psf_large_g = gpuArray(psf_large);

        gpu_time = [];
        for k = 1:3
            tic;
            bl_large_tapered = edge_taper_auto(bl_large_g, psf_large_g);
            wait(gpuDevice);
            gpu_time(k) = toc;
        end
        t_gpu = min(gpu_time);

        cpu_time = [];
        cpu_test_possible = false; % set to true to actually test CPU
        if cpu_test_possible
            fprintf('\x1b[1;33m⚠️  WARNING:\x1b[0m Running large CPU filter, may take minutes...\n');
            for k = 1:2
                tic;
                bl_cpu = bl_large;
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

    %% 9. Direct CPU vs GPU edge_taper_auto on identical large 3D block
    cpu_gpu_identical_ok = true;
    try
        fprintf('--- 3D CPU vs GPU edge_taper_auto test on large block ---\n');
        test_sz = [512, 512, 512];
        test_psf_sz = [11, 11, 9];

        rng(1);
        bl_cpu = rand(test_sz, 'single');
        psf = rand(test_psf_sz, 'single');
        psf = psf / sum(psf(:));

        tic;
        bl_cpu_tapered = edge_taper_auto(bl_cpu, psf);
        t_cpu = toc;

        bl_gpu = gpuArray(bl_cpu);
        psf_gpu = gpuArray(psf);
        wait(gpuDevice);
        tic;
        bl_gpu_tapered = edge_taper_auto(bl_gpu, psf_gpu);
        wait(gpuDevice);
        t_gpu = toc;
        bl_gpu_tapered_cpu = gather(bl_gpu_tapered);

        diff = abs(bl_cpu_tapered - bl_gpu_tapered_cpu);
        max_diff = max(diff(:));
        mean_diff = mean(diff(:));
        thresh_max = 5e-4; thresh_mean = 1e-4;

        fprintf('CPU time: %.2fs, GPU time: %.2fs\n', t_cpu, t_gpu);
        fprintf('Max abs diff: %.6g, Mean abs diff: %.6g\n', max_diff, mean_diff);

        if max_diff < thresh_max && mean_diff < thresh_mean
            print_pass(sprintf('CPU and GPU edge_taper_auto agree (max %.3g, mean %.3g)', max_diff, mean_diff));
        else
            print_fail(sprintf('CPU and GPU edge_taper_auto differ! (max %.3g, mean %.3g)', max_diff, mean_diff));
            cpu_gpu_identical_ok = false;
        end
    catch ME
        print_fail(['Large 3D CPU vs GPU edge_taper_auto test failed: ' ME.message]);
        cpu_gpu_identical_ok = false;
    end

    %% Final overall summary
    summary_ok = twod_ok && all_ok && maxdiff < pass_thresh && meandiff < mean_thresh && ...
        max(et_gpu_cpu(:)) <= 1 && min(et_gpu_cpu(:)) >= 0 && ...
        isequal(size(et_gpu_cpu), sz) && conv_ok && large_block_ok && cpu_gpu_identical_ok;
    if summary_ok
        print_pass('ALL TESTS PASSED 🎉');
    else
        print_fail('Some tests failed.');
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