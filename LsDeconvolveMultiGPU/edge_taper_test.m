all_ok = true;

try
    pass_thresh = 0.2;     % Acceptable max diff (for 3D)
    mean_thresh = 0.03;    % Acceptable mean diff

    %% 1. 2D (512x512) CPU vs GPU edge_taper_auto test
    try
        fprintf('\n--- 2D (512x512) CPU vs GPU edge_taper_auto test ---\n');
        img_sz = [512, 512];
        psf_sz = [13, 13];

        rng(42);
        img = rand(img_sz, 'single');
        psf = fspecial('gaussian', psf_sz, 2);
        psf = single(psf / sum(psf(:)));

        % CPU
        tic;
        cpu_tapered = edge_taper_auto(img, psf);
        t_cpu = toc;

        % GPU
        img_gpu = gpuArray(img);
        psf_gpu = gpuArray(psf);
        wait(gpuDevice);
        tic;
        gpu_tapered = edge_taper_auto(img_gpu, psf_gpu);
        wait(gpuDevice);
        t_gpu = toc;
        gpu_tapered_cpu = gather(gpu_tapered);

        % Compare
        diff = abs(cpu_tapered - gpu_tapered_cpu);
        max_diff = max(diff(:));
        mean_diff = mean(diff(:));
        thresh_max = 1e-6; thresh_mean = 1e-7;

        fprintf('CPU time: %.3fs, GPU time: %.3fs\n', t_cpu, t_gpu);
        fprintf('Max abs diff: %.6g, Mean abs diff: %.6g\n', max_diff, mean_diff);

        if max_diff < thresh_max && mean_diff < thresh_mean
            print_pass(sprintf('2D (512x512) CPU and GPU edge_taper_auto are identical (max %.2g, mean %.2g)', max_diff, mean_diff));
        else
            print_fail(sprintf('2D (512x512) CPU and GPU edge_taper_auto differ! (max %.2g, mean %.2g)', max_diff, mean_diff));
            all_ok = false;
        end
    catch ME
        print_fail(['2D (512x512) CPU vs GPU edge_taper_auto test failed: ' ME.message]);
        all_ok = false;
    end

    %% 2. Standard 3D test
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
        all_ok = false;
    end

    %% 3. Full 3D value range
    if max(et_gpu_cpu(:)) <= 1 && min(et_gpu_cpu(:)) >= 0
        print_pass('GPU-tapered 3D volume values are within expected [0,1] range');
    else
        print_fail('GPU-tapered 3D volume values out of expected range!');
        all_ok = false;
    end

    %% 4. Small array edge case
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
        all_ok = false;
    end

    %% 5. Output size matches input
    if isequal(size(et_gpu_cpu), sz)
        print_pass('Output size matches input size for 3D GPU test');
    else
        print_fail('Output size does not match input size for 3D GPU test');
        all_ok = false;
    end

    %% 6. Test conv3d_mex against MATLAB's CPU reference (convn + replicate padding)
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
            else
                print_fail(sprintf('conv3d_mex vs convn+replicate: max diff %.2g, mean %.2g', max_conv_diff, mean_conv_diff));
                all_ok = false;
            end
        else
            print_fail('conv3d_mex output shape does not match CPU reference!');
            all_ok = false;
        end
    catch ME
        print_fail(['conv3d_mex test failed: ' ME.message]);
        all_ok = false;
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
        all_ok = false;
    end

    %% 8. Large 3D block edge_taper_auto GPU performance/stress test
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

        bl_large_tapered_cpu = gather(bl_large_tapered);

        if all(isfinite(bl_large_tapered_cpu(:))) && ...
           max(bl_large_tapered_cpu(:)) <= 1 && ...
           min(bl_large_tapered_cpu(:)) >= 0
            print_pass(sprintf('Large 3D edge_taper_auto: GPU %.3fs', t_gpu));
        else
            print_fail('Large 3D block: edge_taper_auto output contains non-finite or out-of-bounds values!');
            all_ok = false;
        end
    catch ME
        print_fail(['Large 3D block edge_taper_auto test failed: ' ME.message]);
        all_ok = false;
    end

    %% 9. Direct CPU vs GPU edge_taper_auto on identical large 3D block
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
            all_ok = false;
        end
    catch ME
        print_fail(['Large 3D CPU vs GPU edge_taper_auto test failed: ' ME.message]);
        all_ok = false;
    end

    %% Final overall summary
    if all_ok
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
