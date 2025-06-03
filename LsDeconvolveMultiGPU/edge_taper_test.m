all_ok = true;

try
    pass_thresh = 1e-5;     % Acceptable max diff (for 3D)
    mean_thresh = 1e-6;     % Acceptable mean diff

    %% 1. Large 3D edge_taper_auto: CPU vs GPU, performance + correctness
    try
        fprintf('\n--- Large 3D edge_taper_auto: CPU vs GPU test ---\n');
        sz = [512, 512, 512];
        psf_sz = [9, 9, 21];

        rng(42);
        bl = rand(sz, 'single');
        psf = rand(psf_sz, 'single');
        psf = psf / sum(psf(:));

        % CPU
        tic;
        bl_tapered_cpu = edge_taper_auto(bl, psf);
        t_cpu = toc;

        % GPU
        bl_gpu = gpuArray(bl);
        psf_gpu = gpuArray(psf);
        wait(gpuDevice);
        tic;
        bl_tapered_gpu = edge_taper_auto(bl_gpu, psf_gpu);
        wait(gpuDevice);
        t_gpu = toc;
        bl_tapered_gpu_cpu = gather(bl_tapered_gpu);

        % Compare
        diff = abs(bl_tapered_cpu - bl_tapered_gpu_cpu);
        max_diff = max(diff(:));
        mean_diff = mean(diff(:));
        fprintf('CPU time: %.2fs, GPU time: %.2fs\n', t_cpu, t_gpu);
        fprintf('Max abs diff: %.6g, Mean abs diff: %.6g\n', max_diff, mean_diff);

        if max_diff < pass_thresh && mean_diff < mean_thresh
            print_pass(sprintf('Large 3D CPU and GPU edge_taper_auto outputs are identical (max %.2g, mean %.2g)', max_diff, mean_diff));
        else
            print_fail(sprintf('Large 3D CPU and GPU edge_taper_auto differ! (max %.2g, mean %.2g)', max_diff, mean_diff));
            all_ok = false;
        end
    catch ME
        print_fail(['Large 3D edge_taper_auto CPU vs GPU test failed: ' ME.message]);
        all_ok = false;
    end

    %% 2. Standard 3D test (central slice)
    sz = [64, 64, 32];
    A = rand(sz, 'single');
    PSF = fspecial('gaussian', 15, 2);
    PSF3 = zeros(15, 15, 7, 'single');
    for z = 1:7
        PSF3(:,:,z) = fspecial('gaussian', 15, 2) * exp(-((z-4).^2)/6);
    end
    PSF3 = PSF3 / sum(PSF3(:));

    Ag = gpuArray(A);
    PSFg = gpuArray(PSF3);
    et_cpu = edge_taper_auto(A, PSF3);
    et_gpu = edge_taper_auto(Ag, PSFg);
    et_gpu_cpu = gather(et_gpu);

    diff_central = abs(et_cpu(:,:,round(sz(3)/2)) - et_gpu_cpu(:,:,round(sz(3)/2)));
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
