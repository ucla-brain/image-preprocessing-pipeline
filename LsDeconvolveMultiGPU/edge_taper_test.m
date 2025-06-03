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
        out_small = edge_taper_auto(Asmall_gpu, PSFsmall_gpu);
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

catch ME
    print_fail(['Test script crashed: ' ME.message]);
end

function print_pass(msg)
    fprintf('\x1b[1;32m✅ PASS:\x1b[0m %s\n', msg);
end

function print_fail(msg)
    fprintf('\x1b[1;31m❌ FAIL:\x1b[0m %s\n', msg);
end
