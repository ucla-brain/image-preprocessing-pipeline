function gauss3d_gpu_test()
% Test harness for gauss3d_gpu vs imgaussfilt3 (MATLAB GPU, single-precision).
% Reports absolute and relative error, and speedup/slowdown.

    gpuDeviceObject = gpuDevice(2);
    reset(gpuDeviceObject);

    hasCprintfFunction = exist('cprintf','file') == 2;
    colorFormat = @(colorName, inputString) format_colored_string(colorName, inputString, hasCprintfFunction);

    testVolumeSizes = {[900, 900, 500]};
    sigmaTestCases = {[0.5 0.5 1.5], [0.5 0.5 2.5], 3.0, 0.25};
    SINGLE_PRECISION_PASS_THRESHOLD = 5e-5;
    paddingMode = 'symmetric';

    testCounter = 0;
    passDirect = 0; failDirect = 0;
    testOOMCounter = 0; testKernelSkipCounter = 0;

    % Add speedup column
    fprintf('%-4s %-5s %-12s %-20s %-14s %-8s %-8s %-8s %-8s %-10s\n', ...
        'PF', 'ID', 'Type', 'Size', 'Sigma', ...
        'maxErr', 'RMS', 'relErr', 'Time(s)', 'Speedup');

    for volumeSizeIndex = 1:numel(testVolumeSizes)
        volumeSize = testVolumeSizes{volumeSizeIndex};
        dataTypeName = 'single';

        for sigmaIndex = 1:numel(sigmaTestCases)
            testCounter = testCounter + 1;
            gaussianSigma = sigmaTestCases{sigmaIndex};

            % Compute kernel size array (for reference/MATLAB call)
            kernelSizeArray = compute_odd_kernel_size(gaussianSigma);
            kernelPaddingAmount = floor(kernelSizeArray / 2);

            rng(0);
            testInputVolume = gpuArray.rand(volumeSize, 'single');

            % Reference output (MATLAB imgaussfilt3 GPU)
            refStartTime = tic;
            directReferenceOutput = imgaussfilt3(testInputVolume, gaussianSigma, ...
                'Padding', paddingMode, 'FilterSize', kernelSizeArray, 'FilterDomain', 'spatial');
            wait(gpuDeviceObject);
            refElapsedSeconds = toc(refStartTime);

            % Direct path: pad input, unpad output
            directInputPadded = padarray(testInputVolume, kernelPaddingAmount, paddingMode, 'both');
            % Unpad indices
            unpadIdxY = (1+kernelPaddingAmount(1)):(size(directInputPadded,1)-kernelPaddingAmount(1));
            unpadIdxX = (1+kernelPaddingAmount(2)):(size(directInputPadded,2)-kernelPaddingAmount(2));
            unpadIdxZ = (1+kernelPaddingAmount(3)):(size(directInputPadded,3)-kernelPaddingAmount(3));
            % Check for valid indices
            if isempty(unpadIdxY) || isempty(unpadIdxX) || isempty(unpadIdxZ)
                fprintf('%s Test skipped (direct): Padding too large for available data. (ID %d)\n', colorFormat('yellow','[SKIP]'), testCounter);
                testKernelSkipCounter = testKernelSkipCounter + 1;
                reset(gpuDeviceObject);
                continue;
            end
            
            directStartTime = tic;
            try
                directFilteredOutput = gauss3d_gpu(directInputPadded, gaussianSigma);
            catch ME
                disp(gaussianSigma)
                fprintf('%s Test failed (direct): %s (ID %d)\n', colorFormat('red','[FAIL]'), ME.message, testCounter);
                reset(gpuDeviceObject);
                continue;
            end
            wait(gpuDeviceObject);
            directElapsedSeconds = toc(directStartTime);
            directFilteredUnpadded = directFilteredOutput(unpadIdxY, unpadIdxX, unpadIdxZ);

            % Error metrics (direct)
            maxAbsErrorDirect = max(abs(directFilteredUnpadded(:) - directReferenceOutput(:)));
            rmsErrorDirect    = sqrt(mean((directFilteredUnpadded(:) - directReferenceOutput(:)).^2));
            relErrorDirect    = rmsErrorDirect / (max(abs(directReferenceOutput(:))) + eps);

            % Speedup
            speedup = refElapsedSeconds / directElapsedSeconds;
            if isfinite(speedup)
                if speedup > 1
                    speedupString = sprintf('+%.0f%%', 100*(speedup-1));
                else
                    speedupString = sprintf('-%.0f%%', 100*(1/speedup-1));
                end
            else
                speedupString = 'N/A';
            end

            % Print Direct
            if maxAbsErrorDirect < SINGLE_PRECISION_PASS_THRESHOLD
                pfDirect = colorFormat('green', '✔️');
                passDirect = passDirect + 1;
            else
                pfDirect = colorFormat('red', '❌');
                failDirect = failDirect + 1;
            end
            fprintf('%-4s %-5d %-12s %-20s %-14s %-7.2e %-8.2e %-8.2e %-8.3f %-10s\n', ...
                pfDirect, testCounter, dataTypeName, mat2str(volumeSize), mat2str(gaussianSigma), ...
                maxAbsErrorDirect, rmsErrorDirect, relErrorDirect, ...
                directElapsedSeconds, speedupString);

            reset(gpuDeviceObject);
        end
    end

    % Suite summary
    fprintf('\n%s\n', colorFormat('yellow', repmat('=',1,80)));
    fprintf('%s\n', colorFormat('magenta', 'TEST SUITE SUMMARY'));
    fprintf('%-24s %-8d\n', colorFormat('green', 'Direct passed:'), passDirect);
    fprintf('%-24s %-8d\n', colorFormat('red',   'Direct failed:'), failDirect);
    fprintf('%-24s %-8d\n', colorFormat('blue',  'OOM/Skipped:'), testOOMCounter);
    fprintf('%-24s %-8d\n', colorFormat('red',   'Kernel size skip:'), testKernelSkipCounter);
    fprintf('%s\n', colorFormat('yellow', repmat('=',1,80)));
end

function kernelSizeArray = compute_odd_kernel_size(gaussianSigma)
    if isscalar(gaussianSigma)
        gaussianSigma = [gaussianSigma gaussianSigma gaussianSigma];
    end
    kernelSizeArray = 2*ceil(3*gaussianSigma) + 1;
    kernelSizeArray = max(kernelSizeArray, 3);
end

function formattedString = format_colored_string(colorName, inputString, hasCprintfFunction)
    if hasCprintfFunction
        formattedString = evalc(['cprintf(''', colorName, ''','' ', 'inputString', ')']);
        formattedString = formattedString(1:end-1);
    else
        formattedString = inputString;
    end
end
