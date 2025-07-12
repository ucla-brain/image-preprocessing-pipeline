function gauss3d_gpu_test()
% Robust test harness for gauss3d_gpu vs imgaussfilt3 (GPU, single-precision).
% Prints and counts direct and fft path tests separately.

    gpuDeviceObject = gpuDevice(1);
    reset(gpuDeviceObject);

    hasCprintfFunction = exist('cprintf','file') == 2;
    colorFormat = @(colorName, inputString) format_colored_string(colorName, inputString, hasCprintfFunction);

    testVolumeSizes = {[128, 192, 64], [934, 934, 697]};
    %testVolumeSizes = {[128, 192, 64]};
    sigmaTestCases = {[0.5 0.5 1.5], [0.5 0.5 2.5], 2.5, 0.25, 8};
    kernelSizeTestCases = {'auto', [13 13 25], 3, 51};
    SINGLE_PRECISION_PASS_THRESHOLD = 5e-5;
    paddingMode = 'circular';

    testCounter = 0;
    passDirect = 0; failDirect = 0;
    passFFT = 0; failFFT = 0;
    testOOMCounter = 0; testKernelSkipCounter = 0;

    % Print summary header
    fprintf('%-4s %-5s %-9s %-12s %-20s %-14s %-20s %-8s %-8s %-8s %-8s %-9s %-10s\n', ...
        'PF', 'ID', 'Path', 'Type', 'Size', 'Sigma', 'Kernel', ...
        'maxErr', 'RMS', 'relErr', 'Time(s)', 'Speedup');

    for volumeSizeIndex = 1:numel(testVolumeSizes)
        volumeSize = testVolumeSizes{volumeSizeIndex};
        dataTypeName = 'single';

        for sigmaIndex = 1:numel(sigmaTestCases)
            gaussianSigma = sigmaTestCases{sigmaIndex};
            for kernelSizeIndex = 1:numel(kernelSizeTestCases)
                testCounter = testCounter + 1;
                kernelSizeInput = kernelSizeTestCases{kernelSizeIndex};
                if ischar(kernelSizeInput) || isstring(kernelSizeInput)
                    kernelSizeDescription = char(kernelSizeInput);
                else
                    kernelSizeDescription = mat2str(kernelSizeInput);
                end

                % Compute kernel size array
                if ischar(kernelSizeInput) || isstring(kernelSizeInput)
                    kernelSizeArray = compute_odd_kernel_size(gaussianSigma);
                else
                    kernelSizeArray = kernelSizeInput;
                    if isscalar(kernelSizeArray)
                        kernelSizeArray = repmat(kernelSizeArray,1,3);
                    end
                end
                kernelPaddingAmount = floor(kernelSizeArray / 2);

                rng(0);
                testInputVolume = gpuArray.rand(volumeSize, 'single');

                % Direct reference (spatial domain, replicate pad)
                directReferenceOutput = imgaussfilt3(testInputVolume, gaussianSigma, ...
                    'Padding', paddingMode, 'FilterSize', kernelSizeArray, 'FilterDomain', 'spatial');
                % FFT reference (frequency domain, replicate pad)
                fftReferenceOutput = imgaussfilt3(testInputVolume, gaussianSigma, ...
                    'Padding', paddingMode, 'FilterSize', kernelSizeArray, 'FilterDomain', 'frequency');

                %% --- Direct path: pad input, unpad output ---
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
                directFilteredOutput = gauss3d_gpu(directInputPadded, gaussianSigma, kernelSizeArray, 'direct');
                wait(gpuDeviceObject);
                directElapsedSeconds = toc(directStartTime);
                directFilteredUnpadded = directFilteredOutput(unpadIdxY, unpadIdxX, unpadIdxZ);

                % Error metrics (direct)
                maxAbsErrorDirect = max(abs(directFilteredUnpadded(:) - directReferenceOutput(:)));
                rmsErrorDirect    = sqrt(mean((directFilteredUnpadded(:) - directReferenceOutput(:)).^2));
                relErrorDirect    = rmsErrorDirect / (max(abs(directReferenceOutput(:))) + eps);

                %% --- FFT path (no manual pad/unpad) ---
                fftStartTime = tic;
                fftFilteredOutput = gauss3d_gpu(directInputPadded, gaussianSigma, kernelSizeArray, 'fft');
                wait(gpuDeviceObject);
                fftElapsedSeconds = toc(fftStartTime);
                fftFilteredOutput = fftFilteredOutput(unpadIdxY, unpadIdxX, unpadIdxZ);

                maxAbsErrorFFT = max(abs(fftFilteredOutput(:) - fftReferenceOutput(:)));
                rmsErrorFFT    = sqrt(mean((fftFilteredOutput(:) - fftReferenceOutput(:)).^2));
                relErrorFFT    = rmsErrorFFT / (max(abs(fftReferenceOutput(:))) + eps);

                % Speedup (fft vs direct)
                speedupRatio = directElapsedSeconds / fftElapsedSeconds;
                if isfinite(speedupRatio)
                    if speedupRatio > 1
                        speedupString = sprintf('+%.0f%%', 100*(speedupRatio-1));
                    else
                        speedupString = sprintf('-%.0f%%', 100*(1-speedupRatio));
                    end
                else
                    speedupString = 'N/A';
                end

                %% --- Print Direct ---
                if maxAbsErrorDirect < SINGLE_PRECISION_PASS_THRESHOLD
                    pfDirect = colorFormat('green', '✔️');
                    passDirect = passDirect + 1;
                else
                    pfDirect = colorFormat('red', '❌');
                    failDirect = failDirect + 1;
                end
                fprintf('%-4s %-5d %-9s %-12s %-20s %-14s %-20s %-7.2e %-8.2e %-8.2e %-8.3f %-10s\n', ...
                    pfDirect, testCounter, 'direct', dataTypeName, mat2str(volumeSize), mat2str(gaussianSigma), ...
                    kernelSizeDescription, maxAbsErrorDirect, rmsErrorDirect, relErrorDirect, ...
                    directElapsedSeconds, speedupString);

                %% --- Print FFT ---
                if maxAbsErrorFFT < SINGLE_PRECISION_PASS_THRESHOLD
                    pfFFT = colorFormat('green', '✔️');
                    passFFT = passFFT + 1;
                else
                    pfFFT = colorFormat('red', '❌');
                    failFFT = failFFT + 1;
                end
                fprintf('%-4s %-5d %-9s %-12s %-20s %-14s %-20s %-8.2e %-8.2e %-8.2e %-8.3f %-10s\n', ...
                    pfFFT, testCounter, 'fft', dataTypeName, mat2str(volumeSize), mat2str(gaussianSigma), ...
                    kernelSizeDescription, maxAbsErrorFFT, rmsErrorFFT, relErrorFFT, ...
                    fftElapsedSeconds, speedupString);

                reset(gpuDeviceObject);
            end
        end
    end

    % --- Suite summary ---
    fprintf('\n%s\n', colorFormat('yellow', repmat('=',1,80)));
    fprintf('%s\n', colorFormat('magenta', 'TEST SUITE SUMMARY'));
    fprintf('%-24s %-8d\n', colorFormat('green', 'Direct passed:'), passDirect);
    fprintf('%-24s %-8d\n', colorFormat('red',   'Direct failed:'), failDirect);
    fprintf('%-24s %-8d\n', colorFormat('green', 'FFT passed:'), passFFT);
    fprintf('%-24s %-8d\n', colorFormat('red',   'FFT failed:'), failFFT);
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
