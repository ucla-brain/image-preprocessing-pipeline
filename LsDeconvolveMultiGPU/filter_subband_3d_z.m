% function bl = filter_subband_3d_z(bl, sigma, levels, wavelet)
%     % Applies filter_subband to each XZ slice (along Y-axis)
%     % In-place update version to avoid extra allocation
%     % start_time = tic;
%     [X, Y, Z] = size(bl);
%     % device = "CPU";
%     if isgpuarray(bl)
%         % Use underlyingType to get the data type inside the gpuArray
%         original_class = underlyingType(bl);
%         % dev = gpuDevice();
%         % device = sprintf("GPU%d", dev.Index);
%     else
%         original_class = class(bl);
%     end
%     if ~strcmp(original_class, 'single')
%         bl = single(bl);
%     end
% 
%     % Dynamic range compression
%     bl = log1p(bl);
% 
%     % Apply filtering across Y axis
%     for y = 1:Y
%         slice = reshape(bl(:, y, :), [X, Z]);
%         slice = filter_subband(slice, sigma, levels, wavelet, 2);
%         bl(:, y, :) = slice;
%     end
% 
%     for x = 1:X
%         slice = reshape(bl(x, :, :), [Y, Z]);
%         slice = filter_subband(slice, sigma, levels, wavelet, 2);
%         bl(x, :, :) = slice;
%     end
% 
%     % Undo compression
%     bl = expm1(bl);
% 
%     % Restore original data type
%     % Restore original data type (remains on GPU if started as gpuArray)
%     bl_class = underlyingType(bl);
%     if ~strcmp(bl_class, original_class)
%         bl = cast(bl, original_class);
%     end
%     % fprintf("%s: destripe ΔT: %.1f s\n", device, toc(start_time));
% end
% 
% function img = filter_subband(img, sigma, levels, wavelet, axes)
%     wl = lower(string(wavelet));
%     if startsWith(wl, ["db", "coif","bior","rbio"])
%         dwtmode('per','nodisp');
%     else
%         dwtmode('sym','nodisp');
%     end
% 
%     if ~isa(img,'single'), img = single(img); end
%     pad = [0 0];
% 
%     if levels == 0
%         levels = wmaxlev(size(img), wavelet);
%     end
%     [C, S] = wavedec2(img, levels, wavelet);
% 
%     start_idx = prod(S(1, :));
%     for n = 1:levels
%         sz = prod(S(n + 1, :));
% 
%         idxH = start_idx + (1:sz);
%         idxV = idxH(end) + (1:sz);
% 
%         if ismember(2, axes)
%             H = reshape(C(idxH), S(n + 1, :));
%             H = filter_coefficient(H, sigma / size(img, 1), 2);
%             C(idxH) = H(:);
%         end
%         if ismember(1, axes)
%             V = reshape(C(idxV), S(n + 1, :));
%             V = filter_coefficient(V, sigma / size(img, 2), 1);
%             C(idxV) = V(:);
%         end
% 
%         start_idx = idxV(end) + sz;
%     end
% 
%     img = waverec2(C, S, wavelet);
% 
%     idx = arrayfun(@(d) 1:(size(img, d) - pad(d)), 1:ndims(img), 'UniformOutput', false);
%     img = img(idx{:});
% end
% 
% function mat = filter_coefficient(mat, sigma, axis)
%     sigma = max(sigma, eps('single'));
%     n = size(mat, axis);
%     sigma = (floor(n/2) + 1) * sigma;
% 
%     % Build notch vector in the correct class/device
%     g = gaussian_notch_filter_1d(n, sigma, isgpuarray(mat), underlyingType(mat));
% 
%     if axis == 1
%         % Already contiguous (dim 1). Fast path.
%         mat = fft(mat, n, 1);
%         mat = mat .* g(:);                        % broadcast along dim1
%         mat = ifft(mat, n, 1, 'symmetric');      % avoids extra real()
%     elseif axis == 2
%         % Make axis 2 contiguous by swapping dims 1<->2, do work along dim1, swap back.
%         mat = permute(mat, [2 1 3]);             % now target axis is dim1 (contiguous)
%         mat = fft(mat, n, 1);
%         mat = mat .* g(:);                        % broadcast along dim1
%         mat = ifft(mat, n, 1, 'symmetric');
%         mat = ipermute(mat, [2 1 3]);            % restore original layout
%     else
%         error('Invalid axis');
%     end
% end
% 
% function g = gaussian_notch_filter_1d(n, sigma, use_gpu, underlying_class)
%     m = floor(n/2);
% 
%     if use_gpu
%         x = gpuArray.colon(cast(0, underlying_class), cast(1, underlying_class), cast(m, underlying_class));
%     else
%         x = cast(0:m, underlying_class);
%     end
% 
%     sigma = cast(sigma, 'like', x);
%     gpos = 1 - exp(-x.^2 ./ (2 * sigma.^2));
% 
%     if use_gpu
%         g = gpuArray.zeros(1, n, underlying_class);
%     else
%         g = zeros(1, n, underlying_class);
%     end
% 
%     g(1:m+1) = gpos;
%     if mod(n,2) == 0
%         g(m+2:n) = gpos(m-1:-1:1);
%     else
%         g(m+2:n) = gpos(m:-1:1);
%     end
% end

% ============================== new version ==============================

function volumeIn = filter_subband_3d_z(volumeIn, sigmaValue, decompositionLevels, waveletName, axes_to_filter)
% FILTER_SUBBAND_3D_Z
% 3D destriping via batched 2D wavelet filtering across Y-slices and X-slices.
% Runs on CPU if 'volumeIn' is on CPU; runs on GPU if 'volumeIn' is gpuArray.
% Axis control:
%   axes_to_filter: [1] => vertical (rows/LH), [2] => horizontal (cols/HL), [1 2] => both.
% Default matches your original intent: filter along axis=2 (horizontal / HL).

    if nargin < 5 || isempty(axes_to_filter), axes_to_filter = 2; end
    axes_to_filter = sort(unique(axes_to_filter(:).'));

    % ---------- dtype + setup ----------
    [sizeX, sizeY, sizeZ] = size(volumeIn);
    isInputOnGPU = isgpuarray(volumeIn);
    if isInputOnGPU
        originalDataClass = underlyingType(volumeIn);
    else
        originalDataClass = class(volumeIn);
    end
    if ~strcmp(originalDataClass, 'single'), volumeIn = single(volumeIn); end

    % ---------- dynamic range compression ----------
    volumeIn = log1p(volumeIn);

    % ---------- padding mode & levels for Y-pass ([X Z]) ----------
    paddingModeString = compute_padding_mode_string(waveletName);

    % ---------- VRAM-aware batching (uses your project helper free_GPU_vRAM) ----------
    if isInputOnGPU
        currentGpuDevice = gpuDevice();
        freeMemoryGB = free_GPU_vRAM(currentGpuDevice.Index, currentGpuDevice);
        bytesFreeApprox = max(0, freeMemoryGB * 1e9);   % helper uses 1e9 scale
        bytesPerElement = 4;                            % single precision
        % Conservative per-slice working-set estimate (FFT/DWT temps)
        bytesPerSliceEstimate = max(1, 16 * max(sizeX, sizeY) * sizeZ * bytesPerElement);
        batchSize = max(1, floor( 0.5 * bytesFreeApprox / bytesPerSliceEstimate )); % 50% headroom
    else
        batchSize = 1; % CPU
    end
    
    % ---------- pass 1: process Y-slices (XZ planes @ fixed Y) ----------
    effectiveLevels_Ypass = compute_effective_levels([sizeX sizeZ], waveletName, decompositionLevels);
    for batchStartY = 1:batchSize:sizeY
        batchIndicesY = batchStartY : min(batchStartY + batchSize - 1, sizeY);

        % Pack slab of XZ images: [sizeX sizeZ batch]
        slab = permute(volumeIn(:, batchIndicesY, :), [1 3 2]);  % [X Z B]

        % Batched DWT → notch → IDWT (device follows input)
        slab = process_slab_batch(slab, sigmaValue, effectiveLevels_Ypass, waveletName, paddingModeString, axes_to_filter);

        % Scatter back in place into [X Y Z]
        volumeIn(:, batchIndicesY, :) = ipermute(slab, [1 3 2]);
    end

    % ---------- pass 2: process X-slices (YZ planes @ fixed X) ----------
    effectiveLevels_Xpass = compute_effective_levels([sizeY sizeZ], waveletName, decompositionLevels);
    for batchStartX = 1:batchSize:sizeX
        batchIndicesX = batchStartX : min(batchStartX + batchSize - 1, sizeX);

        % Pack slab of YZ images: [sizeY sizeZ batch]
        slab = permute(volumeIn(batchIndicesX, :, :), [2 3 1]);  % [Y Z B]

        % Batched DWT → notch → IDWT (device follows input)
        slab = process_slab_batch(slab, sigmaValue, effectiveLevels_Xpass, waveletName, paddingModeString, axes_to_filter);

        % Scatter back in place into [X Y Z]
        volumeIn(batchIndicesX, :, :) = ipermute(slab, [2 3 1]);
    end

    % ---------- undo compression ----------
    volumeIn = expm1(volumeIn);

    % ---------- restore dtype ----------
    if ~strcmp(underlyingType(volumeIn), originalDataClass)
        volumeIn = cast(volumeIn, originalDataClass);
    end
end


function imageOut = filter_subband_2d(imageIn, sigmaValue, decompositionLevels, waveletName, axes_to_filter)
% FILTER_SUBBAND_2D
% 2D counterpart for visual sanity checks. CPU/GPU follows input device.
% Axis control same as 3D:
%   axes_to_filter: [1] => vertical (rows/LH), [2] => horizontal (cols/HL), [1 2] => both (default).

    if nargin < 5 || isempty(axes_to_filter), axes_to_filter = [1 2]; end
    axes_to_filter = sort(unique(axes_to_filter(:).'));

    isInputOnGPU = isgpuarray(imageIn);
    if isgpuarray(imageIn)
        originalDataClass = underlyingType(imageIn);
    else
        originalDataClass = class(imageIn);
    end
    if ~strcmp(originalDataClass,'single'), imageIn = single(imageIn); end

    % Compression
    imageIn = log1p(imageIn);

    % Use the same batched core with batch=1
    numRows = size(imageIn,1); numCols = size(imageIn,2);
    paddingModeString = compute_padding_mode_string(waveletName);
    effectiveLevels   = compute_effective_levels([numRows numCols], waveletName, decompositionLevels);

    % Shape to [rows cols 1] slab and process
    batchSlabXZB = reshape(imageIn, numRows, numCols, 1);
    if isInputOnGPU && ~isgpuarray(batchSlabXZB), batchSlabXZB = gpuArray(batchSlabXZB); end

    reconstructedBatchXZB = process_slab_batch(batchSlabXZB, sigmaValue, effectiveLevels, waveletName, paddingModeString, axes_to_filter);

    % Back to 2D
    imageOut = reconstructedBatchXZB(:,:,1);

    % Undo compression
    imageOut = expm1(imageOut);

    % Restore dtype
    if ~strcmp(underlyingType(imageOut), originalDataClass)
        imageOut = cast(imageOut, originalDataClass);
    end
end


% =============================================================================
% Core batched processor (shared by 2D and 3D)
% =============================================================================
function slab = process_slab_batch(slab, sigmaValue, effectiveLevels, waveletName, paddingModeString, axes_to_filter)
% PROCESS_SLAB_BATCH (VRAM-optimized)
% In-place style: accepts/returns numeric CPU/GPU 'slab' shaped [rows, cols, batch].
% Steps: reshape → DWT → notch HL/LH → IDWT → reshape back (robust to squeezing).

    % Preserve device & class exactly (no changes if already single gpuArray)
    inputIsOnGpu = isgpuarray(slab);
    if inputIsOnGpu
        originalUnderlyingClass = underlyingType(slab);
    else
        originalUnderlyingClass = class(slab);
    end

    % Canonical SSCB view with minimal scalars
    rows = size(slab,1);
    cols = size(slab,2);
    slab = dlarray(reshape(slab, rows, cols, 1, []), "SSCB");   % [rows cols 1 batch]

    % Forward DWT (SSCB ⇒ details are [r c C B] with C=3)
    [approximationCoeffs, detailCoeffs] = dldwt(slab, Wavelet=waveletName, Level=effectiveLevels, PaddingMode=paddingModeString, FullTree=true);

    % Notch HL/LH per level in place, channel-wise (no page packing)
    if iscell(detailCoeffs)
        for levelIndex = 1:effectiveLevels
            tmp = stripdims(detailCoeffs{levelIndex});       % numeric on CPU/GPU, [r c C B]
            tmp = notch_HV_in_batch(tmp, sigmaValue, axes_to_filter);
            detailCoeffs{levelIndex} = dlarray(tmp, "SSCB");
        end
    else
        tmp = stripdims(detailCoeffs);
        tmp = notch_HV_in_batch(tmp, sigmaValue, axes_to_filter);
        detailCoeffs = dlarray(tmp, "SSCB");
    end

    % Inverse DWT → numeric and reshape back without guessing batch
    slab = dlidwt(approximationCoeffs, detailCoeffs, Wavelet=waveletName, PaddingMode=paddingModeString);  % [rows cols 1 batch] or squeezed
    slab = extractdata(slab);
    slab = reshape(slab, rows, cols, []);   % handles both [rows cols 1 B] and [rows cols B]

    % Restore device/class exactly
    if inputIsOnGpu && ~isgpuarray(slab), slab = gpuArray(slab); end
    if inputIsOnGpu
        currentClass = underlyingType(slab);
    else
        currentClass = class(slab);
    end
    if ~strcmp(currentClass, originalUnderlyingClass), slab = cast(slab, originalUnderlyingClass); end
end

% =============================================================================
% Helpers: padding/levels
% =============================================================================
function paddingModeString = compute_padding_mode_string(waveletName)
    lowerWaveletName = lower(string(waveletName));
    if startsWith(lowerWaveletName, ["db","coif","bior","rbio"])
        paddingModeString = "periodic";   % 'per'
    else
        paddingModeString = "symmetric";  % 'sym'
    end
end

function effectiveLevels = compute_effective_levels(sizeVectorRC, waveletName, requestedLevels)
    maximumAllowedLevels = wmaxlev(sizeVectorRC, waveletName); % may be 0 for very small sizes
    if requestedLevels <= 0
        effectiveLevels = max(1, maximumAllowedLevels);
    else
        effectiveLevels = min( max(1, floor(double(requestedLevels))) , max(1, maximumAllowedLevels) );
    end
end

% =============================================================================
% Helper: orientation-aware notch (axis control & batching)
% =============================================================================
function detailTensor = notch_HV_in_batch(detailTensor, sigmaValue, axes_to_filter)
% NOTCH_HV_IN_BATCH
% In-place style: accepts and returns 'detailTensor' with SSCB layout [rows, cols, channels, batch].
% Channels are [HL, LH, HH]. We apply:
%   - if 2 ∈ axes_to_filter : HL (channel 1) → notch along columns (axis=2), sigma scaled by rows
%   - if 1 ∈ axes_to_filter : LH (channel 2) → notch along rows    (axis=1), sigma scaled by cols

    % Ensure 4-D shape [rows cols channels batch]
    sizeVector = size(detailTensor);
    if numel(sizeVector) < 4
        sizeVector(4) = 1;
        detailTensor = reshape(detailTensor, sizeVector);
    end

    numRows     = size(detailTensor, 1);
    numCols     = size(detailTensor, 2);
    numChannels = size(detailTensor, 3);
    numBatch    = size(detailTensor, 4);

    % ---- HL (channel 1): filter along columns (axis=2) ----
    if any(axes_to_filter == 2) && numChannels >= 1
        hlStack = reshape(detailTensor(:, :, 1, :), numRows, numCols, numBatch);          % [rows cols batch]
        hlStack = filter_coefficient_pages(hlStack, sigmaValue / numRows, 2);             % notch along columns
        detailTensor(:, :, 1, :) = reshape(hlStack, numRows, numCols, 1, numBatch);
    end

    % ---- LH (channel 2): filter along rows (axis=1) ----
    if any(axes_to_filter == 1) && numChannels >= 2
        lhStack = reshape(detailTensor(:, :, 2, :), numRows, numCols, numBatch);          % [rows cols batch]
        lhStack = filter_coefficient_pages(lhStack, sigmaValue / numCols, 1);             % notch along rows
        detailTensor(:, :, 2, :) = reshape(lhStack, numRows, numCols, 1, numBatch);
    end

    % HH (channel ≥3) untouched
end


% =============================================================================
% Helpers: FFT-based coefficient filtering (batch-friendly)
% =============================================================================
function stack = filter_coefficient_pages(stack, sigmaUnitless, fftAxis)
% stackIn: [rows cols pages], operate along fftAxis ∈ {1,2}.
% Uses FFT along a contiguous dimension for speed and implicit expansion for scaling.

    assert(ismember(fftAxis, [1, 2]), 'fftAxis must be 1 or 2.');
    sigmaUnitless = max(sigmaUnitless, eps('single'));
    transformLength = size(stack, fftAxis);
    sigmaPixels = (floor(transformLength/2) + 1) * sigmaUnitless;
    if fftAxis == 2, stack = permute(stack, [2 1 3]); end   % make target axis contiguous
    stack = fft(stack, transformLength, 1);
    stack = stack .* reshape(gaussian_notch_filter_1d(transformLength, sigmaPixels, isgpuarray(stack), underlyingType(stack)), [], 1, 1);
    stack = ifft(stack, transformLength, 1, 'symmetric');
    if fftAxis == 2, stack = ipermute(stack, [2 1 3]); end
end


% =============================================================================
% Helper: Gaussian notch generator (no caching; CPU/GPU aware)
% =============================================================================
function notchVector = gaussian_notch_filter_1d(transformLength, sigmaPixels, isOnGPU, underlyingClassName)
    halfLength = floor(transformLength/2);
    if isOnGPU
        coordinateVector = gpuArray.colon(cast(0, underlyingClassName), cast(1, underlyingClassName), cast(halfLength, underlyingClassName));
        notchVector = gpuArray.zeros(1, transformLength, underlyingClassName);
    else
        coordinateVector = cast(0:halfLength, underlyingClassName);
        notchVector = zeros(1, transformLength, underlyingClassName);
    end
    sigmaPixels = cast(sigmaPixels, 'like', coordinateVector);
    positiveHalf = 1 - exp(-(coordinateVector.^2) ./ (2 * sigmaPixels.^2));
    notchVector(1:halfLength+1) = positiveHalf;
    if mod(transformLength,2) == 0
        notchVector(halfLength+2:transformLength) = positiveHalf(halfLength-1:-1:1);
    else
        notchVector(halfLength+2:transformLength) = positiveHalf(halfLength:-1:1);
    end
end
