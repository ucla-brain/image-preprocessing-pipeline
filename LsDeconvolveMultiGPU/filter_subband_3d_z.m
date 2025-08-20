function bl = filter_subband_3d_z(bl, sigma, levels, wavelet)
    % Applies filter_subband to each XZ slice (along Y-axis)
    % In-place update version to avoid extra allocation
    % start_time = tic;
    [X, Y, Z] = size(bl);
    % device = "CPU";
    if isgpuarray(bl)
        % Use underlyingType to get the data type inside the gpuArray
        original_class = underlyingType(bl);
        % dev = gpuDevice();
        % device = sprintf("GPU%d", dev.Index);
    else
        original_class = class(bl);
    end
    if ~strcmp(original_class, 'single')
        bl = single(bl);
    end

    % Dynamic range compression
    bl = log1p(bl);

    % Apply filtering across Y axis
    for y = 1:Y
        slice = reshape(bl(:, y, :), [X, Z]);
        slice = filter_subband(slice, sigma, levels, wavelet, 2);
        bl(:, y, :) = slice;
    end

    for x = 1:X
        slice = reshape(bl(x, :, :), [Y, Z]);
        slice = filter_subband(slice, sigma, levels, wavelet, 2);
        bl(x, :, :) = slice;
    end

    % Undo compression
    bl = expm1(bl);

    % Restore original data type
    % Restore original data type (remains on GPU if started as gpuArray)
    bl_class = underlyingType(bl);
    if ~strcmp(bl_class, original_class)
        bl = cast(bl, original_class);
    end
    % fprintf("%s: destripe ΔT: %.1f s\n", device, toc(start_time));
end

function img = filter_subband(img, sigma, levels, wavelet, axes)
    wl = lower(string(wavelet));
    if startsWith(wl, ["db", "coif","bior","rbio"])
        dwtmode('per','nodisp');
    else
        dwtmode('sym','nodisp');
    end

    if ~isa(img,'single'), img = single(img); end
    pad = [0 0];

    if levels == 0
        levels = wmaxlev(size(img), wavelet);
    end
    [C, S] = wavedec2(img, levels, wavelet);

    start_idx = prod(S(1, :));
    for n = 1:levels
        sz = prod(S(n + 1, :));

        idxH = start_idx + (1:sz);
        idxV = idxH(end) + (1:sz);

        if ismember(2, axes)
            H = reshape(C(idxH), S(n + 1, :));
            H = filter_coefficient(H, sigma / size(img, 1), 2);
            C(idxH) = H(:);
        end
        if ismember(1, axes)
            V = reshape(C(idxV), S(n + 1, :));
            V = filter_coefficient(V, sigma / size(img, 2), 1);
            C(idxV) = V(:);
        end

        start_idx = idxV(end) + sz;
    end

    img = waverec2(C, S, wavelet);

    idx = arrayfun(@(d) 1:(size(img, d) - pad(d)), 1:ndims(img), 'UniformOutput', false);
    img = img(idx{:});
end

function mat = filter_coefficient(mat, sigma, axis)
    sigma = max(sigma, eps('single'));
    n = size(mat, axis);
    sigma = (floor(n/2) + 1) * sigma;

    % Build notch vector in the correct class/device
    g = gaussian_notch_filter_1d(n, sigma, isgpuarray(mat), underlyingType(mat));

    if axis == 1
        % Already contiguous (dim 1). Fast path.
        mat = fft(mat, n, 1);
        mat = mat .* g(:);                        % broadcast along dim1
        mat = ifft(mat, n, 1, 'symmetric');      % avoids extra real()
    elseif axis == 2
        % Make axis 2 contiguous by swapping dims 1<->2, do work along dim1, swap back.
        mat = permute(mat, [2 1 3]);             % now target axis is dim1 (contiguous)
        mat = fft(mat, n, 1);
        mat = mat .* g(:);                        % broadcast along dim1
        mat = ifft(mat, n, 1, 'symmetric');
        mat = ipermute(mat, [2 1 3]);            % restore original layout
    else
        error('Invalid axis');
    end
end

function g = gaussian_notch_filter_1d(n, sigma, use_gpu, underlying_class)
    m = floor(n/2);

    if use_gpu
        x = gpuArray.colon(cast(0, underlying_class), cast(1, underlying_class), cast(m, underlying_class));
    else
        x = cast(0:m, underlying_class);
    end

    sigma = cast(sigma, 'like', x);
    gpos = 1 - exp(-x.^2 ./ (2 * sigma.^2));

    if use_gpu
        g = gpuArray.zeros(1, n, underlying_class);
    else
        g = zeros(1, n, underlying_class);
    end

    g(1:m+1) = gpos;
    if mod(n,2) == 0
        g(m+2:n) = gpos(m-1:-1:1);
    else
        g(m+2:n) = gpos(m:-1:1);
    end
end

% function bl = filter_subband_3d_z(bl, sigma, levels, wavelet)
%     % GPU-batched version using dldwt/dlidwt so multiple Y-slices run concurrently.
%     % In-place semantics preserved at the volume level; batching kept VRAM-aware.
% 
%     % ---------- dtype + setup ----------
%     [X, Y, Z] = size(bl);
%     if isgpuarray(bl)
%         original_class = underlyingType(bl);
%         gdev = gpuDevice();
%     else
%         original_class = class(bl);
%     end
%     if ~strcmp(original_class, 'single'); bl = single(bl); end
% 
%     % ---------- dynamic range compression ----------
%     bl = log1p(bl);
% 
%     % ---------- choose padding mode per your rule ----------
%     wl = lower(string(wavelet));
%     if startsWith(wl, ["db","coif","bior","rbio"])
%         paddingMode = "periodic";   % per
%     else
%         paddingMode = "symmetric";  % sym
%     end
% 
%     % ---------- VRAM-aware batching across Y ----------
%     if isgpuarray(bl)
%         memGB = free_GPU_vRAM(gdev.Index, gdev);           % your helper
%         bytes_free = max(0, memGB * 1e9);                  % your helper uses 1e9 scale
%         bytes_per_elem = 4;                                 % single
%         % Conservative per-slice working-set for DWT+FFT temps (rows=X, cols=Z):
%         mem_per_slice_bytes = max(1, 8 * X * Z * bytes_per_elem);
%         batch_size_y = max(1, floor( (0.5 * bytes_free) / mem_per_slice_bytes )); % 50% headroom
%     else
%         batch_size_y = 1;  % CPU fallback
%     end
% 
%     % ---------- process Y-slices in VRAM-sized batches ----------
%     for y0 = 1:batch_size_y:Y
%         ys = y0 : min(y0+batch_size_y-1, Y);
% 
%         % pack a slab of B slices: [X Z B]
%         slab = permute(bl(:, ys, :), [1 3 2]);   % [X Z B]
%         if ~isgpuarray(slab); slab = gpuArray(slab); end
%         slab = single(slab);
% 
%         % --- batched 2D DWT on GPU: convert to SSCB (S=spatial, C=1, B=batch) ---
%         slabSSCB = dlarray(slab, "SSCB");        % [X Z 1 B]
%         maxLev = wmaxlev([X Z], wavelet);           % can be 0 for small sizes / long filters
%         if levels <= 0
%             levelsEff = max(1, maxLev);
%         else
%             levelsEff = min( max(1, floor(double(levels))) , max(1, maxLev) );
%         end
%         [a, d] = dldwt(slabSSCB, Wavelet=wavelet, Level=levelsEff, PaddingMode=paddingMode, FullTree=true);  % GPU
% 
%         % ---------- levelsEff > 1 ----------
%         if iscell(d)
%             for lev = 1:levelsEff
%                 det_u = stripdims(d{lev});                 % numeric gpu array
% 
%                 r = size(det_u,1); c = size(det_u,2);
%                 sz = size(det_u); if numel(sz)<4, sz(4)=1; end
%                 % Find orientation (==3) and batch dims (the other non-spatial)
%                 idxOrient = find(sz==3,1,'first'); 
%                 rest = 3:numel(sz); rest(rest==idxOrient)=[];
%                 idxBatch = rest(find(sz(rest)~=1,1,'first')); 
%                 if isempty(idxBatch), idxBatch = setdiff(3:4, idxOrient); end
%                 % Canonicalize to [r c 3 B]
%                 det_u = permute(det_u, [1 2 idxOrient idxBatch]);
%                 B = size(det_u,4);
% 
%                 % Pack (channel,batch) into pages and notch H/V
%                 det_u = reshape(det_u, r, c, 3*B);                 % pages: HL1 LH1 HH1 HL2 LH2 HH2 ...
%                 selH = 1:3:(3*B); if ~isempty(selH), det_u(:,:,selH) = filter_coefficient_pages(det_u(:,:,selH), sigma / X, 2); end
%                 selV = 2:3:(3*B); if ~isempty(selV), det_u(:,:,selV) = filter_coefficient_pages(det_u(:,:,selV), sigma / Z, 1); end
%                 det_u = reshape(det_u, r, c, 3, B);
% 
%                 % Restore original orientation/batch ordering
%                 invperm = [1 2 idxOrient idxBatch];
%                 [~,invperm] = sort(invperm);
%                 det_u = permute(det_u, invperm);
% 
%                 d{lev} = dlarray(det_u, "SSCB");
%             end
% 
%         % ---------- levelsEff == 1 ----------
%         else
%             det_u = stripdims(d);                              % [r c ? ?]
%             r = size(det_u,1); c = size(det_u,2);
%             sz = size(det_u); if numel(sz)<4, sz(4)=1; end
%             idxOrient = find(sz==3,1,'first'); 
%             rest = 3:numel(sz); rest(rest==idxOrient)=[];
%             idxBatch = rest(find(sz(rest)~=1,1,'first')); 
%             if isempty(idxBatch), idxBatch = setdiff(3:4, idxOrient); end
%             det_u = permute(det_u, [1 2 idxOrient idxBatch]);
%             B = size(det_u,4);
% 
%             det_u = reshape(det_u, r, c, 3*B);
%             selH = 1:3:(3*B); if ~isempty(selH), det_u(:,:,selH) = filter_coefficient_pages(det_u(:,:,selH), sigma / X, 2); end
%             selV = 2:3:(3*B); if ~isempty(selV), det_u(:,:,selV) = filter_coefficient_pages(det_u(:,:,selV), sigma / Z, 1); end
%             det_u = reshape(det_u, r, c, 3, B);
% 
%             invperm = [1 2 idxOrient idxBatch];
%             [~,invperm] = sort(invperm);
%             det_u = permute(det_u, invperm);
% 
%             d = dlarray(det_u, "SSCB");
%         end
% 
%         % --- inverse batched DWT on GPU ---
%         slabRec = dlidwt(a, d, Wavelet=wavelet, PaddingMode=paddingMode);  % [X Z 1 B] dlarray
%         slabOut = extractdata(slabRec);                                     % numeric gpu array [X Z 1 B]
%         slabOut = reshape(slabOut, X, Z, []);                               % [X Z B]
% 
%         % scatter back in place
%         bl(:, ys, :) = ipermute(slabOut, [1 3 2]);  % back to [X Y Z]
%     end
% 
%     % ---------- undo compression ----------
%     bl = expm1(bl);
% 
%     % ---------- restore dtype ----------
%     if ~strcmp(underlyingType(bl), original_class)
%         bl = cast(bl, original_class);
%     end
% end
% 
% % ===================================================================================
% % Internal helpers (minimal, focused). Kept in-file to avoid new dependencies.
% % ===================================================================================
% 
% function mat = filter_coefficient_pages(mat, sigma, axis)
%     sigma = max(sigma, eps('single'));
%     n = size(mat, axis);
%     sigma = (floor(n/2) + 1) * sigma;
%     g = gaussian_notch_filter_1d(n, sigma, isgpuarray(mat), underlyingType(mat));
% 
%     if axis == 1
%         mat = fft(mat, n, 1);
%         scale = reshape(g, [], 1, 1);      %  [n 1 1], same device/class
%         mat = mat .* scale;                %  implicit expansion (GPU-friendly)
%         mat = ifft(mat, n, 1, 'symmetric');
%     elseif axis == 2
%         mat = permute(mat, [2 1 3]);       % make target axis contiguous
%         mat = fft(mat, n, 1);
%         scale = reshape(g, [], 1, 1);
%         mat = mat .* scale;
%         mat = ifft(mat, n, 1, 'symmetric');
%         mat = ipermute(mat, [2 1 3]);
%     else
%         error('Invalid axis');
%     end
% end
% 
% function g = gaussian_notch_filter_1d(n, sigma, use_gpu, underlying_class)
% % Same math as your original, no caching (per your request).
%     m = floor(n/2);
%     if use_gpu
%         x = gpuArray.colon(cast(0, underlying_class), cast(1, underlying_class), cast(m, underlying_class));
%         g = gpuArray.zeros(1, n, underlying_class);
%     else
%         x = cast(0:m, underlying_class);
%         g = zeros(1, n, underlying_class);
%     end
%     sigma = cast(sigma, 'like', x);
%     gpos = 1 - exp(-x.^2 ./ (2 * sigma.^2));
%     g(1:m+1) = gpos;
%     if mod(n,2) == 0, g(m+2:n) = gpos(m-1:-1:1); else, g(m+2:n) = gpos(m:-1:1); end
% end
