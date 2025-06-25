function maxN = test_max_otf_size_gaussian_fast_descend
fprintf('Testing maximum *fast* OTF cube size for sigma = [0.5 0.5 2.5], psf size 9x9x21\n');

psf_sz = [9 9 21];
sigma = [0.5 0.5 2.5];
maxNpossible = 1290;
startN = 64;

% Prepare PSF only once
center = (psf_sz+1)/2;
[x, y, z] = ndgrid(1:psf_sz(1), 1:psf_sz(2), 1:psf_sz(3));
psf = exp(-0.5*((x-center(1))/sigma(1)).^2 ...
         -0.5*((y-center(2))/sigma(2)).^2 ...
         -0.5*((z-center(3))/sigma(3)).^2 );
psf = psf / sum(psf(:));
psf_shifted = ifftshift(single(psf)); % Only shift once!

% Get all fast cube sizes up to maxNpossible, sorted descending
candidates = next_fast_len(startN:maxNpossible);
candidates = unique(candidates);
candidates = sort(candidates, 'descend');

maxN = NaN; % default, in case none succeed

for idx = 1:numel(candidates)
    tryN = candidates(idx);
    fprintf('Trying N = %d... ', tryN);
    try
        psf_shifted_gpu = gpuArray(psf_shifted); % Only allocate/copy here
        fft_shape = [tryN tryN tryN];
        otf = otf_gpu(psf_shifted_gpu, fft_shape);
        otf_conj = conj_gpu(otf);
        wait(gpuDevice);
        clear otf otf_conj psf_shifted_gpu;
        fprintf('OK\n');
        maxN = tryN;
        break; % Stop at first success!
    catch ME
        fprintf('FAILED (%s)\n', ME.message);
        reset(gpuDevice);
    end
end

if ~isnan(maxN)
    fprintf('\nMax fast cube size N: %d (%.2f GB for OTF+conj)\n', maxN, 2*4*maxN^3/2^30);
else
    fprintf('\nNo cube size succeeded in this range.\n');
end
end

function n_vec = next_fast_len(n_vec)
    for i = 1:numel(n_vec)
        n = n_vec(i);
        while true
            f = factor(n);
            if all(f <= 7)
                n_vec(i) = n;
                break;
            end
            n = n + 1;
        end
    end
end
