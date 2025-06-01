function test_gauss3d_mex_headless()
    % Check for imgaussfilt3 (Image Processing Toolbox)
    if ~exist('imgaussfilt3','file')
        error('Requires Image Processing Toolbox for validation.');
    end

    sizes = {[32 32 16], [33 40 29], [24 18 21]};
    types = {@single, @double};
    sigmas = {1.5, [1 2 3]};
    tol = 2e-5;  % Acceptable max error

    failures = 0;
    total = 0;

    for isz = 1:numel(sizes)
        sz = sizes{isz};
        for ityp = 1:numel(types)
            castf = types{ityp};
            for isig = 1:numel(sigmas)
                sigma = sigmas{isig};
                total = total + 1;

                x = rand(sz, 'single');
                x = castf(x);

                y_mex = gauss3d_mex(x, sigma);

                y_ref = imgaussfilt3(x, sigma, ...
                    'Padding','replicate','FilterSize',odd_kernel_size(sigma));

                maxerr = max(abs(y_mex(:) - y_ref(:)));
                rmserr = sqrt(mean((y_mex(:)-y_ref(:)).^2));

                tag = sprintf('size=%s, type=%s, sigma=%s', ...
                    mat2str(sz), func2str(castf), mat2str(sigma));
                if maxerr > tol || isnan(maxerr)
                    fprintf('FAIL: %s | maxerr=%.2e, rmserr=%.2e\n', tag, maxerr, rmserr);
                    failures = failures + 1;
                else
                    fprintf('PASS: %s | maxerr=%.2e, rmserr=%.2e\n', tag, maxerr, rmserr);
                end
            end
        end
    end

    fprintf('\n%d/%d tests passed. %d failed.\n', total-failures, total, failures);
    if failures > 0
        error('Some tests FAILED.');
    else
        disp('All tests PASSED.');
    end
end

function sz = odd_kernel_size(sigma)
    if numel(sigma)==1, sigma=[sigma sigma sigma]; end
    sz = 2*ceil(3*sigma)+1;
    sz = max(sz,3);
end
