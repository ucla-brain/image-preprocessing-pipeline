function test_gauss3d_mex()
    % Check for imgaussfilt3 (Image Processing Toolbox)
    if ~exist('imgaussfilt3','file')
        error('Requires Image Processing Toolbox for validation.');
    end

    % Test parameters
    sizes = {[32 32 16], [33 40 29]};  % even and odd dimensions
    types = {@single, @double};
    sigmas = {1.5, [1.0 2.0 3.0]};
    tol = 1e-5;   % Acceptable error for comparison

    for isz = 1:numel(sizes)
        sz = sizes{isz};
        for ityp = 1:numel(types)
            castf = types{ityp};
            for isig = 1:numel(sigmas)
                sigma = sigmas{isig};
                fprintf('Testing size %s, type %s, sigma %s ... ', ...
                    mat2str(sz), func2str(castf), mat2str(sigma));

                % Generate synthetic 3D data
                x = rand(sz, 'single');
                x = castf(x);

                % Run your CUDA filter
                y_cuda = gauss3d_mex(x, sigma);

                % Run MATLAB reference
                y_ref = imgaussfilt3(x, sigma, ...
                    'Padding','replicate','FilterSize',odd_kernel_size(sigma));

                % Error metrics
                maxerr = max(abs(y_cuda(:) - y_ref(:)));
                rmserr = sqrt(mean((y_cuda(:)-y_ref(:)).^2));
                fprintf('maxerr = %.2e, rmserr = %.2e\n', maxerr, rmserr);

                % Assert
                if maxerr > tol
                    warning('Test failed: maxerr > tolerance!');
                end

                % Visual check (optional)
                if isz == 1 && ityp == 1 && isig == 1
                    figure(1); clf;
                    subplot(1,3,1); imagesc(squeeze(y_ref(:,:,end/2))); axis image; title('MATLAB imgaussfilt3');
                    subplot(1,3,2); imagesc(squeeze(y_cuda(:,:,end/2))); axis image; title('gauss3d\_mex');
                    subplot(1,3,3); imagesc(squeeze(y_ref(:,:,end/2))-squeeze(y_cuda(:,:,end/2))); axis image; title('Difference');
                    colorbar;
                    drawnow;
                end
            end
        end
    end

    disp('All tests done.');

end

function sz = odd_kernel_size(sigma)
    % Compute an odd kernel size for fair comparison
    if numel(sigma)==1, sigma=[sigma sigma sigma]; end
    sz = 2*ceil(3*sigma)+1;
    sz = max(sz,3); % minimum filter size is 3 for imgaussfilt3
end
