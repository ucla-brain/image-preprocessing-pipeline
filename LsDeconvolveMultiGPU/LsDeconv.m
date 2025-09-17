% Program for Deconvolution of Light Sheet Microscopy Stacks.
%
% Originally written in MATLAB 2018b by Klaus Becker klaus.becker at tuwien.ac.at
%
% Enhanced by Keivan Moradi at UCLA B.R.A.I.N. (Dong lab) in MATLAB 2023a.
% Contact: kmoradi at mednet.ucla.edu
%
% Main modifications by Keivan Moradi:
%   - Multi-GPU support
%   - Resume capability
%   - 3D Gaussian filter support
%   - Destripe function specifically along the z-axis
%   - lots of C/C++/CUDA functions
%   - adaptive PSF
%
% LsDeconv is free software: you can redistribute it and/or modify it
% under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% LsDeconv is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%
% See the GNU General Public License for more details.
% You should have received a copy of the GNU General Public License
% along with this file. If not, see <http://www.gnu.org/licenses/>.

function [] = LsDeconv(varargin)
    try
        disp(' ');
        disp('LsDeconv: Deconvolution tool for Light Sheet Microscopy.');
        disp('TU Wien, 2019: This program was was initially written in MATLAB V2018b by klaus.becker@tuwien.ac.at');
        disp('Keivan Moradi, 2023-5: Rewrote in MATLAB V2025a. kmoradi@mednet.ucla.edu. UCLA B.R.A.I.N (Dong lab)');
        disp('Multi-GPU and Multi-CPU parallel processing, Resume, Destripe, Gaussian, 3D edge taper, and, adaptive PSF.');
        disp(' ');
        disp(datetime('now'));

        % make sure correct number of parameters specified
        if nargin ~= 33
            showinfo();
            return
        end

        %read command line parameters
        disp('assigning command line parameter strings')
        inpath                     = varargin{1};
        dxy                        = varargin{2};
        dz                         = varargin{3};
        numit                      = varargin{4};
        NA                         = varargin{5};
        rf                         = varargin{6};
        lambda_ex                  = varargin{7};
        lambda_em                  = varargin{8};
        fcyl                       = varargin{9};
        slitwidth                  = varargin{10};
        damping                    = varargin{11};
        clipval                    = varargin{12};
        stop_criterion             = varargin{13};
        block_size_max             = double(varargin{14});
        gpus                       = varargin{15};
        amplification              = single(varargin{16});

        filter.gaussian_sigma      = varargin{17};
        filter.dark                = varargin{18};

        filter.destripe_sigma      = varargin{19};

        filter.fibermetric_sigma   = varargin{20};
        filter.fibermetric_alpha   = varargin{21};
        filter.fibermetric_beta    = varargin{22};
        filter.fibermetric_gamma   = varargin{23};
        filter.fibermetric_method  = varargin{24};

        filter.regularize_interval = varargin{25};
        resume                     = varargin{26};
        starting_block             = varargin{27};
        convert_to_8bit            = varargin{28};
        convert_to_16bit           = varargin{29};
        filter.use_fft             = varargin{30};
        filter.adaptive_psf        = varargin{31};
        filter.accelerate          = varargin{32};
        cache_drive                = varargin{33};
        if ~exist(cache_drive, 'dir')
            mkdir(cache_drive);
            disp('Cache drive dir created: ' + cache_drive)
        else
            disp('Cache drive dir exists: ' + cache_drive)
        end
        
        assert(isa(inpath     , 'string'), "wrong type " + class(inpath));
        assert(isa(dxy        , 'double'), "wrong type " + class(dxy));
        assert(isa(dz         , 'double'), "wrong type " + class(dz));
        assert(isa(numit      , 'double'), "wrong type " + class(numit));
        assert(isa(lambda_ex  , 'double'), "wrong type " + class(lambda_ex));
        assert(isa(lambda_em  , 'double'), "wrong type " + class(lambda_em));
        assert(isa(cache_drive, 'string'), "wrong type " + class(cache_drive));

        if isfolder(inpath)
            % make folder for results and make sure the outpath is writable
            outpath = fullfile(inpath, 'deconvolved');
            if ~exist(outpath, 'dir')
                disp('making folder ' + outpath )
                mkdir(outpath);
            end
            disp('outpath folder created and/or exists ' + outpath);
            log_file_path = fullfile(outpath, 'log.txt');
            log_file = fopen(log_file_path, 'w');
            disp('made log file: ' + log_file_path)
            % get available resources
            [ram_available, ram_total]  = get_memory();
            p_log(log_file, 'system information ...');
            p_log(log_file, ['   total RAM (GB): ' num2str(ram_total/1024^3)]);
            p_log(log_file, ['   available RAM (GB): ' num2str(ram_available/1024^3)]);
            p_log(log_file, ['   data processed on GPUs: ' num2str(unique(gpus(gpus>0)))]);
            p_log(log_file, ['   data processed on CPUs: ' num2str(size(gpus(gpus<1), 2)) ' cores']);
            p_log(log_file, ' ');

            p_log(log_file, 'image stack info ...');
            [stack_info.x, stack_info.y, stack_info.z, stack_info.bit_depth] = getstackinfo(inpath); % n is volume dimension
            stack_info.dxy = dxy;
            stack_info.dz = dz;
            stack_info.convert_to_8bit = convert_to_8bit;
            stack_info.convert_to_16bit = convert_to_16bit;

            if resume && numel(dir(fullfile(outpath, 'img_*.tif'))) == stack_info.z
                disp('it seems all the files are already deconvolved!');
                return
            elseif ~resume
                delete(fullfile(outpath, '*.tif'));
            end
        else
            error('provided path did not exist!');
        end

        if stack_info.x * stack_info.y * stack_info.z == 0
            error('No valid TIFF files could be found!');
        end

        if ~exist(cache_drive, 'dir')
            mkdir(cache_drive);
        elseif ~resume
            delete(fullfile(cache_drive, "*.mat"));
            delete(fullfile(cache_drive, "*.lz4"));
            delete(fullfile(cache_drive, "*.tmp"));
        end

        p_log(log_file, ['   image size (voxels): ' num2str(stack_info.x)  'x * ' num2str(stack_info.y) 'y * ' num2str(stack_info.z) 'z = ' num2str(stack_info.x * stack_info.y * stack_info.z)]);
        p_log(log_file, ['   voxel size (nm^3): ' num2str(dxy)  'x * ' num2str(dxy) 'y * ' num2str(dz) 'z = ' num2str(dxy^2*dz)]);
        p_log(log_file, ['   image bit depth: ' num2str(stack_info.bit_depth)]);
        p_log(log_file, ' ');

        p_log(log_file, 'imaging system parameters ...')
        p_log(log_file, ['   focal length of cylinder lens (mm): ' num2str(fcyl)]);
        p_log(log_file, ['   width of slit aperture (mm): ' num2str(slitwidth)]);
        p_log(log_file, ['   numerical aperture: ' num2str(NA)]);
        p_log(log_file, ['   excitation wavelength (nm): ' num2str(lambda_ex)]);
        p_log(log_file, ['   emission wavelength (nm): ' num2str(lambda_em)]);
        p_log(log_file, ['   refractive index: ' num2str(rf)]);
        p_log(log_file, ' ');

        % generate PSF
        p_log(log_file, 'calculating PSF ...');
        Rxy = 0.61 * lambda_em / NA;
        dxy_corr = min(dxy, Rxy / 3);
        [psf.psf, FWHMxy, FWHMz] = LsMakePSF(dxy_corr, dz, NA, rf, lambda_ex, lambda_em, fcyl, slitwidth);
        psf.inv = psf.psf(end:-1:1, end:-1:1, end:-1:1);
        p_log(log_file, ['   size of PSF (pixel): ' num2str(size(psf.psf, 1))  ' x ' num2str(size(psf.psf, 2)) ' x ' num2str(size(psf.psf, 3))]);
        p_log(log_file, ['   FWHHM of PSF lateral (nm): ' num2str(FWHMxy)]);
        p_log(log_file, ['   FWHHM of PSF axial (nm): ' num2str(FWHMz)]);
        p_log(log_file, ['   Rayleigh range of objective lateral (nm): ' num2str(Rxy)]);
        p_log(log_file, ['   Rayleigh range of objective axial (nm): ' num2str((2 * lambda_em * rf) / NA^2)]);
        clear Rxy dxy_corr FWHMxy FWHMz;
        p_log(log_file, ' ');

        % === split the image into smaller blocks ===
        % x and y pads are interpolated since they are smaller than z
        % z pad is from actual image to avoid artifacts

        block_path = fullfile(cache_drive, 'block.mat');
        if resume && exist(block_path, 'file')
            p_log(log_file, 'Resuming by loading block info ...')
            loaded = load(block_path);
            block = loaded.block;
            clear loaded;
            % After loading block struct, assert its split is valid for the current stack
            assert(all(size(block.p1) == [block.nx * block.ny * block.nz, 3]), ...
                'block.p1 shape mismatch with block.nx, block.ny, block.nz');
            assert(all(size(block.p2) == [block.nx * block.ny * block.nz, 3]), ...
                'block.p2 shape mismatch with block.nx, block.ny, block.nz');
            fields = fieldnames(stack_info);
            for f = 1:numel(fields)
                if ~isequal(block.stack_info.(fields{f}), stack_info.(fields{f}))
                    error('Loaded block.mat stack_info.%s (%s) does not match current stack_info (%s)', ...
                        fields{f}, mat2str(block.stack_info.(fields{f})), mat2str(stack_info.(fields{f})));
                end
            end
        else
            p_log(log_file, 'partitioning the image into blocks ...')
            output_bytes = 2;
            if convert_to_8bit
                output_bytes = 1;
            elseif stack_info.bit_depth == 8
                output_bytes = 1;
            end

            block.stack_info = stack_info;
            [block.nx, block.ny, block.nz, block.x, block.y, block.z, ...
             block.x_pad, block.y_pad, block.z_pad, block.fft_shape] = ...
                autosplit(stack_info, size(psf.psf), filter, block_size_max, ram_available, numit, output_bytes);

            [block.p1, block.p2] = split_stack(stack_info, block);
            save(block_path, 'block');
        end
        check_block_coverage_planes(stack_info, block);

        p_log(log_file, ['   block numbers: ' num2str(block.nx) 'x * ' num2str(block.ny) 'y * ' num2str(block.nz) 'z = ' num2str(block.nx * block.ny * block.nz) ' blocks.']);
        p_log(log_file, ['   block size loaded image: ' num2str(block.x) 'x * ' num2str(block.y) 'y * ' num2str(block.z) 'z = ' num2str(block.x * block.y * block.z) ' voxels.']);
        p_log(log_file, ['   block size deconvolved image: ' num2str(block.x) 'x * ' num2str(block.y) 'y * ' num2str(block.z - 2*block.z_pad) 'z = ' num2str(block.x * block.y * (block.z - 2*block.z_pad)) ' voxels.']);
        p_log(log_file, ['   block size in ram: ' num2str(block.x+2*block.x_pad) 'x * ' num2str(block.y+2*block.y_pad) 'y * ' num2str(block.z) 'z = ' num2str((block.x+2*block.x_pad) * (block.y+2*block.y_pad) * block.z) ' voxels.']);
        p_log(log_file, ['   padding on x: ' num2str(block.x_pad) ' steps.']);
        p_log(log_file, ['   padding on y: ' num2str(block.y_pad) ' steps.']);
        p_log(log_file, ['   padding on z: ' num2str(block.z_pad) ' steps.']);
        p_log(log_file, ' ');

        % start image processing
        filter.dark = dark(filter, stack_info.bit_depth);
        p_log(log_file, 'preprocessing params ...')
        p_log(log_file, ['   3D gaussian filter sigma: ' num2str(filter.gaussian_sigma, 3)]);
        p_log(log_file, ['   post filter baseline subtraction (denoising): ' num2str(filter.dark)]);
        p_log(log_file, ['   destrip along the z-axis sigma: ' num2str(filter.destripe_sigma)]);
        p_log(log_file, ' ');

        p_log(log_file, 'deconvolution params ...')
        p_log(log_file, ['   max. iterations: ' num2str(numit)]);
        p_log(log_file, ['   regularization interval: ' num2str(filter.regularize_interval)]);
        p_log(log_file, ['   Tikhonov regularization lambda (0 to 1): ' num2str(damping)]);
        p_log(log_file, ['   early stopping criterion (%): ' num2str(stop_criterion)]);
        if filter.use_fft
        p_log(log_file, ['   deconvolve in frequency domain: yes']);
        p_log(log_file, ['   fft shape: ' num2str(block.fft_shape)]);
        else
        p_log(log_file, ['   deconvolve in frequency domain: no']);
        end
        if filter.accelerate
        p_log(log_file, ['   Nesterov/Anderson-style acceleration: yes']);
        else
        p_log(log_file, ['   Nesterov/Anderson-style acceleration: no']);
        end
        if filter.adaptive_psf
        p_log(log_file, ['   adaptive psf using Weiner method: yes']);
        else
        p_log(log_file, ['   adaptive psf using Weiner method: no']);
        end
        p_log(log_file, ['   destripe sigma: ' num2str(filter.destripe_sigma)]);
        p_log(log_file, ' ');
        
        p_log(log_file, 'postprocessing params ...')
        p_log(log_file, ['   histogram clipping percentiles: [lb=' num2str(100 - clipval) ' ub=' num2str(clipval) ']' ]);
        p_log(log_file, ['   signal amplification: ' num2str(amplification) 'x']);
        p_log(log_file, ['   post deconvolution dark subtraction: ' num2str(amplification)]);
        if convert_to_8bit
        p_log(log_file, '   convert to 8-bit: yes');
        else
        p_log(log_file, '   convert to 8-bit: no');
        end
        if convert_to_16bit
        p_log(log_file, '   convert to 16-bit: yes');
        else
        p_log(log_file, '   convert to 16-bit: no');
        end
        p_log(log_file, ' ');

        p_log(log_file, 'paths ...')
        p_log(log_file, ['   source image: ' char(inpath)]);
        p_log(log_file, ['   deconvolved image: ' char(outpath)]);
        p_log(log_file, ['   blocks: ' char(cache_drive)]);
        p_log(log_file, ['   logs: ' char(log_file_path)]);
        p_log(log_file, ' ');

        start_time = datetime('now');
        process(inpath, outpath, log_file, stack_info, block, psf, numit, ...
            damping, clipval, stop_criterion, gpus, cache_drive, ...
            amplification, filter, resume, starting_block);

        p_log(log_file, ['deconvolution finished at ' char(datetime)]);
        p_log(log_file, ['elapsed time: ' char(duration(datetime('now') - start_time, 'Format', 'dd:hh:mm:ss'))]);
        disp('----------------------------------------------------------------------------------------');
        delete(gcp('nocreate'));
        try
            status = rmdir(cache_drive, 's');
            if status
                p_log(log_file, sprintf('[Cache folder deleting]: succeeded for "%s"!', cache_drive));
            else
                p_log(log_file, sprintf('[Cache folder deleting]: failed for "%s"!', cache_drive));
            end
        catch ME
            p_log(log_file, sprintf('[Cache folder deleting]: failed for "%s": %s', cache_drive, ME.message));
        end
        fclose(log_file);
    catch ME
        % error handling
        disp(ME);
        if exist('log_file', 'var')
            p_log(log_file, getReport(ME, 'extended', 'hyperlinks', 'off'));
        end
        exit(1)
    end
end

function [nx,ny,nz,x,y,z,x_pad,y_pad,z_pad,fft_shape] = autosplit( ...
    stack_info, psf_size, filter, block_size_max, ram_available, numit, output_bytes)
%AUTOSPLIT  Choose an (x,y,z) block size that fits in RAM and ≤ 2 147 483 647 elements.
%
%   The function searches from the largest feasible block downwards and
%   returns the best‐scoring candidate (largest core volume).  Two run-time
%   assertions at the end double-check that the chosen block really obeys
%   MATLAB’s hard array-size limit and the user-supplied block_size_max.

    % =====  CONFIGURABLE CONSTANTS  =====================================
    physCores          = feature('numCores');
    socketCount        = get_num_cpu_sockets();
    ram_fraction       = 0.5;        % ≤ 50 % of per-socket RAM
    ram_reserved       = ram_available * ram_fraction / socketCount;
    bl_bytes           = 4;          % single-precision
    max_total_elements = 2^31 - 1;   % MATLAB hard cap (≈2.147 billion)
    block_size_max = min(block_size_max, max_total_elements);
    use_fft            = filter.use_fft;

    % =====  INPUT-DERIVED CONSTANTS  ====================================
    x_dim = stack_info.x;  y_dim = stack_info.y;  z_dim = stack_info.z;
    slice_pixels = x_dim * y_dim;
    z_max_ram = floor(ram_reserved / (output_bytes * slice_pixels));
    z_max     = min(z_max_ram, z_dim);

    xy_min = max(min([psf_size(1)*2, x_dim, y_dim]), 1);
    z_min  = max(min(psf_size(3)*2, z_dim), 1);

    % =====  PAD SIZES  ===================================================
    pad = [0 0 0];
    if filter.destripe_sigma > 0, pad = [1 1 1]; end
    if numit > 0, pad = max(pad, decon_pad_size(psf_size)); end
    unwanted_pad = 0; % reshape(gaussian_pad_size(filter.gaussian_sigma), 1, []);

    % =====  SEARCH  ======================================================
    best = [];              best_score   = -Inf;
    fail_count = 0;         mem_core_mult = physCores * bl_bytes;

    for zz = z_max:-1:z_min
        if zz > z_dim, continue; end

        xy_max = min([floor(sqrt(floor(block_size_max/zz))), x_dim, y_dim]);
        if xy_max < xy_min, continue; end

        % memory that scales only with z (constant inside inner loop)
        slice_mem  = output_bytes * slice_pixels * zz;

        for xy = xy_max:-1:xy_min
            block_core  = [xy xy zz];
            block_shape = block_core + 2*pad;
            if use_fft
                block_shape = next_fast_len(block_shape);
            end

            % ---------- guards -------------------------------------------
            if  any(block_shape                   > [x_dim y_dim z_dim]), continue; end
            if prod(block_shape)                  > block_size_max      , continue; end
            if prod(block_shape + 2*unwanted_pad) > max_total_elements  , continue; end

            % workspace memory (per-core) is proportional to block_core volume
            mem_needed = slice_mem + prod(block_core) * mem_core_mult;
            if mem_needed > ram_available, continue; end
            % --------------------------------------------------------------

            score = prod(block_core);            % favour larger core vols
            if score > best_score
                best        = struct('core',block_core, 'pad',pad, ...
                                     'fft_shape',block_shape);
                best_score  = score;
                fail_count  = 0;
            else
                if ~isempty(best)
                    fail_count = fail_count + 1;
                    if fail_count > 10, break; end
                end
            end
        end
    end

    if isempty(best)
        error(['autosplit: No block shape fits in memory. ', ...
               'Increase block_size_max or reduce psf_size.']);
    end

    % =====  RETURN VALUES  ==============================================
    [x ,y ,z ]           = deal(best.core (1), best.core (2), best.core (3));
    [x_pad,y_pad,z_pad]  = deal(best.pad  (1), best.pad  (2), best.pad  (3));
    fft_shape            = best.fft_shape;

    % ==================  FINAL DEFENSIVE ASSERTS  ========================
    total_shape = [x y z] + 2*[x_pad y_pad z_pad] + 2*unwanted_pad;     % FIX #3
    assert(prod(total_shape(:)) <= max_total_elements, ...              % FIX #4
        'autosplit: internal bug — selected block exceeds MATLAB array limit');

    if use_fft
        assert(prod(fft_shape(:)) <= block_size_max, ...
            'autosplit: internal bug — FFT shape exceeds block_size_max');
    end

    nx = ceil(x_dim / x);
    ny = ceil(y_dim / y);
    nz = ceil(z_dim / z);
end


% function pad_size = gaussian_pad_size(image_size, sigma, kernel)
%     % Accepts sigma (scalar or vector), computes pad_size for each dimension.
%     if isscalar(sigma)
%         sigma = repmat(sigma, size(image_size));
%     end
%     % Kernel size: covers ~99.7% of Gaussian (3 sigma each side)
%     ksize = 2 * ceil(3 * sigma(:)) + 1;
%     pad_size = floor(ksize / 2);    % Column vector
%     if numel(pad_size) ~= numel(image_size)
%         pad_size = [pad_size; zeros(numel(image_size) - numel(pad_size), 1)];
%     end
%     pad_size = ceil(max(pad_size(:).', kernel(:).'));
% end

function pad_size = gaussian_pad_size(sigma)
    % Accepts sigma (scalar or vector), computes pad_size for each dimension.
    if isscalar(sigma)
        sigma = repmat(sigma, 3);
    end
    % Kernel size: covers ~99.7% of Gaussian (3 sigma each side)
    ksize = 2 * ceil(3 * sigma(:)) + 1;
    pad_size = floor(ksize / 2);    % Column vector
end

function pad = decon_pad_size(psf_sz)
    pad = ceil(psf_sz(:).' * 1.5);
end

function n_vec = next_fast_len(n_vec)
    % Accepts a scalar or a vector of positive integers and returns, for each,
    % the next integer >= n_in(i) whose factors are all <= 7.
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

function check_block_coverage_planes(stack_info, block)
    % disp('Checking block coverage for errors ...');

    p1 = block.p1;
    p2 = block.p2;
    errors = {};

    % 1. Block boundary checks
    for k = 1:size(p1,1)
        if any(p1(k,:) < 1) || any(p2(k,:) > [stack_info.x, stack_info.y, stack_info.z])
            errors{end+1} = sprintf('Block %d out of bounds: p1=%s, p2=%s', ...
                k, mat2str(p1(k,:)), mat2str(p2(k,:)));
        end
        if any(p1(k,:) > p2(k,:))
            errors{end+1} = sprintf('Block %d has p1 > p2: p1=%s, p2=%s', ...
                k, mat2str(p1(k,:)), mat2str(p2(k,:)));
        end
    end

    % 2. Z-plane coverage
    z_cover = zeros(1, stack_info.z);
    for k = 1:size(p1,1)
        z_range = p1(k,3):p2(k,3);
        z_cover(z_range) = z_cover(z_range) + 1;
    end
    missing_z = find(z_cover == 0);
    if ~isempty(missing_z)
        errors{end+1} = sprintf('Missing Z planes: %s', mat2str(missing_z));
    end
    if numel(find(z_cover > 0)) ~= stack_info.z
        errors{end+1} = sprintf('Unique covered Z planes = %d, expected %d', ...
            numel(find(z_cover > 0)), stack_info.z);
    end
    if max(p2(:,3)) > stack_info.z
        errors{end+1} = sprintf('Blocks extend past last Z-plane! Max p2(:,3)=%d, stack_info.z=%d', ...
            max(p2(:,3)), stack_info.z);
    end
    if min(p1(:,3)) < 1
        errors{end+1} = sprintf('Blocks start before Z=1! Min p1(:,3)=%d', min(p1(:,3)));
    end

    % 3. XY coverage at first and last z planes
    for z = [1, stack_info.z]
        covered = zeros(stack_info.x, stack_info.y);
        blocks = find(p1(:,3) <= z & p2(:,3) >= z);
        for k = blocks.'
            xs = p1(k,1):p2(k,1);
            ys = p1(k,2):p2(k,2);
            covered(xs, ys) = covered(xs, ys) + 1;
        end
        num_gaps = sum(covered(:) == 0);
        num_overlaps = sum(covered(:) > 1); % This counts pixels covered by more than 1 block (not face/edge)
        %disp(['   Blocks covering z=' num2str(z) ': ' num2str(numel(blocks))]);
        %disp(['   XY at z=' num2str(z) ': gaps=' num2str(num_gaps) ', overlaps=' num2str(num_overlaps)]);
        if num_gaps > 0
            errors{end+1} = sprintf('XY at z=%d: %d gaps', z, num_gaps);
        end
        % Only error if overlaps are >0 *AND* not just at the faces
        % But for most tiling schemes, face/edge/corner overlaps are fine and expected,
        % so we don't report overlaps here unless you want to.
    end

    % 4. XZ coverage at first and last y planes
    for y = [1, stack_info.y]
        covered = zeros(stack_info.x, stack_info.z);
        blocks = find(p1(:,2) <= y & p2(:,2) >= y);
        for k = blocks.'
            xs = p1(k,1):p2(k,1);
            zs = p1(k,3):p2(k,3);
            covered(xs, zs) = covered(xs, zs) + 1;
        end
        num_gaps = sum(covered(:) == 0);
        num_overlaps = sum(covered(:) > 1);
        %disp(['   Blocks covering y=' num2str(y) ': ' num2str(numel(blocks))]);
        %disp(['   XZ at y=' num2str(y) ': gaps=' num2str(num_gaps) ', overlaps=' num2str(num_overlaps)]);
        if num_gaps > 0
            errors{end+1} = sprintf('XZ at y=%d: %d gaps', y, num_gaps);
        end
    end

    % 5. YZ coverage at first and last x planes
    for x = [1, stack_info.x]
        covered = zeros(stack_info.y, stack_info.z);
        blocks = find(p1(:,1) <= x & p2(:,1) >= x);
        for k = blocks.'
            ys = p1(k,2):p2(k,2);
            zs = p1(k,3):p2(k,3);
            covered(ys, zs) = covered(ys, zs) + 1;
        end
        num_gaps = sum(covered(:) == 0);
        num_overlaps = sum(covered(:) > 1);
        %disp(['   Blocks covering x=' num2str(x) ': ' num2str(numel(blocks))]);
        %disp(['   YZ at x=' num2str(x) ': gaps=' num2str(num_gaps) ', overlaps=' num2str(num_overlaps)]);
        if num_gaps > 0
            errors{end+1} = sprintf('YZ at x=%d: %d gaps', x, num_gaps);
        end
    end

    % 6. True 3D interior-overlap check (ignore face/edge/corner)
    %disp('   Checking for true 3D interior overlaps (ignoring faces/edges/corners)...');
    N = size(p1, 1);
    true_overlaps = [];
    for i = 1:N-1
        p1_i = p1(i, :); p2_i = p2(i, :);
        for j = i+1:N
            p1_j = p1(j, :); p2_j = p2(j, :);
            overlap_x = min(p2_i(1), p2_j(1)) - max(p1_i(1), p1_j(1)) + 1;
            overlap_y = min(p2_i(2), p2_j(2)) - max(p1_i(2), p1_j(2)) + 1;
            overlap_z = min(p2_i(3), p2_j(3)) - max(p1_i(3), p1_j(3)) + 1;
            % All axes must have overlap > 1 (true interior)
            if overlap_x > 1 && overlap_y > 1 && overlap_z > 1
                true_overlaps = [true_overlaps; i, j];
            end
        end
    end
    if isempty(true_overlaps)
        %disp('   No true 3D interior overlaps found between blocks.');
    else
        %disp('   True 3D interior overlaps found between blocks:');
        disp(true_overlaps);
        errors{end+1} = sprintf('Found %d pairs of blocks with true interior overlap.', size(true_overlaps, 1));
    end

    % 7. Warn if any block is much smaller than nominal
    % nominal_block_size = [block.x, block.y, block.z];
    % actual_sizes = block.p2 - block.p1 + 1;
    % axis_labels = 'XYZ';
    % for i = 1:size(actual_sizes, 1)
    %     too_small = actual_sizes(i,:) < 0.5 * nominal_block_size;
    %     if any(too_small)
    %         % MATLAB R2019b+: strjoin(string)
    %         axstr = '';
    %         for dim = 1:3
    %             if too_small(dim)
    %                 axstr = [axstr, axis_labels(dim)];
    %             end
    %         end
    %         if isempty(axstr)
    %             axstr = '-';
    %         end
    %         fprintf('Warning: Block %d is small in axis %s. Size: [%s], Expected: [%s]\n', ...
    %             i, axstr, num2str(actual_sizes(i,:)), num2str(nominal_block_size));
    %     end
    % end

    % Final report
    if ~isempty(errors)
        err_msg = sprintf('Block coverage error(s) detected:\n%s', strjoin(errors, '\n'));
        error(err_msg);
    else
        disp('   Block coverage test: PASSED');
    end
end

function init_min_max_file(min_max_path, bit_depth)
    if isfile(min_max_path)
        % Do NOT clobber on resume or if a previous run left it intact.
        return
    end
    switch bit_depth
        case 8
            rawmax = 255;
        case 16
            rawmax = 65535;
        otherwise
            rawmax = -inf; % will be promoted by workers on first block
    end
    deconvmin = inf;
    deconvmax = -inf;

    tmpfile = [min_max_path, '.tmp.mat'];
    save(tmpfile, "deconvmin", "deconvmax", "rawmax", "-v7.3", "-nocompression");
    if isfile(min_max_path), delete(min_max_path); end
    [ok,msg] = movefile(tmpfile, min_max_path, 'f');
    if ~ok, error("init_min_max_file: move failed: %s", msg); end
end

function process(inpath, outpath, log_file, stack_info, block, psf, numit, ...
    damping, clipval, stop_criterion, gpus, cache_drive, amplification, ...
    filter, resume, starting_block)

    need_post_processing = false;
    if starting_block == 1
        need_post_processing = true;
    end

    % load filelist
    filelist = dir(fullfile(inpath, '*.tif'));
    if numel(filelist) == 0
        filelist = dir(fullfile(inpath, '*.tiff'));
    end
    filelist = natsortfiles(filelist);
    filelist = fullfile(char(inpath), {filelist.name});

    % intermediate variables needed for interprocess communication
    % NOTE: semaphore keys should be a more than zero values.
    % all the existing processes should be killed first before creating a
    % sechamore
    myCluster = parcluster('Processes');
    delete(myCluster.Jobs);
    delete(gcp("nocreate"));
    min_max_path = fullfile(char(cache_drive), 'min_max.mat');
    init_min_max_file(min_max_path, stack_info.bit_depth);
    [unique_gpus, ~, ~] = unique(gpus(:));
    unique_gpus = sort(unique_gpus, 'descend').';
    gpus = repmat(unique_gpus, 1, numel(gpus)/numel(unique_gpus));
    % flatten the gpus array
    gpus = gpus(:)';
    
    % initiate locks and semaphors
    % semkeys are arbitrary non-zero values
    semkey_single = 1e3;
    % semkey_loading_base = 1e4;
    semaphore_create(semkey_single, 1);
    semkey_gpu_base = 1e5;
    % queue('create', semkey_gpu_base, unique_gpus);
    for gpu = unique_gpus
        semaphore_create(semkey_gpu_base + gpu, 1);
        % semaphore_create(gpu + semkey_loading_base, 3);
    end

    % start deconvolution
    num_blocks = block.nx * block.ny * block.nz;
    remaining_blocks = num_blocks;
    while remaining_blocks > 0
        for i = starting_block : num_blocks
            % skip blocks already worked on
            block_path = fullfile(cache_drive, ['bl_' num2str(i) '.lz4']);
            block_path_tmp = fullfile(cache_drive, ['bl_' num2str(i) '.lz4.tmp']);
            if exist(block_path, "file")
                if dir(block_path).bytes > 0
                    remaining_blocks = remaining_blocks - 1;
                else
                    delete(block_path);
                end
            end
            if exist(block_path_tmp, "file")
                try
                    delete(block_path_tmp);
                catch
                    warning([block_path_tmp ' file could not be deleted!']);
                end
            end
        end
        if remaining_blocks > 0
            gpus = gpus(1:min(numel(gpus), remaining_blocks));
            pool = parpool('local', numel(gpus), 'IdleTimeout', Inf);
            dQueue = parallel.pool.DataQueue;
            afterEach(dQueue, @disp);
            deconvolution_start_time = tic;
            parfor idx = 1 : numel(gpus)

                deconvolve( ...
                    filelist, psf, numit, damping, ...
                    block, stack_info, min_max_path, clipval, ...
                    stop_criterion, gpus(idx), semkey_gpu_base, ...
                    cache_drive, filter, starting_block + idx - 1, dQueue);
            end
            p_log(log_file, sprintf('deconvolution time: %s', char(duration(seconds(toc(deconvolution_start_time)), 'Format', 'dd:hh:mm:ss'))));
            delete(pool);
        end
        starting_block = 1;
    end

    % clear locks and semaphors
    semaphore_destroy(semkey_single);
    for gpu = unique_gpus
        semaphore_destroy(semkey_gpu_base + gpu);
    end

    % postprocess and write tif files
    if need_post_processing
        postprocess_save(outpath, cache_drive, min_max_path, clipval, log_file, stack_info, resume, block, amplification);
    end
end

function deconvolve(filelist, psf, numit, damping, ...
    block, stack_info, min_max_path, clipval, ...
    stop_criterion, gpu, semkey_gpu_base, cache_drive, ...
    filter, starting_block, dQueue)
    
    semkey_single = 1e3;
    % semkey_loading_base = 1e4;
    % semkey_loading = semkey_loading_base + gpu;

    if stack_info.bit_depth == 8
        rawmax = 255;
    elseif stack_info.bit_depth == 16
        rawmax = 65535;
    else
        rawmax = -Inf;
    end

    deconvmax = 0; deconvmin = Inf;
    x = 1; y = 2; z = 3;

    num_blocks = block.nx * block.ny * block.nz;
    num_blocks_str = num2str(num_blocks);
    % loading_time_moving_average = 0;

    for blnr = starting_block : num_blocks

        block_processing_start = tic;

        % skip blocks already worked on
        block_path = fullfile(cache_drive, ['bl_' num2str(blnr) '.lz4']);
        block_path_tmp = fullfile(cache_drive, ['bl_' num2str(blnr) '.lz4.tmp']);
        semaphore('wait', semkey_single);
        if num_blocks > 1 && (exist(block_path, "file") || exist(block_path_tmp, "file"))
            semaphore('post', semkey_single);
            continue
        end
        fclose(fopen(block_path, 'w'));
        semaphore('post', semkey_single);

        % begin processing next block and load next block of data into memory
        startp = block.p1(blnr, :); endp = block.p2(blnr, :);
        x1     = startp(x);           x2 = endp(x);
        y1     = startp(y);           y2 = endp(y);
        z1     = startp(z);           z2 = endp(z);

        % decon_step_is_slower_than_loading = numit > 1; % then load slower to prevent RAM overload
        % if decon_step_is_slower_than_loading, semaphore('wait', semkey_loading); end
        % send(dQueue, [current_device(gpu) ': block ' num2str(blnr) ' from ' num_blocks_str ' is loading ...']);
        % loading_time = tic;
        bl = load_block(filelist, x1, x2, y1, y2, z1, z2, block, stack_info);
        % loading_time_moving_average = loading_time_moving_average * 0.9 + toc(loading_time) * 0.1;
        % send(dQueue, [current_device(gpu) ': block ' num2str(blnr) ' from ' num_blocks_str ' is loaded in ' num2str(round(loading_time_moving_average, 1))]);
        % if decon_step_is_slower_than_loading, semaphore('post', semkey_loading); end

        % get min-max of raw data stack
        if ~ismember(stack_info.bit_depth, [8, 16])
            rawmax_start = tic;
            rawmax = max(max(bl(:)), rawmax);
            send(dQueue, [current_device(gpu) ': block ' num2str(blnr) ' from ' num_blocks_str ' rawmax calculated in ' num2str(round(toc(rawmax_start), 1))]);
        end

        % deconvolve current block of data
        if exist(block_path_tmp, "file") || (exist(block_path, "file") && dir(block_path).bytes > 0)
            continue
        end

        expected_size = size(bl);  % Store size before processing
        bl = process_block(bl, block, psf, numit, damping, stop_criterion, filter, gpu, semkey_gpu_base);
        % === Check padded block size is unchanged by process_block ===
        actual_size = size(bl);
        assert(isequal(actual_size, expected_size), ...
            sprintf(['[process_block] Block %d: Output block size mismatch!\n', ...
                     'Expected [%d %d %d], got [%d %d %d].\n', ...
                     'Block X[%d-%d], Y[%d-%d], Z[%d-%d]'], ...
                    blnr, expected_size, actual_size, ...
                    block.p1(blnr,1), block.p2(blnr,1), ...
                    block.p1(blnr,2), block.p2(blnr,2), ...
                    block.p1(blnr,3), block.p2(blnr,3)));

        % === Remove padding before saving ===
        % save_start = tic;
        bl = bl(1 + block.x_pad : end - block.x_pad, ...
                1 + block.y_pad : end - block.y_pad, ...
                1 + block.z_pad : end - block.z_pad);

        % === Check trimmed block matches expected core size ===
        expected_size = [block.p2(blnr,1) - block.p1(blnr,1) + 1, ...
                         block.p2(blnr,2) - block.p1(blnr,2) + 1, ...
                         block.p2(blnr,3) - block.p1(blnr,3) + 1];
        actual_size = size(bl);

        assert(isequal(actual_size, expected_size), ...
            sprintf(['[remove padding] Block %d: Output block size mismatch!\n', ...
                     'Expected [%d %d %d], got [%d %d %d].\n', ...
                     'Block X[%d-%d], Y[%d-%d], Z[%d-%d]'], ...
                    blnr, expected_size, actual_size, ...
                    block.p1(blnr,1), block.p2(blnr,1), ...
                    block.p1(blnr,2), block.p2(blnr,2), ...
                    block.p1(blnr,3), block.p2(blnr,3)));
        [lb, ub] = deconvolved_stats(bl, clipval);
        % find maximum value in other blocks
        semaphore('wait', semkey_single);
        could_not_save = true;
        attempts = 0;
        while could_not_save
            try
                if exist(min_max_path, "file")
                    min_max = load(min_max_path);
                    if isgpuarray(min_max.deconvmax)
                        deconvmax = gather(min_max.deconvmax);
                        deconvmin = gather(min_max.deconvmin);
                        rawmax = gather(min_max.rawmax);
                    else
                        deconvmax = min_max.deconvmax;
                        deconvmin = min_max.deconvmin;
                        rawmax = min_max.rawmax;
                    end
                end
                % consolidate and save block stats
                deconvmax = max(ub, deconvmax);
                deconvmin = min(lb, deconvmin);
                % save(min_max_path, "deconvmin", "deconvmax", "rawmax", "-v7.3", "-nocompression");
                tmpfile = [min_max_path, '.tmp.mat'];
                save(tmpfile, "deconvmin", "deconvmax", "rawmax", "-v7.3", "-nocompression");
                delete(min_max_path);
                [status, msg] = movefile(tmpfile, min_max_path, 'f');  % 'f' forces overwrite
                if ~status
                    error("Failed to move temp file: %s", msg);
                end
                could_not_save = false;
            catch
                send(dQueue, "could not load or save min_max file. Retrying ...")
                attempts = attempts + 1;
                if attempts > 40
                    send(dQueue, "deleting min_max file.")
                    delete(min_max_path);
                else
                    pause(1);
                end
            end
        end
        semaphore('post', semkey_single);

        % save block to disk
        if exist(block_path_tmp, "file") || (exist(block_path, "file") && dir(block_path).bytes > 0)
            continue
        end
        could_not_save = true;
        while could_not_save
            try
                save_lz4(block_path_tmp, bl);
                movefile(block_path_tmp, block_path, 'f');
                % send(dQueue, [current_device(gpu) ': block ' num2str(blnr) ' from ' num_blocks_str ' saved in ' num2str(round(toc(save_start), 1))]);
                could_not_save = false;
            catch ME
                send(dQueue, sprintf('could not save %s! Retrying ... %s: %s', block_path, ME.identifier, ME.message));
                pause(1);
            end
        end
        send(dQueue, sprintf('%s: block %d from %d filters applied in %.1f', current_device(gpu), blnr, num_blocks, toc(block_processing_start)));
    end
end

function bl = load_block(filelist, x1, x2, y1, y2, z1, z2, block, stack_info)
    % Load a padded 3D block from the stack, using only available planes,
    % and pad symmetrically to the target block size

    block_start = [x1, y1, z1];
    block_end   = [x2, y2, z2];
    block_pad   = [block.x_pad, block.y_pad, block.z_pad];
    volume_size = [stack_info.x, stack_info.y, stack_info.z];

    % Request region (may go out of bounds at the edges)
    requested_start = block_start - block_pad;
    requested_end   = block_end   + block_pad;

    % In-bounds region (what we can actually read)
    read_start = max(1, requested_start);
    read_end   = min(volume_size, requested_end);

    % Indices to load for each axis
    x_indices = read_start(1):read_end(1);
    y_indices = read_start(2):read_end(2);
    z_indices = read_start(3):read_end(3);

    % Only allocate space for what we read from disk
    bl = zeros(numel(x_indices), numel(y_indices), numel(z_indices), 'single');

    % -----------------------------
    % Use `load_bl_tif` if available
    % -----------------------------
    use_fast_loader = exist('load_bl_tif', 'file') == 3;  % MEX-file available?

    if use_fast_loader
        try
            % Extract subregion using fast multithreaded native MEX
            subfilelist = filelist(z_indices);
            y0 = y_indices(1);
            x0 = x_indices(1);
            H  = numel(y_indices);
            W  = numel(x_indices);

            % Note: transpose = true to match MATLAB behavior (imread' = X,Y)
            bl = load_bl_tif(subfilelist, y0, x0, H, W, true);
            bl = im2single(bl);  % Ensure consistent type

        catch ME
            warning('[load_block] load_bl_tif failed (%s), falling back to imread.', ME.message);
            use_fast_loader = false;
        end
    end

    % --------------------------------------
    % Fallback: use slow MATLAB imread loop
    % --------------------------------------
    if ~use_fast_loader
        bl = zeros(numel(x_indices), numel(y_indices), numel(z_indices), 'single');
        for k = 1:numel(z_indices)
            slice_idx = z_indices(k);
            slice = imread(filelist{slice_idx}, ...
                'PixelRegion', {[y_indices(1), y_indices(end)], [x_indices(1), x_indices(end)]});
            bl(:, :, k) = im2single(slice)';  % Transpose to [X,Y]
        end
    end

    % ------------------------
    % Symmetric edge padding
    % ------------------------
    % How much to pad before and after in each axis
    pad_before = read_start - requested_start;
    pad_after  = requested_end - read_end;

    % Final block size after all padding
    block_core_size = block_end - block_start + 1;
    block_target_size = block_core_size + 2 * block_pad;

    % Pad as needed to reach the requested size
    if any(pad_before > 0)
        bl = padarray(bl, pad_before, 'symmetric', 'pre');
    end
    if any(pad_after > 0)
        bl = padarray(bl, pad_after, 'symmetric', 'post');
    end

    % ------------------------
    % Final consistency check
    % ------------------------
    assert(isequal(size(bl), block_target_size), ...
        sprintf('[load_block] Output size mismatch! Got [%s], expected [%s]', ...
        num2str(size(bl)), num2str(block_target_size)));
end

function bl = process_block(bl, block, psf, niter, lambda, stop_criterion, filter, gpu, semkey_gpu_base)
    bl_size = size(bl);
    if gpu && (min(filter.gaussian_sigma(:)) > 0 || niter > 0 || filter.destripe_sigma > 0 || min(filter.fibermetric_sigma(:)) > 0)
        % get the next available gpu
        % gpu = queue('wait', semkey_gpu_base);
        semaphore('w', semkey_gpu_base + gpu);
        gpu_device = gpuDevice(gpu);
        bl = gpuArray(bl);
    end

    if any(filter.gaussian_sigma > 0)
        if gpu
            bl = gauss3d_gpu(bl, filter.gaussian_sigma);
        else
            bl = imgaussfilt3(bl, filter.gaussian_sigma, 'Padding', 'symmetric', 'FilterDomain', 'spatial');
        end
        if filter.dark > 0
            bl = bl - filter.dark;
            bl = max(bl, 0);
        end
    end

    if niter > 0 && max(bl(:)) > eps('single')
        % deconvolve block using Lucy-Richardson or blind algorithm
        bl = decon(bl, psf, niter, lambda, stop_criterion, filter, block.fft_shape);
    end

    if filter.destripe_sigma > 0
        bl = filter_subband_3d_z(bl, filter.destripe_sigma, 0, "db9");
    end

    bl = apply_fibermetric_filter(bl, filter);

    if gpu && isgpuarray(bl)
        % Reseting the GPU
        bl = gather(bl);
        reset(gpu_device);  % to free 2 extra copies of bl in gpu
        semaphore('p', semkey_gpu_base + gpu);
    end

    assert(all(size(bl) == bl_size), '[process_block]: block size mismatch!');
end

function postprocess_save(outpath, cache_drive, min_max_path, clipval, log_file, stack_info, resume, block, amplification)
    %POSTPROCESS_SAVE   Final stage: re-assembles cached LZ4 blocks into TIFFs.
    %
    % For each Z-slab (one full X-Y plane stack), all bricks are loaded and
    % decompressed **in C++ threads** via load_slab_lz4_save_as_tif (zero MATLAB-side
    % parallel plumbing). Only one slab lives in RAM at a time.
    %
    % INPUT ARGUMENTS
    %   outpath        – destination folder for img_######.tif files
    %   cache_drive    – folder containing the *.lz4 brick cache
    %   min_max_path   – path to min_max.mat produced earlier (optional)
    %   log_file       – file handle or path for p_log()
    %   stack_info     – struct with fields: x, y, z, convert_to_8bit, convert_to_16bit
    %   resume         – logical, continue an interrupted run
    %   block          – struct with p1, p2, nx, ny, nz describing bricks
    %   amplification  – display-style gain factor applied after clipping

    % 1. Build & sanity-check brick file list
    numBlocks  = size(block.p1, 1);
    blocklist  = cellstr(fullfile(cache_drive, compose("bl_%d.lz4", (1:numBlocks).')));
    missingMask = ~cellfun(@isfile, blocklist);

    if any(missingMask)
        p_log(log_file, sprintf(2,'ERROR: %d block files are missing – aborting.', nnz(missingMask)));
        disp(find(missingMask));
        error('postprocess_save:MissingBlocks','Block cache incomplete.');
    end

    % Axis shorthand for readability
    x = 1; y = 2; z = 3;

    % 2. Load min / max statistics for rescaling
    if isfile(min_max_path)
        S         = load(min_max_path);
        deconvmin = S.deconvmin;
        deconvmax = S.deconvmax;
        rawmax    = S.rawmax;
    else
        warning('min_max.mat not found – using defaults.');
        deconvmin = 0;
        deconvmax = 5.3374;
        rawmax    = 65535;
    end

    % Override for final datatype
    if stack_info.convert_to_8bit
        rawmax = 255;
    elseif stack_info.convert_to_16bit
        rawmax = 65535;
    end

    % Target scale
    if stack_info.convert_to_8bit || rawmax <= 255
        scal = 255;
    elseif stack_info.convert_to_16bit || rawmax <= 65535
        scal = 65535;
    else
        scal = rawmax;
    end

    p_log(log_file,'image stats …');
    p_log(log_file,sprintf('   target data type max value: %g', scal));
    p_log(log_file,sprintf('   %.2f%% max before deconv  : %g', clipval, rawmax));
    p_log(log_file,sprintf('   %.2f%% max after  deconv  : %g', clipval, deconvmax));
    p_log(log_file,sprintf('   %.2f%% min after  deconv  : %g\n', 100 - clipval, deconvmin));

    % 3. Detect already-written TIFFs (resume mode)
    blocksPerSlab = block.nx * block.ny;
    if resume
        files = dir(fullfile(outpath,'img_*.tif'));
    else
        files = [];
    end

    if ~isempty(files)
        nums          = cellfun(@(s) sscanf(s,'img_%6d.tif'), {files.name});
        lastDone      = max(nums);
        nextFileIdx   = lastDone + 1;
        starting_z_block = find(block.p2(1:blocksPerSlab:end, z) >= nextFileIdx, 1);
        p_log(log_file, sprintf('Resuming: last slice = %d ➜ continue at %d (slab %d)', ...
                                lastDone, nextFileIdx, starting_z_block));
    else
        nextFileIdx      = 1;
        starting_z_block = 1;
    end

    % 4. Re-assemble and save Z-slabs as 2D TIFs (calls all-in-one MEX)
    for nz = starting_z_block : block.nz
        p_log(log_file, sprintf('Slab %d / %d – mounting %d blocks …', nz, block.nz, blocksPerSlab));
        block_inds = ((nz-1)*blocksPerSlab + 1) : (nz*blocksPerSlab);

        if ~all(cellfun(@isfile, blocklist(block_inds)))
            error('postprocess_save:MissingDuringResume', ...
                  'Block files vanished during run (slab %d).', nz);
        end

        % Z-range and slab depth
        slab_z1    = block.p1(block_inds(1), z);
        slab_z2    = block.p2(block_inds(1), z);
        slab_depth = slab_z2 - slab_z1 + 1;

        % 1-based brick coordinates **inside the slab**
        p1_slab = uint64([ ...
            block.p1(block_inds,x), ...
            block.p1(block_inds,y), ...
            block.p1(block_inds,z) - slab_z1 + 1 ]);

        p2_slab = uint64([ ...
            block.p2(block_inds,x), ...
            block.p2(block_inds,y), ...
            block.p2(block_inds,z) - slab_z1 + 1 ]);

        slabSize = uint64([ stack_info.x, stack_info.y, slab_depth ]);

        % Prepare filenames for this slab
        file_z1 = nextFileIdx;
        indices = file_z1 + (0:slab_depth-1);
        fileNames = compose("img_%06d.tif", indices);
        fileList = cellstr(fullfile(outpath, fileNames));
        existing = cellfun(@(f) exist(f, 'file'), fileList);

        % Skip if all slices exist
        if all(existing)
            fprintf('   All %d slices already exist in %s\n', slab_depth, outpath);
            nextFileIdx = nextFileIdx + slab_depth;
            continue;
        end

        % Save using all-in-one C++ MEX (no MATLAB fallback, always in XYZ order, always uses .tif)
        % slicesToSave = find(~existing);
        fileListToSave = fileList(~existing);

        % Main call: load, rescale, save slab as TIFFs directly
        % Note: The new MEX must match this signature and perform everything inside C++!
        % You may need to pass more flags (e.g., isXYZLayout, useTiles, compression, nThreads).
        compression = 'deflate'; % or expose as input if you want
        useTiles    = false;     % or expose as input
        nThreads    = feature('numCores');
        isXYZLayout = true;      % shoudl be in sync with load_bl_tif call; isXYZLayout path is much faster

        % Forward all parameters to C++: blocklist, fileListToSave, p1_slab, p2_slab, slabSize, etc.
        % All logic (rescaling, save, datatype, amplification, min/max) is handled inside C++
        load_slab_lz4_save_as_tif( ...
            blocklist(block_inds), ...     % source .lz4 brick files (cellstr)
            fileListToSave, ...            % destination .tif files (cellstr)
            p1_slab, p2_slab, slabSize, ...
            scal, amplification, deconvmin, deconvmax, ...
            compression, useTiles, nThreads, isXYZLayout);

        nextFileIdx = nextFileIdx + slab_depth;
    end

    p_log(log_file, 'postprocess_save completed successfully.');
end

function semaphore_destroy(semkey)
    try
        semaphore('destroy', semkey);
    catch
    end
end

function semaphore_create(semkey, value)
    semaphore_destroy(semkey);
    semaphore('c', semkey, value);
    % disp(['semaphore ' num2str(semkey) ' is created with the initial value of ' num2str(value)]);
end

function [x, y, z, bit_depth] = getstackinfo(datadir)
    filelist = dir(fullfile(datadir, '*.tif'));
    if numel(filelist) == 0
        filelist = dir(fullfile(datadir, '*.tiff'));
    end

    if numel(filelist) == 0
        x = 0; y = 0; z = 0; bit_depth=0;
        disp("empty list of tif files!");
    else
        the_first_file = fullfile(datadir, filelist(1).name);
        test = imread(the_first_file);
        x = size(test, 2);
        y = size(test, 1);
        z = numel(filelist);
        bit_depth = imfinfo(the_first_file).BitDepth;
    end
end

function [ram_available, ram_total]  = get_memory()
    if ispc
        [~, m] = memory;  % check if os supports memory function
        ram_available = m.PhysicalMemory.Available; % returns free memory not physical memory in bytes
        ram_total = m.PhysicalMemory.Total;
    else
        [~, w] = unix('free -b | grep Mem');
        stats = str2double(regexp(w, '[0-9]*', 'match'));
        ram_total = stats(1);
        ram_available = stats(end);
    end
end

function num_cpu_sockets = get_num_cpu_sockets()
    if ispc
        [~, str] = dos('wmic cpu get SocketDesignation');
        num_cpu_sockets = count(str,'CPU');
        if num_cpu_sockets < 1
            num_cpu_sockets = 1;
        end
    else
        [~, num_cpu_sockets_str] = unix("grep 'physical id' /proc/cpuinfo | sort -u | wc -l");
        num_cpu_sockets = str2double(regexp(num_cpu_sockets_str, '[0-9]*', 'match'));
    end
end

function showinfo()
    disp('Usage: LsDeconv TIFDIR DELTAXY DELTAZ nITER NA RI LAMBDA_EX LAMBDA_EM FCYL SLITWIDTH DAMPING HISOCLIP STOP_CRIT BLOCK_SIZE_MAX GPU AMPLIFICATION GAUSSIAN_SIGMA DARK DESTRIPE_SIGMA FIBERMETRIC_SIGMA REGULARIZE_INTERVAL RESUME STARTING_BLOCK CONVERT_TO_8BIT CONVERT_TO_16BIT USE_FFT ADAPTIVE_PSF ACCELERATE CACHE_DRIVE');
    disp(' ');
    disp('TIFFDIR: Directory of 2D TIFF files (8/16/32-bit grayscale). Numerical filenames like xx00001.tif, xx00002.tif, ...');
    disp('DELTAXY DELTAZ: Voxel size in nanometers [XY Z]. E.g., 250 500 means 250 nm XY, 500 nm Z.');
    disp('nITER: Maximum number of Richardson-Lucy iterations.');
    disp('NA: Numerical aperture of objective.');
    disp('RI: Refractive index of imaging medium/sample.');
    disp('LAMBDA_EX / LAMBDA_EM: Excitation / emission wavelength in nm.');
    disp('FCYL: Focal length (mm) of cylinder lens for light sheet.');
    disp('SLITWIDTH: Slit width (mm) before cylinder lens (NaLs = sin(arctan(w/(2*f))).');
    disp('DAMPING: Regularization/damping factor [0-10%]. Higher for noisy data.');
    disp('HISOCLIP: Histogram clip % [0-5]. 0 disables, 0.01 clips 0.01 and 99.99 percentile.');
    disp('STOP_CRIT: Stopping criterion as percent change per iteration.');
    disp('BLOCK_SIZE_MAX: Maximum block size for splitting large data (in elements).');
    disp('GPU: List of GPU IDs to use (0 for CPU only, or e.g. [0 1 2]).');
    disp('AMPLIFICATION: Additional signal scaling factor.');
    disp('GAUSSIAN_SIGMA: 3D Gaussian filter sigma (pixels).');
    disp('DARK: 1 to invert input image for dark objects, 0 for bright.');
    disp('DESTRIPE_SIGMA: Destripe filter sigma (pixels) [0 disables].');
    disp('FIBERMETRIC_SIGMA: [start step end] for vesselness filter (0 disables).');
    disp('REGULARIZE_INTERVAL: Number of iterations between regularization steps.');
    disp('RESUME: 1 to resume from existing cache, 0 to start fresh.');
    disp('STARTING_BLOCK: Index to start block-wise deconvolution from.');
    disp('CONVERT_TO_8BIT / CONVERT_TO_16BIT: Convert output to 8-bit/16-bit (logical).');
    disp('USE_FFT: 1 to use FFT-based convolution, 0 for direct.');
    disp('ADAPTIVE_PSF: 1 to enable adaptive PSF computation.');
    disp('ACCELERATE: 1 to enable acceleration mode (approx. RL or Nesterov).');
    disp('CACHE_DRIVE: Path to cache directory for block outputs/temp files.');
    disp(' ');
    disp('Example:');
    disp('  LsDeconv ./images 250 500 20 1.0 1.333 488 520 50 1.0 0 0.01 2 500000000 0 1.0 1 0 0 0 [1 1 1] 0 1 0 1 0 0 ./cache');
    disp(' ');
    disp('Notes:');
    disp('- Supports block-based, multi-GPU, resumable deconvolution with advanced options.');
    disp('- Cache directory is created automatically if it does not exist.');
    disp('- Input images must be grayscale, single-channel, and ordered numerically.');
end

function p_log(log_file, message)
    disp(message);
    fprintf(log_file, '%s\r\n', message);
end

function baseline_subtraction = dark(filter, bit_depth)
    baseline_subtraction = 0;
    sigma = double(filter.gaussian_sigma);
    % Ensure sigma is a vector of length 3
    if isscalar(sigma), sigma = repmat(sigma,1,3); end
    if any(sigma > 0)
        kernel_size = 2 * ceil(3 * sigma) + 1;
        % Create the zeros array with the correct type
        if bit_depth == 8
            a = zeros(kernel_size, 'uint8');
        elseif bit_depth == 16
            a = zeros(kernel_size, 'uint16');
        else
            warning('unsupported image bit depth');
            a = zeros(kernel_size);
        end
        % Center index
        kernel_center = ceil(kernel_size/2);
        % Set center voxel to filter.dark
        a(kernel_center(1), kernel_center(2), kernel_center(3)) = filter.dark;
        a = im2single(a);
        a = imgaussfilt3(a, sigma, 'Padding', 'symmetric', 'FilterDomain', 'spatial');
        baseline_subtraction = max(a(:));
    end
end

function [lb, ub] = deconvolved_stats(bl, clipval)
    stats = single(fast_twin_tail_orderstat(bl, [(100 - clipval) clipval]));
    lb = stats(1);
    ub = stats(2);
end

function bl = apply_fibermetric_filter(bl, filter)
    sigma = filter.fibermetric_sigma;
    if all(sigma > 0) && sigma(2) > 0 && sigma(1) <= sigma(3)
        sigma_range = sigma(1):sigma(2):sigma(3);
        if isempty(sigma_range)
            warning('Fibermetric sigma range is empty. No filtering applied.');
        else
            bl = fibermetric_gpu(bl, sigma(1), sigma(3), sigma(2), ...
                                 filter.fibermetric_alpha, filter.fibermetric_beta, filter.fibermetric_gamma, ...
                                 'bright', filter.fibermetric_method);
            % bl = fibermetric(bl, sigma_range, ObjectPolarity=polarity, 'StructureSensitivity', filter.fibermetric_gamma);
        end
    else
        if any(sigma > 0)
            warning('Invalid fibermetric_sigma: [start step end] must all be > 0, step > 0, start <= end. No filtering applied.');
        end
        % bl = bl; % (implicit)
    end
end

