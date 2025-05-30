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
        disp('Keivan Moradi, 2023: Patched it in MATLAB V2023a. kmoradi@mednet.ucla.edu. UCLA B.R.A.I.N (Dong lab)');
        disp('Main changes: Improved block size calculareon and Multi-GPU and Multi-CPU parallel processing, Resume, Flip Y axis, and 3D gaussian filter support');
        disp(' ');
        disp(datetime('now'));
        disp(' ');

        % make sure correct number of parameters specified
        if nargin < 28
            showinfo();
            return
        end

        %read command line parameters
        disp("assigning command line parameter strings")
        inpath = varargin{1};
        dxy = varargin{2};
        dz = varargin{3};
        numit = varargin{4};
        NA = varargin{5};
        rf = varargin{6};
        lambda_ex = varargin{7};
        lambda_em = varargin{8};
        fcyl = varargin{9};
        slitwidth = varargin{10};
        damping = varargin{11};
        clipval = varargin{12};
        stop_criterion = varargin{13};
        block_size_max = double(varargin{14});
        gpus = varargin{15};
        amplification = varargin{16};
        filter.gaussian_sigma = varargin{17};
        filter.gaussian_size = varargin{18};
        filter.dark = varargin{19};
        filter.destripe_sigma = varargin{20};
        filter.regularize_interval = varargin{21};
        resume = varargin{22};
        starting_block = varargin{23};
        flip_upside_down = varargin{24};
        convert_to_8bit = varargin{25};
        convert_to_16bit = varargin{26};
        filter.use_fft = varargin{27};
        cache_drive = varargin{28};
        if ~exist(cache_drive, "dir")
            disp("making cache drive dir " + cache_drive)
            mkdir(cache_drive);
        end
        disp("cache drive dir created and/or exists " + cache_drive)
        
        assert(isa(inpath, "string"), "wrong type " + class(inpath));
        assert(isa(dxy, "double"), "wrong type " + class(dxy));
        assert(isa(dz, "double"), "wrong type " + class(dz));
        assert(isa(numit, "double"), "wrong type " + class(numit));
        assert(isa(lambda_ex, "double"), "wrong type " + class(lambda_ex));
        assert(isa(lambda_em, "double"), "wrong type " + class(lambda_em));
        assert(isa(cache_drive, "string"), "wrong type " + class(cache_drive));

        if isfolder(inpath)
            % make folder for results and make sure the outpath is writable
            outpath = fullfile(inpath, 'deconvolved');
            if flip_upside_down
                outpath = fullfile(inpath, 'deconvolved_fliped_upside_down');
            end
            if ~exist(outpath, 'dir')
                disp("making folder " + outpath )
                mkdir(outpath);
            end
            disp("outpath folder created and/or exists " + outpath);
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
            stack_info.flip_upside_down = flip_upside_down;
            stack_info.convert_to_8bit = convert_to_8bit;
            stack_info.convert_to_16bit = convert_to_16bit;

            if resume && numel(dir(fullfile(outpath, 'img_*.tif'))) == stack_info.z
                disp("it seems all the files are already deconvolved!");
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
        % else
        %     files = dir(fullfile(cache_drive, "bl_*"));
        %     files = files(~[files.isdir]); % files only
        %     for idx = 1:length(files)
        %         file = files(idx);
        %         if file.bytes < 10
        %             try
        %                 delete(fullfile(cache_drive, file.name));
        %             catch
        %                 warning('Cannot delete %s', file.name);
        %             end
        %         end
        %     end
        end

        disp("Logging image attributes");
        p_log(log_file, ['   image size (voxels): ' num2str(stack_info.x)  'x * ' num2str(stack_info.y) 'y * ' num2str(stack_info.z) 'z = ' num2str(stack_info.x * stack_info.y * stack_info.z)]);
        p_log(log_file, ['   voxel size (nm^3): ' num2str(dxy)  'x * ' num2str(dxy) 'y * ' num2str(dz) 'z = ' num2str(dxy^2*dz)]);
        p_log(log_file, ['   image bit depth: ' num2str(stack_info.bit_depth)]);
        p_log(log_file, ['   convert to 8bit: ' num2str(stack_info.convert_to_8bit)]);
        p_log(log_file, ['   convert to 16bit: ' num2str(stack_info.convert_to_16bit)]);
        p_log(log_file, ' ');

        disp("Logging imaging system parameters");
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
        p_log(log_file, 'partitioning the image into blocks ...')

        block_path = fullfile(cache_drive, 'block.mat');
        if resume && exist(block_path, 'file')
            loaded = load(block_path);
            block = loaded.block;
            p1 = block.p1;
            p2 = block.p2;
        else
            [block.nx, block.ny, block.nz, block.x, block.y, block.z, ...
             block.x_pad, block.y_pad, block.z_pad, block.fft_shape] = ...
                autosplit(stack_info, size(psf.psf), filter, block_size_max, ram_total);

            [p1, p2] = split(stack_info, block);

            % Embed p1 and p2 directly into block
            block.p1 = p1;
            block.p2 = p2;

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
        p_log(log_file, ['   3D gaussian filter size: ' num2str(filter.gaussian_size, 3)]);
        p_log(log_file, ['   post filter baseline subtraction (denoising): ' num2str(filter.dark)]);
        p_log(log_file, ['   destrip along the z-axis sigma: ' num2str(filter.destripe_sigma)]);
        p_log(log_file, ' ');

        p_log(log_file, 'deconvolution params ...')
        p_log(log_file, ['   max. iterations: ' num2str(numit)]);
        p_log(log_file, ['   regularization interval: ' num2str(filter.regularize_interval)]);
        p_log(log_file, ['   Tikhonov regularization lambda (0 to 1): ' num2str(damping)]);
        p_log(log_file, ['   early stopping criterion (%): ' num2str(stop_criterion)]);
        p_log(log_file, ['   deconvolve in frequency domain: ' num2str(filter.use_fft)]);
        p_log(log_file, ['   destripe sigma: ' num2str(filter.destripe_sigma)]);
        p_log(log_file, ' ');
        
        p_log(log_file, 'postprocessing params ...')
        p_log(log_file, ['   histogram clipping value (%): ' num2str(clipval)]);
        p_log(log_file, ['   signal amplification: ' num2str(amplification)]);
        p_log(log_file, ['   post deconvolution dark subtraction: ' num2str(amplification)]);
        if flip_upside_down
        p_log(log_file, '   flip the image upside down (y-axis): yes');
        else
        p_log(log_file, '   flip the image upside down (y-axis): no');
        end
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
        
        process(inpath, outpath, log_file, stack_info, block, psf, numit, ...
            damping, clipval, stop_criterion, gpus, cache_drive, ...
            amplification, filter, resume, starting_block);

        fclose(log_file);
        % open(log_file_path);
    catch ME
        % error handling
        disp(ME);
        if exist('log_file', 'var')
            p_log(log_file, getReport(ME, 'extended', 'hyperlinks', 'off'));
        end
        exit(1)
    end
end

function [nx, ny, nz, x, y, z, x_pad, y_pad, z_pad, fft_shape] = autosplit(stack_info, psf_size, filter, block_size_max, ram_available)
    % Parameters for RAM and block sizing
    ram_usage_portion = 0.5;               % Use at most 50% of available RAM
    bytes_per_voxel = 8;                   % Use 4 for single, 8 for double (adjust as needed)
    max_elements_per_dim = 1290;           % 3D cube limit from 2^31-1 elements
    max_elements_total  = 2^31 - 1;        % MATLAB's total element limit

    % Compute the max z size that fits in RAM (capped at 1290 and stack_info.z)
    z_max_ram = floor(ram_available * ram_usage_portion / (bytes_per_voxel * stack_info.x * stack_info.y));
    z_max = min(z_max_ram, stack_info.z);

    % Set min and max block sizes, capping to allowed per-dimension limit
    min_block = min([floor(max_elements_per_dim/2) floor(max_elements_per_dim/2) floor(max_elements_per_dim/2);
                     stack_info.x                  stack_info.y                  stack_info.z]);

    max_block = [min(max_elements_per_dim, stack_info.x), ...
                 min(max_elements_per_dim, stack_info.y), ...
                 min(max_elements_per_dim, z_max       )];

    % Cap total block size to max allowed elements
    block_size_max = min(block_size_max, max_elements_total);

    % For later use in dimension checks
    max_dim = max_elements_per_dim;

    best_score = -Inf;
    best = struct();

    % Use coarse step for initial sweep (square xy blocks)
    for z = max_block(3):-1:min_block(3)
        for xy = max_block(1):-1:min_block(1)
            x = xy; y = xy;
            bl_core = [x y z];

            d_pad = decon_pad_size(psf_size);
            bl_shape = bl_core + 2*d_pad;

            if filter.use_fft
                bl_shape = next_fast_len(bl_shape);
            end

            g_pad = gaussian_pad_size(bl_shape, filter.gaussian_size);
            bl_shape_max = bl_shape + 2*g_pad;

            if any(bl_shape > max_dim)
                continue;
            end
            if prod(bl_shape_max) > block_size_max
                continue;
            end

            score = prod(bl_core);
            if score > best_score
                best_score = score;
                best = struct('bl_core', bl_core, 'd_pad', d_pad, 'fft_shape', bl_shape);
            end
        end
    end

    if isempty(fieldnames(best))
        error('autosplit: No block shape fits in memory. Try increasing block_size_max or reducing min_block.');
    end

    x = best.bl_core(1); y = best.bl_core(2); z = best.bl_core(3);
    x_pad = best.d_pad(1); y_pad = best.d_pad(2); z_pad = best.d_pad(3);
    fft_shape = best.fft_shape;
    nx = ceil(stack_info.x / x);
    ny = ceil(stack_info.y / y);
    nz = ceil((stack_info.z - 2*z_pad) / (z - 2*z_pad));
end

function pad_size = gaussian_pad_size(image_size, filter_size)
    % Returns the required padding for a Gaussian filter of size filter_size
    % for each image dimension in image_size.
    % filter_size can be scalar or vector; if scalar, applies to all dims.
    if isscalar(filter_size)
        filter_size = repmat(filter_size, size(image_size));
    end
    pad_size = floor(filter_size(:) ./ 2);    % Column vector output
    if numel(pad_size) ~= numel(image_size)
        pad_size = [pad_size; zeros(numel(image_size) - numel(pad_size), 1)];
    end
    pad_size = pad_size(:).'; % Row vector (consistent with image_size)
end

function pad = decon_pad_size(psf_sz)
    % Returns padding size for deconvolution given PSF size in each dimension.
    % For a PSF of size N, the pad is ceil(N/2) for each dimension.
    pad = ceil(psf_sz / 2);
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


%provides coordinates of sub-blocks after splitting
function [p1, p2] = split(stack_info, block)
    % Calculate starting indices for each block (x, y, z)
    x_step = block.x - 2*block.x_pad;
    y_step = block.y - 2*block.y_pad;
    z_step = block.z - 2*block.z_pad;

    x_starts = 1 : x_step : (stack_info.x - block.x + 1);
    if isempty(x_starts) || (x_starts(end) + block.x - 1 < stack_info.x)
        last_start = stack_info.x - block.x + 1;
        if isempty(x_starts) || x_starts(end) ~= last_start
            x_starts = [x_starts, last_start];
        end
    end

    y_starts = 1 : y_step : (stack_info.y - block.y + 1);
    if isempty(y_starts) || (y_starts(end) + block.y - 1 < stack_info.y)
        last_start = stack_info.y - block.y + 1;
        if isempty(y_starts) || y_starts(end) ~= last_start
            y_starts = [y_starts, last_start];
        end
    end

    z_starts = 1 : z_step : (stack_info.z - block.z + 1);
    if isempty(z_starts) || (z_starts(end) + block.z - 1 < stack_info.z)
        last_start = stack_info.z - block.z + 1;
        if isempty(z_starts) || z_starts(end) ~= last_start
            z_starts = [z_starts, last_start];
        end
    end

    nx = numel(x_starts);
    ny = numel(y_starts);
    nz = numel(z_starts);

    p1 = zeros(nx * ny * nz, 4);
    p2 = zeros(nx * ny * nz, 3);

    blnr = 0;
    for iz = 1:nz
        for iy = 1:ny
            for ix = 1:nx
                blnr = blnr + 1;
                xs = x_starts(ix);
                ys = y_starts(iy);
                zs = z_starts(iz);

                p1(blnr, 1) = xs;
                p2(blnr, 1) = min(xs + block.x - 1, stack_info.x);

                p1(blnr, 2) = ys;
                p2(blnr, 2) = min(ys + block.y - 1, stack_info.y);

                p1(blnr, 3) = zs;
                p2(blnr, 3) = min(zs + block.z - 1, stack_info.z);

                p1(blnr, 4) = 0; % optional debug value
            end
        end
    end
end

function process(inpath, outpath, log_file, stack_info, block, psf, numit, ...
    damping, clipval, stop_criterion, gpus, cache_drive, amplification, ...
    filter, resume, starting_block)

    start_time = datetime('now');

    need_post_processing = false;
    if starting_block == 1
        need_post_processing = true;
    end
    
    % split it to chunks
    [p1, p2] = split(stack_info, block);

    % load filelist
    filelist = dir(fullfile(inpath, '*.tif'));
    if numel(filelist) == 0
        filelist = dir(fullfile(inpath, '*.tiff'));
    end
    filelist = natsortfiles(filelist);
    filelist = fullfile(inpath, {filelist.name});

    % flatten the gpus array
    gpus = gpus(:)';
    
    % intermediate variables needed for interprocess communication
    % NOTE: semaphore keys should be a more than zero values.
    % all the existing processes should be killed first before creating a
    % sechamore
    myCluster = parcluster('Processes');
    delete(myCluster.Jobs);
    delete(gcp("nocreate"));
    min_max_path = fullfile(cache_drive, "min_max.mat");
    [unique_gpus, ~, ~] = unique(gpus(:));
    unique_gpus = sort(unique_gpus, 'descend').';
    % [unique_gpus, ~, gpus_vertical] = unique(sort(gpus(gpus>0)));
    % gpu_count = accumarray(gpus_vertical, 1).';
    clear gpus_vertical;
    
    % initiate locks and semaphors
    % semkeys are arbitrary non-zero values
    cleanupSemaphoresFromCache();
    semkey_single = 1e3;
    semkey_loading_base = 1e4;
    semaphore_create(semkey_single, 1);
    gpu_queue_key = 6969;
    queue('create', gpu_queue_key, unique_gpus);
    for gpu = unique_gpus
        % semaphore_create(gpu, 1);
        semaphore_create(gpu + semkey_loading_base, 3);
    end

    % start deconvolution
    num_blocks = block.nx * block.ny * block.nz;
    remaining_blocks = num_blocks;
    while remaining_blocks > 0
        for i = starting_block : num_blocks
            % skip blocks already worked on
            block_path = fullfile(cache_drive, ['bl_' num2str(i) '.mat']);
            block_path_tmp = fullfile(cache_drive, ['bl_' num2str(i) '.mat.tmp']);
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
            parfor idx = 1 : numel(gpus)
                deconvolve( ...
                    filelist, psf, numit, damping, ...
                    block, stack_info, p1, p2, min_max_path, ...
                    stop_criterion, gpus(idx), gpu_queue_key, ...
                    cache_drive, filter, starting_block + idx - 1, dQueue);
            end
            delete(pool);
        end
        starting_block = 1;
    end

    % clear locks and semaphors
    semaphore_destroy(semkey_single);
    for gpu = unique_gpus
        % semaphore_destroy(gpu);
        semaphore_destroy(gpu + semkey_loading_base);
    end
    queue('destroy', gpu_queue_key);

    % postprocess and write tif files
    % delete(gcp('nocreate'));
    if need_post_processing
        postprocess_save(...
            outpath, cache_drive, min_max_path, log_file, clipval, ...
            p1, p2, stack_info, resume, block, amplification);
    end
    
    p_log(log_file, ['deconvolution finished at ' char(datetime)]);
    p_log(log_file, ['elapsed time: ' char(duration(datetime('now') - start_time, 'Format', 'dd:hh:mm:ss'))]);
    disp('----------------------------------------------------------------------------------------');
    delete(gcp('nocreate'));
end

function deconvolve(filelist, psf, numit, damping, ...
    block, stack_info, p1, p2, min_max_path, ...
    stop_criterion, gpu, gpu_queue_key, cache_drive, ...
    filter, starting_block, dQueue)
    
    semkey_single = 1e3;
    semkey_loading_base = 1e4;
    semkey_loading = semkey_loading_base + gpu;

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

    for blnr = starting_block : num_blocks
        % skip blocks already worked on
        block_path = fullfile(cache_drive, ['bl_' num2str(blnr) '.mat']);
        block_path_tmp = fullfile(cache_drive, ['bl_' num2str(blnr) '.mat.tmp']);
        semaphore('wait', semkey_single);
        if num_blocks > 1 && (exist(block_path, "file") || exist(block_path_tmp, "file"))
            semaphore('post', semkey_single);
            continue
        end
        fclose(fopen(block_path, 'w'));
        semaphore('post', semkey_single);

        % begin processing next block and load next block of data into memory
        startp = p1(blnr, :); endp = p2(blnr, :);
        x1     = startp(x);     x2 = endp(x);
        y1     = startp(y);     y2 = endp(y);
        z1     = startp(z);     z2 = endp(z);
        semaphore('wait', semkey_loading);
        send(dQueue, [current_device(gpu) ': block ' num2str(blnr) ' from ' num_blocks_str ' is loading ...']);
        loading_start = tic;
        bl = load_block(filelist, x1, x2, y1, y2, z1, z2, block, stack_info);
        send(dQueue, [current_device(gpu) ': block ' num2str(blnr) ' from ' num_blocks_str ' is loaded in ' num2str(round(toc(loading_start), 1))]);
        semaphore('post', semkey_loading);

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
        block_processing_start = tic;
        expected_size = size(bl);  % Store size before processing
        [bl, lb, ub] = process_block(bl, block, psf, numit, damping, stop_criterion, gpu, gpu_queue_key, filter);
        % === Check padded block size is unchanged by process_block ===
        actual_size = size(bl);
        assert(isequal(actual_size, expected_size), ...
            sprintf(['[process_block] Block %d: Output block size mismatch!\n', ...
                     'Expected [%d %d %d], got [%d %d %d].\n', ...
                     'Block X[%d-%d], Y[%d-%d], Z[%d-%d]'], ...
                    blnr, expected_size, actual_size, ...
                    p1(blnr,1), p2(blnr,1), ...
                    p1(blnr,2), p2(blnr,2), ...
                    p1(blnr,3), p2(blnr,3)));

        % Report status
        send(dQueue, [current_device(gpu) ': block ' num2str(blnr) ...
            ' from ' num_blocks_str ' filters applied in ' ...
            num2str(round(toc(block_processing_start), 1))]);

        % === Remove padding before saving ===
        save_start = tic;
        bl = bl(1 + block.x_pad : end - block.x_pad, ...
                1 + block.y_pad : end - block.y_pad, ...
                1 + block.z_pad : end - block.z_pad);

        % === Check trimmed block matches expected core size ===
        expected_size = [p2(blnr,1) - p1(blnr,1) + 1, ...
                         p2(blnr,2) - p1(blnr,2) + 1, ...
                         p2(blnr,3) - p1(blnr,3) + 1];
        actual_size = size(bl);

        assert(isequal(actual_size, expected_size), ...
            sprintf(['[remove padding] Block %d: Output block size mismatch!\n', ...
                     'Expected [%d %d %d], got [%d %d %d].\n', ...
                     'Block X[%d-%d], Y[%d-%d], Z[%d-%d]'], ...
                    blnr, expected_size, actual_size, ...
                    p1(blnr,1), p2(blnr,1), ...
                    p1(blnr,2), p2(blnr,2), ...
                    p1(blnr,3), p2(blnr,3)));

        % find maximum value in other blocks
        semaphore('wait', semkey_single);
        could_not_save = true;
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
                save(min_max_path, "deconvmin", "deconvmax", "rawmax", "-v7.3", "-nocompression");
                could_not_save = false;
            catch
                send(dQueue, "could not load or save min_max file. Retrying ...")
                pause(1);
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
                save(block_path_tmp, 'bl', '-v7.3');  % , '-nocompression'
                movefile(block_path_tmp, block_path, 'f');
                send(dQueue, [current_device(gpu) ': block ' num2str(blnr) ' from ' num_blocks_str ' saved in ' num2str(round(toc(save_start), 1))]);
                could_not_save = false;
            catch
                send(dQueue, ['could not save ' block_path '! Retrying ...']);
                pause(1);
            end
        end
    end
end

function [bl, lb, ub] = process_block(bl, block, psf, niter, lambda, stop_criterion, gpu, gpu_queue_key, filter)
    bl_size = size(bl);
    gpu_id = 0;
    if gpu && (min(filter.gaussian_sigma(:)) > 0 || niter > 0)
        % get the next available gpu
        gpu_id = queue('wait', gpu_queue_key);
        gpu_device = gpuDevice(gpu_id);
        bl = gpuArray(bl);
    end

    if min(filter.gaussian_sigma(:)) > 0
        bl = imgaussfilt3(bl, filter.gaussian_sigma, 'FilterSize', filter.gaussian_size, 'Padding', 'symmetric'); %circular
        bl = bl - filter.dark;
        bl = max(bl, 0);
    end

    if niter > 0 && max(bl(:)) > eps('single')
        pad_pre = [0 0 0];
        pad_post = [0 0 0];
        if filter.use_fft
            [bl, pad_pre, pad_post] = pad_block_to_fft_shape(bl, block.fft_shape);
        end
    
        % deconvolve block using Lucy-Richardson or blind algorithm
        bl = decon(bl, psf, niter, lambda, stop_criterion, filter.regularize_interval, gpu_id, filter.use_fft);

        % remove padding
        if filter.use_fft
            bl = bl(...
                pad_pre(1)+1 : end-pad_post(1), ...
                pad_pre(2)+1 : end-pad_post(2), ...
                pad_pre(3)+1 : end-pad_post(3));
        end
    end

    % since prctile function needs high vram usage gather it to avoid low
    % memory error
    if gpu && isgpuarray(bl) && free_GPU_vRAM(gpu_id, gpu_device) < 60
        % Reseting the GPU
        bl = gather(bl);
        reset(gpu_device);  % to free 2 extra copies of bl in gpu
        if free_GPU_vRAM(gpu_id, gpu_device) > 43
            bl = gpuArray(bl);
        else
            queue('post', gpu_queue_key, gpu_id);
        end
    end

    [lb, ub] = deconvolved_stats(bl);

    if gpu && isgpuarray(bl)
        bl = gather(bl);
        reset(gpu_device);  % to free gpu memory
        queue('post', gpu_queue_key, gpu_id);
    end

    if filter.destripe_sigma > 0
        bl = filter_subband_3d_z(bl, filter.destripe_sigma, 0, "db9");
    end

    assert(all(size(bl) == bl_size), '[process_block]: block size mismatch!');
end

function [bl, pad_pre, pad_post] = pad_block_to_fft_shape(bl, fft_shape)
    sz = size(bl);
    if numel(sz) < 3, sz(3) = 1; end
    pad_pre = zeros(1,3);
    pad_post = zeros(1,3);
    for k = 1:3
        missing = fft_shape(k) - sz(k);
        if missing > 0
            pad_pre(k) = floor(missing/2);
            pad_post(k) = ceil(missing/2);
        end
    end
    if any(pad_pre > 0 | pad_post > 0)
        bl = padarray(bl, pad_pre, 'replicate', 'pre');
        bl = padarray(bl, pad_post, 'replicate', 'post');
    end
end

function postprocess_save(...
    outpath, cache_drive, min_max_path, log_file, clipval, ...
    p1, p2, stack_info, resume, block, amplification)

    semkey_single = 1e3;
    semkey_multi = 1e4;
    semaphore_create(semkey_single, 1);
    semaphore_create(semkey_multi, 32);

    % Use cell array for blocklist for compatibility with file functions
    blocklist = cell(size(p1, 1), 1);
    missing_blocks = [];
    for i = 1 : size(p1, 1)
        blocklist{i} = fullfile(cache_drive, ['bl_' num2str(i) '.mat']);
        if ~exist(blocklist{i}, 'file')
            missing_blocks(end+1) = i; %#ok<AGROW>
        end
    end

    %%% Fail early if any expected block files are missing
    if ~isempty(missing_blocks)
        disp('ERROR: The following blocks are missing and postprocessing cannot continue:');
        disp(missing_blocks);
        error('Aborting postprocess_save due to missing block files.');
    end

    x = 1; y = 2; z = 3;
    if exist(min_max_path, "file")
        min_max = load(min_max_path);
        deconvmin = min_max.deconvmin;
        deconvmax = min_max.deconvmax;
        rawmax = min_max.rawmax;
    else
        warning("min_max.mat not found!")
        deconvmin = 0;
        deconvmax = 5.3374;
        rawmax = 65535;
    end

    if stack_info.convert_to_8bit
        rawmax = 255;
    elseif stack_info.convert_to_16bit
        rawmax = 65535;
    end

    % rescale deconvolved data
    if stack_info.convert_to_8bit
        scal = 255;
    elseif stack_info.convert_to_16bit
        scal = 65535;
    elseif rawmax <= 255
        scal = 255;
    elseif rawmax <= 65535
        scal = 65535;
    else
        scal = rawmax; % scale to maximum of input data
    end
    p_log(log_file, 'image stats ...');
    p_log(log_file, ['   max image value based on data type: ' num2str(scal)]);
    p_log(log_file, ['   max block 99.99% percentile before deconvolution: ' num2str(rawmax)]);
    p_log(log_file, ['   max block 99.99% percentile after deconvolution: ' num2str(deconvmax)]);
    p_log(log_file, ['   min value in image after deconvolution: ' num2str(deconvmin)]);
    p_log(log_file,  ' ');

    % Estimate the global histogram and upper and lower clipping values
    if clipval > 0
        nbins = 1e6;
        binwidth = deconvmax / nbins;
        bins = 0 : binwidth : deconvmax;

        chist = zeros(1, nbins);

        disp('calculating histogram...');
        parfor i = 1:numel(blocklist)
            S = load_bl(blocklist{i}, semkey_multi);
            chist = chist + histcounts(S.bl, bins);
        end
        chist = cumsum(chist);

        % normalize cumulative histogram to 0..100%
        chist = chist / max(chist) * 100;
        % determine upper and lower histogram clipping values
        low_clip = findClosest(chist, clipval) * binwidth;
        high_clip = findClosest(chist, 100 - clipval) * binwidth;
    end

    %%%%%%%%%%%%%% mount data and save data layer by layer %%%%%%%%%%%%%%

    % Robust resume logic
    starting_z_block = 1;
    imagenr = 0;
    num_tif_files = numel(dir(fullfile(outpath, '*.tif')));
    blnr = 1;
    num_blocks_per_z_slab = block.nx * block.ny;
    if resume && num_tif_files
        last_completed_z = num_tif_files;
        starting_z_block = 0;

        % Find the z-chunk where last completed z-plane resides
        for blnr = 1:length(p1)
            if p1(blnr, x) == 1 && p1(blnr, y) == 1
                % potential start of slab
                slab_start_z = p1(blnr, z);
                slab_end_z = p2(blnr, z);

                if slab_end_z <= last_completed_z
                    starting_z_block = starting_z_block + 1;
                    imagenr = slab_end_z;
                else
                    break;
                end
            end
        end

        if imagenr ~= last_completed_z
            % mismatch, possible partial save, rollback one slab to be safe
            starting_z_block = max(1, starting_z_block); % avoid zero-index
            imagenr = p1((starting_z_block - 1) * num_blocks_per_z_slab + 1, z) - 1;
        end

        disp(['number of existing tif files: ' num2str(num_tif_files)]);
        disp(['resuming from slab ' num2str(starting_z_block) ', image number ' num2str(imagenr + 1)]);
    end
    clear num_tif_files;

    num_workers = feature('numcores');
    if isempty(gcp('nocreate'))
        pool = parpool('local', num_workers, 'IdleTimeout', Inf);
    else
        pool = gcp();
    end

    async_load(num_blocks_per_z_slab) = parallel.FevalFuture;
    for nz = starting_z_block : block.nz
        disp(['layer ' num2str(nz) ' from ' num2str(block.nz) ': mounting blocks ...']);

        % Indices for blocks in this Z slab
        block_inds = ((nz-1)*block.nx*block.ny + 1):(nz*block.nx*block.ny);

        % Verify existence of all required block files
        missing_files = false;
        for idx = block_inds
            if ~exist(blocklist{idx}, 'file')
                disp(['Missing block file: ' blocklist{idx}]);
                missing_files = true;
            end
        end

        if missing_files
            error('Missing files detected in slab %d. Resume aborted.', nz);
        end

        slab_z1 = p1(block_inds(1), z);
        slab_z2 = p2(block_inds(1), z);
        slab_depth = slab_z2 - slab_z1 + 1;

        R = zeros(stack_info.x, stack_info.y, slab_depth, 'single');

        % Async load all blocks in this Z slab
        for j = 1:length(block_inds)
            async_load(j) = pool.parfeval(@load_bl, 1, blocklist{block_inds(j)}, semkey_multi);
        end

        % Assign each block directly using p1/p2 indices
        for j = 1:length(block_inds)
            asigment_time_start = tic;
            blnr = block_inds(j);
            R(p1(blnr, x):p2(blnr, x), ...
              p1(blnr, y):p2(blnr, y), ...
              p1(blnr, z)-slab_z1+1 : p2(blnr, z)-slab_z1+1) = async_load(j).fetchOutputs;

            filepath = blocklist{blnr};
            [~, name, ext] = fileparts(filepath);
            filename = strcat(name, ext);
            if iscell(filename), filename = filename{1}; end

            fprintf('   block %d/%d file: %s loaded and assigned in %.1f s\n', ...
                j, num_blocks_per_z_slab, filename, round(toc(asigment_time_start), 1));

        end
        
        % since R matrix can be very large memory mangement is important.
        % combining operations should be avoided to save RAM.
        if clipval > 0
            %perform histogram clipping
            R = R - low_clip;
            R = min(R, high_clip - low_clip);
            R = R .* (scal .* amplification ./ (high_clip - low_clip));
        else
            %otherwise scale using min.max method
            if deconvmin > 0
                R = R - deconvmin;
                R = R .* (scal .* amplification ./ (deconvmax - deconvmin));
            else
                R = R .* (scal .* amplification ./ deconvmax);
            end
        end
        R = R - amplification;
        R = round(R);
        R = min(R, scal);
        R = max(R, 0);
        if stack_info.flip_upside_down
            R = flip(R, 2);
        end

        if rawmax <= 255       % 8bit data
            R = uint8(R);
        elseif rawmax <= 65535 % 16bit data
            R = uint16(R);
        end

        %write images to output path
        disp(['layer ' num2str(nz) ' from ' num2str(block.nz) ': saving ' num2str(size(R, 3)) ' images ...']);
        parfor k = 1 : size(R, 3)
            save_time = tic;
            % file path
            s = num2str(imagenr + k - 1);
            while length(s) < 6
                s = strcat('0', s);
            end
            save_path = fullfile(outpath, ['img_' s '.tif']);
            if exist(save_path, "file")
                continue
            end

            message = save_image_2d(R(:,:,k), save_path, s, rawmax, save_time);
            disp(message);
        end

        imagenr = imagenr + size(R, 3);
        % delete(gcp('nocreate'));
        % pool = parpool('local', 16, 'IdleTimeout', Inf);
        % async_load(1 : num_blocks_per_z_slab) = parallel.FevalFuture;
    end
    semaphore_destroy(semkey_single);
    semaphore_destroy(semkey_multi);
    % delte tmp files
    % for i = 1 : blnr
    %     delete(convertStringsToChars(blocklist(i)));
    % end
end

function bl = load_bl(path, semkey)
    semaphore('wait', semkey);
    cleanup = onCleanup(@() semaphore('post', semkey));
    try
        bl = importdata(path);
    catch
        % Use warning or simple log
        delete(path);
        error('Deleting corrupted file: %s\n', path);
    end
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
    disp(['semaphore ' num2str(semkey) ' is created with the initial value of ' num2str(value)]);
end

function device = current_device(gpu)
    device = 'CPU';
    if gpu > 0
        device = 'GPU';
    end
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
    disp('Usage: LsDeconv TIFDIR DELTAXY DELTAZ nITER NA RI LAMBDA_EX LAMBDA_EM FCYL SLITWIDTH DAMPING HISOCLIP STOP_CRIT MEM_PERCENT');
    disp(' ');
    disp('TIFFDIR: Directory containing the 2D-tiff files to be deconvolved(16-bit 0r 32bit float grayscale images supported).');
    disp('The images are expected to be formated with numerical endings, as e.g. xx00001.tif, xx00002.tif, ... xx00010.tif....');
    disp(' ');
    disp('DELTAXY DELTAZ: xy- and z-Size of a voxel in nanometer. Choosing e.g. 250 500 means that a voxel is 250 nm x 250 nm wide in x- and y-direction');
    disp('and 500 nm wide in z-direction (vertical direction). Values depend on the total magnification of the microscope and the camera chip.');
    disp(' ');
    disp('nITER: max. number of iterations for Lucy-Richardson algorithm. Deconvolution stops before if stop_crit is reached.');
    disp(' ');
    disp('NA: numerical aperture of the objective.');
    disp(' ');
    disp('RI: refractive index of imaging medium and sample');
    disp(' ');
    disp('LAMBDA_EX: fluorescence excitation wavelength in nanometer.');
    disp(' ');
    disp('LAMBDA_EM: fluorescence emmision wavelength in nanometer.');
    disp(' ');
    disp('FCYL: focal length f in millimeter of the cylinder lens used for light sheet generation.');
    disp(' ');
    disp('SLITWIDTH: full width w in millimeter of the slit aperture placed in front of the cylinder lens. The NA of the light sheet');
    disp('generator system is calculated as NaLs = sin(arctan(w/(2*f))');
    disp(' ');
    disp('DAMPING: parameter between 0% and 10%. Increase value for images that are noisy. For images with');
    disp('good signal to noise ratio the damping parameter should be set to zero (no damping at all)');
    disp(' ');
    disp('HISTOCLIP: percent value between 0 and 5 percent. If HISTOCLIP is set e.g. to 0.01% then the histogram of the deconvolved');
    disp('stack is clipped at the 0.01 and the 99.99 percentile and the remaininginte intensity values are scaled to the full range (0...65535');
    disp('in case of of 16 bit images, and 0...Imax in case of 32 bit float images, where Imax is the highest intensity value');
    disp('occuring in the source stack');
    disp(' ');
    disp('STOP_CRIT: I the pecentual change to the last iteration step becomes lower than STOP_CRIT the deconvolution of the current');
    disp('block is finished. If STOP_CRIT is e.g. set to 2% then the iteration stops, if there is less than 2% percent');
    disp('change compared to the last iteration step.');
    disp(' ');
    disp('MEM_PERCENT: percent of RAM (or GPU memory, respectivbely, that can maximally by occopied by a data block. If the size of the image stack');
    disp('is larger than MEM_PERCENT * RAMSIZE / 100, the data set is split into blocks that are deconvolved sequentially and then stitched.');
    disp('A value of 4% usually is a good choice when working on CPU, a value of 50% when using the GPU. Decrease this value if other memory consuming');
	disp('programs are active.');
    disp(' ');
    disp('GPU: 0 = peform convolutions on CPU, 1 = perform convolutions on GPU');
end

function p_log(log_file, message)
    disp(message);
    fprintf(log_file, '%s\r\n', message);
end

function dark_ = dark(filter, bit_depth)
    if bit_depth == 8
        a=zeros(filter.gaussian_size, "uint8");
    elseif bit_depth == 16
        a=zeros(filter.gaussian_size, "uint16");
    else
        warning('unsupported image bit depth');
        a=zeros(filter.gaussian_size);
    end
    % dark is a value greater than 10 surrounded by zeros
    a(ceil(filter.gaussian_size(1)/2), ceil(filter.gaussian_size(2)/2), ceil(filter.gaussian_size(3)/2)) = filter.dark;
    a=im2single(a);
    a=imgaussfilt3(a, filter.gaussian_sigma, 'FilterSize', filter.gaussian_size, 'Padding', 'circular');
    dark_ = max(a(:));
end

%writes 32bit float tiff-imges
function writeTiff32(img, fname)
    t = Tiff(fname, 'w');
    tag.ImageLength = size(img, 1);
    tag.ImageWidth = size(img, 2);
    tag.Compression = Tiff.Compression.LZW;
    tag.SampleFormat = Tiff.SampleFormat.IEEEFP;
    tag.Photometric = Tiff.Photometric.MinIsBlack;
    tag.BitsPerSample = 32;
    tag.SamplesPerPixel = 1;
    tag.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    t.setTag(tag);
    t.write(img);
    t.close();
end

function index = findClosest(data, x)
    [~,index] = min(abs(data-x));
end

function [lb, ub] = deconvolved_stats(deconvolved)
    stats = prctile(deconvolved, [0.1 99.99], "all");
    if isgpuarray(stats)
        stats = gather(stats);
    end
    lb = stats(1);
    ub = stats(2);
end

function message = save_image_2d(im, path, s, rawmax, save_time)
    im = squeeze(im);
    im = im';
    for num_retries = 1 : 40
        try
            if rawmax <= 255       % 8bit data
                imwrite(im, path, 'Compression', 'Deflate');
            elseif rawmax <= 65535 % 16bit data
                imwrite(im, path, 'Compression', 'Deflate');
            else                   % 32 bit data
                writeTiff32(im, path) %im must be single;
            end
            break
        catch
            pause(1);
        end
    end
    message = ['   saved img_' s ' in ' num2str(round(toc(save_time), 1)) ' seconds and after ' num2str(num_retries) ' attempts.'];
end

function check_block_coverage_planes(stack_info, block)
    disp('checking for potential issues ...');

    [p1, p2] = split(stack_info, block);

    % Planes to check (boundaries only)
    planes = {'XY', 'XZ', 'YZ'};
    errors = {};

    % XY at z=1 and z=end
    for z = [1, stack_info.z]
        covered = false(stack_info.x, stack_info.y);
        for k = 1:size(p1,1)
            if p1(k,3) <= z && p2(k,3) >= z
                xs = p1(k,1):p2(k,1);
                ys = p1(k,2):p2(k,2);
                covered(xs, ys) = covered(xs, ys) + 1;
            end
        end
        overlaps = sum(covered(:) > 1);
        gaps = sum(covered(:) == 0);
        if overlaps > 0 || gaps > 0
            errors{end+1} = sprintf('XY plane at z=%d: %d gaps, %d overlaps', z, gaps, overlaps);
        end
    end

    % XZ at y=1 and y=end
    for y = [1, stack_info.y]
        covered = false(stack_info.x, stack_info.z);
        for k = 1:size(p1,1)
            if p1(k,2) <= y && p2(k,2) >= y
                xs = p1(k,1):p2(k,1);
                zs = p1(k,3):p2(k,3);
                covered(xs, zs) = covered(xs, zs) + 1;
            end
        end
        overlaps = sum(covered(:) > 1);
        gaps = sum(covered(:) == 0);
        if overlaps > 0 || gaps > 0
            errors{end+1} = sprintf('XZ plane at y=%d: %d gaps, %d overlaps', y, gaps, overlaps);
        end
    end

    % YZ at x=1 and x=end
    for x = [1, stack_info.x]
        covered = false(stack_info.y, stack_info.z);
        for k = 1:size(p1,1)
            if p1(k,1) <= x && p2(k,1) >= x
                ys = p1(k,2):p2(k,2);
                zs = p1(k,3):p2(k,3);
                covered(ys, zs) = covered(ys, zs) + 1;
            end
        end
        overlaps = sum(covered(:) > 1);
        gaps = sum(covered(:) == 0);
        if overlaps > 0 || gaps > 0
            errors{end+1} = sprintf('YZ plane at x=%d: %d gaps, %d overlaps', x, gaps, overlaps);
        end
    end

    if ~isempty(errors)
        err_msg = sprintf('Block coverage error(s) detected:\n%s', strjoin(errors, '\n'));
        error(err_msg);
    end
end

function bl = load_block(filelist, x1, x2, y1, y2, z1, z2, block, stack_info)
    % Load a padded 3D block from a stack of 2D image slices.
    % Assumes filelist is a cell array of file paths, one per z-plane.

    % ---- Vectorized setup ----
    starts = [x1, y1, z1];
    ends   = [x2, y2, z2];
    pads   = [block.x_pad, block.y_pad, block.z_pad];
    vol_sz = [stack_info.x, stack_info.y, stack_info.z];

    % Full requested block (possibly out-of-bounds)
    q1 = starts - pads;         % block window start for x/y/z
    q2 = ends   + pads;         % block window end   for x/y/z

    % Actual in-bounds region to load from disk
    src1 = max(1, q1);
    src2 = min(vol_sz, q2);

    % Padding needed (pre and post for each axis)
    pad_pre  = src1 - q1;
    pad_post = q2 - src2;

    % Indices for each axis (in-bounds)
    x_src = src1(1):src2(1);
    y_src = src1(2):src2(2);
    z_src = src1(3):src2(3);

    % Preallocate buffer for real data
    nx = numel(x_src);
    ny = numel(y_src);
    nz = numel(z_src);
    bl_real = zeros(nx, ny, nz, 'single');

    % Load 2D slices from disk
    for k = 1:nz
        img_k = z_src(k);
        try
            slice = imread(filelist{img_k}, 'PixelRegion', ...
                {[y_src(1), y_src(end)], [x_src(1), x_src(end)]});
        catch ME
            error('[load_block] Error reading slice %d: %s', img_k, ME.message);
        end
        bl_real(:, :, k) = im2single(slice)';
    end

    % Target (padded) block size
    core_sz   = ends - starts + 1;
    target_sz = core_sz + 2 * pads;

    % Apply symmetric padding only where needed
    bl = bl_real;
    if any(pad_pre  > 0),  bl = padarray(bl, pad_pre,  'symmetric', 'pre');  end
    if any(pad_post > 0),  bl = padarray(bl, pad_post, 'symmetric', 'post'); end

    % Final safety check
    assert(isequal(size(bl), target_sz), ...
        sprintf('[load_block] Output size mismatch! Got [%s], expected [%s]', ...
        num2str(size(bl)), num2str(target_sz)));
end
