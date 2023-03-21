% Program for Deconvolution of Light Sheet Microscopy Stacks. Copyright
% TU-Wien 2019, written using MATLAB 2018b by Klaus
% Becker(klaus.becker@tuwien.ac.at) Keivan Moradi at UCLA B.R.A.I.N (Dong
% lab) patched it in MATLAB V2022b. kmoradi@mednet.ucla.edu. Main changes
% by Keivan Moradi: Multi-GPU, resume, and 3D gaussian filter support
% LsDeconv is free software: you can redistribute it and/or modify it under
% the terms of the GNU General Public License as published by the Free
% Software Foundation, either version 3 of the License, or at your option)
% any later version. LsDeconv is distributed in the hope that it will be
% useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
% Public License for more details. You should have received a copy of the
% GNU General Public License along with this file. If not, see
% <http://www.gnu.org/licenses/>.

% TODO: save both flipped and non-flipped
% TODO: Test SmartSPIM PSF
% TODO: resume saving consider images saved already in save_image function
% TODO: Update showinfo function
% TODO: test applying gaussian every 10 it

function [] = LsDeconv(varargin)
    try
        disp(' ');
        disp('LsDeconv: Deconvolution tool for Light Sheet Microscopy.');
        disp('TU Wien, 2019: This program was was initially written in MATLAB V2018b by klaus.becker@tuwien.ac.at');
        disp('Keivan Moradi, 2023: Patched it in MATLAB V2022b. kmoradi@mednet.ucla.edu. UCLA B.R.A.I.N (Dong lab)');
        disp('Main changes: Multi-GPU, Resume, and 3D gaussian filter support');
        disp(' ');
        disp(datetime('now'));
        disp(' ');

        if nargin < 20
            showinfo();
            if isdeployed
                exit(1);
            end
            return
        end

        %read command line parameters
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
        block_size_max = varargin{14};
        gpus = varargin{15};
        amplification = varargin{16};
        sigma = varargin{17};
        resume = varargin{18};
        starting_block = varargin{19};
        flip_upside_down = varargin{20};
        cache_drive = tempdir;
        if nargin > 20
            cache_drive=varargin{21};
            if ~exist(cache_drive, "dir")
                mkdir(cache_drive);
            end
        end

        %convert command line parameters from string to double
        if isdeployed
            dxy = str2double(strrep(dxy, ',', '.'));
            dz = str2double(strrep(dz, ',', '.'));
            numit = str2double(strrep(numit, ',', '.'));
            NA = str2double(strrep(NA, ',', '.'));
            rf = str2double(strrep(rf, ',', '.'));
            lambda_ex = str2double(strrep(lambda_ex, ',', '.'));
            lambda_em = str2double(strrep(lambda_em, ',', '.'));
            fcyl = str2double(strrep(fcyl, ',', '.'));
            slitwidth = str2double(strrep(slitwidth, ',', '.'));
            damping = str2double(strrep(damping, ',', '.'))/100;
            clipval = str2double(strrep(clipval, ',', '.'));
            stop_criterion = str2double(strrep(stop_criterion, ',', '.'));
            block_size_max = str2double(strrep(block_size_max, ',', '.'));
            % gpus = str2double(strrep(gpus, ',', '.'));
            amplification = str2double(strrep(amplification, ',', '.'));
            resume = str2double(strrep(resume, ',', '.'));
            starting_block = str2double(strrep(starting_block, ',', '.'));
            flip_upside_down = str2double(strrep(flip_upside_down, ',', '.'));
        end

        if isfolder(inpath)
            % make folder for results and make sure the outpath is writable
            outpath = fullfile(inpath, 'deconvolved');
            if flip_upside_down
                outpath = fullfile(inpath, 'deconvolved_fliped_upside_down');
            end
            if ~exist(outpath, 'dir')
                mkdir(outpath);
            end

            log_file_path = fullfile(outpath, 'log.txt');
            log_file = fopen(log_file_path, 'w');
            % get available resources
            [ram_available, ram_total]  = get_memory();
            p_log(log_file, 'system information ...');
            p_log(log_file, ['   total RAM (GB): ' num2str(ram_total/1024^3)]);
            p_log(log_file, ['   available RAM (GB): ' num2str(ram_available/1024^3)]);
            p_log(log_file, ['   data processed on GPUs: ' num2str(unique(gpus(gpus>0)))]);
            p_log(log_file, ['   data processed on CPUs: ' num2str(size(gpus(gpus<1), 2)) ' cores']);
            p_log(log_file, ' ');

            p_log(log_file, 'image stack info ...');
            [info.x, info.y, info.z, info.bit_depth] = getstackinfo(inpath); % n is volume dimension
            info.flip_upside_down = flip_upside_down;

            if resume && numel(dir(fullfile(outpath, '*.tif'))) == info.z
                disp("it seems all the files are already deconvolved!");
                return
            elseif ~resume
                delete(fullfile(outpath, '*.tif'));
            end
        else
            error('provided path did not exist!');
        end

        if info.x * info.y * info.z == 0
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

        p_log(log_file, ['   image size (voxels): ' num2str(info.x)  'x * ' num2str(info.y) 'y * ' num2str(info.z) 'z = ' num2str(info.x * info.y * info.z)]);
        p_log(log_file, ['   voxel size (nm^3): ' num2str(dxy)  'x * ' num2str(dxy) 'y * ' num2str(dz) 'z = ' num2str(dxy^2*dz)]);
        p_log(log_file, ['   image bit depth: ' num2str(info.bit_depth)]);
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
        [psf, FWHMxy, FWHMz] = LsMakePSF(dxy_corr, dz, NA, rf, lambda_ex, lambda_em, fcyl, slitwidth);
        p_log(log_file, ['   size of PSF (pixel): ' num2str(size(psf, 1))  ' x ' num2str(size(psf, 2)) ' x ' num2str(size(psf, 3))]);
        p_log(log_file, ['   FWHHM of PSF lateral (nm): ' num2str(FWHMxy)]);
        p_log(log_file, ['   FWHHM of PSF axial (nm): ' num2str(FWHMz)]);
        p_log(log_file, ['   Rayleigh range of objective lateral (nm): ' num2str(Rxy)]);
        p_log(log_file, ['   Rayleigh range of objective axial (nm): ' num2str((2 * lambda_em * rf) / NA^2)]);
        clear Rxy dxy_corr FWHMxy FWHMz;

        % plot_matrix(psf);
        options.overwrite = true;
        saveastiff(psf, 'psf.tif', options);
        p_log(log_file, ' ');

        % disp(psf)
        % psf_file = Tiff(fullfile(inpath, 'deconvolved', 'psf.tif'), 'w');
        % write(psf_file, psf);
        % close(psf_file);

        % split the image into smaller blocks
        % x and y pads are interpolated since they are smaller than z
        % z pad is from actual image to avoid artifacts
        p_log(log_file, 'partitioning the image into blocks ...')

        block_path = fullfile(cache_drive, 'block.mat');
        if resume && exist(block_path, 'file')
            block = load(block_path).block;
        else
            [block.nx, block.ny, block.nz, block.x, block.y, block.z, block.x_pad, block.y_pad, block.z_pad] = autosplit(info, size(psf), block_size_max, ram_available);
            save(block_path, "block");
        end

        p_log(log_file, ['   block numbers: ' num2str(block.nx) 'x * ' num2str(block.ny) 'y * ' num2str(block.nz) 'z = ' num2str(block.nx * block.ny * block.nz) ' blocks.']);
        p_log(log_file, ['   block size loaded image: ' num2str(block.x) 'x * ' num2str(block.y) 'y * ' num2str(block.z) 'z = ' num2str(block.x * block.y * block.z) ' voxels.']);
        p_log(log_file, ['   block size deconvolved image: ' num2str(block.x) 'x * ' num2str(block.y) 'y * ' num2str(block.z - 2*block.z_pad) 'z = ' num2str(block.x * block.y * (block.z - 2*block.z_pad)) ' voxels.']);
        p_log(log_file, ['   block size in ram: ' num2str(block.x+2*block.x_pad) 'x * ' num2str(block.y+2*block.y_pad) 'y * ' num2str(block.z) 'z = ' num2str((block.x+2*block.x_pad) * (block.y+2*block.y_pad) * block.z) ' voxels.']);
        p_log(log_file, ['   padding on x: ' num2str(block.x_pad) ' steps.']);
        p_log(log_file, ['   padding on y: ' num2str(block.y_pad) ' steps.']);
        p_log(log_file, ['   padding on z: ' num2str(block.z_pad) ' steps.']);
        p_log(log_file, ' ');

        % start image processing
        p_log(log_file, 'preprocessing params ...')
        p_log(log_file, ['   gaussian sigma: ' num2str(sigma, 3)]);
        p_log(log_file, ['   post gaussian baseline subtraction (denoising): ' num2str(dark(sigma))]);
        p_log(log_file, ' ');

        p_log(log_file, 'deconvolution params ...')
        p_log(log_file, ['   max. iterations: ' num2str(numit)]);
        p_log(log_file, ' ');

        p_log(log_file, 'postprocessing params ...')
        p_log(log_file, ['   damping factor (%): ' num2str(damping)]);
        p_log(log_file, ['   stop criterion (%): ' num2str(stop_criterion)]);
        p_log(log_file, ['   histogram clipping value (%): ' num2str(clipval)]);
        p_log(log_file, ['   signal amplification: ' num2str(amplification)]);
        p_log(log_file, ['   post deconvolution dark subtraction: ' num2str(amplification)]);
        if flip_upside_down
        p_log(log_file, '   flip the image upside down (y-axis): yes');
        else
        p_log(log_file, '   flip the image upside down (y-axis): no');
        end
        p_log(log_file, ' ');

        p_log(log_file, 'paths ...')
        p_log(log_file, ['   source image: ' char(inpath)]);
        p_log(log_file, ['   deconvolved image: ' char(outpath)]);
        p_log(log_file, ['   blocks: ' char(cache_drive)]);
        p_log(log_file, ['   logs: ' char(log_file_path)]);
        p_log(log_file, ' ');
        process(inpath, outpath, log_file, info, block, psf, numit, ...
            damping, clipval, stop_criterion, gpus, cache_drive, ...
            amplification, sigma, resume, starting_block);

        fclose(log_file);
        open(log_file_path);

        if isdeployed
            exit(0);
        end
    catch ME
        % error handling
        text = getReport(ME, 'extended', 'hyperlinks', 'off');
        disp(text);
        if isdeployed
            exit(1);
        end
    end
end

% calculate xy plan size based on z step size
function [x, y, x_pad, y_pad] = calculate_xy_size(z, z_pad, info, block_size_max, psf_size)
    % min pad size is half the psf size on the axis
    % x_max is max_value_allowed_by_block_size_and_z - 2 * min_pad
    x_max_in_ram = floor(nthroot(double(block_size_max/ z), 2));
    x_max_deconvolved = x_max_in_ram - psf_size(1);
    x_min_deconvolved = x_max_deconvolved - 5*psf_size(1);
    x = x_min_deconvolved;
    x_pad = pad_size(x, psf_size(1));
    z_deconvolved = z - 2*z_pad;
    max_deconvolved_voxels = x^2 * z_deconvolved;
    for x_ = x_min_deconvolved+1:x_max_deconvolved
        x_pad_ = pad_size(x_, psf_size(1));
        voxel_count_in_ram = (x_+2*x_pad_)^2 * z;
        deconvolved_voxels = x_^2 * z_deconvolved;
        if voxel_count_in_ram <= block_size_max && deconvolved_voxels > max_deconvolved_voxels
            x = x_;
            x_pad = x_pad_;
            max_deconvolved_voxels = deconvolved_voxels;
        end
    end
    y = x;
    y_pad = x_pad;
    if x > info.x
        x = info.x;
    end
    if y > info.y
        y = info.y;
    end

    if info.x > info.y
        % first try to increase x then y
        while x < info.x && (x+1+2*pad_size(x+1, psf_size(1))) * (y+2*y_pad) * z <= block_size_max && (x+1)*y*z_deconvolved > max_deconvolved_voxels
            x = x + 1;
            x_pad = pad_size(x, psf_size(1));
            max_deconvolved_voxels = x*y*z_deconvolved;
        end
        while y < info.y && (x+2*x_pad) * (y+1+2*pad_size(y+1, psf_size(2))) * z <= block_size_max && x*(y+1)*z_deconvolved > max_deconvolved_voxels
            y = y + 1;
            y_pad = pad_size(y, psf_size(2));
            max_deconvolved_voxels = x*y*z_deconvolved;
        end
    else
        % first try to increase y then x
        while y < info.y && (x+2*x_pad) * (y+1+2*pad_size(y+1, psf_size(2))) * z <= block_size_max && x*(y+1)*z_deconvolved > max_deconvolved_voxels
            y = y + 1;
            y_pad = pad_size(y, psf_size(2));
            max_deconvolved_voxels = x*y*z_deconvolved;
        end
        while x < info.x && (x+1+2*pad_size(x+1, psf_size(1))) * (y+2*y_pad) * z <= block_size_max && (x+1)*y*z_deconvolved > max_deconvolved_voxels
            x = x + 1;
            x_pad = pad_size(x, psf_size(1));
            max_deconvolved_voxels = x*y*z_deconvolved;
        end
    end
end

% determine the required number of blocks that are deconvolved sequentially
function [nx, ny, nz, x, y, z, x_pad, y_pad, z_pad] = autosplit(info, psf_size, block_size_max, ram_available)
    if block_size_max <= 0
        error(['block size should be larger than zero and smaller than ' num2str(intmax("int32")) ' for GPU computing']);
    end
    if block_size_max > intmax("int32")
        warning(['block size should be smaller than ' num2str(intmax("int32")) ' for GPU computing']);
    end

    % psf half width was not enouph to eliminate artifacts on z
    psf_size(3) = psf_size(3) .* 2;

    % image will be converted to single precision (8 bit) float during
    % deconvolution but two copies are needed
    % z z-steps of the original image volume will be chunked to
    % smaller 3D blocks. After deconvolution the blocks will be
    % reassembled to save deconvolved z-steps. Therefore, there should be
    % enough ram to reassemble z z-steps in the end.
    z_max = min(floor(ram_available / info.x / info.y / 16), info.z);
    z = z_max;
    % load extra z layers for each block to avoid generating artifacts on z
    % for efficiency of FFT pad data in a way that the largest prime
    % factor becomes <= 5 for each end of the block
    z_pad = ceil(pad_size(z, psf_size(3)));
    [x, y, x_pad, y_pad] = calculate_xy_size(z, z_pad, info, block_size_max, psf_size);
    max_deconvolved_voxels = 0;
    for z_ = psf_size(3):z_max
        z_pad_ = pad_size(z_, psf_size(3));
        [x_, y_, x_pad_, y_pad_] = calculate_xy_size(z_, z_pad_, info, block_size_max, psf_size);
        deconvolved_voxels = x_ * y_ * (z_ - 2 * z_pad_);
        block_size = (x_ + 2 * x_pad_) * (y_ + 2 * y_pad_) * z_;
        if mod(z_pad_, 1)==0 && deconvolved_voxels > max_deconvolved_voxels && block_size <= block_size_max
            x = x_;
            y = y_;
            z = z_;
            x_pad = x_pad_;
            y_pad = y_pad_;
            z_pad = z_pad_;
            max_deconvolved_voxels = deconvolved_voxels;
        end
    end

    % number of blocks on each axis considering z-pad
    nx = ceil(info.x / x);
    ny = ceil(info.y / y);
    nz = ceil((info.z - 2*z_pad) / (z - 2*z_pad));
end

%provides coordinates of sub-blocks after splitting
function [p1, p2] = split(info, block)
    % bounding box coordinate points
    p1 = zeros(block.nx * block.ny * block.nz, 4);
    p2 = zeros(block.nx * block.ny * block.nz, 3);

    blnr = 0;
    for nz = 0 : block.nz-1
        % zs = nz * block.z + 1 - (2*nz-1) * block.z_pad;
        zs = nz * (block.z - 2 * block.z_pad) + 1;
        for ny = 0 : block.ny-1
            ys = ny * block.y + 1;
            for nx = 0 : block.nx-1
                xs = nx * block.x + 1;

                blnr = blnr + 1;
                p1(blnr, 1) = xs;
                p2(blnr, 1) = min([xs + block.x - 1, info.x]);

                p1(blnr, 2) = ys;
                p2(blnr, 2) = min([ys + block.y - 1, info.y]);

                p1(blnr, 3) = zs;
                p2(blnr, 3) = min([zs + block.z - 1, info.z]);

                % imaged processed so far
                p1(blnr, 4) = nz * block.z - max(2*nz-1, 0) * block.z_pad;
            end
        end
    end
end

function process(inpath, outpath, log_file, info, block, psf, numit, ...
    damping, clipval, stop_criterion, gpus, cache_drive, amplification, ...
    sigma, resume, starting_block)

    start_time = datetime('now');

    % split it to chunks
    [p1, p2] = split(info, block);

    % intermediate variables needed for interprocess communication
    min_max_path = fullfile(cache_drive, "min_max.mat");
    dark_threshold = dark(sigma);

    % start deconvolution
    num_blocks = block.nx * block.ny * block.nz;
    remaining_blocks = num_blocks;
    while remaining_blocks > 0
        for i = starting_block : num_blocks
            % skip blocks already worked on
            block_path = fullfile(cache_drive, ['bl_' num2str(i) '_temp.mat']);
            if exist(block_path, "file")
                if dir(block_path).bytes > 10
                    remaining_blocks = remaining_blocks - 1;
                else
                    delete(block_path);
                end
            end
        end
        if remaining_blocks > 0
            delete(gcp('nocreate'));
            gpus = gpus(1:min(size(gpus, 2), remaining_blocks));
            if size(gpus, 2) > 1
                pool = parpool('local', size(gpus, 2), 'IdleTimeout', Inf);
                parallel_deconvolve(1 : size(gpus, 2)) = parallel.FevalFuture;
                idx = 0;
                for gpu = gpus
                    idx = idx + 1;
                    parallel_deconvolve(idx) = pool.parfeval(@deconvolve, 0, ...
                        inpath, psf, numit, damping, ...
                        block, info, p1, p2, min_max_path, ...
                        stop_criterion, gpu, cache_drive, ...
                        sigma, dark_threshold, ...
                        starting_block + idx - 1);
                    pause(10);
                end
                for j = 1:idx
                    %parallel_deconvolve(j).wait
                    parallel_deconvolve(j).fetchOutputs;
                end
                delete(pool);
            else
                deconvolve( ...
                    inpath, psf, numit, damping, ...
                    block, info, p1, p2, min_max_path, ...
                    stop_criterion, gpus(1), cache_drive, ...
                    sigma, dark_threshold, ...
                    starting_block);
            end
            % make sure all the blocks are processed
        end
        starting_block = 1;
    end

    % postprocess and write tif files
    delete(gcp('nocreate'));
    postprocess_save(...
        outpath, cache_drive, min_max_path, log_file, clipval, ...
        p1, p2, info, resume, block, amplification);

    p_log(log_file, ['deconvolution finished at ' char(datetime)]);
    p_log(log_file, ['elapsed time: ' char(duration(datetime('now') - start_time, 'Format', 'dd:hh:mm:ss'))]);
    disp('----------------------------------------------------------------------------------------');
    delete(gcp('nocreate'));
end

function deconvolve(inpath, psf, numit, damping, ...
    block, info, p1, p2, min_max_path, ...
    stop_criterion, gpu, cache_drive, ...
    sigma, dark_threshold, starting_block)

    if info.bit_depth == 8
        rawmax = 255;
    elseif info.bit_depth == 16
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
        block_path = fullfile(cache_drive, ['bl_' num2str(blnr) '_temp.mat']);

        if num_blocks > 1 && exist(block_path, "file")
            continue
        else
            fclose(fopen(block_path, 'w'));
        end

        % begin processing next block
        disp( ['loading block ' num2str(blnr) ' from ' num_blocks_str]);

        % load next block of data into memory
        startp = p1(blnr, :);
        endp = p2(blnr, :);
        x1 = startp(x); x2 = endp(x);
        y1 = startp(y); y2 = endp(y);
        z1 = startp(z); z2 = endp(z);
        bl = load_block(inpath, x1, x2, y1, y2, z1, z2);

        % get min-max of raw data stack
        if ~ismember(info.bit_depth, [8, 16])
            rawmax = max(prctile(bl(:), 99.9, 'all'), rawmax);
        end

        % deconvolve current block of data
        disp(['processing block ' num2str(blnr) ' from ' num_blocks_str]);

        [bl, lb, ub] = process_block(bl, block, psf, numit, damping, stop_criterion, gpu, sigma, dark_threshold);

        % delete block z_pad
        if block.z_pad > 0 && block.nz > 1
            if  z1 == 1
                bl = bl(:, :, 1:end - block.z_pad);
            elseif z2 == info.z
                bl = bl(:, :, 1 + block.z_pad:end);
            else
                bl = bl(:, :, 1 + block.z_pad:end - block.z_pad);
            end
        end

        % find maximum value in other blocks
        if exist(min_max_path, "file")
            min_max = load(min_max_path);
            deconvmax = gather(min_max.deconvmax);
            deconvmin = gather(min_max.deconvmin);
            rawmax = gather(min_max.rawmax);
        end
        % consolidate and save block stats
        deconvmax = max(ub, deconvmax);
        deconvmin = min(lb, deconvmin);
        save(min_max_path, "deconvmin", "deconvmax", "rawmax", "-v7.3", "-nocompression");

        disp(['saving block ' num2str(blnr) ' from ' num_blocks_str]);
        % save block to disk
        save(block_path, 'bl', '-v7.3', '-nocompression');
    end
end

function [bl, lb, ub] = process_block(bl, block, psf, niter, lambda, stop_criterion, gpu, sigma, dark_threshold)
    need_vram_conservation = false;
    if gpu
        gpu_device = gpuDevice(gpu);
        if gpu_device.TotalMemory < 32e9 || gpu_device.TotalMemory > 80e9
            need_vram_conservation = true;
        end
        bl = gpuArray(bl);
    end
    gaussian_start = tic;
    if min(sigma(:)) > 0
        bl = imgaussfilt3(bl, sigma, 'padding', 'circular');
        bl = bl - dark_threshold;
        bl = max(bl, 0);

        % flat = prctile(bl, 25, 3);
        % flat = median(bl, 3);
        % flat = imgaussfilt3(flat, 2, 'padding', 'circular');
        % flat = flat./max(flat(:));
        % bl = bl ./ flat;
        % clear flat;
        % disp("flat applied");

        disp(['3D Gaussian filter applied in ' num2str(toc(gaussian_start)) 's']);
    end

    if niter > 0
        % for efficiency of FFT pad data in a way that the largest prime
        % factor becomes <= 5. z_padding comes from image, which is
        % different from x and y pad that are interpolated based on image.
        % In case z_pad was small for FFT efficiency it will be
        % interpolated slightly
        blx = size(bl, 1);
        bly = size(bl, 2);
        blz = size(bl, 3);
    
        if blx ~= block.x || block.x_pad <= 0
            pad_x = pad_size(blx, size(psf, 1));
        else
            pad_x = block.x_pad;
        end
        if bly ~= block.y || block.y_pad <= 0
            pad_y = pad_size(bly, size(psf, 2));
        else
            pad_y = block.y_pad;
        end
    
        if blz < block.z
            if block.z_pad <= 0
                pad_z = pad_size(blz, size(psf, 3));
            else
                % add minimum pad to make sure FFT is optimized
                pad_z = pad_size(blz, 1);
            end
            if blz + 2*pad_z > block.z
                pad_z = (block.z - blz)/2;
            end
        elseif blz > block.z
            warning('image block on z-axis exceeds the maximum allowed!')
        else
            % FFT optimized z pad is already loaded from the actual image
            % stack no z pad interpolation is needed
            pad_z = 0;
        end
    
        bl = padarray(bl, [floor(pad_x) floor(pad_y) floor(pad_z)], 'pre', 'symmetric');
        bl = padarray(bl, [ceil(pad_x) ceil(pad_y) ceil(pad_z)], 'post', 'symmetric');
    
        % deconvolve block using Lucy-Richardson algorithm
        if gpu
        % GPU accelerated deconvolution needs 4*block_size_max vRAM > 32 GB
        % GPUs having >80 GB of vRAM can handle >6*block_size_max. 
        % To deconvolve two blocks on one GPU, bl is gathered, which only 
        % decelerats denom = bl ./ denom part of deconvolution that is 
        % computationaly least expensive.
            if need_vram_conservation
                bl = gather(bl);
            end
            bl = deconGPU(bl, psf, niter, lambda, stop_criterion);
        else
            bl = deconCPU(bl, psf, niter, lambda, stop_criterion);
        end
    
        %remove padding
        if pad_z > 0
            bl = bl(floor(pad_x) : end-ceil(pad_x)-1, floor(pad_y) : end-ceil(pad_y)-1, floor(pad_z) : end-ceil(pad_z)-1);
        else
            bl = bl(floor(pad_x) : end-ceil(pad_x)-1, floor(pad_y) : end-ceil(pad_y)-1, :);
        end
    end

    % since prctile function needs high vram usage gather it to avoid low
    % memory error
    if isgpuarray(bl)
        bl = gather(bl);
    end
    [lb, ub] = deconvolved_stats(bl);
end

%Lucy-Richardson deconvolution
function deconvolved = deconCPU(bl, psf, niter, lambda, stop_criterion)
    deconvolved = bl;
    OTF = single(psf2otf(psf, size(bl)));
    R = 1/26 * ones(3, 3, 3, 'single');
    R(2,2,2) = single(0);

    for i = 1 : niter
        start_time = tic;
        if stop_criterion > 0
            deconvolved_old = deconvolved;
        end
        denom = convFFT(deconvolved, OTF);
        denom = max(denom, eps('single')); % protect against division by zero
        denom = bl ./ denom;
        denom = convFFT(denom, conj(OTF));
        if lambda > 0
            deconvolved = deconvolved .* denom .* (1 - lambda) + ...
                          convn(deconvolved, R, 'same') .* lambda;
        else
            deconvolved = deconvolved .* denom;
        end
        % clear denom; % for 25% temporarily reduction in RAM memory usage
        deconvolved = abs(deconvolved); % get rid of imaginary artifacts

        if stop_criterion > 0
            % estimate quality criterion
            delta = rmse(deconvolved, deconvolved_old, 'all');
            if i == 1
                delta_rel = 0;
            else
                delta_rel = (deltaL - delta) / deltaL * 100;
            end
            deltaL = delta;

            disp(['iteration: ' num2str(i) ' duration: ' num2str(toc(start_time)) ' delta: ' num2str(delta_rel, 3)]);

            if i > 1 && delta_rel <= stop_criterion
                disp('stop criterion reached. finishing iterations.');
                break
            end
        else
            disp(['iteration: ' num2str(i) ' duration: ' num2str(toc(start_time))]);
        end
    end
end

function deconvolved = deconGPU(bl, psf, niter, lambda, stop_criterion)
    deconvolved = gpuArray(bl);
    psf_inv = psf(end:-1:1, end:-1:1, end:-1:1); % spatially reversed psf
    R = 1/26 * ones(3, 3, 3, 'single');
    R(2,2,2) = single(0);

    for i = 1 : niter
        start_time = tic;
        if stop_criterion > 0
            deconvolved_old = deconvolved;
        end
        denom = convn(deconvolved, psf, 'same');
        denom = max(denom, eps('single')); % protect against division by zero
        denom = bl ./ denom;
        denom = convn(denom, psf_inv, 'same');
        if lambda > 0
            deconvolved = deconvolved .* denom .* (1 - lambda) + ...
                          convn(deconvolved, R, 'same') .* lambda;
        else
            deconvolved = deconvolved .* denom;
        end
        % clear denom; % for 25% temporarily reduction in GPU memory usage
        deconvolved = abs(deconvolved); % get rid of imaginary artifacts

        if stop_criterion > 0
            % estimate quality criterion
            delta = rmse(deconvolved, deconvolved_old, 'all');
            if i == 1
                delta_rel = 0;
            else
                delta_rel = (deltaL - delta) / deltaL * 100;
            end
            deltaL = delta;

            disp(['iteration: ' num2str(i) ' duration: ' num2str(toc(start_time)) ' delta: ' num2str(delta_rel, 3)]);

            if i > 1 && delta_rel <= stop_criterion
                disp('stop criterion reached. Finishing iterations.');
                break
            end
        else
            disp(['iteration: ' num2str(i) ' duration: ' num2str(toc(start_time))]);
        end
    end
end

function postprocess_save(...
    outpath, cache_drive, min_max_path, log_file, clipval, ...
    p1, p2, info, resume, block, amplification)

    blocklist = strings(size(p1, 1), 1);
    for i = 1 : size(p1, 1)
        blocklist(i) = fullfile(cache_drive, ['bl_' num2str(i) '_temp.mat']);
        if ~exist(blocklist(i), 'file')
            warning(['missing block: bl_' num2str(i) '_temp.mat'])
        end
    end

    x = 1; y = 2; z = 3; z_saved = 4;
    if exist(min_max_path, "file")
        min_max = load(min_max_path);
        deconvmin = min_max.deconvmin;
        deconvmax = min_max.deconvmax;
        rawmax = min_max.rawmax;
    else
        warning("min_max.mat not found!")
        deconvmin = 0;
        deconvmax = 5.3374;
        rawmax = 255;
    end

    % rescale deconvolved data
    if rawmax <= 255
        scal = 255;
    elseif rawmax <= 65535
        scal = 65535;
    else
        scal = rawmax; % scale to maximum of input data
    end
    p_log(log_file, 'image stats ...');
    p_log(log_file, ['   max image value based on data type: ' num2str(scal)]);
    p_log(log_file, ['   max block 99.9% percentile before deconvolution: ' num2str(rawmax)]);
    p_log(log_file, ['   max block 99.9% percentile after deconvolution: ' num2str(deconvmax)]);
    p_log(log_file, ['   min value in image after deconvolution: ' num2str(deconvmin)]);
    p_log(log_file,  ' ');

    if clipval > 0
        %estimate the global histogram and upper and lower clipping values
        nbins = 1e6;
        binwidth = deconvmax / nbins;
        bins = 0 : binwidth : deconvmax;

        %calculate cumulative histogram by scanning all blocks
        disp('calculating histogram...');

        for i = size(blocklist, 1) : -1 : 1
            S = load(blocklist(i), 'bl');
            chist = chist + cumsum(histcounts(S.bl, bins));
        end
        clear S;

        % normalize cumulative histogram to 0..100%
        chist = chist / max(chist) * 100;
        % determine upper and lower histogram clipping values
        low_clip = findClosest(chist, clipval) * binwidth;
        high_clip = findClosest(chist, 100-clipval) * binwidth;
    end

    % mount data and save data layer by layer
    blnr = 1;  % block number
    imagenr = 0;  % image number
    starting_z_block = 1;
    num_tif_files = numel(dir(fullfile(outpath, '*.tif')));
    if resume && num_tif_files
        starting_z_block = 0;
        while blnr <= length(p1) && p1(blnr, z_saved) <= num_tif_files
            if p1(blnr, x) == 1 && p1(blnr, y) == 1
                starting_block_number = blnr;
                starting_z_block = starting_z_block + 1;
            end
            blnr = blnr + 1;
        end
        blnr = starting_block_number;
        imagenr = p1(starting_block_number, z_saved); % since imagenr starts from zero but z levels start from 1
        disp(['number of existing tif files ' num2str(num_tif_files)]);
        disp(['resuming from block ' num2str(blnr) ' and image number ' num2str(imagenr)]);
    end
    clear num_tif_files;

    pool = parpool('local', 3, 'IdleTimeout', Inf);
    async_load(1 : block.nx * block.ny) = parallel.FevalFuture;

    for nz = starting_z_block : block.nz
        disp(['mounting layer ' num2str(nz) ' from ' num2str(block.nz)]);

        %load and mount next layer of images
        if block.z_pad > 0 && block.nz > 1
            if  nz == 1 || nz == block.nz
                R = zeros(info.x, info.y, p2(blnr, z) - p1(blnr, z) + 1 - block.z_pad, 'single');
            else
                R = zeros(info.x, info.y, p2(blnr, z) - p1(blnr, z) + 1 - 2*block.z_pad, 'single');
            end
        end

        for j = 1 : block.nx * block.ny
             async_load(j) = pool.parfeval(@my_load, 1, blocklist(blnr+j-1));
        end
        time_out = 120;
        for j = 1 : block.nx * block.ny
            if ispc
                file_path_parts = strsplit(blocklist(blnr), '\');
            else
                file_path_parts = strsplit(blocklist(blnr), '/');
            end
            file_name = char(file_path_parts(end));
            disp(['   loading block ' num2str(blnr) ':' num2str(block.nx * block.ny) ' file ' file_name]);
            % R( p1(blnr, x) : p2(blnr, x), p1(blnr, y) : p2(blnr, y), :) = my_load(blocklist(blnr));
            time_out_start = tic;
            async_load(j).wait('finished', time_out); % timeout in seconds
            R(p1(blnr, x) : p2(blnr, x), p1(blnr, y) : p2(blnr, y), :) = async_load(j).fetchOutputs;
            time_out = 0.9*time_out + 0.3*toc(time_out_start);
            blnr = blnr + 1;
        end
        
        % since R can be a very large matrix memory mangement is important.
        % combining operations should be avoided to save RAM.
        if clipval > 0
            %perform histogram clipping
            R = clip_min_max(R, low_clip, high_clip);
            R = R - low_clip;
            R = R .* (scal .* amplification ./ (high_clip - low_clip));
            R = min(R, scal);
        else
            %otherwise scale using min.max method
            if deconvmin > 0
                R = R - deconvmin;
                R = R .* (scal .* amplification ./ (deconvmax - deconvmin));
            else
                R = R .* (scal .* amplification ./ deconvmax);
            end
            R = R - amplification;
            R = clip_min_max(R, 0, scal);
        end

        %write images to output path
        disp(['saving ' num2str(size(R, 3)) ' images...']);
        save_image(R, outpath, imagenr, rawmax, info.flip_upside_down);
        imagenr = imagenr + size(R, 3);
        clear R;
    end

    % delte tmp files
    % for i = 1 : blnr
    %     delete(convertStringsToChars(blocklist(i)));
    % end
end

%calculates a theoretical point spread function
function [psf, FWHMxy, FWHMz] = LsMakePSF(dxy, dz, NA, nf, lambda_ex, lambda_em, fcyl,slitwidth)
    [nxy, nz, FWHMxy, FWHMz] = DeterminePSFsize(dxy, dz, NA, nf, lambda_ex, lambda_em, fcyl, slitwidth);

    %construct psf
    NAls = sin(atan(slitwidth / (2 * fcyl)));
    psf = samplePSF(dxy, dz, nxy, nz, NA, nf, lambda_ex, lambda_em, NAls);
    % disp('ok');
end

%determine the required grid size (xyz) for psf sampling
function [nxy, nz, FWHMxy, FWHMz] = DeterminePSFsize(dxy, dz, NA, nf, lambda_ex, lambda_em, fcyl, slitwidth)
    %Size of PSF grid is gridsize (xy z) times FWHM
    gridsizeXY = 2;
    gridsizeZ = 2;

    NAls = sin(atan(0.5 * slitwidth / fcyl));
    halfmax = 0.5 .* LsPSFeq(0, 0, 0, NA, nf, lambda_ex, lambda_em, NAls);

    %find zero crossings
    fxy = @(x)LsPSFeq(x, 0, 0, NA, nf, lambda_ex, lambda_em, NAls) - halfmax;
    fz = @(x)LsPSFeq(0, 0, x, NA, nf, lambda_ex, lambda_em, NAls) - halfmax;
    FWHMxy = 2 * abs(fzero(fxy, 100));
    FWHMz = 2 * abs(fzero(fz, 100));

    Rxy = 0.61 * lambda_em / NA;
    dxy_corr = min(dxy, Rxy / 3);

    nxy = ceil(gridsizeXY * FWHMxy / dxy_corr);
    nz = ceil(gridsizeZ * FWHMz / dz);

    %ensure that the grid dimensions are odd
    if mod(nxy, 2) == 0
        nxy = nxy + 1;
    end
    if mod(nz, 2) == 0
        nz = nz + 1;
    end
end

function psf = samplePSF(dxy, dz, nxy, nz, NA_obj, rf, lambda_ex, lambda_em, NA_ls)
	% disp([dxy, dz, nxy, nz, NA_obj, rf, lambda_ex, lambda_em, NA_ls]);
    % fprintf('dxy=%.1f, dz=%.1f, nxy=%.1f, nz=%.1f, NA_obj=%.1f, rf=%.2f, lambda_ex=%.1f, lambda_em=%.1f, NA_ls=%.4f\n', dxy, dz, nxy, nz, NA_obj, rf, lambda_ex, lambda_em, NA_ls);

    if mod(nxy, 2) == 0 || mod(nz, 2) == 0
        error('function samplePSF: nxy and nz must be odd!');
    end

    psf = zeros((nxy - 1) / 2 + 1, (nxy - 1) / 2 + 1, (nz - 1) / 2 + 1, 'single');
    for z = 0 : (nz - 1) / 2
        for y = 0 : (nxy - 1) / 2
            for x = 0 : (nxy - 1) / 2
               psf(x+1, y+1, z+1) = LsPSFeq(x*dxy, y*dxy, z*dz, NA_obj, rf, lambda_ex, lambda_em, NA_ls);
            end
        end
    end

    %Since the PSF is symmetrical around all axes only the first Octand is
    %calculated for computation efficiency. The other 7 Octands are
    %obtained by mirroring around the respective axes
    psf = mirror8(psf);

    %normalize psf to integral one
    psf = psf ./ sum(psf(:));
end

function R = mirror8(p1)
    %mirrors the content of the first quadrant to all other quadrants to
    %obtain the complete PSF.

    sx = 2 * size(p1, 1) - 1; sy = 2 * size(p1, 2) - 1; sz = 2 * size(p1, 3) - 1;
    cx = ceil(sx / 2); cy = ceil(sy / 2); cz = ceil(sz / 2);

    R = zeros(sx, sy, sz, 'single');
    R(cx:sx, cy:sy, cz:sz) = p1;
    R(cx:sx, 1:cy, cz:sz) = flip3D(p1, 0, 1 ,0);
    R(1:cx, 1:cy, cz:sz) = flip3D(p1, 1, 1, 0);
    R(1:cx, cy:sy, cz:sz) = flip3D(p1, 1, 0, 0);
    R(cx:sx, cy:sy, 1:cz) = flip3D(p1, 0, 0, 1);
    R(cx:sx, 1:cy, 1:cz) =  flip3D(p1, 0, 1 ,1);
    R(1:cx, 1:cy, 1:cz) =  flip3D(p1, 1, 1, 1);
    R(1:cx, cy:sy, 1:cz) =  flip3D(p1, 1, 0, 1);
end

%utility function for mirror8
function R = flip3D(data, x, y, z)
    R = data;
    if x
        R = flip(R, 1);
    end
    if y
        R = flip(R, 2);
    end
    if z
        R = flip(R, 3);
    end
end

%calculates PSF at point (x,y,z)
function R = LsPSFeq(x, y, z, NAobj, n, lambda_ex, lambda_em, NAls)
    R = PSF(z, 0, x, NAls, n, lambda_ex) .* PSF(x, y, z, NAobj, n, lambda_em);
end

%utility function for LsPSFeq
function R = PSF(x, y, z, NA, n, lambda)
    f2 = @(p)f1(p, x, y, z, lambda, NA, n);
    f2_integral = integral(f2, 0, 1, 'AbsTol', 1e-3);
    R = 4 .* abs(f2_integral).^2;
end

%utility function for LsPSFeq
function R = f1(p, x, y, z, lambda, NA, n)
    R = besselj(0, 2 .* pi .* NA .* sqrt(x.^2 + y.^2) .* p ./ (lambda .* n))...
        .* exp(1i .* (-pi .* p.^2 .* z .* NA.^2) ./ (lambda .* n.^2)) .* p;
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

% function plot_matrix(psf)
%     % Keivan disp(size(psf));
%     diff = double(squeeze(psf));
%     diff(diff==0)=nan;                              % added line
%     h = slice(diff, [], [], 1:size(diff,3));
%     set(h, 'EdgeColor','none', 'FaceColor','interp')
%     alpha(.1);
%     drawnow;
% end

function [ram_available, ram_total]  = get_memory()
%     if gpu
%         try
%             check = gpuDevice;  % check if CUDA is available
%             if (check.DeviceSupported == 1)
%                 mem = check.TotalMemory;
%             else
%                 disp('Matlab does not support your GPU');
%                 mem = 0;
%             end
%         catch
%             disp('CUDA is not available');
%             mem = 0;
%         end
%     end
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

% function vram = available_gpu_memory(gpu)
%     gpu_device = gpuDevice(gpu);
%     vram = gpu_device.AvailableMemory;
% end

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

function dark_threshold = dark(sigma)
    %sigma=[0.5 0.5 0.5];
    a=zeros(floor(sigma*20));
    a(floor(sigma(1)*10), floor(sigma(2)*10), floor(sigma(3)*10))=1;
    a=imgaussfilt3(a, sigma, 'padding', 'circular');
    % a(:,:,sigma(3)*10)
    % dark_threshold = prctile(a(a>0), 50);
    dark_threshold = mean(a(a>0));
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

function x = findGoodFFTLength(x)
    while max(factor(x)) > 5
        x = x + 1;
    end
end

function pad = pad_size(x, psf_size)
    pad = 0.5 * (findGoodFFTLength(x + psf_size) - x);
end

function [lb, ub] = deconvolved_stats(deconvolved)
    stats = prctile(deconvolved(:), [0.1 99.9], "all");
    lb = stats(1);
    ub = stats(2);
end

% return bounded value clipped between lower (lb) and upper bound (ub)
function y = clip_min_max(x, lb, ub)
    y=min(max(x, lb), ub);
end

function value = my_load(fname)
    data = load(fname);
    value = data.bl;
end

function save_image(R, outpath, imagenr_start, rawmax, flip_upside_down)
    for k = 1 : size(R, 3)
        tic;

        % select tile
        im = squeeze(R(:,:,k)');
        if flip_upside_down
            im = flip(im);
        end

        % file path
        s = num2str(imagenr_start + k - 1);
        while length(s) < 6
            s = strcat('0', s);
        end
        path = fullfile(outpath, ['img_' s '.tif']);

        % save
        if rawmax <= 255       % 8bit data
            imwrite(uint8(im), path); % , 'Compression', 'none'
        elseif rawmax <= 65535 % 16bit data
            imwrite(uint16(im), path); % , 'Compression', 'none'
        else                   % 32 bit data
            writeTiff32(im, path) %im must be single;
        end

        disp(['   saved img_' s ' in ' num2str(toc) ' seconds.'])
    end
end

function bl = load_block(inpath, start_x, end_x, start_y, end_y, start_z, end_z)
    filelist = dir(fullfile(inpath, '*.tif'));
    if numel(filelist) == 0
        filelist = dir(fullfile(inpath, '*.tiff'));
    end

    nx = end_x - start_x;
    ny = end_y - start_y;
    nz = end_z - start_z;

    %disp([start_x, end_x, start_y, end_y, start_z, end_z, nx+1, ny+1, nz+1])
    bl = zeros(nx+1, ny+1, nz+1, 'single');

    path = repmat({''}, nz+1, 1);
    for k = 1 : nz+1
        path{k} = fullfile(inpath, filelist((k-1)+start_z).name);
    end
    for k = 1 : nz+1
        im = im2single((imread(path{k}, 'PixelRegion', {[start_y, end_y],[start_x, end_x]})));
        bl(:, :, k) = im';
    end
end

%deconvolve with OTF
function R = convFFT(data , otf)
    R = ifftn(otf .* fftn(data));
end