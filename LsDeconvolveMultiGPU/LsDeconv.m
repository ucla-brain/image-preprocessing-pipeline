% Program for Deconvolution of Light Sheet Microscopy Stacks.
% Copyright TU-Wien 2019, written using MATLAB 2018b by Klaus Becker
% (klaus.becker@tuwien.ac.at)
% LsDeconv is free software: you can redistribute it and/or modify it under the terms of the
% GNU General Public License as published by the Free Software Foundation, either version 3 of the
% License, or at your option) any later version.
% LsDeconv is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
% even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
% General Public License for more details. You should have received a copy of the GNU General Public
% License along with this file. If not, see <http://www.gnu.org/licenses/>.
% LsDeconv('D:/test_raw_tif_0offset', 422, 1000, 100, 0.40, 1.52, 561, 600, 240, 12.0, 1, 0.01, 2.0, 80.0, 0)
% LsDeconv('F:/20210729_16_18_40_SW210318-07_R-HPC_15x_Zstep1um_50p_4ms/Ex_642_Em_680/143750/143750_153710', 422, 1000, 9, 0.40, 1.52, 642, 680, 240, 12.0, 1, 0.01, 2.0, 20, 1);

function [] = LsDeconv(varargin)
    try
        disp(' ');
        disp('LsDeconv: Deconvolution tool for Light Sheet Microscopy.');
        disp('(c) TU Wien, 2019. This program was was initially written in MATLAB V2018b by klaus.becker@tuwien.ac.at');
        disp('Keivan Moradi at UCLA B.R.A.I.N (Dong lab) patched it in MATLAB V2022b. kmoradi@mednet.ucla.edu. Main changes: Multi-GPU, resume, and 3D gaussian filter support');
        disp(' ');
        disp(datetime('now'));

        if nargin < 19
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
        cache_drive = tempdir;
        if nargin > 19
            cache_drive=varargin{20};
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
            %gpus = str2double(strrep(gpus, ',', '.'));
            amplification = str2double(strrep(amplification, ',', '.'));
            resume = str2double(strrep(resume, ',', '.'));
            starting_block = str2double(strrep(starting_block, ',', '.'));
        end

        if isfolder(inpath)
            disp('path exist! getting stack info ...')
            [nx, ny, nz] = getstackinfo(inpath); % n is volume dimension
        else
            disp('provided path did not exist!')
        end

        if nx * ny * nz == 0
            error('No valid TIFF files could be found!');
        end

        %start data processing
        mem = block_size_max * 1024^3;
        % mem = getmemory(gpus, mem_percent); % free memory in byted
        % if min(gpus, mem)
        %     gpuDevice(gpu);
        % end

        [tx, ty, tz] = autosplit(nx, ny, nz, mem);
        num_blocks = tx * ty * tz;
        if num_blocks > 1
            disp(['deconvolution split into ' num2str(num_blocks) ' = ' num2str(tx) ' x ' num2str(ty) ' x ' num2str(tz) ' xyz blocks.']);
        end

        delete(gcp('nocreate'));
        process(inpath, tx, ty, tz, dxy, dz, numit, NA, rf, ...
            lambda_ex, lambda_em, fcyl, slitwidth, damping, clipval, ...
            stop_criterion, gpus, cache_drive, amplification, sigma, ...
            resume, starting_block);

        if isdeployed
            exit(0);
        end
    catch ME
        %error handling
        text = getReport(ME, 'extended', 'hyperlinks', 'off');
        disp(text);
        if isdeployed
            exit(1);
        end
    end
end

% function mem = getmemory(gpu, mem_percent)
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
%     else
%         try
%             [~, m] = memory;  % check if os supports memory function
%             mem = m.PhysicalMemory.Available; % returns free memory not physical memory in bytes
%         catch
%             disp('Matlab is unable to get free RAM in your OS!')
%             mem = 0;
%         end
%     end
%
%     if ~mem
%         mem = input('Enter available free RAM or vRAM in GB?\n') * 1024^3;
%     end
%
%     if gpu
%         mem = mem * mem_percent / 100 / 5;
%     else
%         mem = mem * mem_percent / 100 / 3;
%     end
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

%determine the required number of blocks that are deconvolved sequentially
function [tx, ty, tz] = autosplit(npix_x, npix_y, npix_z, maxblocksize)
    tx = 0; ty = 0; tz = 0;
    tz_max = max(floor(npix_z / 100), 1);

    if maxblocksize == 0
        return
    end

    while true
        while true
            if npix_x / tx >= npix_y / ty
                tx = tx + 1;
            else
                ty = ty + 1;
            end

            %if tx >= 99 || tz_max >= 99
            %    error('AutoTiles: error during splitting occured!');
            %end

            for tz = 1 : tz_max
                bytes = ceil(npix_x / tx) * ceil(npix_y / ty) * ceil(npix_z / tz) * 8;
                if bytes <= maxblocksize
                    return
                end
            end
        end
    end
end

function  process(inpath, tx, ty, tz, dxy, dz, numit, NA, rf, ...
    lambda_ex, lambda_em, fcyl, slitwidth, damping, ...
    clipval, stop_criterion, gpus, cache_drive, amplification, sigma, ...
    resume, starting_block)

    tic;
    %generate PSF
    Rxy = 0.61 * lambda_em / NA;
    dxy_corr = min(dxy, Rxy / 3);
    [psf, nxy, nz, FWHMxy, FWHMz] = LsMakePSF(dxy_corr, dz, NA, rf, lambda_ex, lambda_em, fcyl, slitwidth);

    % plot_matrix(psf);
    options.overwrite = true;
    saveastiff(psf, 'psf.tif', options);

    % get stack info and split it to chunks
    [info.x, info.y, info.z, info.bit_depth] = getstackinfo(inpath);
    [p1, p2] = split(info, tx, ty, tz);
    if ~exist(cache_drive, 'dir')
        mkdir(cache_drive);
    elseif ~resume
        delete(fullfile(cache_drive, "*.*"));
    else
        files = dir(fullfile(cache_drive, "bl_*"));
        files = files(~[files.isdir]); % files only
        for idx = 1:length(files)
            file = files(idx);
            if file.bytes < 10
                try
                    delete(fullfile(cache_drive, file.name));
                catch
                    warning('Cannot delete %s', file.name);
                end
            end
        end
    end
    min_max_path = fullfile(cache_drive, "min_max.mat");

    % make folder for results and make sure the outpath is writable
    outpath = fullfile(inpath, 'deconvolved');
    if ~exist(outpath, 'dir')
        mkdir(outpath);
    elseif resume && numel(dir(fullfile(outpath, '*.tif'))) == numel(dir(fullfile(inpath, '*.tif')))
        disp("it seems all the files are already deconvolved!");
        return
    elseif ~resume
        delete(fullfile(outpath, '*.tif'));
    end


    % start deconvolution
    num_blocks = tx * ty * tz;
    gpus = gpus(1:min(size(gpus, 2), num_blocks));
    if size(gpus, 2) > 1
        pool = parpool('local', size(gpus, 2));
        parallel_deconvolve(1 : size(gpus, 2)) = parallel.FevalFuture;
        idx = 0;
        for gpu = gpus
            idx = idx + 1;
            parallel_deconvolve(idx) = pool.parfeval(@deconvolve, 2, ...
                inpath, psf, numit, damping, ...
                tx, ty, tz, info, p1, p2, min_max_path, ...
                stop_criterion, gpu, cache_drive, sigma, ...
                resume, starting_block + idx - 1);
            pause(30);
        end
        rawmax = 0;
        for j = idx:-1:1
            parallel_deconvolve(j).wait
            [rawmax_, blocklist] = parallel_deconvolve(j).fetchOutputs;
            rawmax = max(rawmax, rawmax_);
        end
        delete(pool);
    else
        [rawmax, blocklist]  = deconvolve( ...
            inpath, psf, numit, damping, ...
            tx, ty, tz, info, p1, p2, min_max_path, ...
            stop_criterion, gpus(1), cache_drive, sigma, ...
            resume, starting_block);
    end

    % postprocess and write tif files
    delete(gcp('nocreate'));
    scal = postprocess_save(outpath, min_max_path, clipval, blocklist, p1, p2, info, resume, tx, ty, tz, rawmax, amplification);

    %write parameter info file
    disp('generating info file...');
    fid = fopen(fullfile(inpath, 'deconvolved', 'DECONV_parameters.txt'),'w');
    fprintf(fid, '%s\r\n',['deconvolution finished at ' datestr(now)]);
    fprintf(fid, '%s\r\n',['data processed on GPU: ' num2str(gpus)]);
    fprintf(fid, '%s\r\n',['elapsed time: ' datestr(datenum(0,0,0,0,0, toc),'HH:MM:SS')]);
    fprintf(fid, '%s\r\n', '');
    fprintf(fid, '%s\r\n',['highest intensity value in deconvolved data: ' num2str(scal)]);
    fprintf(fid, '%s\r\n',['focal length of cylinder lens (mm): ' num2str(fcyl)]);
    fprintf(fid, '%s\r\n',['width of slit aperture (mm): ' num2str(slitwidth)]);
    fprintf(fid, '%s\r\n',['histogram clipping value (%): ' num2str(clipval)]);
    fprintf(fid, '%s\r\n',['numerical aperture: ' num2str(NA)]);
    fprintf(fid, '%s\r\n',['excitation wavelength (nm): ' num2str(lambda_ex)]);
    fprintf(fid, '%s\r\n',['emission wavelength (nm): ' num2str(lambda_em)]);
    fprintf(fid, '%s\r\n',['refractive index: ' num2str(rf)]);
    fprintf(fid, '%s\r\n',['max. iterations: ' num2str(numit)]);
    fprintf(fid, '%s\r\n',['damping factor (%): ' num2str(damping)]);
    fprintf(fid, '%s\r\n',['stop criterion (%): ' num2str(stop_criterion)]);
    fprintf(fid, '%s\r\n', '');
    fprintf(fid, '%s\r\n',['source data folder: ' inpath]);
    fprintf(fid, '%s\r\n',['number of blocks (x y z): ' num2str(tx)  ' x ' num2str(ty) ' x ' num2str(tz)]);
    fprintf(fid, '%s\r\n', '');
    fprintf(fid, '%s\r\n',['voxel size : ' num2str(dxy)  ' nm x ' num2str(dxy) ' nm x ' num2str(dz) ' nm']);
    fprintf(fid, '%s\r\n',['size of PSF (pixel): ' num2str(nxy)  ' x ' num2str(nxy) ' x ' num2str(nz)]);
    fprintf(fid, '%s\r\n',['FWHHM of PSF lateral (nm): ' num2str(FWHMxy)]);
    fprintf(fid, '%s\r\n',['FWHHM of PSF axial (nm): ' num2str(FWHMz)]);
    Rxy = 0.61 * lambda_em / NA;
    Rz = (2 * lambda_em * rf) / NA^2;
    fprintf(fid, '%s\r\n',['Rayleigh range of objective lateral (nm): ' num2str(Rxy)]);
    fprintf(fid, '%s\r\n',['Rayleigh range of objective axial (nm): ' num2str(Rz)]);
    fclose(fid);

    disp(['deconvolution of: ' strrep(inpath, '\', '/') ' finished successfully']);
    disp(['elapsed time: ' datestr(datenum(0,0,0,0,0, toc),'HH:MM:SS')]);
    disp(datetime('now'));
    disp('----------------------------------------------------------------------------------------');

    % disp(psf)
    % psf_file = Tiff(fullfile(inpath, 'deconvolved', 'psf.tif'), 'w');
    % write(psf_file, psf);
    % close(psf_file);
end

function [rawmax, blocklist] = deconvolve(inpath, psf, numit, damping, ...
    numxblocks, numyblocks, numzblocks, info, p1, p2, min_max_path, ...
    stop_criterion, gpu, cache_drive, ...
    sigma, resume, starting_block)

    if gpu
        gpuDevice(gpu);
    end

    % [info.x, info.y, info.z, info.bit_depth] = getstackinfo(inpath);
    % [p1, p2] = split(info, numxblocks, numyblocks, numzblocks);
    blocklist = strings(size(p1, 1), 1);

    if info.bit_depth == 8
        rawmax = 255;
    elseif info.bit_depth == 16
        rawmax = 65535;
    else
        rawmax = -Inf;
    end

    deconvmax = 0; deconvmin = Inf;
    x = 1; y = 2; z = 3;

    % save_path = cache_drive;
    % if ~exist(save_path, 'dir')
    %     mkdir(save_path);
    % elseif ~resume
    %     delete(fullfile(save_path, "*.*"))
    % end
    % min_max_path = fullfile(save_path, "min_max.mat");

    num_blocks = numxblocks * numyblocks * numzblocks;
    num_blocks_str = num2str(size(blocklist, 1));
    if num_blocks ~= size(blocklist, 1)
        warning("warning blocklist size and num_blocks do not match!");
    end

    remaining_blocks = num_blocks - starting_block + 1;
    while remaining_blocks > 0
        for i = starting_block : size(blocklist, 1)
            % skip blocks already worked on
            block_path = fullfile(cache_drive, join(['bl_', num2str(i), '_temp.mat']));
            try
                if resume && num_blocks > 1 && exist(block_path, "file")
                    % disp(['skipping block ' num2str(i) ' from ' num_blocks_str]);
                    % save block path
                    blocklist(i) = block_path;
                    continue
                else
                    fclose(fopen(block_path, 'w'));
                end

                % begin processing next block
                disp( ['loading block ' num2str(i) ' from ' num_blocks_str]);

                % load next block of data
                startp = p1(i, :); endp = p2(i, :);
                %startpos(x), startpos(y), startps(z) : start coordinates of currrent block (i)
                %endpos(x), endpos(y), endpos(z) : end coordinates of current block

                % load next block into memory
                x1 = startp(x); x2 = endp(x);
                y1 = startp(y); y2 = endp(y);
                z1 = startp(z); z2 = endp(z);
                bl = load_block(inpath, x1, x2, y1, y2, z1, z2);

                % find maximum value in processed data
                if exist(min_max_path, "file")
                    min_max = load(min_max_path);
                    deconvmax = min_max.deconvmax;
                    deconvmin = min_max.deconvmin;
                    rawmax = min_max.rawmax;
                end

                % get min-max of raw data stack
                if ~ismember(info.bit_depth, [8, 16])
                    rawmax = max(max(bl(:)), rawmax);
                end

                % deconvolve current block of data
                disp(['processing block ' num2str(i) ' from ' num_blocks_str]);
                bl = process_block(bl, psf, numit, damping, stop_criterion, gpu, sigma);


                deconvmax = max(max(bl(:)), deconvmax);
                deconvmin = min(min(bl(:)), deconvmin);
                save(min_max_path, "deconvmin", "deconvmax", "rawmax", "-v7.3", "-nocompression");

                disp(['saving block ' num2str(i) ' from ' num_blocks_str]);
                % save block to disk
                blocklist(i) = block_path;
                save(blocklist(i), 'bl', '-v7.3', '-nocompression');
            catch error
                % remove the placeholder file in case of error
                fprintf(1,'The identifier was:\n%s', error.identifier);
                fprintf(1,'There was an error! The message was:\n%s', error.message);
                delete(block_path);
                error(["deconvolution failed for block " num2str(i)]);
            end
        end

        % make sure all the blocks are processed
        for i = starting_block : size(blocklist, 1)
            % skip blocks already worked on
            block_path = fullfile(cache_drive, join(['bl_', num2str(i), '_temp.mat']));
            if dir(block_path).bytes > 0
                remaining_blocks = remaining_blocks - 1;
            else
                delete(block_path);
            end
        end
    end
end

function value = my_load(fname)
    data = load(fname);
    value = data.bl;
end

function scal = postprocess_save(outpath, min_max_path, clipval, blocklist, p1, p2, info, resume, numxblocks, numyblocks, numzblocks, rawmax, amplification)
    x = 1; y = 2; z = 3;
    if exist(min_max_path, "file")
        min_max = load(min_max_path);
        deconvmin = min_max.deconvmin;
        deconvmax = min_max.deconvmax;
    else
        warning("min_max.mat not found!")
        deconvmin = 0;
        deconvmax = 5.3374;
    end

    % rescale deconvolved data
    if rawmax <= 255
        scal = 255;
    elseif rawmax <= 65535
        scal = 65535;
    else
        scal = rawmax; % scale to maximum of input data
    end

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
        while blnr <= length(p1) && p1(blnr, z)-1 <= num_tif_files
            if p1(blnr, x) == 1 && p1(blnr, y) == 1
                starting_block_number = blnr;
                starting_z_block = starting_z_block + 1;
            end
            blnr = blnr + 1;
        end
        blnr = starting_block_number;
        imagenr = p1(starting_block_number, z) - 1; % since imagenr starts from zero but z levels start from 1
        disp(['number of existin tif files ' num2str(num_tif_files)]);
        disp(['resuming from block ' num2str(blnr) ' and image number ' num2str(imagenr)]);
    end
    clear num_tif_files;

    pool = parpool('local', 3);
    pool.IdleTimeout = 120; % 120=2h
    async_load(1 : numxblocks * numyblocks) = parallel.FevalFuture;

    for i = starting_z_block : numzblocks
        disp(['mounting layer ' num2str(i) ' from ' num2str(numzblocks)]);
        %load and mount next layer of images
        R = zeros(info.x, info.y, p2(blnr, z) - p1(blnr, z) + 1, 'single');
        for j = 1 : numxblocks * numyblocks
             async_load(j) = pool.parfeval(@my_load, 1, blocklist(blnr+j-1));
        end
        for j = 1 : numxblocks * numyblocks
            if ispc
                file_path_parts = strsplit(blocklist(blnr), '\');
            else
                file_path_parts = strsplit(blocklist(blnr), '/');
            end
            file_name = string(file_path_parts(end));
            disp(append('   loading block ', num2str(blnr), ' file ', file_name));
        %    R( p1(blnr, x) : p2(blnr, x), p1(blnr, y) : p2(blnr, y), :) = my_load(blocklist(blnr));
            async_load(j).wait('finished', 1200); % timeout in seconds
            R( p1(blnr, x) : p2(blnr, x), p1(blnr, y) : p2(blnr, y), :) = async_load(j).fetchOutputs;
            blnr = blnr + 1;
        end

        if clipval > 0
            %perform histogram clipping
            R(R < low_clip) = low_clip;
            R(R > high_clip) = high_clip;
            R = (R - low_clip) ./ (high_clip - low_clip) .* (scal .* amplification);
            R(R>scal) = scal;
        else
            %otherwise scale using min.max method
            if deconvmin > 0
                R = (R - deconvmin) ./ (deconvmax - deconvmin) .* (scal .* amplification);
            else
                R = R  .* (scal .* amplification ./ deconvmax);
            end
            R(R>scal) = scal;
        end

        %write images to output path
        disp(['saving ' num2str(size(R, 3)) ' images...']);
        save_image(R, outpath, imagenr, rawmax);
        imagenr = imagenr + size(R, 3);
        clear R;
    end

    % delte tmp files
    % for i = 1 : blnr
    %     delete(convertStringsToChars(blocklist(i)));
    % end
end

function save_image(R, outpath, imagenr_start, rawmax)
    for k = 1 : size(R, 3)
        tic;

        % select tile
        im = flip(squeeze(R(:,:,k)'));

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

function block = process_block(block, psf, numit, damping, stopcrit, gpu, sigma)
    %for efficiency of FFT pad data in a way that the largest prime factor becomes <= 5
    blx = size(block, 1);
    bly = size(block, 2);
    blz = size(block, 3);
    pad_x = 0.5 * (findGoodFFTLength(blx + 4 * size(psf, 1)) - blx);
    pad_y = 0.5 * (findGoodFFTLength(bly + 4 * size(psf, 2)) - bly);
    pad_z = 0.5 * (findGoodFFTLength(blz + 4 * size(psf, 3)) - blz);

    block = padarray(block, [floor(pad_x) floor(pad_y) floor(pad_z)], 'pre', 'symmetric');
    block = padarray(block, [ceil(pad_x) ceil(pad_y) ceil(pad_z)], 'post', 'symmetric');

    %deconvolve block using Lucy-Richardson algorithm
    if gpu
        block = deconGPU(block, psf, numit, damping, stopcrit, sigma);
    else
        block = deconCPU(block, psf, numit, damping, stopcrit, sigma);
    end

    %remove padding
    block = block(floor(pad_x) : end-ceil(pad_x)-1, floor(pad_y) : end-ceil(pad_y)-1, floor(pad_z) : end-ceil(pad_z)-1);
end

%provides coordinates of subblocks after splitting
function [p1, p2] = split(info, nx, ny, nz)
    xw = ceil(info.x / nx);
    yw = ceil(info.y / ny);
    zw = ceil(info.z / nz);

    p1 = zeros(nx*ny*nz, 3);
    p2 = zeros(nx*ny*nz, 3);

    n = 0;
    for i = 0 : nz-1
        zs = i * zw + 1;
        for j = 0 : ny-1
            ys = j * yw + 1;
            for k = 0 : nx-1
                xs = k * xw + 1;
                n = n + 1;
                p1(n, 1) = xs;
                p2(n, 1) = min([xs + xw - 1, info.x]);

                p1(n, 2) = ys;
                p2(n, 2) = min([ys + yw - 1, info.y]);

                p1(n, 3) = zs;
                p2(n, 3) = min([zs + zw - 1, info.z]);
                % disp([xs + xw - 1, info.x, ys + yw - 1, info.y, zs + zw - 1, info.z])
            end
        end
    end
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

%Lucy-Richardson deconvolution
function deconvolved = deconCPU(stack, psf, niter, lambda, stop_criterion, sigma)
    if min(sigma>0)
        stack = imgaussfilt3(stack, sigma);  % , 'padding', 'symmetric'
        disp("3D Gaussian filter applied");
    end
    deconvolved = stack;
    OTF = single(psf2otf(psf, size(stack)));
    R = 1/26 * ones(3, 3, 3, 'single'); R(2,2,2) = single(0);

    for i = 1 : niter
        denom = convFFT(deconvolved, OTF);
        denom(denom < eps('single')) = eps('single'); % protect against division by zero
        if stop_criterion > 0
            if lambda == 0
                deconvolved_new = convFFT(stack ./ denom, conj(OTF)) .* deconvolved;
            else
                deconvolved_new = (1 - lambda) .* convFFT(stack ./ denom, conj(OTF)) .* deconvolved ...
                                + lambda .* convn(deconvolved, R, 'same');
            end

            %estimate quality criterion
            delta = sqrt(sum((deconvolved(:) - deconvolved_new(:)).^2));
            if i == 1
                delta_rel = 0;
            else
                delta_rel = (deltaL - delta) / deltaL * 100;
            end

            deconvolved = deconvolved_new;
            deltaL = delta;

            disp(['iteration: ' num2str(i), ' delta: ' num2str(delta_rel, 3)]);

            if i > 1 && delta_rel <= stop_criterion
                disp('stop criterion reached. Finishing iterations.');
                break
            end
        else
            if lambda > 0
                deconvolved = (1 - lambda) .* convFFT(stack ./ denom, conj(OTF)) .* deconvolved ...
                            + lambda .* convn(deconvolved, R, 'same');
            else
                deconvolved = convFFT(stack ./ denom, conj(OTF)) .* deconvolved;
            end
            disp(['iteration: ' num2str(i)]);
        end
    end

    %get rid of imaginary artifacts
    deconvolved = abs(deconvolved);
end

function deconvolved = deconGPU(stack, psf, niter, lambda, stop_criterion, sigma)
    if min(sigma>0)
        stack = gather(imgaussfilt3(gpuArray(stack), sigma));  % , 'padding', 'symmetric'
        disp("3D Gaussian filter applied");
    end
    deconvolved = stack;
    psf_inv = psf(end:-1:1, end:-1:1, end:-1:1); % spatially reversed psf
    R = 1/26 * ones(3, 3, 3, 'single');
    R(2,2,2) = single(0);

    for i = 1 : niter
        % tic;
        denom = convGPU(deconvolved, psf);
        denom(denom < eps('single')) = eps('single'); % protect against division by zero
        if stop_criterion > 0
            if lambda == 0
                deconvolved_new = gather(convGPU(stack ./ denom, psf_inv)) .* deconvolved;
            else
                deconvolved_new = (1 - lambda) .* gather(convGPU(stack ./ denom, psf_inv)) .* deconvolved ...
                                + lambda .* convn(deconvolved, R, 'same');
            end

            %estimate quality criterion
            delta = sqrt(sum((deconvolved(:) - deconvolved_new(:)).^2));
            if i == 1
                delta_rel = 0;
            else
                delta_rel = (deltaL - delta) / deltaL * 100;
            end

            deconvolved = deconvolved_new;
            deltaL = delta;

            disp(['iteration: ' num2str(i), ' delta: ' num2str(delta_rel, 3)]);

            if i > 1 && delta_rel <= stop_criterion
                disp('stop criterion reached. Finishing iterations.');
                deconvolved = gather(deconvolved);
                break
            end
        else
            if lambda > 0
                deconvolved = (1 - lambda) .* convGPU(stack ./ denom, psf_inv) .* deconvolved ...
                            + lambda .* convn(deconvolved, R, 'same');
            else
                % deconvolved = convGPU(stack ./ denom, psf_inv) .* deconvolved;
                denom = stack ./ denom;
                denom = convn(denom, psf_inv, 'same');
                deconvolved = denom .* deconvolved;
            end
            % disp(['iteration: ' num2str(i) ' took ' datestr(datenum(0,0,0,0,0, toc),'HH:MM:SS')]);
            disp(['iteration: ' num2str(i)]);
        end
    end

    %get rid of imaginary artifacts
    deconvolved = abs(gather(deconvolved));
end

%deconvolve with OTF
function R = convFFT(data , otf)
    R = ifftn(otf .* fftn(data));
end

function R = convGPU(data, psf)
    % R = gather(convn(gpuArray(data), psf, 'same'));
    R = convn(gpuArray(data), psf, 'same');
end

%calculates a theoretical point spread function
function [psf, nxy, nz, FWHMxy, FWHMz] = LsMakePSF(dxy, dz, NA, nf, lambda_ex, lambda_em, fcyl,slitwidth)
    disp('calculating PSF...');
    [nxy, nz, FWHMxy, FWHMz] = DeterminePSFsize(dxy, dz, NA, nf, lambda_ex, lambda_em, fcyl, slitwidth);

    %construct psf
    NAls = sin(atan(slitwidth / (2 * fcyl)));
    psf = samplePSF(dxy, dz, nxy, nz, NA, nf, lambda_ex, lambda_em, NAls);
    disp('ok');
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

    %Since the PSF is symmetrical around all axes only the first Octand
    %is calculated for computation efficiency. The other 7 Octands are obtained by mirroring around
    %the respecitve axes
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
