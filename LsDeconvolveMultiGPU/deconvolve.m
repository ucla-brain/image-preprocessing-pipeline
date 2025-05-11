function [] = deconvolve(folderPath, dxy, dz, numit, lambda_ex, lambda_em, ...
                        cache_drive_path)
    disp('deconvolve:')
    calledAs = "deconvolve(folderPath=" + folderPath + ", dxy=" + dxy + ...
        ", dz=" + dz + ", numit=" + numit + ", lambda_ex=" + lambda_ex + ...
        ", lambda_em=" + lambda_em + ", cache_drive_path=" + ...
        cache_drive_path + ")";        
    disp("Hello I was called as ");
    disp(calledAs);
    LsDeconv(...
        convertCharsToStrings(folderPath), ... % e.g. "/data/20230724_15_54_32_SM230601_05_LS_15x_800z_stitched/Ex_488_Em_525_tif", ...
        str2double(strrep(dxy, ',', '.')),  ... % dxy (nm) e.g. 400 for 15x, 800 for 9x
        str2double(strrep(dz, ',', '.')),  ... % dz (nm)  e.g. 800 for 15x, 800 for 9x
        str2double(strrep(numit, ',', '.')),   ... % numit e.g. 10
        0.40, ... % NA
        1.42, ... % rf
        str2double(strrep(lambda_ex, ',', '.')),  ... % lambda_ex 488 561 642  e.g. 488
        str2double(strrep(lambda_em, ',', '.')),  ... % lambda_em 525 600 690  e.g. 525
        240,  ... % fcyl 80, 240 for the older unaligned stage
        12.0, ... % slitwidth (mm)
        0,    ... % damping percent or lambda
        0,    ... % clipval
        0,    ... % stop_criterion
        intmax('int32')-10^6,... % block_size_max=max number of elements in the block. GPU arrays on MATLAB are limited to intmax('int32') as the max size.
        [repmat(1:8, 1, floor(feature('numcores') / 8))],  ... % [repmat(1:8, 1, 5) zeros(1, 64)] gpu index in gpuDeviceTable, 0 means CPU
        1.0,  ... % signal amplification if clipval=0. clipval=1 means no amplification.
        [0.5, 0.5, 1.5], ... % x y z sigma of the 3D gaussian filter applied before deconvolution. filter_size = ceil(sigma * 4 + 1)
        [5, 5, 9], ... % filter_size
        1,   ... % denoising and background subtraction strength [0 to 255] for 8bit and [0 to 65535] for 16bit images
        1,    ... % 0 not resume, 1 = resume
        1,    ... % starting block should be greater than 0 for multiGPU processing
        0,    ... % 1 flip the deconvolved image upside down. 0 do not.
        0,    ... % 1 convert_to_8bit, 0 keep as is
        convertCharsToStrings(cache_drive_path) ... % cache drive (optional)
    );
end
