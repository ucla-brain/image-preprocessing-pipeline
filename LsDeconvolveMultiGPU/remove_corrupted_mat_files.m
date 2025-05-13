
% === Script: remove_corrupted_mat_files.m ===

% Specify folder containing .mat files
folder_path = 'V:\tif\ADT5_9x_AAV_ds\cache_deconvolution_ADT5_Ex_561_Em_600';  % <-- change this path

% Get list of all .mat files in the folder
mat_files = natsortfiles(dir(fullfile(folder_path, '*.mat')));

% Loop through each file
for k = 894:length(mat_files)
    file_name = mat_files(k).name;
    full_path = fullfile(folder_path, file_name);

    fprintf('Checking file: %s\n', file_name);

    try
        % Try loading the .mat file
        load(full_path, '-mat');
    catch ME
        % If there's an error, assume the file is corrupted
        fprintf('Corrupted file detected and deleted: %s\n', file_name);
        delete(full_path);
    end
end

fprintf('Done checking all files.\n');
