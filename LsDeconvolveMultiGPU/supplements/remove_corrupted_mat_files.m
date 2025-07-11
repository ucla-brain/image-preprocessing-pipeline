% === Script: remove_corrupted_mat_files_parallel.m ===

% Specify folder
folder_path = '/path/to/file';

% Get sorted list of .mat files
mat_structs = natsortfiles(dir(fullfile(folder_path, '*.mat')));
file_names = {mat_structs.name};

% Use parallel loop
parfor k = 1:length(file_names)
    file_name = file_names{k};
    full_path = fullfile(folder_path, file_name);
    try
        % Load to a struct instead of workspace
        if exist(full_path, 'file')
            bl = importdata(full_path);
        end
    catch
        % Use warning or simple log
        warning('Deleting corrupted file: %s\n', file_name);
        delete(full_path);
    end
end

fprintf('Done checking all files.\n');
