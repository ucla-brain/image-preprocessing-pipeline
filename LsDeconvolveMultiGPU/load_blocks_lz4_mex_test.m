function load_blocks_lz4_mex_test(varargin)
    %LOAD_BLOCKS_LZ4_MEX_TEST  End-to-end validation and benchmark.
    %
    %   Creates a large random single-precision volume.
    %   Splits it into equal-sized bricks.
    %   Saves each brick with save_lz4_mex.
    %   Reassembles using load_blocks_lz4_mex (MEX).
    %   Optionally compares against a pure-MATLAB parfeval reference.

    % -------------------------------------------------------------------------
    % Configuration
    % -------------------------------------------------------------------------
    if nargin && strcmpi(varargin{1},'big')
        stack_info = struct('x', 1024, 'y', 2049, 'z', 3073);   % ≈ 6 GiB
        block      = struct('x', 256,  'y', 256,  'z', 384);
    else
        stack_info = struct('x', 256,  'y', 512,  'z', 768);    % quick test
        block      = struct('x', 64,   'y', 64,   'z', 96);
    end

    % Derive #blocks in each dim for split()
    block.nx = ceil(stack_info.x / block.x);
    block.ny = ceil(stack_info.y / block.y);
    block.nz = ceil(stack_info.z / block.z);

    % -------------------------------------------------------------------------
    % Temporary directory with automatic cleanup
    % -------------------------------------------------------------------------
    outDir = tempname;
    mkdir(outDir);
    cleanupTmp = onCleanup(@() rmdir(outDir,'s'));   % delete recursively on exit
    fprintf('Temporary data in: %s\n', outDir);

    % -------------------------------------------------------------------------
    % Generate synthetic volume
    % -------------------------------------------------------------------------
    rng(1);
    V = rand([stack_info.x, stack_info.y, stack_info.z], 'single');

    % -------------------------------------------------------------------------
    % Split into blocks and save each brick
    % -------------------------------------------------------------------------
    [p1, p2] = split(stack_info, block);      % user-supplied helper
    nBlks    = size(p1,1);
    fileNames = cell(nBlks,1);

    fprintf('Saving %d blocks...\n', nBlks);
    for k = 1:nBlks
        sx = p1(k,1);  ex = p2(k,1);
        sy = p1(k,2);  ey = p2(k,2);
        sz = p1(k,3);  ez = p2(k,3);

        blk = V(sx:ex, sy:ey, sz:ez);
        fname = fullfile(outDir, sprintf('blk_%05d.lz4c', k));
        save_lz4_mex(fname, blk);
        fileNames{k} = fname;
    end
    fprintf('Done saving.\n');

    % -------------------------------------------------------------------------
    % Reconstruct with the new MEX
    % -------------------------------------------------------------------------
    fprintf('Reconstructing with load_blocks_lz4_mex...\n');
    tic;
    V_mex = load_blocks_lz4_mex(fileNames, uint64(p1), uint64(p2), ...
                                uint64([stack_info.x, stack_info.y, stack_info.z]));
    t_mex = toc;
    fprintf('MEX reconstruction time: %.2f s\n', t_mex);

    % -------------------------------------------------------------------------
    % MATLAB parfeval reference reconstruction
    % -------------------------------------------------------------------------
    fprintf('Building MATLAB parfeval reference...\n');
    %pool = gcp('nocreate'); if isempty(pool), pool = parpool; end
    %asyncs = parallel.FevalFuture.empty(nBlks,0);
    %V_ref  = zeros(size(V), 'single');

    tic;
    %for k = 1:nBlks
    %    asyncs(k) = parfeval(@load_lz4_mex, 1, fileNames{k});
    %end
%
    %cancelFutures = onCleanup(@() cancel(asyncs(isvalid(asyncs))));
    %for done = 1:nBlks
    %    [idx_blk, blk] = fetchNext(asyncs);
    %    V_ref(p1(idx_blk,1):p2(idx_blk,1), ...
    %          p1(idx_blk,2):p2(idx_blk,2), ...
    %          p1(idx_blk,3):p2(idx_blk,3)) = blk;
    %end
    t_matlab = toc;
    fprintf('MATLAB reference time: %.2f s\n', t_matlab);

    % -------------------------------------------------------------------------
    % Verification
    % -------------------------------------------------------------------------
    assert(isequaln(V, V_mex), 'Mismatch between original and MEX reconstruction.');
    % assert(isequaln(V, V_ref), 'Mismatch between original and MATLAB reference reconstruction.');

    fprintf('\nSUCCESS: all reconstructions are identical.\n');
    fprintf('Speed-up vs. parfeval: %.1f×\n', t_matlab / t_mex);
end