function load_slab_lz4_test(varargin)
% End-to-end validation & benchmark for load_slab_lz4.
%   • Generates a random 3-D single array.
%   • Splits & saves bricks with save_lz4_mex (char filenames).
%   • Reassembles with threaded MEX.
%   • Builds a reference either with a process pool (parfeval) or serial loop.
%   • Verifies byte-wise identity and prints speed-up.

% ─── 1. CONFIG ───────────────────────────────────────────────────────────
if nargin && strcmpi(varargin{1},'big')
    stack = struct('x',1024,'y',2049,'z',3073);  brick = struct('x',256,'y',256,'z',384);
else
    stack = struct('x',256,'y',512,'z',768);     brick = struct('x',64 ,'y',64 ,'z',96);
end
brick.nx = ceil(stack.x/brick.x);
brick.ny = ceil(stack.y/brick.y);
brick.nz = ceil(stack.z/brick.z);

% ─── 2. TEMP DIR ─────────────────────────────────────────────────────────
tmpDir  = tempname;  mkdir(tmpDir);
cleanup = onCleanup(@() rmdir(tmpDir,'s'));
fprintf("Temp dir : %s\n",tmpDir);

% ─── 3. SYNTHETIC VOLUME ────────────────────────────────────────────────
rng(1);  V = rand([stack.x stack.y stack.z],'single');

% ─── 4. SPLIT & SAVE BRICKS (char filenames) ─────────────────────────────
[p1d,p2d] = split(stack,brick);   % double indices for MATLAB
nBricks   = size(p1d,1);
fnames    = cell(nBricks,1);

fprintf("Saving %d bricks …\n",nBricks);
for k = 1:nBricks
    blk   = V(p1d(k,1):p2d(k,1), p1d(k,2):p2d(k,2), p1d(k,3):p2d(k,3));
    fname = char(fullfile(tmpDir,sprintf('blk_%05d.lz4c',k)));
    save_lz4_mex(fname, blk);
    fnames{k} = fname;
end
fprintf("Done saving\n");

% ─── 5. RECONSTRUCT WITH THREADED MEX ────────────────────────────────────
fprintf("Reconstructing with load_slab_lz4 …\n");
tic;
V_mex = load_slab_lz4(fnames, uint64(p1d), uint64(p2d), uint64([stack.x stack.y stack.z]));
t_mex = toc;
fprintf("MEX time : %.2f s\n",t_mex);

% ─── 6. BUILD REFERENCE (process pool if possible) ───────────────────────
fprintf("Building reference …\n");
refMode = "serial";              % default fallback
t_ref   = NaN;

try
    % Try to start a **process-based** pool (not threads)
    if isempty(gcp('nocreate')) || strcmp(gcp.Type,"thread-based")
        delete(gcp('nocreate'));          % ensure none
        parpool("Processes");             % will error if profile broken
    end
    pool = gcp();
    refMode = "parfeval";
catch ME
    warning("Process pool unavailable – will build reference serially.\n%s",ME.message);
    pool = [];
end

if refMode == "parfeval"
    fut   = parallel.FevalFuture.empty(nBricks,0);
    V_ref = zeros(size(V),'single');
    tic;
    for k = 1:nBricks
        fut(k) = parfeval(@load_lz4_mex,1,fnames{k});
    end
    cancelOnExit = onCleanup(@() cancel(fut(isvalid(fut))));
    for done = 1:nBricks
        [idx, blk] = fetchNext(fut);
        V_ref(p1d(idx,1):p2d(idx,1), ...
              p1d(idx,2):p2d(idx,2), ...
              p1d(idx,3):p2d(idx,3)) = blk;
    end
    t_ref = toc;
    fprintf("parfeval (Processes) time : %.2f s\n",t_ref);

else   % ── serial fallback ──
    tic;
    V_ref = zeros(size(V),'single');
    for k = 1:nBricks
        blk = load_lz4_mex(fnames{k});
        V_ref(p1d(k,1):p2d(k,1), ...
              p1d(k,2):p2d(k,2), ...
              p1d(k,3):p2d(k,3)) = blk;
    end
    t_ref = toc;
    fprintf("Serial reference time     : %.2f s\n",t_ref);
end

% ─── 7. VERIFY ───────────────────────────────────────────────────────────
assert(isequaln(V,V_mex),  "MEX reconstruction mismatch");
assert(isequaln(V,V_ref), "Reference reconstruction mismatch");

fprintf("\nSUCCESS: all reconstructions identical.\n");
fprintf("Speed-up vs. %s: %.2fx\n", refMode, t_ref/t_mex);
end
