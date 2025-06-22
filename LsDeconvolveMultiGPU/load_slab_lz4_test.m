function load_slab_lz4_test(varargin)
% End-to-end validation & benchmark for load_slab_lz4, including scaling/clipping postprocessing.

% ─── 1. CONFIG ───────────────────────────────────────────────────────────
if nargin && strcmpi(varargin{1},'big')
    stack = struct('x',1024,'y',2049,'z',3073);  brick = struct('x',256,'y',256,'z',384);
else
    stack = struct('x',256,'y',512,'z',768);     brick = struct('x',64 ,'y',64 ,'z',96);
end
brick.nx = ceil(stack.x/brick.x);
brick.ny = ceil(stack.y/brick.y);
brick.nz = ceil(stack.z/brick.z);

% --- TEST SCALING/CLIPPING PARAMETERS (you may change these for other scenarios) ---
clipval       = 0;     % set >0 for hard clipping, or 0 for plain scale
scal          = 255;   % 255 for uint8 output, 65535 for uint16 output, etc.
amplification = 1;
deconvmin     = 0;
deconvmax     = 1;
low_clip      = 0.1;
high_clip     = 0.9;

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
V_mex = load_slab_lz4( ...
    fnames, uint64(p1d), uint64(p2d), uint64([stack.x stack.y stack.z]), ...
    clipval, scal, amplification, deconvmin, deconvmax, low_clip, high_clip, feature('numCores'));
t_mex = toc;
fprintf("MEX time : %.2f s\n",t_mex);

% ─── 6. BUILD REFERENCE IN MATLAB (for final 8/16-bit result) ─────────────
fprintf("Building MATLAB reference (with postprocessing) …\n");
tic;
V_ref = V;

if clipval > 0
    V_ref = V_ref - low_clip;
    V_ref = min(V_ref, high_clip - low_clip);
    V_ref = V_ref .* (scal .* amplification ./ (high_clip - low_clip));
else
    if deconvmin > 0
        V_ref = (V_ref - deconvmin) .* (scal .* amplification ./ (deconvmax - deconvmin));
    else
        V_ref = V_ref .* (scal .* amplification ./ deconvmax);
    end
end
V_ref = round(V_ref - amplification);
V_ref = min(max(V_ref, 0), scal); % clamp

if scal <= 255
    V_ref = uint8(V_ref);
elseif scal <= 65535
    V_ref = uint16(V_ref);
end
t_ref = toc;
fprintf("MATLAB reference time : %.2f s\n",t_ref);

% ─── 7. VERIFY ───────────────────────────────────────────────────────────
eq = isequaln(V_mex, V_ref);
if eq
    fprintf('\n✅ SUCCESS: load_slab_lz4 output matches MATLAB reference exactly!\n');
    fprintf('Speed-up: %.2fx\n', t_ref/t_mex);
else
    fprintf('\n❌ MISMATCH: load_slab_lz4 output does not match MATLAB postprocessing.\n');
    dif = nnz(V_mex ~= V_ref);
    maxabs = double(max(abs(double(V_mex(:)) - double(V_ref(:)))));
    fprintf('  #mismatched voxels: %d (max abs diff: %.4g)\n', dif, maxabs);
end

end
