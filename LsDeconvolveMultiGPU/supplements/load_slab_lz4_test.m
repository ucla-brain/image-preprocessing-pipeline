function load_slab_lz4_fulltest(varargin)
% Exhaustive test-bench for save_lz4_mex ↔ load_slab_lz4 (no clipping branch).
% Generates a random stack, splits into LZ4 bricks, reloads with every
% scale branch, and compares against a MATLAB reference
% that uses *exactly* the user-supplied formula.

% ─────────── geometry ───────────
if nargin && strcmpi(varargin{1},'big')
    stack = struct('x',1024,'y',2049,'z',3073);
    brick = struct('x',256 ,'y',256 ,'z',384 );
else
    stack = struct('x',959,'y',611,'z',287);
    brick = struct('x',2 ,'y',1 ,'z',1 );
end
brick.nx = ceil(stack.x/brick.x);
brick.ny = ceil(stack.y/brick.y);
brick.nz = ceil(stack.z/brick.z);

% ─────────── sweep parameters (single) ───────────
scalS = single([255 65535]);          % → uint8 / uint16
dminS = single([0 0.1]);              % deconvmin
amplification = single(1);
deconvmax = single(2.5);

% ─────────── temp dir & bricks ───────────
tmpDir = tempname;  mkdir(tmpDir);
cleanupObj = onCleanup(@() rmdir(tmpDir,'s'));
feval('rng',1);   % deterministic
R0 = rand([stack.x stack.y stack.z],'single');   % original volume

[p1d,p2d] = split_stack(stack,brick);
nBricks   = size(p1d,1);
fn        = cell(nBricks,1);
for k = 1:nBricks
    blk   = R0(p1d(k,1):p2d(k,1), p1d(k,2):p2d(k,2), p1d(k,3):p2d(k,3));
    fn{k} = fullfile(tmpDir,sprintf('blk_%05d.lz4c',k));
    save_lz4_mex(fn{k},blk);
end

% ─────────── exhaustive sweep (no clipping) ───────────
total = 0;  fail = 0;
for scal = scalS
    for deconvmin = dminS
        total = total + 1;

        % ---- call MEX ----
        try
            [V_mex, t_mex] = load_slab_lz4( ...
                fn, uint64(p1d), uint64(p2d), ...
                uint64([stack.x stack.y stack.z]), ...
                scal, amplification, deconvmin, deconvmax, ...
                feature('numCores'));
        catch ME
            fprintf('❌  MEX error (scal=%g dmin=%g): %s\n', ...
                    double(scal), double(deconvmin), ME.message);
            fail = fail + 1;  continue
        end

        % ---- MATLAB reference (uses EXACT formula) ----
        R = R0;                                   %#ok<NASGU> base name must be R
        if deconvmin > 0
            R = (R - deconvmin) .* (scal .* amplification ./ (deconvmax - deconvmin));
        else
            R = R .* (scal .* amplification ./ deconvmax);
        end
        R = round(R - amplification);
        R = min(max(R,0),scal);
        if scal <= 255
            R = uint8(R);
        elseif scal <= 65535
            R = uint16(R);
        end
        V_ref = R;    %#ok<NASGU> keep name consistent

        % ---- strict type check ----
        if ~strcmp(class(V_mex), class(V_ref))
            error('Type mismatch: V_mex is %s, V_ref is %s', ...
                   class(V_mex), class(V_ref));
        end

        % ---- compare ----
        nerr   = nnz(V_mex ~= V_ref);
        maxabs = max(abs(single(V_mex(:)) - single(V_ref(:))));

        if maxabs <= 1
            fprintf('✅ scal=%g dmin=%g (%.2fs  #diff=%d max|Δ|=%g)\n', ...
                    double(scal), double(deconvmin), ...
                    t_mex, nerr, maxabs);
        else
            fprintf('❌ scal=%g dmin=%g  (#diff=%d max|Δ|=%g)\n', ...
                    double(scal), double(deconvmin), ...
                    nerr, maxabs);
            fail = fail + 1;
        end
    end
end

fprintf('\nSUMMARY: %d cases, %d failed\n', total, fail);
end
