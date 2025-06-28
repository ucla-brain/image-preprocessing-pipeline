function load_slab_lz4_fulltest(varargin)
% Exhaustive test-bench for save_lz4_mex ↔ load_slab_lz4.
% Generates a random stack, splits into LZ4 bricks, reloads with every
% clip / scale branch, and compares against a MATLAB reference
% that uses *exactly* the user-supplied formula.

% ─────────── geometry ───────────
if nargin && strcmpi(varargin{1},'big')
    stack = struct('x',1024,'y',2049,'z',3073);
    brick = struct('x',256 ,'y',256 ,'z',384 );
else
    stack = struct('x',128,'y',128,'z',192);
    brick = struct('x',32 ,'y',32 ,'z',48 );
end
brick.nx = ceil(stack.x/brick.x);
brick.ny = ceil(stack.y/brick.y);
brick.nz = ceil(stack.z/brick.z);

% ─────────── sweep parameters (single) ───────────
scalS = single([255 65535]);          % → uint8 / uint16
clipS = single([0 1]);                % clip off / on
dminS = single([0 0.1]);              % deconvmin
low_clip  = single(0.1);
high_clip = single(0.9);
amplification = single(1);
deconvmax = single(1);

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

% ─────────── exhaustive sweep ───────────
total = 0;  fail = 0;
for scal = scalS
    for clipval = clipS
        for deconvmin = dminS
            total = total + 1;

            % ---- call MEX ----
            try
                [V_mex, t_mex] = load_slab_lz4( ...
                    fn, uint64(p1d), uint64(p2d), ...
                    uint64([stack.x stack.y stack.z]), ...
                    clipval, scal, amplification, ...
                    deconvmin, deconvmax, low_clip, high_clip, ...
                    feature('numCores'));
            catch ME
                fprintf('❌  MEX error (scal=%g clip=%g dmin=%g): %s\n', ...
                        double(scal), double(clipval), double(deconvmin), ME.message);
                fail = fail + 1;  continue
            end

            % ---- MATLAB reference (uses EXACT formula) ----
            R = R0;                                   %#ok<NASGU> base name must be R
            if clipval > 0
                R = R - low_clip;
                R = min(R, high_clip - low_clip);
                R = R .* (scal .* amplification ./ (high_clip - low_clip));
            else
                if deconvmin > 0
                    R = (R - deconvmin) .* (scal .* amplification ./ ...
                                              (deconvmax - deconvmin));
                else
                    R = R .* (scal .* amplification ./ deconvmax);
                end
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
                fprintf('✅ scal=%g clip=%g dmin=%g (%.2fs  #diff=%d max|Δ|=%g)\n', ...
                        double(scal), double(clipval), double(deconvmin), ...
                        t_mex, nerr, maxabs);
            else
                fprintf('❌ scal=%g clip=%g dmin=%g  (#diff=%d max|Δ|=%g)\n', ...
                        double(scal), double(clipval), double(deconvmin), ...
                        nerr, maxabs);
                fail = fail + 1;
            end
        end
    end
end

fprintf('\nSUMMARY: %d cases, %d failed\n', total, fail);
end
