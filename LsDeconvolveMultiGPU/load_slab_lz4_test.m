function load_slab_lz4_fulltest(varargin)
% Comprehensive test for load_slab_lz4: covers all scaling/clip/cast logic paths.
% All temporary files are deleted on exit or error.

if nargin && strcmpi(varargin{1},'big')
    stack = struct('x',1024,'y',2049,'z',3073);  brick = struct('x',256,'y',256,'z',384);
else
    stack = struct('x',128,'y',128,'z',192);     brick = struct('x',32 ,'y',32 ,'z',48);
end
brick.nx = ceil(stack.x/brick.x);
brick.ny = ceil(stack.y/brick.y);
brick.nz = ceil(stack.z/brick.z);

% --- Parameters to sweep ---
scal_options      = [255, 65535];
clipval_options   = [0, 1];           % 0 = no clip, 1 = with clip
deconvmin_options = [0, 0.1];         % test with and without deconvmin > 0
low_clip          = 0.1;
high_clip         = 0.9;
amplification     = 1;
deconvmax         = 1;

% ─── TEMP DIR ───────────────────────────────────────────────────────────
tmpDir  = tempname;  mkdir(tmpDir);
cleanup = onCleanup(@() mycleanup(tmpDir)); % Robust cleanup

% ─── SYNTHETIC VOLUME ──────────────────────────────────────────────────
rng(1);  V = rand([stack.x stack.y stack.z],'single');

% ─── SPLIT & SAVE BRICKS ───────────────────────────────────────────────
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
fprintf("Done saving\n\n");

all_pass = true; total_cases = 0; failed_cases = 0;

for scal = scal_options
    for clipval = clipval_options
        for deconvmin = deconvmin_options
            if clipval > 0 && deconvmin > 0
                continue
            end

            fprintf("Testing: scal=%d, clipval=%g, deconvmin=%g\n", scal, clipval, deconvmin);

            tic;
            V_mex = load_slab_lz4( ...
                fnames, uint64(p1d), uint64(p2d), uint64([stack.x stack.y stack.z]), ...
                clipval, scal, amplification, deconvmin, deconvmax, low_clip, high_clip, feature('numCores'));
            t_mex = toc;

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
            V_ref = single(V_ref);    % <-- Add this line
            V_ref = round(V_ref - amplification);
            V_ref = min(max(V_ref, 0), scal); % clamp

            if scal <= 255
                V_ref = uint8(V_ref);
            elseif scal <= 65535
                V_ref = uint16(V_ref);
            end

            total_cases = total_cases + 1;
            eq = isequaln(V_mex, V_ref);

            if eq
                fprintf("   ✅  PASS   |  t=%.2fs\n", t_mex);
            else
                all_pass = false; failed_cases = failed_cases + 1;
                dif = nnz(V_mex ~= V_ref);
                maxabs = double(max(abs(double(V_mex(:)) - double(V_ref(:)))));
                fprintf("   ❌  FAIL   |  #diff=%d, maxabs=%.4g\n", dif, maxabs);
            end
        end
    end
end

fprintf("\n-------------------------------------------------\n");
if all_pass
    fprintf("🎉 All %d test cases PASSED.\n", total_cases);
else
    fprintf("❌ %d of %d test cases FAILED.\n", failed_cases, total_cases);
end

end

function mycleanup(d)
    if exist(d, 'dir')
        fprintf('[cleanup] Removing tempdir %s\n', d);
        rmdir(d, 's');
    end
end
