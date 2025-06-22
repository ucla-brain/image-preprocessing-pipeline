function load_slab_lz4_fulltest(varargin)
% One-file test-suite for load_slab_lz4 / save_lz4_mex
% - Generates a synthetic volume, bricks it, saves bricks,
%   reloads with load_slab_lz4 under every scale/clip path,
%   compares against a MATLAB reference (float-identical),
%   prints PASS/FAIL and cleans up temporary files.
%
% Usage:  load_slab_lz4_fulltest            % quick
%         load_slab_lz4_fulltest big        % big volume

% ─── volume / brick sizes ────────────────────────────────────────────────
if nargin && strcmpi(varargin{1},'big')
    stack = struct('x',1024,'y',2049,'z',3073); brick = struct('x',256,'y',256,'z',384);
else
    stack = struct('x',128,'y',128,'z',192);    brick = struct('x',32 ,'y',32 ,'z',48 );
end

% ─── sweep parameters ────────────────────────────────────────────────────
scal_opts      = [255 65535];      % uint8 / uint16
clip_opts      = [0   1];          % off / on
deconvmin_opts = [0   0.1];        % off / on (only when clip==0)
low_clip = 0.1; high_clip = 0.9;
amplification = 1;   deconvmax = 1;

% ─── temp directory & synthetic volume ──────────────────────────────────
tmpDir = tempname;  mkdir(tmpDir);
cleanupObj = onCleanup(@() rmdir(tmpDir,'s'));
rng(1);  V = rand([stack.x stack.y stack.z],'single');

% ─── brick index lists and save bricks ───────────────────────────────────
[p1d,p2d] = split(stack,brick);
fnames = cell(size(p1d,1),1);
fprintf("Saving %d bricks …\n",numel(fnames))
for k = 1:numel(fnames)
    blk = V(p1d(k,1):p2d(k,1), p1d(k,2):p2d(k,2), p1d(k,3):p2d(k,3));
    fn  = fullfile(tmpDir,sprintf('blk_%05d.lz4c',k));
    save_lz4_mex(fn,blk);  fnames{k}=fn;
end
fprintf("Done saving\n\n")

% ─── exhaustive sweep ────────────────────────────────────────────────────
total=0; fail=0; t0_all=tic;
for scal=scal_opts
  for clipval=clip_opts
    for deconvmin=deconvmin_opts
      if clipval>0 && deconvmin>0, continue, end
      total=total+1;
      fprintf('Testing: scal=%d, clip=%g, dmin=%g … ',scal,clipval,deconvmin)

      % —— call MEX safely ——
      try
        t_mex=tic;
        V_mex = load_slab_lz4( ...
          fnames,uint64(p1d),uint64(p2d),uint64([stack.x stack.y stack.z]), ...
          single(clipval),single(scal),single(amplification), ...
          single(deconvmin),single(deconvmax), ...
          single(low_clip),single(high_clip),feature('numCores'));
        t_mex = toc(t_mex);
      catch ME
        fprintf("❌  MEX error: %s\n",ME.message)
        fail=fail+1;  continue
      end

      % —— MATLAB reference (single-precision) ——
      V_ref = single(V);  scalS=single(scal); clipS=single(clipval);
      dminS = single(deconvmin); dmaxS=single(deconvmax);
      lowS  = single(low_clip);  highS=single(high_clip);
      ampS  = single(amplification);

      if clipS>0
          rngS = highS-lowS;
          sf   = single(double(scalS*ampS)/double(rngS));
          V_ref = min(max(V_ref-lowS,0),rngS).*sf;
      else
          if dminS>0
              rngS = dmaxS-dminS;
              sf   = single(double(scalS*ampS)/double(rngS));
              V_ref = (V_ref-dminS).*sf;
          else
              sf = single(double(scalS*ampS)/double(dmaxS));
              V_ref = V_ref.*sf;
          end
      end
      V_ref = round(V_ref-ampS);
      V_ref = min(max(V_ref,single(0)),scalS);
      V_ref = (scalS<=255)*uint8(V_ref) + (scalS>255)*uint16(V_ref);

      % —— compare ——
      if isequaln(V_mex,V_ref)
          fprintf("✅  PASS (%.2fs)\n",t_mex)
      else
          fail=fail+1;
          fprintf("❌  FAIL  (#diff=%d, max|Δ|=%d)\n", ...
             nnz(V_mex~=V_ref), ...
             max(abs(double(V_mex(:))-double(V_ref(:)))))
      end
    end
  end
end
fprintf("\nTotal %d cases, %d failed, wall-time %.2fs\n",total,fail,toc(t0_all))
end