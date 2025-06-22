function load_slab_lz4_fulltest(varargin)
% Exhaustive test for save_lz4_mex ↔ load_slab_lz4.
% Verifies all scaling + clipping paths using temporary LZ4 brick files.

% ── geometry ────────────────────────────────────────────────────────────
if nargin && strcmpi(varargin{1},'big')
    stack = struct('x',1024,'y',2049,'z',3073);
    brick = struct('x',256 ,'y',256 ,'z',384 );
else
    stack = struct('x',128,'y',128,'z',192);
    brick = struct('x',32 ,'y',32 ,'z',48 );
end
brick.nx = ceil(stack.x / brick.x);
brick.ny = ceil(stack.y / brick.y);
brick.nz = ceil(stack.z / brick.z);

% ── sweep parameters (all SINGLE) ───────────────────────────────────────
scalS  = single([255 65535]);     % uint8 / uint16 paths
clipS  = single([0 1]);           % off / on
dminS  = single([0 0.1]);         % 0 & >0 (used only when clip==0)
lowS   = single(0.1);  highS = single(0.9);
ampS   = single(1);     dmaxS = single(1);

% ── temp dir, synthetic volume, bricks ──────────────────────────────────
tmpDir = tempname;  mkdir(tmpDir);  cleanup = onCleanup(@() rmdir(tmpDir,'s'));
feval('rng',1);  V = rand([stack.x stack.y stack.z],'single');

[p1d,p2d] = split(stack,brick);   % your helper, requires .nx/.ny/.nz in brick
nBricks = size(p1d,1);  fn = cell(nBricks,1);
for k = 1:nBricks
    blk   = V(p1d(k,1):p2d(k,1), p1d(k,2):p2d(k,2), p1d(k,3):p2d(k,3));
    fn{k} = fullfile(tmpDir,sprintf('blk_%05d.lz4c',k));
    save_lz4_mex(fn{k}, blk);
end

% ── exhaustive sweep ────────────────────────────────────────────────────
total=0; fail=0;
for scal = scalS
  for clipval = clipS
    for deconvmin = dminS
      total = total + 1;
      t0 = tic;
      try
          V_mex = load_slab_lz4( ...
              fn, uint64(p1d), uint64(p2d), uint64([stack.x stack.y stack.z]), ...
              clipval, scal, ampS, deconvmin, dmaxS, lowS, highS, feature('numCores'));
          t_mex = toc(t0);
      catch ME
          fprintf('❌ MEX error (scal=%g clip=%g dmin=%g): %s\n', scal, clipval, deconvmin, ME.message);
          fail = fail + 1; continue
      end

      % ── reference (same math as in MEX) ───────────────────────────────
      Vref = V;
      if clipval > 0
          rng = highS - lowS;
          sf  = single(double(scal * ampS) / double(rng));
          Vref = min(max(Vref - lowS, 0), rng) .* sf;
      else
          if deconvmin > 0
              rng = dmaxS - deconvmin;
              sf  = single(double(scal * ampS) / double(rng));
              Vref = (Vref - deconvmin) .* sf;
          else
              sf  = single(double(scal * ampS) / double(dmaxS));
              Vref = Vref .* sf;
          end
      end
      Vref = round(Vref - ampS);
      Vref = min(max(Vref, 0), scal);
      if scal <= 255
          Vref = uint8(Vref);
      else
          Vref = uint16(Vref);
      end

      % ── compare ───────────────────────────────────────────────────────
      if isequaln(V_mex, Vref)
          fprintf('✅ scal=%g clip=%g dmin=%g (%.2fs)\n', scal, clipval, deconvmin, t_mex);
      else
          fail = fail + 1;
          nerr = nnz(V_mex ~= Vref);
          maxabs = max(abs(double(V_mex(:)) - double(Vref(:))));
          fprintf('❌ scal=%g clip=%g dmin=%g  (#diff=%d max|Δ|=%.4g)\n', ...
              scal, clipval, deconvmin, nerr, maxabs);
      end
    end
  end
end

fprintf('\nSUMMARY: %d cases, %d failed\n', total, fail);
end
