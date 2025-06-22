function load_slab_lz4_fulltest(varargin)
% Exhaustive test for save_lz4_mex ↔ load_slab_lz4.
% Verifies all scaling + clipping paths using temporary LZ4 brick files.
%
% The test FAILS immediately if either the MEX result or the reference
% result is not single precision.

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
scalS  = single([255 65535]);   clipS = single([0 1]);
dminS  = single([0   0.1]);
lowS   = single(0.1);  highS = single(0.9);
ampS   = single(1);    dmaxS = single(1);

% ── temp dir, synthetic volume, bricks ──────────────────────────────────
tmpDir = tempname;  mkdir(tmpDir);
cleanup = onCleanup(@() rmdir(tmpDir,'s'));
feval('rng',1);  V = rand([stack.x stack.y stack.z],'single');

[p1d,p2d] = split(stack,brick);          % helper must use brick.nx/ny/nz
nBricks   = size(p1d,1);  fn = cell(nBricks,1);
for k = 1:nBricks
    blk   = V(p1d(k,1):p2d(k,1), p1d(k,2):p2d(k,2), p1d(k,3):p2d(k,3));
    fn{k} = fullfile(tmpDir,sprintf('blk_%05d.lz4c',k));
    save_lz4_mex(fn{k}, blk);
end

% ── exhaustive sweep ────────────────────────────────────────────────────
total = 0; fail = 0;
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
          fprintf('❌ MEX error (scal=%g clip=%g dmin=%g): %s\n', ...
                  double(scal), double(clipval), double(deconvmin), ME.message);
          fail = fail + 1;  continue
      end

      % ── build MATLAB reference (single math) ────────────────────────────
      Vref = V;
      if clipval > 0
          rng = highS - lowS;
          sf  = scal * ampS / rng;
          Vref = min(max(Vref - lowS, 0), rng) .* sf;
      else
          if deconvmin > 0
              rng = dmaxS - deconvmin;
              sf  = scal * ampS / rng;
              Vref = (Vref - deconvmin) .* sf;
          else
              sf  = scal * ampS / dmaxS;
              Vref = Vref .* sf;
          end
      end
      Vref = round(Vref - ampS);
      Vref = min(max(Vref, 0), scal);
      Vref = (scal <= 255) * uint8(Vref) + (scal > 255) * uint16(Vref);

      % ── type check ─────────────────────────────────────────────────────
      assert(isa(V_mex,'single'), 'V_mex is not single precision');
      assert(isa(Vref,'single'), 'Vref is not single precision');

      % ── comparison (no auto-casting) ───────────────────────────────────
      nerr   = nnz(V_mex ~= Vref);
      maxabs = max(abs(V_mex(:) - Vref(:)));

      if maxabs <= 1          % accept ±1 LSB
          fprintf('✅ scal=%g clip=%g dmin=%g (%.2fs  #diff=%d max|Δ|=%g)\n', ...
                  double(scal), double(clipval), double(deconvmin), ...
                  t_mex, nerr, maxabs);
      else
          fail = fail + 1;
          fprintf('❌ scal=%g clip=%g dmin=%g  (#diff=%d max|Δ|=%g)\n', ...
                  double(scal), double(clipval), double(deconvmin), ...
                  nerr, maxabs);
      end
    end
  end
end

fprintf('\nSUMMARY: %d cases, %d failed\n', total, fail);
end
