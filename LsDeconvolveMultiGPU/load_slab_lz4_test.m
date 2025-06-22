function load_slab_lz4_fulltest(varargin)
% Exhaustive test for save_lz4_mex ↔ load_slab_lz4.
% Creates a temp volume, bricks & saves it, reloads through every
% clip / deconvmin / uint8-uint16 branch, byte-compares to MATLAB.

% ── geometry ────────────────────────────────────────────────────────────
if nargin && strcmpi(varargin{1},'big')
    stack = struct('x',1024,'y',2049,'z',3073);
    brick = struct('x',256 ,'y',256 ,'z',384 );
else
    stack = struct('x',128,'y',128,'z',192);
    brick = struct('x',32 ,'y',32 ,'z',48 );
end

% ── sweep parameters (all SINGLE) ───────────────────────────────────────
scalS  = single([255 65535]);     % uint8 / uint16 paths
clipS  = single([0 1]);           % off / on
dminS  = single([0 0.1]);         % 0 & >0 (used only when clip==0)
lowS   = single(0.1);  highS = single(0.9);
ampS   = single(1);     dmaxS = single(1);

% ── temp dir, synthetic volume, bricks ─────────────────────────────────-
tmpDir = tempname;  mkdir(tmpDir);  c = onCleanup(@() rmdir(tmpDir,'s'));
feval('rng',1);  V = rand([stack.x stack.y stack.z],'single');

[p1d,p2d] = split(stack,brick);          % your existing helper
fn = cell(size(p1d,1),1);
for k = 1:numel(fn)
    blk   = V(p1d(k,1):p2d(k,1), p1d(k,2):p2d(k,2), p1d(k,3):p2d(k,3));
    fn{k} = fullfile(tmpDir,sprintf('blk_%05d.lz4c',k));
    save_lz4_mex(fn{k},blk);
end

% ── exhaustive sweep ────────────────────────────────────────────────────
total=0; fail=0;
for s = scalS
  for cval = clipS
    for dmin = dminS
      total=total+1;
      t = tic;
      try
          V_mex = load_slab_lz4(fn,uint64(p1d),uint64(p2d), ...
                   uint64([stack.x stack.y stack.z]), ...
                   cval,s,ampS,dmin,dmaxS,lowS,highS,feature('numCores'));
      catch ME
          fprintf('❌ MEX error (scal=%g clip=%g dmin=%g): %s\n',s,cval,dmin,ME.message);
          fail=fail+1;   continue
      end
      t_mex = toc(t);

      % reference (single math, double only for scale factor division)
      Vref = V;
      if cval > 0
          rng = highS - lowS;
          sf  = single(double(s*ampS)/double(rng));
          Vref = min(max(Vref-lowS,0),rng).*sf;
      else
          if dmin > 0
              rng = dmaxS - dmin;
              sf  = single(double(s*ampS)/double(rng));
              Vref = (Vref-dmin).*sf;
          else
              sf  = single(double(s*ampS)/double(dmaxS));
              Vref = Vref.*sf;
          end
      end
      Vref = round(Vref-ampS);
      Vref = min(max(Vref,0),s);
      Vref = (s<=255)*uint8(Vref) + (s>255)*uint16(Vref);

      if isequaln(V_mex,Vref)
          fprintf('✅ scal=%g clip=%g dmin=%g (%.2fs)\n',s,cval,dmin,t_mex);
      else
          fail=fail+1;
          fprintf('❌ scal=%g clip=%g dmin=%g  (#diff=%d)\n', ...
              s,cval,dmin,nnz(V_mex~=Vref));
      end
    end
  end
end
fprintf('\nSUMMARY: %d cases, %d failed\n',total,fail)
end
