%calculates a theoretical point spread function
function [psf, FWHMxy, FWHMz] = LsMakePSF(dxy, dz, NA, nf, lambda_ex, lambda_em, fcyl, slitwidth)
    [nxy, nz, FWHMxy, FWHMz] = DeterminePSFsize(dxy, dz, NA, nf, lambda_ex, lambda_em, fcyl, slitwidth);

    %construct psf
    NAls = sin(atan(slitwidth / (2 * fcyl)));
    psf = samplePSF(dxy, dz, nxy, nz, NA, nf, lambda_ex, lambda_em, NAls);
    % disp('ok');
end

%determine the required grid size (xyz) for psf sampling
function [nxy, nz, FWHMxy, FWHMz] = DeterminePSFsize(dxy, dz, NA, nf, lambda_ex, lambda_em, fcyl, slitwidth)
    %Size of PSF grid is gridsize (xy z) times FWHM
    gridsizeXY = 2;
    gridsizeZ = 2;

    NAls = sin(atan(0.5 * slitwidth / fcyl));
    halfmax = 0.5 .* LsPSFeq(0, 0, 0, NA, nf, lambda_ex, lambda_em, NAls);

    %find zero crossings
    fxy = @(x)LsPSFeq(x, 0, 0, NA, nf, lambda_ex, lambda_em, NAls) - halfmax;
    fz = @(x)LsPSFeq(0, 0, x, NA, nf, lambda_ex, lambda_em, NAls) - halfmax;
    FWHMxy = 2 * abs(fzero(fxy, 100));
    FWHMz = 2 * abs(fzero(fz, 100));

    Rxy = 0.61 * lambda_em / NA;
    dxy_corr = min(dxy, Rxy / 3);

    nxy = ceil(gridsizeXY * FWHMxy / dxy_corr);
    nz = ceil(gridsizeZ * FWHMz / dz);

    %ensure that the grid dimensions are odd
    if mod(nxy, 2) == 0
        nxy = nxy + 1;
    end
    if mod(nz, 2) == 0
        nz = nz + 1;
    end
end

function psf = samplePSF(dxy, dz, nxy, nz, NA_obj, rf, lambda_ex, lambda_em, NA_ls)
	% disp([dxy, dz, nxy, nz, NA_obj, rf, lambda_ex, lambda_em, NA_ls]);
    % fprintf('dxy=%.1f, dz=%.1f, nxy=%.1f, nz=%.1f, NA_obj=%.1f, rf=%.2f, lambda_ex=%.1f, lambda_em=%.1f, NA_ls=%.4f\n', dxy, dz, nxy, nz, NA_obj, rf, lambda_ex, lambda_em, NA_ls);

    if mod(nxy, 2) == 0 || mod(nz, 2) == 0
        error('function samplePSF: nxy and nz must be odd!');
    end

    psf = zeros((nxy - 1) / 2 + 1, (nxy - 1) / 2 + 1, (nz - 1) / 2 + 1, 'single');
    for z = 0 : (nz - 1) / 2
        for y = 0 : (nxy - 1) / 2
            for x = 0 : (nxy - 1) / 2
               psf(x+1, y+1, z+1) = LsPSFeq(x*dxy, y*dxy, z*dz, NA_obj, rf, lambda_ex, lambda_em, NA_ls);
            end
        end
    end

    %Since the PSF is symmetrical around all axes only the first Octand is
    %calculated for computation efficiency. The other 7 Octands are
    %obtained by mirroring around the respective axes
    psf = mirror8(psf);

    %normalize psf to integral one
    psf = psf ./ sum(psf(:));
end

function R = mirror8(p1)
    %mirrors the content of the first quadrant to all other quadrants to
    %obtain the complete PSF.

    sx = 2 * size(p1, 1) - 1; sy = 2 * size(p1, 2) - 1; sz = 2 * size(p1, 3) - 1;
    cx = ceil(sx / 2); cy = ceil(sy / 2); cz = ceil(sz / 2);

    R = zeros(sx, sy, sz, 'single');
    R(cx:sx, cy:sy, cz:sz) = p1;
    R(cx:sx, 1:cy, cz:sz) = flip3D(p1, 0, 1 ,0);
    R(1:cx, 1:cy, cz:sz) = flip3D(p1, 1, 1, 0);
    R(1:cx, cy:sy, cz:sz) = flip3D(p1, 1, 0, 0);
    R(cx:sx, cy:sy, 1:cz) = flip3D(p1, 0, 0, 1);
    R(cx:sx, 1:cy, 1:cz) =  flip3D(p1, 0, 1 ,1);
    R(1:cx, 1:cy, 1:cz) =  flip3D(p1, 1, 1, 1);
    R(1:cx, cy:sy, 1:cz) =  flip3D(p1, 1, 0, 1);
end

%utility function for mirror8
function R = flip3D(data, x, y, z)
    R = data;
    if x
        R = flip(R, 1);
    end
    if y
        R = flip(R, 2);
    end
    if z
        R = flip(R, 3);
    end
end

%calculates PSF at point (x,y,z)
function R = LsPSFeq(x, y, z, NAobj, n, lambda_ex, lambda_em, NAls)
    R = PSF(z, 0, x, NAls, n, lambda_ex) .* PSF(x, y, z, NAobj, n, lambda_em);
end

%utility function for LsPSFeq
function R = PSF(x, y, z, NA, n, lambda)
    f2 = @(p)f1(p, x, y, z, lambda, NA, n);
    f2_integral = integral(f2, 0, 1, 'AbsTol', 1e-3);
    R = 4 .* abs(f2_integral).^2;
end

%utility function for LsPSFeq
function R = f1(p, x, y, z, lambda, NA, n)
    R = besselj(0, 2 .* single(pi) .* NA .* sqrt(x.^2 + y.^2) .* p ./ (lambda .* n))...
        .* exp(1i .* (-single(pi) .* p.^2 .* z .* NA.^2) ./ (lambda .* n.^2)) .* p;
end