function [p1, p2] = split_stack(stack_info, block)
    %% provides coordinates of sub-blocks after splitting

    % bounding box coordinate points
    p1 = zeros(block.nx * block.ny * block.nz, 3);
    p2 = zeros(block.nx * block.ny * block.nz, 3);

    blnr = 0;
    for nz = 0 : block.nz-1
        zs = nz * block.z + 1;
        for ny = 0 : block.ny-1
            ys = ny * block.y + 1;
            for nx = 0 : block.nx-1
                xs = nx * block.x + 1;

                blnr = blnr + 1;
                p1(blnr, 1) = xs;
                p2(blnr, 1) = min([xs + block.x - 1, stack_info.x]);

                p1(blnr, 2) = ys;
                p2(blnr, 2) = min([ys + block.y - 1, stack_info.y]);

                p1(blnr, 3) = zs;
                p2(blnr, 3) = min([zs + block.z - 1, stack_info.z]);
            end
        end
    end
end