from __future__ import annotations
import os
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np
import torch
import tensorly as tl
from tensorly.decomposition import parafac, tucker
from tensorly.cp_tensor import cp_to_tensor as kruskal_to_tensor
from tensorly.tucker_tensor import tucker_to_tensor
from sklearn.utils.extmath import randomized_svd
from itertools import cycle
import gc

tl.set_backend("pytorch")

# ------------------------- GPU discovery & memory -------------------------

@dataclass(frozen=True)
class DeviceInfo:
    id: int
    name: str
    total: int   # bytes
    free: int    # bytes

def discover_cuda_devices() -> List[DeviceInfo]:
    if not torch.cuda.is_available():
        return []
    devices: List[DeviceInfo] = []
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            free, total = torch.cuda.mem_get_info()
        name = torch.cuda.get_device_name(i)
        devices.append(DeviceInfo(id=i, name=name, total=total, free=free))
    # Sort by free memory desc so we schedule heavier work first
    devices.sort(key=lambda d: d.free, reverse=True)
    return devices

def refresh_free_mem(dev_id: int) -> Tuple[int, int]:
    # returns (free, total) in bytes
    with torch.cuda.device(dev_id):
        return torch.cuda.mem_get_info()

def free_all_cuda_mem() -> None:
    torch.cuda.empty_cache()
    gc.collect()

# ------------------------- Block planner -------------------------

@dataclass(frozen=True)
class Block:
    z0: int; z1: int
    y0: int; y1: int
    x0: int; x1: int

def _voxels_to_edge(max_voxels: int, shape: Tuple[int,int,int]) -> Tuple[int,int,int]:
    """Heuristic: target (z,y,x) edges so that z*y*x <= max_voxels and respects dims."""
    Z, Y, X = shape
    # Start with a cube-ish edge
    edge = int(round(max_voxels ** (1/3)))
    ez = max(1, min(Z, edge))
    ey = max(1, min(Y, edge))
    ex = max(1, min(X, edge))
    # Adjust down if we overshoot (rare with rounding)
    while ez*ey*ex > max_voxels and (ez>1 or ey>1 or ex>1):
        # shrink the largest dimension first
        if ez >= ey and ez >= ex and ez > 1:
            ez -= 1
        elif ey >= ez and ey >= ex and ey > 1:
            ey -= 1
        elif ex > 1:
            ex -= 1
    return ez, ey, ex

def plan_blocks(shape: Tuple[int,int,int],
                bytes_per_voxel: int,
                devices: List[DeviceInfo],
                safety_factor: float = 0.45,
                work_parallelism: int = 2) -> Tuple[List[Block], Tuple[int,int,int]]:
    """
    Compute block edges based on free VRAM.

    memory model (heuristic, conservative):
      per-block working set ~ bytes_per_voxel * block_voxels * K
    K ~ 6–8 for Tucker/CP with temps; we fold parallelism and a safety margin.
    """
    if not devices:
        # CPU only; keep blocks modest
        target_bytes = 2 * 1024**3  # ~2GB worth of voxels for CPU RAM (tune)
    else:
        # Use the best device (index 0 after sorting) as sizing baseline
        best = devices[0]
        free, _total = refresh_free_mem(best.id)
        target_bytes = int(free * safety_factor)

    # Effective multiplier for temps & intermediates; tuned conservatively.
    K = 2.0
    per_block_bytes_budget = max(target_bytes // work_parallelism, bytes_per_voxel * 8)

    max_voxels = max(int(per_block_bytes_budget // (bytes_per_voxel * K)), 512)
    block_edges = _voxels_to_edge(max_voxels, shape)

    Z, Y, X = shape
    bz, by, bx = block_edges

    blocks: List[Block] = []
    for z0 in range(0, Z, bz):
        z1 = min(z0 + bz, Z)
        for y0 in range(0, Y, by):
            y1 = min(y0 + by, Y)
            for x0 in range(0, X, bx):
                x1 = min(x0 + bx, X)
                blocks.append(Block(z0, z1, y0, y1, x0, x1))

    return blocks, block_edges

# ------------------------- Skip logic -------------------------

def is_trivially_full_or_empty(block: np.ndarray, nan_ratio_thresh: float = 0.002) -> bool:
    # Skip if no NaNs (nothing to complete) or virtually all NaNs (can’t infer)
    nan_ratio = np.isnan(block).mean()
    return (nan_ratio == 0.0) or (nan_ratio >= 1.0 - 1e-9) or (nan_ratio <= nan_ratio_thresh)

def is_low_variance(block: np.ndarray, eps: float = 1e-8) -> bool:
    # Compute variance on observed entries only
    m = ~np.isnan(block)
    if not m.any():
        return True
    v = np.var(block[m])
    return v <= eps

# ------------------------- Rank estimation (3D) -------------------------

def _fill_missing_gpu(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.any():
        mean_val = torch.nanmean(x)
        # torch.nan_to_num is fast and in-place-capable with out=
        return torch.nan_to_num(x, nan=float(mean_val))
    # no observed? just zeros
    return torch.nan_to_num(x, nan=0.0)

def estimate_rank_3d_gpu(block: np.ndarray,
                         energy_threshold: float = 0.95,
                         max_rank: int = 10,
                         device_id: int = 0) -> Tuple[int,int,int]:
    """
    Estimate Tucker ranks (rZ, rY, rX) via low-rank SVD on unfoldings, on GPU if possible.
    Falls back to CPU randomized SVD. Caps ranks by max_rank and dimension.
    """
    Z, Y, X = block.shape
    rcap = max_rank

    try:
        device = torch.device(f"cuda:{device_id}")
        xt = torch.tensor(block, dtype=torch.float32, device=device)
        mask = ~torch.isnan(xt)
        xt = _fill_missing_gpu(xt, mask)

        def rank_from_svd(shape_mat: torch.Tensor, dim_cap: int) -> int:
            q = int(min(rcap, min(shape_mat.shape)))
            # svd_lowrank may OOM; let it bubble to outer handler
            U, S, V = torch.svd_lowrank(shape_mat, q=q)
            energy = torch.cumsum(S**2, dim=0) / (torch.sum(S**2) + 1e-12)
            r = int(torch.searchsorted(energy, torch.tensor(energy_threshold, device=device)).item() + 1)
            return max(1, min(r, q, dim_cap))

        # mode-0 unfolding: (Z, Y*X)
        rZ = rank_from_svd(xt.reshape(Z, Y*X), Z)
        # mode-1 unfolding: (Y, Z*X)
        rY = rank_from_svd(xt.permute(1,0,2).reshape(Y, Z*X), Y)
        # mode-2 unfolding: (X, Z*Y)
        rX = rank_from_svd(xt.permute(2,0,1).reshape(X, Z*Y), X)

        # Cleanup
        del xt, mask
        torch.cuda.empty_cache()
        return rZ, rY, rX

    except torch.cuda.OutOfMemoryError as e:
        # Free aggressively and fall back to CPU
        try:
            del xt, mask  # type: ignore
        except Exception:
            pass
        torch.cuda.empty_cache()
        # Fall back CPU randomized SVD
    except Exception:
        # Other GPU errors -> fallback CPU
        try:
            del xt, mask  # type: ignore
        except Exception:
            pass
        torch.cuda.empty_cache()

    # CPU fallback with randomized SVD
    arr = np.nan_to_num(block, nan=np.nanmean(block) if np.isnan(block).sum() else 0.0)
    def cpu_rank_from_svd(mat: np.ndarray, dim_cap: int) -> int:
        q = int(min(rcap, min(mat.shape)))
        U, S, Vt = randomized_svd(mat, n_components=q)
        energy = np.cumsum(S**2) / (S**2).sum()
        r = int(np.searchsorted(energy, energy_threshold) + 1)
        return max(1, min(r, q, dim_cap))

    rZ = cpu_rank_from_svd(arr.reshape(Z, Y*X), Z)
    rY = cpu_rank_from_svd(arr.transpose(1,0,2).reshape(Y, Z*X), Y)
    rX = cpu_rank_from_svd(arr.transpose(2,0,1).reshape(X, Z*Y), X)
    return rZ, rY, rX

# ------------------------- 3D completion (CP/Tucker) -------------------------

def complete_block_3d(block: np.ndarray,
                      method: str,
                      ranks: Tuple[int,int,int],
                      device_id: Optional[int]) -> np.ndarray:
    """
    Complete a 3D block with CP or Tucker on GPU if device_id is not None.
    Robust to OOM: retries with smaller ranks, then CPU fallback.
    """
    rZ, rY, rX = ranks
    r_min = max(1, min(rZ, rY, rX))

    def _do(block_np: np.ndarray, dev: Optional[int], rZ: int, rY: int, rX: int) -> np.ndarray:
        device = torch.device(f"cuda:{dev}") if dev is not None else torch.device("cpu")
        xt = torch.tensor(block_np, dtype=torch.float32, device=device)
        mask = ~torch.isnan(xt)
        xt = _fill_missing_gpu(xt, mask) if device.type == "cuda" else torch.tensor(
            np.nan_to_num(block_np, nan=np.nanmean(block_np) if np.isnan(block_np).sum() else 0.0),
            dtype=torch.float32, device=device
        )

        if method == "cp":
            # CP rank uses a single scalar r
            r = max(1, min(r_min, min(block_np.shape)))
            # tensorly expects dense; factors = (weights, [A,B,C])
            factors = parafac(xt, rank=r)
            rec = kruskal_to_tensor(factors)
        elif method == "tucker":
            # Use provided per-mode ranks
            rZc = max(1, min(rZ, block_np.shape[0]))
            rYc = max(1, min(rY, block_np.shape[1]))
            rXc = max(1, min(rX, block_np.shape[2]))
            core, factors = tucker(xt, rank=(rZc, rYc, rXc))
            rec = tucker_to_tensor((core, factors))
        else:
            raise ValueError(f"Unknown method '{method}'")

        # Keep observed values; fill only missing (avoid needless copies)
        out = torch.where(mask, xt, rec)
        result = out.detach().cpu().numpy()

        # Cleanup
        del xt, mask, rec
        try:
            del factors
        except Exception:
            pass
        try:
            del core
        except Exception:
            pass
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        return result

    # Try on GPU (if provided), shrinking rank on OOM
    shrink_steps = [1.0, 0.75, 0.5, 0.35]
    if device_id is not None:
        for s in shrink_steps:
            try:
                rz = max(1, int(rZ * s))
                ry = max(1, int(rY * s))
                rx = max(1, int(rX * s))
                return _do(block, device_id, rz, ry, rx)
            except torch.cuda.OutOfMemoryError:
                # free and try smaller ranks
                torch.cuda.empty_cache()
            except Exception:
                # Any other GPU error -> break to CPU fallback
                break

    # CPU fallback
    return _do(block, None, rZ, rY, rX)

# ------------------------- Public API -------------------------

def complete_volume_3d_multi_gpu(
    volume: np.ndarray,
    method: str = "tucker",            # "tucker" or "cp"
    energy_threshold: float = 0.85,
    max_rank: int = 10,                # hard cap per your note
    skip_nan_ratio_leq: float = 0.002, # skip blocks with ~no missing
    skip_low_variance: bool = True,
    workers_per_device: int = 1,
    verbose: bool = True,
) -> np.ndarray:
    """
    Blockwise 3D completion with automatic multi-GPU scheduling and VRAM-aware chunking.

    volume: [Z, Y, X] with NaNs as missing values.
    returns: completed volume, same shape.
    """
    assert volume.ndim == 3, "volume must be 3D [Z,Y,X]"
    Z, Y, X = volume.shape
    out = np.array(volume, copy=True)  # we'll write blocks back in place

    # Discover devices and plan blocks
    devices = discover_cuda_devices()
    if verbose:
        if devices:
            print("CUDA devices (sorted by free VRAM):")
            for d in devices:
                print(f"  cuda:{d.id} - {d.name} | free={d.free/1e9:.2f} GB / total={d.total/1e9:.2f} GB")
        else:
            print("No CUDA devices detected; running on CPU.")

    bytes_per_voxel = 4  # float32
    blocks, (bz, by, bx) = plan_blocks((Z, Y, X), bytes_per_voxel, devices, safety_factor=0.45, work_parallelism=max(1, len(devices)*workers_per_device))
    if verbose:
        print(f"Planned {len(blocks)} blocks with target block size ~ ({bz},{by},{bx}).")

    # Prepare a device iterator for round-robin scheduling
    dev_iter = cycle([d.id for d in devices] if devices else [None])

    # Helper for per-block processing
    def process_block(b: Block, device_id_opt: Optional[int]) -> Tuple[Block, np.ndarray]:
        z0,z1,y0,y1,x0,x1 = b.z0,b.z1,b.y0,b.y1,b.x0,b.x1
        blk = out[z0:z1, y0:y1, x0:x1]  # view

        # Skip heuristics
        if is_trivially_full_or_empty(blk, nan_ratio_thresh=skip_nan_ratio_leq):
            return b, blk
        if is_uniform_3d(blk):  # user-provided
            return b, blk
        if skip_low_variance and is_low_variance(blk):
            return b, blk

        # Rank estimation (GPU first, CPU fallback inside)
        dev_for_rank = device_id_opt if device_id_opt is not None else (devices[0].id if devices else None)
        dev_for_rank = dev_for_rank if dev_for_rank is not None else 0  # dummy for signature
        rZ, rY, rX = estimate_rank_3d_gpu(blk, energy_threshold=energy_threshold,
                                          max_rank=max_rank, device_id=dev_for_rank)

        # Completion
        completed = complete_block_3d(blk, method=method, ranks=(rZ, rY, rX),
                                      device_id=device_id_opt)

        return b, completed

    # Scheduler loop
    max_workers = min(len(blocks), max(1, len(devices) * workers_per_device))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_block = {}
        for b in blocks:
            dev = next(dev_iter)
            fut = ex.submit(process_block, b, dev)
            future_to_block[fut] = b

        completed_cnt = 0
        total = len(blocks)
        for fut in as_completed(future_to_block):
            b = future_to_block[fut]
            try:
                _b, blk_completed = fut.result()
                out[b.z0:b.z1, b.y0:b.y1, b.x0:b.x1] = blk_completed
                completed_cnt += 1
                if verbose and (completed_cnt % 25 == 0 or completed_cnt == total):
                    print(f"Completed {completed_cnt}/{total} blocks...")
            except Exception as e:
                if verbose:
                    print(
                        f"Block z[{b.z0}:{b.z1}] y[{b.y0}:{b.y1}] x[{b.x0}:{b.x1}] "
                        f"failed: {e.__class__.__name__}: {e}"
                    )
                continue

            # Write back in place
            out[b.z0:b.z1, b.y0:b.y1, b.x0:b.x1] = blk_completed
            completed_cnt += 1
            if verbose and (completed_cnt % 25 == 0):
                print(f"Completed {completed_cnt}/{len(blocks)} blocks...")

    # Force VRAM cleanup before returning
    if devices:
        free_all_cuda_mem()

    return out