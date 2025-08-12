# matrix_completion_3d.py
# Robust, multi‑GPU, VRAM‑aware 3D matrix completion (CP/Tucker) with careful cleanup.

from __future__ import annotations

import gc
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import cycle
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
import torch
import tensorly as tl
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.utils.extmath import randomized_svd
from tensorly.cp_tensor import cp_to_tensor as kruskal_to_tensor
from tensorly.decomposition import parafac, tucker
from tensorly.tucker_tensor import tucker_to_tensor

from pystripe.core import is_uniform_3d

# Some IDEs warn on set_backend due to dynamic dispatch; it is callable at runtime.
tl.set_backend("pytorch")  # noqa: FBT003

LOGGER = logging.getLogger(__name__)


# ------------------------- GPU discovery & memory -------------------------

@dataclass(frozen=True)
class DeviceInfo:
    device_id: int
    name: str
    total_bytes: int
    free_bytes: int


def discover_cuda_devices() -> List[DeviceInfo]:
    if not torch.cuda.is_available():
        return []
    devices: List[DeviceInfo] = []
    for idx in range(torch.cuda.device_count()):
        with torch.cuda.device(idx):
            free, total = torch.cuda.mem_get_info()
        devices.append(
            DeviceInfo(
                device_id=idx,
                name=torch.cuda.get_device_name(idx),
                total_bytes=total,
                free_bytes=free,
            )
        )
    devices.sort(key=lambda d: d.free_bytes, reverse=True)
    return devices


def refresh_free_mem(device_id: int) -> Tuple[int, int]:
    with torch.cuda.device(device_id):
        free, total = torch.cuda.mem_get_info()
    return free, total


def free_all_cuda_mem() -> None:
    torch.cuda.empty_cache()
    gc.collect()


# ------------------------- Block planner -------------------------

@dataclass(frozen=True)
class Block:
    z0: int
    z1: int
    y0: int
    y1: int
    x0: int
    x1: int


def _voxels_to_edge(max_voxels: int, volume_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Choose (z,y,x) edges so z*y*x <= max_voxels with a near-cubic footprint."""
    z_dim, y_dim, x_dim = volume_shape
    edge = max(1, int(round(max_voxels ** (1 / 3))))
    ez, ey, ex = min(z_dim, edge), min(y_dim, edge), min(x_dim, edge)
    while ez * ey * ex > max_voxels and (ez > 1 or ey > 1 or ex > 1):
        if ez >= ey and ez >= ex and ez > 1:
            ez -= 1
        elif ey >= ez and ey >= ex and ey > 1:
            ey -= 1
        elif ex > 1:
            ex -= 1
    return max(1, ez), max(1, ey), max(1, ex)


def plan_blocks(
    volume_shape: Tuple[int, int, int],
    bytes_per_voxel: int,
    devices: List[DeviceInfo],
    safety_factor: float = 0.45,
    work_parallelism: int = 2,
) -> Tuple[List[Block], Tuple[int, int, int]]:
    """
    Compute non-overlapping blocks based on available VRAM.
    Heuristic: per-block working set ~= bytes_per_voxel * block_voxels * K
    K is conservative due to intermediates; combined with safety_factor to avoid OOM.
    """
    if devices:
        baseline_device = devices[0]
        free_bytes, _ = refresh_free_mem(baseline_device.device_id)
        target_bytes = int(free_bytes * safety_factor)
    else:
        # CPU-only fallback budget (RAM). Tune if needed.
        target_bytes = 2 * 1024**3  # ~2GB

    # Effective multiplier for intermediates. Conservative.
    k_mult = 2.0
    per_block_budget = max(target_bytes // max(1, work_parallelism), bytes_per_voxel * 8)
    max_voxels = max(int(per_block_budget // (bytes_per_voxel * k_mult)), 512)

    bz, by, bx = _voxels_to_edge(max_voxels, volume_shape)
    z_dim, y_dim, x_dim = volume_shape

    blocks: List[Block] = []
    for z0 in range(0, z_dim, bz):
        z1 = min(z0 + bz, z_dim)
        for y0 in range(0, y_dim, by):
            y1 = min(y0 + by, y_dim)
            for x0 in range(0, x_dim, bx):
                x1 = min(x0 + bx, x_dim)
                blocks.append(Block(z0, z1, y0, y1, x0, x1))

    return blocks, (bz, by, bx)


# ------------------------- Skip logic -------------------------

def is_trivially_full_or_empty(block: np.ndarray, nan_ratio_threshold: float = 0.002) -> bool:
    """Skip if block has no NaNs (nothing to complete), ~no observed data, or trivially small NaN ratio."""
    nan_ratio = float(np.isnan(block).mean())
    return (nan_ratio == 0.0) or (nan_ratio >= 1.0 - 1e-9) or (nan_ratio <= nan_ratio_threshold)


def is_low_variance(block: np.ndarray, eps: float = 1e-8) -> bool:
    observed = ~np.isnan(block)
    if not np.any(observed):
        return True
    return float(np.var(block[observed])) <= eps


# ------------------------- Rank estimation (3D) -------------------------

def _fill_missing_gpu_inplace(x_tensor: torch.Tensor) -> None:
    """Fill NaNs with the observed mean on GPU, in-place via out=."""
    has_data = bool(torch.any(~torch.isnan(x_tensor)).item())
    mean_val = float(torch.nanmean(x_tensor).item()) if has_data else 0.0
    torch.nan_to_num(x_tensor, nan=mean_val, posinf=None, neginf=None, out=x_tensor)


def estimate_ranks_3d_gpu(
    block: np.ndarray,
    energy_threshold: float = 0.95,
    max_rank: int = 10,
    device_id: int = 0,
) -> Tuple[int, int, int]:
    """
    Estimate Tucker ranks (rank_z, rank_y, rank_x) using torch.svd_lowrank on each unfolding.
    Falls back to CPU randomized SVD on OOM or if GPU path fails. Caps each rank by (max_rank, mode_dim).
    """
    z_dim, y_dim, x_dim = block.shape
    hard_cap = int(max_rank)

    x_tensor: Optional[torch.Tensor] = None
    try:
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device_id)

        x_tensor = torch.tensor(block, dtype=torch.float32, device=device)
        _fill_missing_gpu_inplace(x_tensor)

        def rank_from_unfold(mat_2d: torch.Tensor, dim_cap: int) -> int:
            q = int(max(1, min(hard_cap, min(mat_2d.shape))))
            if hasattr(torch, "svd_lowrank"):
                u, s, v = torch.svd_lowrank(mat_2d, q=q)  # s: (q,)
            else:
                u, s, v = torch.linalg.svd(mat_2d, full_matrices=False)  # fallback
                s = s[:q]
            cumulative = torch.cumsum(s**2, dim=0)
            denom = torch.sum(s**2) + 1e-12
            energy = cumulative / denom
            cutoff = torch.tensor(energy_threshold, device=mat_2d.device, dtype=energy.dtype)
            idx = int(torch.searchsorted(energy, cutoff).item()) + 1
            return max(1, min(idx, q, dim_cap))

        rank_z = rank_from_unfold(x_tensor.reshape(z_dim, y_dim * x_dim), z_dim)
        rank_y = rank_from_unfold(x_tensor.permute(1, 0, 2).reshape(y_dim, z_dim * x_dim), y_dim)
        rank_x = rank_from_unfold(x_tensor.permute(2, 0, 1).reshape(x_dim, z_dim * y_dim), x_dim)

        return rank_z, rank_y, rank_x

    except torch.cuda.OutOfMemoryError as exc:
        LOGGER.warning("GPU OOM during rank estimation on cuda:%d; falling back to CPU. %s", device_id, exc)
    except RuntimeError as exc:
        LOGGER.warning("GPU RuntimeError during rank estimation on cuda:%d; falling back to CPU. %s", device_id, exc)
    except Exception as exc:
        LOGGER.exception("Unexpected GPU error during rank estimation on cuda:%d; falling back to CPU. %s", device_id, exc)
    finally:
        if x_tensor is not None:
            del x_tensor
        torch.cuda.empty_cache()
        gc.collect()

    # CPU fallback (randomized SVD)
    arr = block.copy()
    if np.isnan(arr).any():
        observed = ~np.isnan(arr)
        if observed.any():
            mean_val = float(np.nanmean(arr))
            arr = np.where(observed, arr, mean_val)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)

    def cpu_rank_from_unfold(mat_2d: np.ndarray, dim_cap: int) -> int:
        q = int(max(1, min(hard_cap, min(mat_2d.shape))))
        u, s, vt = randomized_svd(mat_2d, n_components=q)
        cumulative = np.cumsum(s**2)
        denom = float((s**2).sum() + 1e-12)
        energy = cumulative / denom
        idx = int(np.searchsorted(energy, energy_threshold) + 1)
        return max(1, min(idx, q, dim_cap))

    rank_z = cpu_rank_from_unfold(arr.reshape(z_dim, y_dim * x_dim), z_dim)
    rank_y = cpu_rank_from_unfold(arr.transpose(1, 0, 2).reshape(y_dim, z_dim * x_dim), y_dim)
    rank_x = cpu_rank_from_unfold(arr.transpose(2, 0, 1).reshape(x_dim, z_dim * y_dim), x_dim)
    return rank_z, rank_y, rank_x


# ------------------------- 3D completion (CP/Tucker) -------------------------

def complete_block_3d(
    block: np.ndarray,
    method: str,
    ranks_zyx: Tuple[int, int, int],
    device_id: Optional[int],
) -> np.ndarray:
    """
    Complete a 3D block with CP or Tucker on GPU if device_id is not None.
    Robust to OOM: retries with smaller ranks, then CPU fallback. In-place style ops where possible.
    """
    rank_z, rank_y, rank_x = ranks_zyx

    def finish_tensor(mask_t: torch.Tensor, observed_t: torch.Tensor, rec_t: torch.Tensor) -> np.ndarray:
        out_t = torch.where(mask_t, observed_t, rec_t)
        result_np = out_t.detach().cpu().numpy()
        # cleanup
        del out_t
        torch.cuda.empty_cache()
        gc.collect()
        return result_np

    def complete_once(block_np: np.ndarray, dev: Optional[int], rz: int, ry: int, rx: int) -> np.ndarray:
        use_gpu = dev is not None and torch.cuda.is_available()
        device = torch.device(f"cuda:{dev}") if use_gpu else torch.device("cpu")
        if use_gpu:
            torch.cuda.set_device(int(dev))
        with (torch.cuda.device(int(dev)) if use_gpu else nullcontext()):
            x_t = torch.tensor(block_np, dtype=torch.float32, device=device)
            mask_t = ~torch.isnan(x_t)
            if use_gpu:
                _fill_missing_gpu_inplace(x_t)
            else:
                has_obs = bool(torch.any(~torch.isnan(x_t)).item())
                mean_val = float(torch.nanmean(x_t).item()) if has_obs else 0.0
                torch.nan_to_num(x_t, nan=mean_val, posinf=None, neginf=None, out=x_t)

            if method.lower() == "cp":
                # Single scalar CP rank from per-mode ranks (simple heuristic: mean)
                cp_rank = max(1, min(int(round((rz + ry + rx) / 3)), min(block_np.shape)))
                factors = parafac(x_t, rank=cp_rank)
                rec_t = kruskal_to_tensor(factors)
                try:
                    del factors
                except Exception:
                    pass

            elif method.lower() == "tucker":
                # Per-mode ranks; tensorly.tucker uses argument name 'rank'
                rz_c = max(1, min(rz, block_np.shape[0]))
                ry_c = max(1, min(ry, block_np.shape[1]))
                rx_c = max(1, min(rx, block_np.shape[2]))
                core_t, factors = tucker(x_t, rank=(rz_c, ry_c, rx_c))
                rec_t = tucker_to_tensor((core_t, factors))
                try:
                    del factors, core_t
                except Exception:
                    pass

            else:
                raise ValueError(f"Unknown method '{method}'. Use 'tucker' or 'cp'.")

            result_np = finish_tensor(mask_t, x_t, rec_t)

            # cleanup
            del x_t, mask_t, rec_t
            if use_gpu:
                torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            gc.collect()
            return result_np

    # Try on GPU first, shrinking ranks on OOMs
    shrink_factors: Iterable[float] = (1.0, 0.75, 0.5, 0.35)
    if device_id is not None:
        for shrink in shrink_factors:
            try:
                rz_try = max(1, int(rank_z * shrink))
                ry_try = max(1, int(rank_y * shrink))
                rx_try = max(1, int(rank_x * shrink))
                return complete_once(block, device_id, rz_try, ry_try, rx_try)
            except torch.cuda.OutOfMemoryError as exc:
                LOGGER.warning("GPU OOM on cuda:%s (shrink=%.2f): %s", device_id, shrink, exc)
                torch.cuda.empty_cache()
                gc.collect()
            except RuntimeError as exc:
                LOGGER.warning("GPU RuntimeError on cuda:%s (shrink=%.2f): %s", device_id, shrink, exc)
                torch.cuda.empty_cache()
                gc.collect()
            except Exception:
                LOGGER.exception("GPU completion failed on cuda:%s; falling back to CPU.", device_id)
                break  # fall back to CPU

    # CPU fallback
    return complete_once(block, None, rank_z, rank_y, rank_x)


# ------------------------- Public API -------------------------

def complete_volume_3d_multi_gpu(
    volume: np.ndarray,
    method: str = "tucker",             # "tucker" or "cp"
    energy_threshold: float = 0.85,
    max_rank: int = 10,                 # hard cap per requirement
    skip_nan_ratio_leq: float = 0.002,  # skip blocks with ~no missing
    skip_low_variance_flag: bool = True,
    workers_per_device: int = 2,        # >1 keeps each GPU busier
    verbose: bool = True,
) -> np.ndarray:
    """
    Blockwise 3D completion with automatic multi-GPU scheduling, VRAM-aware chunking,
    and careful cleanup. volume shape: [Z, Y, X], NaNs mark missing values.
    Returns a float32 volume of the same shape.
    """
    if volume.ndim != 3:
        raise ValueError("volume must be 3D [Z, Y, X].")
    if method.lower() not in {"tucker", "cp"}:
        raise ValueError("method must be 'tucker' or 'cp'.")

    # Ensure float32 for better perf/memory. Keep NaNs.
    vol_np = volume.astype(np.float32, copy=False)

    z_dim, y_dim, x_dim = vol_np.shape
    out_np = np.array(vol_np, copy=True)  # write back in place per block

    devices = discover_cuda_devices()
    if verbose:
        if devices:
            print("CUDA devices (sorted by free VRAM):")
            for d in devices:
                print(
                    f"  cuda:{d.device_id} - {d.name} | free={d.free_bytes/1e9:.2f} GB / total={d.total_bytes/1e9:.2f} GB"
                )
        else:
            print("No CUDA devices detected; running on CPU.")

    bytes_per_voxel = int(np.dtype(np.float32).itemsize)
    planned_blocks, block_edges = plan_blocks(
        (z_dim, y_dim, x_dim),
        bytes_per_voxel,
        devices,
        safety_factor=0.45,
        work_parallelism=max(1, len(devices) * max(1, workers_per_device)),
    )
    if verbose:
        bz, by, bx = block_edges
        print(f"Planned {len(planned_blocks)} blocks with target block size ~ ({bz},{by},{bx}).")

    # Assign a dedicated thread pool per device for load balance
    device_ids: List[Optional[int]] = [d.device_id for d in devices] if devices else [None]
    per_device_pools: Dict[Optional[int], ThreadPoolExecutor] = {
        dev_id: ThreadPoolExecutor(max_workers=max(1, workers_per_device)) for dev_id in device_ids
    }
    device_cycle = cycle(device_ids)

    def process_one_block(block_desc: Block, dev_id: Optional[int]) -> Tuple[Block, np.ndarray]:
        z0, z1, y0, y1, x0, x1 = block_desc.z0, block_desc.z1, block_desc.y0, block_desc.y1, block_desc.x0, block_desc.x1
        block_view = out_np[z0:z1, y0:y1, x0:x1]  # view into output

        # Skip cheap blocks
        # if is_trivially_full_or_empty(block_view, nan_ratio_threshold=skip_nan_ratio_leq):
        #     return block_desc, block_view
        if is_uniform_3d(block_view):
            return block_desc, block_view
        # if skip_low_variance_flag and is_low_variance(block_view):
        #     return block_desc, block_view

        # Pin device in this thread (if any)
        if dev_id is not None:
            torch.cuda.set_device(int(dev_id))
        with (torch.cuda.device(int(dev_id)) if dev_id is not None else nullcontext()):
            # Estimate ranks on the same device (GPU if available)
            rank_z, rank_y, rank_x = estimate_ranks_3d_gpu(
                block_view,
                energy_threshold=energy_threshold,
                max_rank=max_rank,
                device_id=dev_id if dev_id is not None else 0,
            )
            # Complete this block
            completed_block = complete_block_3d(
                block_view,
                method=method,
                ranks_zyx=(rank_z, rank_y, rank_x),
                device_id=dev_id,
            )
        return block_desc, completed_block

    # Dispatch and collect
    futures = {}
    try:
        for idx, blk in enumerate(planned_blocks):
            dev_choice = next(device_cycle)
            fut = per_device_pools[dev_choice].submit(process_one_block, blk, dev_choice)
            futures[fut] = blk

        finished = 0
        total = len(planned_blocks)
        for fut in as_completed(futures):
            blk = futures[fut]
            try:
                returned_blk, completed = fut.result()
                out_np[
                    returned_blk.z0:returned_blk.z1,
                    returned_blk.y0:returned_blk.y1,
                    returned_blk.x0:returned_blk.x1,
                ] = completed
            except Exception as exc:
                LOGGER.exception(
                    "Block z[%d:%d] y[%d:%d] x[%d:%d] failed. %s",
                    blk.z0, blk.z1, blk.y0, blk.y1, blk.x0, blk.x1,
                    exc
                )
                # leave original data in place
            finally:
                finished += 1
                if verbose and (finished % 25 == 0 or finished == total):
                    print(f"Completed {finished}/{total} blocks...")

    finally:
        for pool in per_device_pools.values():
            pool.shutdown(wait=True)
        free_all_cuda_mem()

    # NOTE: keep final return outside try/finally; plainly reachable.
    return out_np
