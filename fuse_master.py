import numpy as np
from scipy.ndimage import gaussian_filter

# ---- Simple caches keyed by blur_sigma signature (guarded) ----
_STACK_CACHE  = globals().get("_STACK_CACHE",  {})
_SORTED_CACHE = globals().get("_SORTED_CACHE", {})

def _sigma_key(blur_sigma):
    if blur_sigma is None:
        return ("none",)
    # Support scalar or iterable sigma
    try:
        it = tuple(float(x) for x in (blur_sigma if np.iterable(blur_sigma) else (blur_sigma,)))
    except Exception:
        it = (float(blur_sigma),)
    return it

def flush_fusion_cache():
    """Clear cached stack and sorted arrays."""
    _STACK_CACHE.clear()
    _SORTED_CACHE.clear()

# ---------- Preprocessing ----------
def normalize_p01(vol, p1=1, p99=99):
    lo, hi = np.percentile(vol, [p1, p99])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    return vol

def _prep_stack(volumes, blur_sigma=None):
    vols = [normalize_p01(v.astype(np.float32, copy=False)) for v in volumes]
    if blur_sigma is not None:
        vols = [gaussian_filter(v, blur_sigma, mode='nearest') for v in vols]
    return np.stack(vols, axis=-1)  # [Z, Y, X, K]

def _get_stack(volumes, blur_sigma):
    key = _sigma_key(blur_sigma)
    cached = _STACK_CACHE.get(key)
    if cached is not None:
        return cached
    stack = _prep_stack(volumes, blur_sigma=blur_sigma)
    _STACK_CACHE[key] = stack
    return stack

def _get_sorted_vals(volumes, blur_sigma):
    key = _sigma_key(blur_sigma)
    cached = _SORTED_CACHE.get(key)
    if cached is not None:
        return cached
    stack = _get_stack(volumes, blur_sigma)
    sorted_vals = np.sort(stack, axis=-1)
    _SORTED_CACHE[key] = sorted_vals
    return sorted_vals

# ---------- Rank-weight helpers ----------
def make_gaussian_rank_weights(K, center_q=0.2, sigma_q=0.08, normalize=True):
    ranks = np.arange(K, dtype=np.float32)
    mu = float(center_q) * (K - 1)
    sigma = max(1e-6, float(sigma_q) * (K - 1))
    w = np.exp(-0.5 * ((ranks - mu) / sigma) ** 2)
    if normalize:
        s = w.sum()
        if s > 0:
            w /= s
    return w.astype(np.float32)

# ---------- Fusion (rank-gaussian) ----------
def fuse_master(
    volumes,
    *,
    blur_sigma=None,
    center_q=0.2,
    sigma_q=0.08,
    debug=False
):
    K = len(volumes)
    rank_weights = make_gaussian_rank_weights(K, center_q=center_q, sigma_q=sigma_q, normalize=True)
    if debug:
        print("rank_weights:", rank_weights)

    sorted_vals = _get_sorted_vals(volumes, blur_sigma)    # [Z,Y,X,K]
    out = np.sum(sorted_vals * rank_weights, axis=-1)
    return out.astype(np.float32)