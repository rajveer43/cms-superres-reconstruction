"""
calo_dataset.py — importable dataset + point-cloud helpers for the CMS
calorimeter super-resolution notebook.

This module exists so that DataLoader worker processes (num_workers > 0) can
*pickle* the Dataset class and its helpers. Classes/functions defined inside a
Jupyter notebook live in __main__ and cannot be unpickled by spawned workers on
macOS — moving them into a real module fixes that, enabling parallel loading.

Nothing here changes the model or loss. It is a faithful copy of the notebook's
point-cloud helpers plus a Dataset that caches the expensive HR point cloud once
per sample so each __getitem__ only does cheap work.
"""

import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Point cloud helpers (copied verbatim in behaviour from the notebook)
# ─────────────────────────────────────────────────────────────────────────────
def image_to_pointcloud(img: np.ndarray,
                        zero_suppression_threshold: float = 1e-6,
                        max_points: int = 1024) -> np.ndarray:
    """Convert a (C, H, W) image to a point cloud (max_points, 2 + C)."""
    C, H, W = img.shape
    eta_coords = np.linspace(-1, 1, H)
    phi_coords = np.linspace(-1, 1, W)
    ETA, PHI = np.meshgrid(eta_coords, phi_coords, indexing='ij')

    energy_sum = img.sum(axis=0)
    mask = energy_sum > zero_suppression_threshold

    eta_active = ETA[mask]
    phi_active = PHI[mask]
    e_active   = img[:, mask].T

    points = np.concatenate([eta_active[:, None],
                             phi_active[:, None],
                             e_active], axis=-1)

    if len(points) >= max_points:
        # Keep highest-energy points (ascontiguousarray: argsort[::-1] is a view)
        order = np.argsort(points[:, 2])[::-1][:max_points]
        points = np.ascontiguousarray(points[order])
    else:
        pad = np.zeros((max_points - len(points), 2 + C), dtype=np.float32)
        points = np.concatenate([points, pad], axis=0)

    return np.ascontiguousarray(points, dtype=np.float32)


def pointcloud_to_image(points: np.ndarray,
                        target_H: int,
                        target_W: int,
                        C: int = 1,
                        aggregation: str = 'sum') -> np.ndarray:
    """Rasterise a point cloud (N, 2+C) -> image (C, H, W) at any resolution."""
    img = np.zeros((C, target_H, target_W), dtype=np.float32)
    count = np.zeros((target_H, target_W), dtype=np.float32)

    eta = points[:, 0]
    phi = points[:, 1]
    energies = points[:, 2:2 + C]

    i_idx = np.clip(np.round((eta + 1) / 2 * (target_H - 1)).astype(int), 0, target_H - 1)
    j_idx = np.clip(np.round((phi + 1) / 2 * (target_W - 1)).astype(int), 0, target_W - 1)

    # Vectorised scatter-add (replaces the per-point Python loop — much faster).
    active = energies.sum(axis=1) != 0
    if active.any():
        flat = i_idx[active] * target_W + j_idx[active]
        for c in range(C):
            np.add.at(img[c].reshape(-1), flat, energies[active, c])
        np.add.at(count.reshape(-1), flat, 1.0)

    if aggregation == 'mean':
        m = count > 0
        img[:, m] /= count[m]

    return img


def apply_zero_suppression(img: np.ndarray, threshold: float) -> np.ndarray:
    """Zero out pixels below hardware threshold."""
    out = img.copy()
    out[out < threshold] = 0.0
    return out


def downsample_via_pointcloud(img: np.ndarray,
                              target_H: int,
                              target_W: int,
                              zero_thresh: float) -> np.ndarray:
    """Downsample (C,H,W) -> (C, target_H, target_W) via point cloud rounding."""
    C, H, W = img.shape
    pc  = image_to_pointcloud(img, zero_suppression_threshold=zero_thresh, max_points=H * W)
    out = pointcloud_to_image(pc, target_H, target_W, C=C, aggregation='sum')
    out = apply_zero_suppression(out, zero_thresh)
    return out


def pad_to_resolution(img: np.ndarray, target_H: int, target_W: int,
                      zero_thresh: float = 0.0) -> np.ndarray:
    """Resize (C, H, W) -> (C, target_H, target_W); downscale or center-pad."""
    C, H, W = img.shape
    if H == target_H and W == target_W:
        return img
    if H > target_H or W > target_W:
        return downsample_via_pointcloud(img, target_H, target_W, zero_thresh=zero_thresh)
    out = np.zeros((C, target_H, target_W), dtype=img.dtype)
    pad_h = (target_H - H) // 2
    pad_w = (target_W - W) // 2
    out[:, pad_h:pad_h + H, pad_w:pad_w + W] = img
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Lazy HDF5 index builder
# ─────────────────────────────────────────────────────────────────────────────
def build_hdf5_index(paths, showers_key='showers', energy_key='incident_energies',
                     chunk=10000):
    """
    Stream the HDF5 files WITHOUT loading showers into RAM. Returns:
      index    : list of (file_path, local_row) for every shower
      e_max    : global max cell energy (computed in chunks)
      energies : (N, 1) float32 incident energies (tiny — kept in RAM for beam_e)

    Only `incident_energies` (200k × 1 ≈ 1.6 MB) is materialised. Showers are
    read block-by-block solely to compute e_max, then discarded.
    """
    import h5py
    index, energy_blocks, e_max = [], [], 0.0
    for path in paths:
        with h5py.File(path, 'r') as f:
            n = f[showers_key].shape[0]
            for r in range(n):
                index.append((path, r))
            # chunked max scan (never holds the whole showers array)
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                block = f[showers_key][start:end]
                m = float(block.max())
                if m > e_max:
                    e_max = m
                del block
            energy_blocks.append(f[energy_key][:].astype(np.float32))
    energies = np.concatenate(energy_blocks, axis=0)
    if e_max <= 0:
        e_max = 1.0
    return index, e_max, energies


def sample_raw_images(index, e_max, n_sample, cell_shape=(1, 45, 144),
                      showers_key='showers', seed=0):
    """Read `n_sample` random raw showers from the HDF5 index for viz/stats.
    Returns (images_norm (n,C,H,W) float32, rows_chosen)."""
    import h5py
    rng = np.random.default_rng(seed)
    n_sample = min(n_sample, len(index))
    chosen = rng.choice(len(index), n_sample, replace=False)
    out = np.empty((n_sample, *cell_shape), dtype=np.float32)
    cache = {}
    for k, gi in enumerate(chosen):
        path, row = index[gi]
        f = cache.get(path) or cache.setdefault(path, h5py.File(path, 'r'))
        out[k] = f[showers_key][row].reshape(cell_shape).astype(np.float32) / e_max
    for f in cache.values():
        f.close()
    return out, chosen


def build_image_cache(index, e_max, hr_res, lr_res, zero_thresh, cache_dir,
                      cell_shape=(1, 45, 144), showers_key='showers',
                      subset_n=None, seed=0, rebuild=False):
    """
    Precompute the *deterministic* HR + LR images once and store them on disk as
    memory-mapped .npy arrays. This removes the ~31 ms/sample point-cloud work
    from __getitem__ (the real training bottleneck) — at load time we only do the
    near-free φ-flip + query sampling.

    Returns dict(hr_path, lr_path, beam_path, rows, hr_res, lr_res, e_max).
    Arrays are float32, memmapped (RAM stays flat; OS pages them in on demand).
    """
    import os, json, h5py
    os.makedirs(cache_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    n_total = len(index)
    if subset_n is not None and subset_n < n_total:
        rows = np.sort(rng.choice(n_total, subset_n, replace=False))
    else:
        rows = np.arange(n_total)
    n = len(rows)

    C = cell_shape[0]
    HR_H, HR_W = hr_res
    LR_H, LR_W = lr_res
    tag = f'{n}_{HR_H}x{HR_W}_{LR_H}x{LR_W}_zt{zero_thresh}'
    hr_path   = os.path.join(cache_dir, f'hr_{tag}.npy')
    lr_path   = os.path.join(cache_dir, f'lr_{tag}.npy')
    beam_path = os.path.join(cache_dir, f'beam_{tag}.npy')
    meta_path = os.path.join(cache_dir, f'meta_{tag}.json')

    if (not rebuild and os.path.exists(hr_path) and os.path.exists(lr_path)
            and os.path.exists(beam_path) and os.path.exists(meta_path)):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f'[cache] reusing existing cache: {hr_path}  (n={meta["n"]})')
        meta['rows'] = np.load(os.path.join(cache_dir, f'rows_{tag}.npy'))
        return meta

    print(f'[cache] building image cache for n={n} samples -> {cache_dir}')
    hr_mm = np.lib.format.open_memmap(hr_path, mode='w+', dtype=np.float32, shape=(n, C, HR_H, HR_W))
    lr_mm = np.lib.format.open_memmap(lr_path, mode='w+', dtype=np.float32, shape=(n, C, LR_H, LR_W))
    beam  = np.empty((n, 1), dtype=np.float32)

    # Group rows by file to read each HDF5 once, in order (fast sequential reads).
    by_file = {}
    for k, gi in enumerate(rows):
        path, r = index[gi]
        by_file.setdefault(path, []).append((k, r))

    t0 = time.time()
    done = 0
    for path, items in by_file.items():
        with h5py.File(path, 'r') as f:
            dset = f[showers_key]
            for k, r in items:
                img = dset[r].reshape(cell_shape).astype(np.float32) / e_max
                hr  = pad_to_resolution(img, HR_H, HR_W, zero_thresh=0.0)
                lr  = downsample_via_pointcloud(hr, LR_H, LR_W, zero_thresh)
                hr_mm[k] = hr
                lr_mm[k] = lr
                done += 1
                if done % 2000 == 0:
                    el = time.time() - t0
                    print(f'  {done}/{n}  ({el:.0f}s, {done/el:.0f}/s)')
    # beam energies for the chosen rows (caller supplies energies separately if wanted)
    hr_mm.flush(); lr_mm.flush()
    np.save(os.path.join(cache_dir, f'rows_{tag}.npy'), rows)

    meta = dict(hr_path=hr_path, lr_path=lr_path, beam_path=beam_path,
                hr_res=list(hr_res), lr_res=list(lr_res), e_max=float(e_max),
                zero_thresh=float(zero_thresh), n=int(n), C=int(C))
    with open(meta_path, 'w') as f:
        json.dump(meta, f)
    meta['rows'] = rows
    print(f'[cache] done in {time.time()-t0:.0f}s  ->  {hr_path}')
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class CalorimeterSRDataset(Dataset):
    """
    Returns per sample:
      lr_pc    : (max_points, 2+C) float32
      hr_img   : (C, HR_H, HR_W) float32
      lr_img   : (C, LR_H, LR_W) float32
      query_pts: (n_query, 2) float32
      query_e  : (n_query, C) float32
      beam_e   : scalar float32

    Memory model: NOTHING is precomputed. Only the raw (N, C, 45, 144) images are
    held (~4 GB for 170k). Everything else is built per-sample inside DataLoader
    workers, so RAM stays flat regardless of dataset size. The earlier
    precompute=True path materialised a full HR point cloud per sample
    (~188 KB × N ≈ 32 GB) which is what exhausted RAM — it is gone.

    The full HR point cloud is never materialised: query points are sampled
    directly from the HR grid via flat indices, so only n_query rows are built.

    Two backends (same output, identical normalisation):
      - in-memory : pass `images` as a dense (N, C, H, W) float32 array.
      - lazy HDF5 : pass `index` = list of (file_path, local_row), plus `e_max`
                    and `cell_shape`. Each __getitem__ reads ONE row from disk
                    (opened lazily per worker), so resident RAM stays ~flat
                    regardless of dataset size. This is the default for the big
                    CaloChallenge files (~20 GB if loaded dense).
    """

    def __init__(self,
                 images: np.ndarray = None,    # (N, C, H, W) float32 normalised (in-memory mode)
                 beam_energies: np.ndarray = None,  # (N, 1) incident energies
                 hr_res: tuple = (125, 125),
                 lr_res: tuple = (64, 64),
                 max_points: int = 512,
                 n_query: int = 512,
                 zero_thresh: float = 1e-3,
                 nonzero_frac: float = 0.5,    # fraction of queries drawn from active (non-zero) HR pixels
                 augment: bool = True,
                 precompute: bool = False,     # kept for API compat; ignored
                 index: list = None,           # [(file_path, local_row), ...] for lazy HDF5
                 e_max: float = None,           # global normalisation constant (lazy mode)
                 cell_shape: tuple = (1, 45, 144),  # (C, H, W) of a raw shower row
                 showers_key: str = 'showers',
                 cache: dict = None,            # meta dict from build_image_cache (fast mode)
                 cache_rows: np.ndarray = None):  # which cache rows this split uses

        self.beam_energies = beam_energies
        self.hr_res        = hr_res
        self.lr_res        = lr_res
        self.max_points    = max_points
        self.n_query       = n_query
        self.zero_thresh   = zero_thresh
        self.nonzero_frac  = nonzero_frac
        self.augment       = augment

        # ── Backend selection ────────────────────────────────────────────────
        self.images      = images
        self.index       = index
        self.e_max       = float(e_max) if e_max is not None else 1.0
        self.cell_shape  = cell_shape
        self.showers_key = showers_key
        self._cache_meta = cache
        self._cached     = cache is not None
        self._lazy       = (images is None) and not self._cached

        if self._cached:
            # Memmapped HR + LR images precomputed by build_image_cache.
            self._hr_path = cache['hr_path']
            self._lr_path = cache['lr_path']
            self.C        = cache['C']
            # cache_rows selects this split's positions within the cache arrays.
            self._crows   = (cache_rows if cache_rows is not None
                             else np.arange(cache['n']))
            self._len     = len(self._crows)
            self._hr_mm   = None   # opened lazily per worker (memmap not always picklable)
            self._lr_mm   = None
        elif self._lazy:
            assert index is not None, 'lazy mode needs an index list'
            self.C = cell_shape[0]
            self._len = len(index)
            self._h5_handles = {}   # per-worker cache of open h5py.File, keyed by path
        else:
            _, C, H, W = images.shape
            self.C = C
            self._len = len(images)

        # Precompute the HR query grid ONCE (shared, tiny: (HR_H*HR_W, 2) ≈ 125 KB).
        HR_H, HR_W = hr_res
        eta = np.linspace(-1, 1, HR_H, dtype=np.float32)
        phi = np.linspace(-1, 1, HR_W, dtype=np.float32)
        ETA, PHI = np.meshgrid(eta, phi, indexing='ij')
        self._grid = np.stack([ETA.ravel(), PHI.ravel()], axis=-1)  # (HR_H*HR_W, 2)
        self._n_full = HR_H * HR_W

    def __len__(self):
        return self._len

    def __getstate__(self):
        # Open h5py handles / memmaps can't be pickled to workers. Send empties;
        # each worker re-opens lazily on first access.
        state = self.__dict__.copy()
        if '_h5_handles' in state:
            state['_h5_handles'] = {}
        state['_hr_mm'] = None
        state['_lr_mm'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if getattr(self, '_lazy', False):
            self._h5_handles = {}
        if getattr(self, '_cached', False):
            self._hr_mm = None
            self._lr_mm = None

    def _get_cache_mm(self):
        if self._hr_mm is None:
            self._hr_mm = np.load(self._hr_path, mmap_mode='r')
            self._lr_mm = np.load(self._lr_path, mmap_mode='r')
        return self._hr_mm, self._lr_mm

    def _sample_query_idx(self, hr_img):
        """Flat indices for query points, OVERSAMPLING non-zero HR pixels.

        The data is ~99% empty, so uniform sampling gives ~2-3 real deposits per
        batch and the model collapses to all-zeros. We draw `nonzero_frac` of the
        queries from active cells (the actual shower) and the rest uniformly, so
        every sample carries real structure to learn. Resolution-invariant (flat
        indices into the HR grid)."""
        energy = hr_img.reshape(self.C, -1).sum(axis=0)        # (HR_H*HR_W,)
        active = np.flatnonzero(energy > 0)
        nq = self.n_query
        n_pos = int(round(nq * self.nonzero_frac))
        if len(active) == 0:
            n_pos = 0
        pos = (np.random.choice(active, n_pos, replace=len(active) < n_pos)
               if n_pos > 0 else np.empty(0, dtype=np.int64))
        rest = np.random.choice(self._n_full, nq - n_pos, replace=False)
        return np.concatenate([pos, rest]).astype(np.int64)

    def _getitem_cached(self, idx):
        """Fast path: HR+LR already on disk. Only flip + pointcloud + query."""
        crow = int(self._crows[idx])
        hr_mm, lr_mm = self._get_cache_mm()
        hr_img = np.array(hr_mm[crow], dtype=np.float32)   # copy out of memmap
        lr_img = np.array(lr_mm[crow], dtype=np.float32)

        grid = self._grid
        if self.augment and random.random() < 0.5:
            # φ-flip: mirror HR, LR (downsample is φ-symmetric) and the query grid.
            hr_img = hr_img[:, :, ::-1].copy()
            lr_img = lr_img[:, :, ::-1].copy()
            grid = grid.copy(); grid[:, 1] = -grid[:, 1]

        lr_pc = image_to_pointcloud(lr_img, zero_suppression_threshold=self.zero_thresh,
                                    max_points=self.max_points)
        q_idx     = self._sample_query_idx(hr_img)
        query_pts = np.ascontiguousarray(grid[q_idx])
        query_e   = np.ascontiguousarray(hr_img.reshape(self.C, -1)[:, q_idx].T)

        return {
            'lr_pc':     torch.from_numpy(lr_pc),
            'hr_img':    torch.from_numpy(np.ascontiguousarray(hr_img)),
            'lr_img':    torch.from_numpy(np.ascontiguousarray(lr_img)),
            'query_pts': torch.from_numpy(query_pts),
            'query_e':   torch.from_numpy(query_e),
            'beam_e':    torch.tensor(float(self.beam_energies[idx, 0]), dtype=torch.float32),
        }

    def _get_h5(self, path):
        """Lazily open (and cache) an HDF5 file handle in THIS process/worker.
        Open handles can't be pickled across workers, so each worker opens its own."""
        import h5py
        h = self._h5_handles.get(path)
        if h is None:
            h = h5py.File(path, 'r')
            self._h5_handles[path] = h
        return h

    def _raw_image(self, idx):
        """Return one normalised (C, H, W) float32 image — from RAM or from disk."""
        if not self._lazy:
            return self.images[idx]
        path, row = self.index[idx]
        f = self._get_h5(path)
        shower = f[self.showers_key][row]                     # reads ONE row from disk
        img = shower.reshape(self.cell_shape).astype(np.float32) / self.e_max
        return img

    def __getitem__(self, idx):
        if self._cached:
            return self._getitem_cached(idx)
        # Build HR image on the fly (cheap: downscale 45×144 -> 125×125).
        hr_img = pad_to_resolution(self._raw_image(idx), *self.hr_res, zero_thresh=0.0)
        hr_img = np.ascontiguousarray(hr_img, dtype=np.float32)

        # φ-flip augmentation: flip image; mirror φ on the shared grid via a copy.
        grid = self._grid
        if self.augment and random.random() < 0.5:
            hr_img = hr_img[:, :, ::-1].copy()
            grid = grid.copy()
            grid[:, 1] = -grid[:, 1]

        # LR image + LR point cloud (the model input).
        lr_img = downsample_via_pointcloud(hr_img, *self.lr_res, self.zero_thresh)
        lr_pc  = image_to_pointcloud(lr_img,
                                     zero_suppression_threshold=self.zero_thresh,
                                     max_points=self.max_points)

        # Sample n_query HR pixels (oversampling non-zero) — no full PC built.
        q_idx     = self._sample_query_idx(hr_img)
        query_pts = np.ascontiguousarray(grid[q_idx])                       # (n_query, 2)
        # query_e = HR energy at those pixels (hr_img is (C, HR_H, HR_W)).
        hr_flat   = hr_img.reshape(self.C, -1)                              # (C, HR_H*HR_W)
        query_e   = np.ascontiguousarray(hr_flat[:, q_idx].T)              # (n_query, C)

        return {
            'lr_pc':     torch.from_numpy(lr_pc),
            'hr_img':    torch.from_numpy(hr_img),
            'lr_img':    torch.from_numpy(lr_img),
            'query_pts': torch.from_numpy(query_pts),
            'query_e':   torch.from_numpy(query_e),
            'beam_e':    torch.tensor(float(self.beam_energies[idx, 0]), dtype=torch.float32),
        }
